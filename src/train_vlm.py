import time
import math
import warnings

warnings.filterwarnings('ignore')
import os
import sys
import torch
import torch.distributed as dist
import yaml
from pathlib import Path

# 项目根目录路径设置
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModel

# 导入项目模块
try:
    # 导入数据集模块
    from src.dataset.vlm_dataset import VLMDataset
except ImportError:
    try:
        # 备选路径
        from dataset.vlm_dataset import VLMDataset
    except ImportError as e:
        print(f"错误: 无法导入VLMDataset: {e}")
        print("请确保src/dataset/vlm_dataset.py文件存在")
        sys.exit(1)

try:
    # 导入实际的VLM模型
    from src.models.vision_encoder import VLM, VLLMconfig
    print("✓ 成功导入VLM模型和配置")
except ImportError as e:
    print(f"错误: 无法导入VLM模型: {e}")
    sys.exit(1)

# 导入配置加载器
try:
    from src.configs.config_loader import load_yaml_config
    from src.configs.training_models import TrainingConfigModel
except ImportError:
    try:
        # 备选路径
        from configs.config_loader import load_yaml_config
        from configs.training_models import TrainingConfigModel
    except ImportError:
        # 如果配置模块不存在，提供简单的YAML加载功能
        def load_yaml_config(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        
        class SimpleConfig:
            def __init__(self, config_dict):
                for key, value in config_dict.items():
                    if isinstance(value, dict):
                        setattr(self, key, SimpleConfig(value))
                    else:
                        setattr(self, key, value)
        
        TrainingConfigModel = SimpleConfig


def Logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def train_epoch(epoch, swanlab_run, config, device="cuda:0"):
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    for step, (X, Y, loss_mask, pixel_values) in enumerate(train_loader):
        X = X.to(device)
        Y = Y.to(device)
        loss_mask = loss_mask.to(device)
        pixel_values = pixel_values.to(device)
        lr = get_lr(epoch * iter_per_epoch + step, config.training.num_epochs * iter_per_epoch, config.training.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with ctx:
            res = model(X, pixel_values=pixel_values)
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())

            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss += res.aux_loss
            loss = loss / config.training.gradient_accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % config.training.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step % config.checkpoints.logging_steps == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    config.training.num_epochs,
                    step,
                    iter_per_epoch,
                    loss.item(),
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            if (swanlab_run is not None) and (not ddp or dist.get_rank() == 0):
                swanlab_run.log({
                    "loss": loss.item(),
                    "lr": optimizer.param_groups[-1]['lr'],
                    "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60,
                    "epoch": epoch + 1,
                    "step": step
                })

        if (step + 1) % config.checkpoints.save_steps == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if config.model.use_moe else ''
            ckp = f'{config.checkpoints.output_dir}/pretrain_vlm_{config.model.hidden_size}{moe_path}.pth'
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            clean_state_dict = {
                key: value for key, value in state_dict.items() if not key.startswith('vision_encoder.')
            }
            clean_state_dict = {k: v.half() for k, v in clean_state_dict.items()}  # 半精度保存
            torch.save(clean_state_dict, ckp)
            model.train()


def init_model(config):
    # 从配置读取模型参数，避免硬编码
    model_config = VLLMconfig(
        dim=getattr(config.model, 'hidden_size', 512),
        n_layers=getattr(config.model, 'num_hidden_layers', 8),
        max_seq_len=getattr(config.model, 'max_seq_len', getattr(config.data, 'max_seq_length', 640)),
        use_moe=getattr(config.model, 'use_moe', False)
    )
    
    # 基于项目根目录处理 tokenizer 路径
    tok_dir = project_root / 'tokenizer'
    if not tok_dir.exists():
        raise FileNotFoundError(f"Tokenizer 目录不存在: {tok_dir}")
    
    tokenizer = AutoTokenizer.from_pretrained(str(tok_dir), use_fast=True)
    
    # 处理视觉模型路径 - 支持配置或默认值
    vision_model_path = getattr(config.model, 'vision_model_path', 'src/models/vision_model')
    vision_path = Path(vision_model_path)
    if not vision_path.is_absolute():
        vision_path = project_root / vision_path
    vision_model_path = str(vision_path.resolve())
    
    moe_path = '_moe' if model_config.use_moe else ''
    
    # 基于项目根目录处理权重文件路径
    output_dir = Path(config.checkpoints.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    ckp = output_dir / f'llm_{model_config.dim}{moe_path}.pth'
    
    model = VLM(model_config, vision_model_path=vision_model_path)
    
    # 如果权重文件存在则加载
    if ckp.exists():
        state_dict = torch.load(str(ckp), map_location="cuda:0" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"警告: 权重文件 {ckp} 不存在，使用随机初始化")

    # 应用冻结策略
    freeze_llm = getattr(config, 'vlm_specific', None) and getattr(config.vlm_specific, 'freeze_llm', True)
    freeze_vision = getattr(config, 'vlm_specific', None) and getattr(config.vlm_specific, 'freeze_vision_encoder', True)
    trainable_modules = getattr(config, 'vlm_specific', None) and getattr(config.vlm_specific, 'trainable_modules', ['vision_proj'])
    
    if freeze_llm or freeze_vision:
        for name, param in model.named_parameters():
            should_freeze = True
            if trainable_modules:
                for module_name in trainable_modules:
                    if module_name in name:
                        should_freeze = False
                        break
            param.requires_grad = not should_freeze

    Logger(f'zerollm-v VLM可训练参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    _, preprocess = model.vision_encoder, model.processor
    return model.to(device), tokenizer, preprocess, model_config


def init_distributed_mode():
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


def load_config(config_path: str = "../configs/vlm_training.yaml"):
    """加载训练配置"""
    try:
        config_dict = load_yaml_config(config_path)
        # 使用pydantic模型验证配置
        return TrainingConfigModel(**config_dict)
    except Exception as e:
        print(f"错误: 无法加载配置文件 {config_path}: {e}")
        print("请确保配置文件存在且格式正确")
        sys.exit(1)


def init_swanlab(config):
    """初始化SwanLab监控"""
    try:
        import swanlab
        
        # 创建实验名称
        if hasattr(config.swanlab, 'experiment_name') and config.swanlab.experiment_name:
            experiment_name = config.swanlab.experiment_name
        else:
            experiment_name = f"zerollm-v-vlm-{config.training.num_epochs}epochs-bs{config.data.batch_size}-lr{config.training.learning_rate}"
        
        # 安全读取配置，使用 getattr 防止属性缺失
        swanlab_config = {
            "model_type": getattr(config.model, 'model_type', 'vlm'),
            "hidden_size": getattr(config.model, 'hidden_size', 512),
            "num_layers": getattr(config.model, 'num_hidden_layers', 8),
            "batch_size": config.data.batch_size,
            "learning_rate": config.training.learning_rate,
            "num_epochs": config.training.num_epochs,
            "max_seq_len": getattr(config.model, 'max_seq_len', getattr(config.data, 'max_seq_length', 640)),
            "use_moe": getattr(config.model, 'use_moe', False)
        }
        
        run = swanlab.init(
            project=config.swanlab.project,
            experiment_name=experiment_name,
            description=getattr(config.swanlab, 'description', ''),
            config=swanlab_config,
            logdir=getattr(config.swanlab, 'logdir', None)
        )
        
        print(f"SwanLab 初始化成功: {experiment_name}")
        return run
        
    except ImportError:
        print("警告: 未安装swanlab，跳过实验监控")
        return None
    except Exception as e:
        print(f"警告: SwanLab初始化失败: {e}")
        return None

if __name__ == "__main__":
    # 加载配置文件 - 支持位置参数和 --config 选项
    import argparse
    parser = argparse.ArgumentParser(description="VLM训练脚本")
    parser.add_argument('config_path', nargs='?', default=None, help="配置文件路径（位置参数）")
    parser.add_argument("--config", dest='config', default=None, help="配置文件路径")
    args = parser.parse_args()
    
    config_path = args.config or args.config_path or 'configs/vlm_training.yaml'
    
    print(f"zerollm-v VLM 训练启动 - 加载配置: {config_path}")
    config = load_config(config_path)
    
    # 设置随机种子
    if hasattr(config, 'system') and hasattr(config.system, 'random_seed'):
        torch.manual_seed(config.system.random_seed)
    else:
        torch.manual_seed(1337)
    
    # 设置输出目录
    os.makedirs(config.checkpoints.output_dir, exist_ok=True)
    
    # 设置设备
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device_type = "cuda" if "cuda" in device else "cpu"
    
    # 设置上下文
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()
    
    # 分布式训练设置
    ddp = int(os.environ.get("RANK", -1)) != -1
    ddp_local_rank, DEVICE = 0, device
    if ddp:
        init_distributed_mode()
        device = torch.device(DEVICE)
    
    # 初始化SwanLab监控
    swanlab_run = None
    if not ddp or ddp_local_rank == 0:
        swanlab_run = init_swanlab(config)
    
    # 初始化模型
    model, tokenizer, preprocess, model_config = init_model(config)
    
    # 处理图像路径 - 优先使用配置中的 images_path
    images_path = getattr(config.data, 'images_path', None)
    if not images_path:
        # 回退到智能推断
        images_path = config.data.data_path.replace('pretrain_data.jsonl', 'pretrain_images').replace('debug_data.jsonl', 'debug_images')
    
    # 确保图像路径基于项目根目录
    img_path = Path(images_path)
    if not img_path.is_absolute():
        img_path = project_root / img_path
    train_ds = VLMDataset(
        config.data.data_path, 
        str(img_path), 
        tokenizer, 
        preprocess=preprocess,
        image_special_token=getattr(config, 'vlm_specific', None) and getattr(config.vlm_specific, 'image_special_token', '<image>') or '<image>',
        max_length=config.data.max_seq_length
    )
    
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=config.data.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=config.data.num_workers,
        sampler=train_sampler
    )
    
    # 安全检查：防止空数据集
    iter_per_epoch = len(train_loader)
    if iter_per_epoch == 0:
        raise ValueError(f'训练集为空: {config.data.data_path}')
    
    # AMP 和 CUDA 兼容性处理
    use_amp = getattr(config.training, 'use_amp', True) and torch.cuda.is_available()
    dtype_mapping = {'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float32': torch.float32}
    amp_dtype = dtype_mapping.get(getattr(config.training, 'dtype', 'float16'), torch.float16)
    
    # 混合精度设置
    ctx = nullcontext() if not use_amp else torch.cuda.amp.autocast(dtype=amp_dtype)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    # 优化器设置
    optimizer_params = {
        'lr': config.training.learning_rate,
        'weight_decay': getattr(config.training, 'weight_decay', 0.01),
        'betas': (getattr(config.training, 'beta1', 0.9), getattr(config.training, 'beta2', 0.999)),
        'eps': getattr(config.training, 'eps', 1e-8)
    }
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        **optimizer_params
    )
    
    # 分布式训练设置
    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])
    
    print(f"\n开始 zerollm-v VLM 训练:")
    print(f"- 训练轮数: {config.training.num_epochs}")
    print(f"- 批处理大小: {config.data.batch_size}")
    print(f"- 学习率: {config.training.learning_rate}")
    print(f"- 每轮步数: {iter_per_epoch}")
    print(f"- 输出目录: {config.checkpoints.output_dir}\n")
    
    for epoch in range(config.training.num_epochs):
        Logger(f"\n=== Epoch {epoch + 1}/{config.training.num_epochs} ====")
        train_epoch(epoch, swanlab_run, config, device)
    
    print("\nzerollm-v VLM 训练完成!")
    if swanlab_run:
        swanlab_run.finish()