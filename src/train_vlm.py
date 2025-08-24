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
    # 尝试导入模型模块（需要用户提供）
    from model.model_vlm import MiniMindVLM, VLMConfig
    from dataset.lm_dataset import VLMDataset
except ImportError as e:
    print(f"警告: 无法导入模型模块: {e}")
    print("请确保model.model_vlm和dataset.lm_dataset模块可用")
    sys.exit(1)

# 导入配置加载器
try:
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


def train_epoch(epoch, swanlab_run, config):
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    for step, (X, Y, loss_mask, pixel_values) in enumerate(train_loader):
        X = X.to(config.training.device)
        Y = Y.to(config.training.device)
        loss_mask = loss_mask.to(config.training.device)
        pixel_values = pixel_values.to(config.training.device)
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
            loss = loss / config.training.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % config.training.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step % config.checkpoints.log_interval == 0:
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

        if (step + 1) % config.checkpoints.save_interval == 0 and (not ddp or dist.get_rank() == 0):
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
    model_config = VLMConfig(
        hidden_size=config.model.hidden_size, 
        num_hidden_layers=config.model.num_hidden_layers,
        max_seq_len=config.model.max_seq_len,
        use_moe=config.model.use_moe
    )
    
    tokenizer = AutoTokenizer.from_pretrained('../model', use_fast=True)
    moe_path = '_moe' if model_config.use_moe else ''
    # 加载纯语言模型权重
    ckp = f'{config.model.llm_weights_dir}/llm_{model_config.hidden_size}{moe_path}.pth'
    model = MiniMindVLM(model_config, vision_model_path=config.model.vision_model_path)
    state_dict = torch.load(ckp, map_location=config.training.device)
    model.load_state_dict(state_dict, strict=False)

    # 冻结除 vision_proj 外的所有参数
    for name, param in model.named_parameters():
        if 'vision_proj' not in name:
            param.requires_grad = False

    Logger(f'zerollm-v VLM可训练参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')

    _, preprocess = model.vision_encoder, model.processor
    return model.to(config.training.device), tokenizer, preprocess, model_config


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
        # 使用SimpleConfig如果没有pydantic验证
        if hasattr(TrainingConfigModel, '__call__'):
            return TrainingConfigModel(config_dict)
        else:
            # 使用验证模型
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
        
        # 初始化SwanLab运行
        swanlab_config = {
            "model_type": config.model.model_type,
            "hidden_size": config.model.hidden_size,
            "num_layers": config.model.num_hidden_layers,
            "batch_size": config.data.batch_size,
            "learning_rate": config.training.learning_rate,
            "num_epochs": config.training.num_epochs,
            "max_seq_len": config.model.max_seq_len,
            "use_moe": config.model.use_moe
        }
        
        run = swanlab.init(
            project=config.swanlab.project,
            experiment_name=experiment_name,
            description=config.swanlab.description,
            config=swanlab_config,
            logdir=config.swanlab.logdir if hasattr(config.swanlab, 'logdir') else None
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
    # 加载配置文件
    config_path = "../configs/vlm_training.yaml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
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
    device_type = "cuda" if "cuda" in config.training.device else "cpu"
    
    # 设置上下文
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()
    
    # 分布式训练设置
    ddp = int(os.environ.get("RANK", -1)) != -1
    ddp_local_rank, DEVICE = 0, config.training.device
    if ddp:
        init_distributed_mode()
        config.training.device = torch.device(DEVICE)
    
    # 初始化SwanLab监控
    swanlab_run = None
    if not ddp or ddp_local_rank == 0:
        swanlab_run = init_swanlab(config)
    
    # 初始化模型
    model, tokenizer, preprocess, model_config = init_model(config)
    
    # 创建数据集
    train_ds = VLMDataset(
        config.data.data_path, 
        config.data.images_path, 
        tokenizer, 
        preprocess=preprocess,
        image_special_token=getattr(config.vlm_specific, 'image_special_token', '<image>') if hasattr(config, 'vlm_specific') else '<image>',
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
    
    # 初始化优化器和混合精度
    scaler = torch.cuda.amp.GradScaler(enabled=(config.training.dtype in ['float16', 'bfloat16']))
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=config.training.learning_rate,
        weight_decay=getattr(config.training, 'weight_decay', 0.01)
    )
    
    # 分布式训练设置
    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])
    
    # 开始训练
    iter_per_epoch = len(train_loader)
    print(f"\n\u5f00始 zerollm-v VLM 训练:")
    print(f"- 训练轮数: {config.training.num_epochs}")
    print(f"- 批处理大小: {config.data.batch_size}")
    print(f"- 学习率: {config.training.learning_rate}")
    print(f"- 每轮步数: {iter_per_epoch}")
    print(f"- 输出目录: {config.checkpoints.output_dir}\n")
    
    for epoch in range(config.training.num_epochs):
        Logger(f"\n=== Epoch {epoch + 1}/{config.training.num_epochs} ====")
        train_epoch(epoch, swanlab_run, config)
    
    print("\nzerollm-v VLM 训练完成!")
    if swanlab_run:
        swanlab_run.finish()