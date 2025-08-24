"""
适配JSONL数据格式的训练器
"""

import argparse
import time
import math
import warnings

warnings.filterwarnings('ignore')
import os
import sys
import torch
import torch.distributed as dist

from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer
from vlm_dataset import create_vlm_dataset


def Logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def train_epoch(epoch, wandb):
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    for step, (X, Y, loss_mask, pixel_values) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)
        pixel_values = pixel_values.to(args.device)
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with ctx:
            # 注意：这里需要根据你的实际模型接口调整
            # res = model(X, pixel_values=pixel_values)
            # 由于我们没有实际的模型，这里用伪代码表示
            logits = torch.randn(X.shape[0], X.shape[1], 50257, device=args.device)  # 示例logits
            aux_loss = torch.tensor(0.0, device=args.device)  # 示例辅助损失
            
            loss = loss_fct(
                logits.view(-1, logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())

            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss += aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item(),
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            # 保存模型代码 - 需要根据实际模型调整
            Logger(f"保存检查点到 {args.save_dir}")
            # model.eval()
            # ckp = f'{args.save_dir}/pretrain_vlm_{args.hidden_size}.pth'
            # torch.save(model.state_dict(), ckp)
            # model.train()


def init_model():
    """初始化模型和tokenizer - 需要根据你的实际模型调整"""
    
    # 加载tokenizer
    tokenizer_path = getattr(args, 'tokenizer_path', 'tokenizer')
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
        Logger(f'✅ 加载自定义tokenizer: {tokenizer_path}')
    except Exception as e:
        Logger(f'⚠️  加载自定义tokenizer失败: {e}')
        Logger('使用GPT2 tokenizer作为备选')
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        special_tokens = {
            "pad_token": "<pad>",
            "additional_special_tokens": ["<|user|>", "<|assistant|>", "<|endoftext|>", "<image>"]
        }
        tokenizer.add_special_tokens(special_tokens)
    
    Logger(f'Tokenizer词汇表大小: {tokenizer.vocab_size}')
    
    # 这里需要根据你的实际模型来初始化
    # model = MiniMindVLM(model_config, vision_model_path="...")
    # 由于没有实际模型，创建一个简单的占位符
    class DummyModel(nn.Module):
        def __init__(self, vocab_size):
            super().__init__()
            self.linear = nn.Linear(vocab_size, vocab_size)
            
        def parameters(self):
            return self.linear.parameters()
            
        def train(self):
            self.linear.train()
            
        def eval(self):
            self.linear.eval()
    
    model = DummyModel(tokenizer.vocab_size)
    Logger(f'模型参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.3f} 百万')
    
    return model.to(args.device), tokenizer


def init_distributed_mode():
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind-V JSONL Pretrain")
    parser.add_argument("--out_dir", type=str, default="out")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)  # 减小batch_size用于测试
    parser.add_argument("--learning_rate", type=float, default=4e-4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", default=False, action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-V-JSONL")
    parser.add_argument("--num_workers", type=int, default=2)  # 减少worker数量
    
    # 数据路径参数
    parser.add_argument("--data_path", type=str, default="test_data/test_data.jsonl")
    parser.add_argument("--images_path", type=str, default="test_data/images")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer")
    
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=1)  # 更频繁的日志
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=128, type=int)  # 减小序列长度用于测试
    
    args = parser.parse_args()

    max_seq_len = args.max_seq_len
    args.save_dir = args.out_dir
    os.makedirs(args.save_dir, exist_ok=True)
    
    torch.manual_seed(1337)
    device_type = "cuda" if "cuda" in args.device else "cpu"

    args.wandb_run_name = f"MiniMind-V JSONL-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"

    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()
    ddp = int(os.environ.get("RANK", -1)) != -1
    ddp_local_rank, DEVICE = 0, "cuda:0"
    
    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)

    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    # 初始化模型和tokenizer
    model, tokenizer = init_model()

    # 创建数据集 - 使用我们的JSONL数据集
    train_ds = create_vlm_dataset(
        data_path=args.data_path,
        images_path=args.images_path,
        tokenizer_path=args.tokenizer_path,
        max_length=max_seq_len
    )
    
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False if ddp else True,
        num_workers=args.num_workers,
        sampler=train_sampler
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    if ddp:
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    iter_per_epoch = len(train_loader)
    Logger(f"每个epoch有 {iter_per_epoch} 个batch")
    
    for epoch in range(args.epochs):
        Logger(f"开始训练 Epoch {epoch + 1}/{args.epochs}")
        train_epoch(epoch, wandb)
        
    Logger("训练完成！")