#!/bin/bash
# 快速环境检查脚本 - check_env.sh
# 在云服务器上运行前的环境验证

echo "=== 环境检查脚本 ==="

# 1. 检查Python版本
echo "1. Python环境:"
python --version || python3 --version || echo "Python未安装"
echo ""

# 2. 检查GPU
echo "2. GPU信息:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "NVIDIA GPU不可用或未安装nvidia-smi"
echo ""

# 3. 检查CUDA
echo "3. CUDA环境:"
nvcc --version 2>/dev/null | grep "release" || echo "CUDA未安装"
echo ""

# 4. 检查必要的Python包
echo "4. 关键依赖包检查:"
python3 -c "
import sys
packages = ['torch', 'torchvision', 'transformers', 'yaml', 'PIL']
for pkg in packages:
    try:
        __import__(pkg)
        print(f'✓ {pkg}')
    except ImportError:
        print(f'✗ {pkg} - 未安装')
"
echo ""

# 5. 检查PyTorch CUDA支持
echo "5. PyTorch CUDA支持:"
python3 -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU数量: {torch.cuda.device_count()}')
    print(f'当前GPU: {torch.cuda.get_device_name()}')
"
echo ""

echo "环境检查完成"