#!/bin/bash
# 调试训练测试脚本 - debug_training.sh
# 用于验证VLM训练流程是否正常工作

set -e  # 出现错误时退出

echo "=== Zero2LLMV 调试训练测试脚本 ==="
echo "开始时间: $(date)"
echo ""

# 1. 检查Python环境
echo "1. 检查Python环境..."
python --version
echo "PyTorch版本:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}')"
echo ""

# 2. 检查项目依赖
echo "2. 检查关键依赖..."
python -c "
try:
    import torch, torchvision, transformers, yaml, swanlab
    print('✓ 核心依赖检查通过')
except ImportError as e:
    print(f'✗ 依赖检查失败: {e}')
    exit(1)
"
echo ""

# 3. 创建调试数据目录
echo "3. 创建调试数据目录..."
mkdir -p dataset/debug_images
mkdir -p debug_out
mkdir -p debug_logs
echo "✓ 目录创建完成"
echo ""

# 4. 创建minimal测试数据
echo "4. 创建测试数据..."
cat > dataset/debug_data.jsonl << 'EOF'
{"image": "test1.jpg", "text": "这是一个测试图片"}
{"image": "test2.jpg", "text": "另一个测试样本"}
{"image": "test3.jpg", "text": "第三个测试数据"}
{"image": "test4.jpg", "text": "最后一个测试"}
EOF

# 创建简单的测试图片（纯色图）
python -c "
from PIL import Image
import numpy as np
import os

os.makedirs('dataset/debug_images', exist_ok=True)
for i, color in enumerate([(255,0,0), (0,255,0), (0,0,255), (255,255,0)], 1):
    img = Image.new('RGB', (224, 224), color)
    img.save(f'dataset/debug_images/test{i}.jpg')
print('✓ 测试图片创建完成')
"
echo ""

# 5. 验证配置文件
echo "5. 验证配置文件..."
if [ -f "configs/debug_training.yaml" ]; then
    echo "✓ 调试配置文件存在"
    python -c "
import yaml
with open('configs/debug_training.yaml', 'r') as f:
    config = yaml.safe_load(f)
    print(f'✓ 配置文件解析成功，模型类型: {config[\"model\"][\"model_type\"]}')
"
else
    echo "✗ 调试配置文件不存在"
    exit 1
fi
echo ""

# 6. 检查模型文件
echo "6. 检查必要的模型文件..."
if [ -d "src/models/vision_model" ]; then
    echo "✓ 视觉模型目录存在"
else
    echo "⚠ 视觉模型目录不存在，训练时可能需要下载"
fi
echo ""

# 7. 运行干燥测试（dry run）
echo "7. 执行训练流程干燥测试..."
echo "运行命令: python src/train_vlm.py --config configs/debug_training.yaml --dry-run"
echo ""

# 如果用户同意，执行实际训练
read -p "是否执行实际的调试训练？(y/N): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "8. 开始调试训练..."
    echo "注意：这将运行最多100步的训练"
    
    # 执行训练
    python src/train_vlm.py --config configs/debug_training.yaml
    
    echo ""
    echo "=== 训练完成 ==="
    echo "检查输出目录: debug_out/"
    echo "检查日志目录: debug_logs/"
    ls -la debug_out/ 2>/dev/null || echo "输出目录为空或不存在"
else
    echo "跳过实际训练"
fi

echo ""
echo "=== 调试测试脚本执行完成 ==="
echo "结束时间: $(date)"