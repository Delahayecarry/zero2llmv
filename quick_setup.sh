#!/bin/bash
# zerollm-v 快速配置脚本 (在现有项目中运行)
# 适用于已克隆项目，只需下载数据和模型

set -Eeuo pipefail
IFS=$'\n\t'

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "\n${YELLOW}=== $1 ===${NC}"; }

DRY_RUN="${DRY_RUN:-}"

# 下载文件函数
download_with_progress() {
    local url="$1"
    local output="$2"
    local name="$3"
    
    if [ -f "$output" ]; then
        log_info "$name 已存在，跳过下载"
        return 0
    fi
    
    log_info "下载 $name..."
    if [ -n "$DRY_RUN" ]; then
        log_info "[DRY_RUN] 将下载: $url -> $output"
        mkdir -p "$(dirname "$output")"
        : > "$output" 2>/dev/null || true
        return 0
    fi
    mkdir -p "$(dirname "$output")"
    
    if curl -L --progress-bar "$url" -o "$output"; then
        log_success "$name 下载完成"
    else
        log_error "$name 下载失败"
        return 1
    fi
}

echo -e "${BLUE}🚀 zerollm-v 快速配置脚本${NC}\n"

# 1. 创建必要目录（与训练脚本/配置一致）
log_step "创建目录结构"
mkdir -p dataset model/vision_model/clip-vit-base-patch16
log_success "目录创建完成"

# 2. 下载训练数据
log_step "下载训练数据"
download_with_progress \
    "https://www.modelscope.cn/datasets/gongjy/minimind-v_dataset/resolve/master/pretrain_vlm_data.jsonl" \
    "dataset/pretrain_data.jsonl" \
    "训练数据"

download_with_progress \
    "https://www.modelscope.cn/datasets/gongjy/minimind-v_dataset/resolve/master/pretrain_images.zip" \
    "dataset/pretrain_images.zip" \
    "图像压缩包"

# 解压图像数据
if [ -f "dataset/pretrain_images.zip" ] && [ ! -d "dataset/pretrain_images" ]; then
    if [ -n "$DRY_RUN" ]; then
        log_info "[DRY_RUN] 将解压: dataset/pretrain_images.zip -> dataset/pretrain_images/"
    else
        log_info "解压图像数据..."
        (cd dataset && unzip -q pretrain_images.zip)
        log_success "图像数据解压完成"
    fi
fi

# 3. 下载CLIP模型
log_step "下载CLIP视觉模型"
base_url="https://huggingface.co/openai/clip-vit-base-patch16/resolve/main"
model_dir="model/vision_model/clip-vit-base-patch16"

files=("config.json" "preprocessor_config.json" "pytorch_model.bin" \
       "tokenizer_config.json" "tokenizer.json" "vocab.json" \
       "merges.txt" "special_tokens_map.json")

for file in "${files[@]}"; do
    download_with_progress "$base_url/$file" "$model_dir/$file" "CLIP: $file"
done

# 4. 配置环境
log_step "配置环境"
if [ -n "${SWANLAB_API_KEY:-}" ]; then
  export SWANLAB_API_KEY="$SWANLAB_API_KEY"
  log_info "已设置 SWANLAB_API_KEY（来自环境变量）"
else
  log_info "未提供 SWANLAB_API_KEY（可选）"
fi
if [ -n "$DRY_RUN" ]; then
  log_info "[DRY_RUN] 将写入 .env 和 PYTHONPATH"
fi
{
  [ -n "${SWANLAB_API_KEY:-}" ] && echo "SWANLAB_API_KEY=$SWANLAB_API_KEY"
  echo "PYTHONPATH=\${PYTHONPATH}:$(pwd)/src"
} > .env
log_success "环境变量配置完成"

# 5. 验证文件
log_step "验证文件完整性"
required_files=(
    "dataset/pretrain_data.jsonl"
    "dataset/pretrain_images"
    "model/vision_model/clip-vit-base-patch16/pytorch_model.bin"
    "configs/vlm_training.yaml"
)

all_good=true
for file in "${required_files[@]}"; do
    if [ -e "$file" ]; then
        log_success "✓ $file"
    else
        log_error "✗ $file 缺失"
        all_good=false
    fi
done

if [ "$all_good" = true ]; then
    echo -e "\n${GREEN}🎉 配置完成！现在可以开始训练：${NC}"
    echo -e "${BLUE}source .env && python src/train_vlm.py configs/vlm_training.yaml${NC}\n"
else
    echo -e "\n${RED}❌ 配置不完整，请检查错误信息${NC}\n"
fi
