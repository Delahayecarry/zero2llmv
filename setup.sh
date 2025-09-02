#!/bin/bash
# zerollm-v 一键部署脚本
# 自动下载数据、模型并配置环境

set -Eeuo pipefail  # 严格错误处理
IFS=$'\n\t'

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 全局变量
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="zerollm-v"
# 可选：通过环境变量传入，不再在脚本中硬编码
SWANLAB_API_KEY="${SWANLAB_API_KEY:-}"
# 支持干跑模式：设置 DRY_RUN=1 跳过实际下载与写操作
DRY_RUN="${DRY_RUN:-}"

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "\n${PURPLE}=== $1 ===${NC}"
}

# 进度条函数
show_progress() {
    local duration=$1
    local message="$2"
    local progress=0
    local bar_length=50
    
    echo -ne "${CYAN}$message${NC}"
    while [ $progress -le $duration ]; do
        local filled=$((progress * bar_length / duration))
        local empty=$((bar_length - filled))
        printf "\r${CYAN}$message${NC} ["
        printf "%*s" $filled | tr ' ' '='
        printf "%*s" $empty | tr ' ' '-'
        printf "] %d%%" $((progress * 100 / duration))
        sleep 0.1
        ((progress++))
    done
    echo ""
}

# 检查命令是否存在
check_command() {
    if ! command -v "$1" &> /dev/null; then
        log_error "$1 未安装，请先安装 $1"
        exit 1
    fi
}

# 环境检查
check_environment() {
    log_step "环境检查"
    
    # 检查Python版本
    log_info "检查Python版本..."
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
        log_info "Python版本: $PYTHON_VERSION"
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
            log_success "Python版本符合要求 (>=3.10)"
        else
            log_error "Python版本太低，需要 >= 3.10，当前版本: $PYTHON_VERSION"
            exit 1
        fi
    else
        log_error "未找到python3，请先安装Python 3.10+"
        exit 1
    fi
    
    # 检查必要命令
    log_info "检查必要命令..."
    check_command "git"
    check_command "curl"
    check_command "unzip"
    
    # 检查uv或pip
    if command -v uv &> /dev/null; then
        log_success "找到 uv 包管理器"
        PACKAGE_MANAGER="uv"
    elif command -v pip3 &> /dev/null || command -v pip &> /dev/null; then
        log_success "找到 pip 包管理器"
        PACKAGE_MANAGER="pip"
    else
        log_error "未找到包管理器 (uv 或 pip)"
        exit 1
    fi
    
    # 检查网络连接（针对实际使用域名）
    log_info "检查网络连接..."
    local ok=false
    for url in "https://www.modelscope.cn" "https://huggingface.co"; do
        if curl -s --connect-timeout 5 "$url" > /dev/null; then
            ok=true; break
        fi
    done
    if [ "$ok" = true ]; then
        log_success "网络连接正常（可访问模型源）"
    else
        log_warning "可能无法访问模型/数据源，将尝试继续"
    fi
    
    log_success "环境检查完成"
}

# 克隆或进入项目
setup_project() {
    log_step "项目设置"
    
    if [ -d "$PROJECT_NAME" ]; then
        log_info "项目目录已存在，进入项目目录"
        cd "$PROJECT_NAME"
    else
        log_info "克隆项目..."
        if [ -n "$DRY_RUN" ]; then
            log_info "[DRY_RUN] 将执行: git clone https://github.com/Delahayecarry/zero2llmv.git $PROJECT_NAME"
            mkdir -p "$PROJECT_NAME"
        else
            git clone https://github.com/Delahayecarry/zero2llmv.git "$PROJECT_NAME"
        fi
        log_success "项目克隆完成"
        cd "$PROJECT_NAME"
    fi
    
    log_info "当前目录: $(pwd)"
    log_success "项目设置完成"
}

# 安装依赖
install_dependencies() {
    log_step "安装依赖"
    
    if [ "$PACKAGE_MANAGER" = "uv" ]; then
        log_info "使用 uv 安装依赖..."
        if [ -n "$DRY_RUN" ]; then
            log_info "[DRY_RUN] 将执行: uv sync 或 uv pip install -e ."
        elif [ -f "uv.lock" ]; then
            uv sync
        else
            uv pip install -e .
        fi
    else
        log_info "使用 pip 安装依赖..."
        if [ -n "$DRY_RUN" ]; then
            log_info "[DRY_RUN] 将执行: pip3 install -e ."
        else
            pip3 install -e .
        fi
    fi
    
    log_success "依赖安装完成"
}

# 下载文件函数
download_file() {
    local url="$1"
    local output="$2"
    local description="$3"
    
    if [ -f "$output" ]; then
        log_info "$description 已存在，跳过下载"
        return 0
    fi
    
    log_info "下载 $description..."
    if [ -n "$DRY_RUN" ]; then
        log_info "[DRY_RUN] 将下载: $url -> $output"
        mkdir -p "$(dirname "$output")"
        : > "$output" 2>/dev/null || true
        return 0
    fi
    mkdir -p "$(dirname "$output")"
    
    # 使用curl下载，显示进度
    if curl -L --progress-bar "$url" -o "$output"; then
        log_success "$description 下载完成"
    else
        log_error "$description 下载失败"
        return 1
    fi
}

# 下载训练数据
download_datasets() {
    log_step "下载训练数据"
    
    # 目标目录与训练脚本/配置保持一致
    # 期望路径：dataset/pretrain_data.jsonl 和 dataset/pretrain_images/
    mkdir -p dataset
    
    # 下载训练数据文件
    download_file \
        "https://www.modelscope.cn/datasets/gongjy/minimind-v_dataset/resolve/master/pretrain_vlm_data.jsonl" \
        "dataset/pretrain_data.jsonl" \
        "训练数据文件"
    
    # 下载图像数据压缩包
    download_file \
        "https://www.modelscope.cn/datasets/gongjy/minimind-v_dataset/resolve/master/pretrain_images.zip" \
        "dataset/pretrain_images.zip" \
        "图像数据压缩包"
    
    # 解压图像数据
    if [ -f "dataset/pretrain_images.zip" ] && [ ! -d "dataset/pretrain_images" ]; then
        if [ -n "$DRY_RUN" ]; then
            log_info "[DRY_RUN] 将解压: dataset/pretrain_images.zip -> dataset/pretrain_images/"
            mkdir -p dataset/pretrain_images
        else
            log_info "解压图像数据..."
            (cd dataset && unzip -q pretrain_images.zip)
            log_success "图像数据解压完成"
        fi
    else
        log_info "图像数据已存在，跳过解压"
    fi
    
    log_success "训练数据下载完成"
}

# 下载视觉模型
download_vision_model() {
    log_step "下载视觉模型"
    
    # 目标路径与训练脚本保持一致：model/vision_model/clip-vit-base-patch16
    local model_dir="model/vision_model/clip-vit-base-patch16"
    mkdir -p "$model_dir"
    
    # CLIP模型文件列表
    local base_url="https://huggingface.co/openai/clip-vit-base-patch16/resolve/main"
    local files=(
        "config.json"
        "preprocessor_config.json"
        "pytorch_model.bin"
        "tokenizer_config.json"
        "tokenizer.json"
        "vocab.json"
        "merges.txt"
        "special_tokens_map.json"
    )
    
    log_info "下载CLIP视觉模型文件..."
    for file in "${files[@]}"; do
        download_file \
            "$base_url/$file" \
            "$model_dir/$file" \
            "CLIP模型文件: $file"
    done
    
    log_success "视觉模型下载完成"
}

# 配置环境变量
setup_environment() {
    log_step "配置环境变量"
    
    # 设置SwanLab API Key（如果外部传入）
    if [ -n "${SWANLAB_API_KEY}" ]; then
        export SWANLAB_API_KEY="$SWANLAB_API_KEY"
        log_info "已设置 SWANLAB_API_KEY 环境变量"
    else
        log_info "未提供 SWANLAB_API_KEY（可选）"
    fi
    
    # 创建.env文件
    if [ -n "$DRY_RUN" ]; then
        log_info "[DRY_RUN] 将写入 .env 和 PYTHONPATH"
    fi
    cat > .env << EOF
# zerollm-v 环境配置
$( [ -n "$SWANLAB_API_KEY" ] && echo "SWANLAB_API_KEY=$SWANLAB_API_KEY" )
PYTHONPATH=\${PYTHONPATH}:$(pwd)/src
EOF
    
    log_success "环境变量配置完成"
    log_info "环境配置已保存到 .env 文件"
}

# 验证安装
verify_installation() {
    log_step "验证安装"
    
    # 检查必要文件
    local required_files=(
        "dataset/pretrain_data.jsonl"
        "dataset/pretrain_images"
        "model/vision_model/clip-vit-base-patch16/config.json"
        "model/vision_model/clip-vit-base-patch16/pytorch_model.bin"
        "configs/vlm_training.yaml"
        "src/train_vlm.py"
    )
    
    log_info "检查必要文件..."
    local missing_files=()
    for file in "${required_files[@]}"; do
        if [ -e "$file" ]; then
            log_success "✓ $file"
        else
            log_error "✗ $file (缺失)"
            missing_files+=("$file")
        fi
    done
    
    if [ ${#missing_files[@]} -eq 0 ]; then
        log_success "所有必要文件都已就位"
    else
        log_error "缺失 ${#missing_files[@]} 个文件，请检查下载过程"
        return 1
    fi
    
    # 测试Python导入
    # DRY_RUN 模式下跳过依赖导入测试
    if [ -n "$DRY_RUN" ]; then
        log_info "[DRY_RUN] 跳过 Python 依赖导入测试"
        return 0
    fi
    log_info "测试Python模块导入..."
    if python3 -c "
import sys
sys.path.insert(0, 'src')
try:
    from configs.config_loader import load_yaml_config
    from configs.training_models import TrainingConfigModel
    print('✓ 配置模块导入成功')
    
    config = load_yaml_config('configs/vlm_training.yaml')
    print('✓ 配置文件加载成功')
    
    import yaml
    print('✓ YAML模块可用')
    
    print('✓ 所有核心模块导入成功')
except Exception as e:
    print(f'✗ 模块导入失败: {e}')
    exit(1)
    "; then
        log_success "Python模块测试通过"
    else
        log_error "Python模块测试失败"
        return 1
    fi
    
    log_success "安装验证完成"
}

# 显示使用说明
show_usage() {
    log_step "安装完成！"
    
    cat << EOF

${GREEN}🎉 zerollm-v 环境配置完成！${NC}

${YELLOW}📁 项目结构:${NC}
├── dataset/                # 训练数据
│   ├── pretrain_data.jsonl # 训练文本数据
│   └── pretrain_images/    # 训练图像数据
├── model/vision_model/clip-vit-base-patch16/ # 视觉模型权重
├── configs/               # 配置文件
└── src/train_vlm.py      # 训练脚本

${YELLOW}🚀 快速开始:${NC}
1. 激活环境变量:
   ${CYAN}source .env${NC}

2. 开始训练:
   ${CYAN}python src/train_vlm.py configs/vlm_training.yaml${NC}

3. 使用 uv 运行 (推荐):
   ${CYAN}uv run src/train_vlm.py configs/vlm_training.yaml${NC}

${YELLOW}📊 监控训练:${NC}
- SwanLab 项目: ${CYAN}https://swanlab.cn${NC}
- 如需启用，请先导出: ${CYAN}export SWANLAB_API_KEY=\"<your_key>\"${NC}

${YELLOW}⚠️  注意事项:${NC}
- 确保您有足够的GPU显存（推荐8GB+）
- 训练过程会在 SwanLab 上实时显示
 - 模型检查点将保存到 out/ 目录（见 configs 配置）

${GREEN}Happy Training! 🎯${NC}

EOF
}

# 主函数
main() {
    echo -e "${PURPLE}"
    cat << "EOF"
  ____                _ _            __      __
 |_  / ___ _ __ ___   | | |_   __     \ \    / /
  / / / _ \ '__/ _ \  | | \ \ / /____ \ \  / / 
 / /_|  __/ | | (_) | | | |\ V /______|  \/ /  
/____\___|_|  \___/  |_|_| \_/          \/   
                                              
            一键部署脚本 v1.0
EOF
    echo -e "${NC}\n"
    
    log_info "开始 zerollm-v 环境配置..."
    
    # 执行各个步骤
    check_environment
    setup_project
    install_dependencies
    download_datasets
    download_vision_model
    setup_environment
    verify_installation
    show_usage
    
    log_success "所有步骤完成！zerollm-v 已准备就绪！"
}

# 错误处理
trap 'log_error "脚本执行过程中发生错误，退出码: $?"; exit 1' ERR

# 执行主函数
main "$@"
