#!/bin/bash

# 复现脚本 - 深度学习训练与推理自动化
# 支持环境检查、训练、推理、模型下载等功能

set -e  # 遇到错误立即退出

# 默认配置
GITHUB_PREFIX="https://github.com/your-username/your-repo/raw/main"
EXP_ID=""
MODE=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
EXPERIMENTS_DIR="$PROJECT_ROOT/experiments"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 显示帮助信息
show_help() {
    cat << EOF
使用说明: $0 [选项]

选项:
    -m, --mode MODE          运行模式: check|train|predict|download
    -e, --exp-id ID          实验ID (例如: exp251022_172342)
    -g, --github URL         GitHub链接前缀
    -h, --help              显示此帮助信息

示例:
    $0 -m check                    # 环境检查
    $0 -m train -e exp251022_172342  # 训练模型
    $0 -m predict -e exp251022_172342 # 推理预测
    $0 -m download -e exp251022_172342 # 下载模型文件

EOF
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -e|--exp-id)
            EXP_ID="$2"
            shift 2
            ;;
        -g|--github)
            GITHUB_PREFIX="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            log_error "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# 检查必要参数
check_required_params() {
    if [[ -z "$MODE" ]]; then
        log_error "必须指定运行模式 (-m/--mode)"
        show_help
        exit 1
    fi
    
    if [[ "$MODE" != "check" && -z "$EXP_ID" ]]; then
        log_error "模式 '$MODE' 需要指定实验ID (-e/--exp-id)"
        show_help
        exit 1
    fi
}

# 环境检查
check_environment() {
    log_info "开始环境检查..."
    
    # 检查Python
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c "import sys; print(sys.version.split()[0])")
        log_success "Python 版本: $PYTHON_VERSION"
    else
        log_error "未找到 Python3"
        return 1
    fi
    
    # 检查PyTorch
    if python3 -c "import torch" &> /dev/null; then
        TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
        CUDA_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())")
        log_success "PyTorch 版本: $TORCH_VERSION"
        log_success "CUDA 可用: $CUDA_AVAILABLE"
        
        if [[ "$CUDA_AVAILABLE" == "True" ]]; then
            CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)")
            log_success "CUDA 版本: $CUDA_VERSION"
        fi
    else
        log_error "PyTorch 未安装"
        return 1
    fi
    
    # 检查必要包
    REQUIRED_PACKAGES=("torchvision" "numpy" "matplotlib" "seaborn" "pandas" "yaml" "cv2")
    for package in "${REQUIRED_PACKAGES[@]}"; do
        if python3 -c "import $package" &> /dev/null; then
            log_success "$package 已安装"
        else
            log_warning "$package 未安装"
        fi
    done
    
    # 检查项目文件
    REQUIRED_FILES=("train.py" "predict.py" "evaluater.py" "trainer_ema.py" "utils.py" "models")
    for file in "${REQUIRED_FILES[@]}"; do
        if [[ -e "$PROJECT_ROOT/$file" ]]; then
            log_success "项目文件: $file 存在"
        else
            log_warning "项目文件: $file 不存在"
        fi
    done
    
    log_success "环境检查完成"
}

# 下载配置文件
download_config() {
    local exp_id="$1"
    local exp_dir="$EXPERIMENTS_DIR/$exp_id"
    
    log_info "下载配置文件到: $exp_dir"
    
    # 创建实验目录
    mkdir -p "$exp_dir"
    
    # 下载配置文件
    local config_url="$GITHUB_PREFIX/experiments/$exp_id/config.yaml"
    if curl -f -s "$config_url" -o "$exp_dir/config.yaml"; then
        log_success "配置文件下载成功"
    else
        log_error "配置文件下载失败: $config_url"
        return 1
    fi
}

# 下载模型文件并校验
download_model() {
    local exp_id="$1"
    local exp_dir="$EXPERIMENTS_DIR/$exp_id"
    
    log_info "下载模型文件到: $exp_dir"
    
    # 首先下载information.yaml获取模型信息
    local info_url="$GITHUB_PREFIX/experiments/$exp_id/information.yaml"
    if ! curl -f -s "$info_url" -o "$exp_dir/information.yaml"; then
        log_error "信息文件下载失败: $info_url"
        return 1
    fi
    
    # 解析模型文件名和SHA256
    local weights_name=$(python3 -c "
import yaml
with open('$exp_dir/information.yaml', 'r') as f:
    info = yaml.safe_load(f)
print(info.get('weights', ''))
")
    
    local expected_sha=$(python3 -c "
import yaml
with open('$exp_dir/information.yaml', 'r') as f:
    info = yaml.safe_load(f)
print(info.get('weights_sha256', ''))
")
    
    if [[ -z "$weights_name" ]]; then
        log_error "无法从信息文件中获取模型文件名"
        return 1
    fi
    
    # 下载模型文件
    local model_url="$GITHUB_PREFIX/experiments/$exp_id/$weights_name"
    log_info "下载模型文件: $weights_name"
    
    if curl -f -s "$model_url" -o "$exp_dir/$weights_name"; then
        log_success "模型文件下载成功"
    else
        log_error "模型文件下载失败: $model_url"
        return 1
    fi
    
    # SHA256校验
    log_info "进行SHA256校验..."
    local actual_sha=$(sha256sum "$exp_dir/$weights_name" | cut -d' ' -f1)
    
    if [[ "$actual_sha" == "$expected_sha" ]]; then
        log_success "SHA256校验通过"
    else
        log_error "SHA256校验失败"
        log_error "期望: $expected_sha"
        log_error "实际: $actual_sha"
        # 删除损坏的文件
        rm -f "$exp_dir/$weights_name"
        return 1
    fi
}

# 训练模型
train_model() {
    local exp_id="$1"
    local exp_dir="$EXPERIMENTS_DIR/$exp_id"
    
    log_info "开始训练模型: $exp_id"
    
    # 检查配置文件是否存在
    if [[ ! -f "$exp_dir/config.yaml" ]]; then
        log_warning "本地配置文件不存在，尝试下载..."
        if ! download_config "$exp_id"; then
            log_error "无法获取配置文件，请检查实验ID或网络连接"
            return 1
        fi
    fi
    
    # 切换到项目根目录
    cd "$PROJECT_ROOT"
    
    # 运行训练脚本
    log_info "启动训练进程..."
    if python3 train.py --config "$exp_dir/config.yaml"; then
        log_success "训练完成"
    else
        log_error "训练失败"
        return 1
    fi
}

# 推理预测
predict_model() {
    local exp_id="$1"
    local exp_dir="$EXPERIMENTS_DIR/$exp_id"
    
    log_info "开始推理预测: $exp_id"
    
    # 检查必要文件
    if [[ ! -f "$exp_dir/config.yaml" ]]; then
        log_warning "配置文件不存在，尝试下载..."
        if ! download_config "$exp_id"; then
            log_error "无法获取配置文件"
            return 1
        fi
    fi
    
    # 检查模型文件
    local model_files=($(find "$exp_dir" -name "model_final_epoch_*.pth" 2>/dev/null))
    if [[ ${#model_files[@]} -eq 0 ]]; then
        log_warning "模型文件不存在，尝试下载..."
        if ! download_model "$exp_id"; then
            log_error "无法获取模型文件"
            return 1
        fi
    fi
    
    # 切换到项目根目录
    cd "$PROJECT_ROOT"
    
    # 运行推理脚本
    log_info "启动推理进程..."
    if python3 predict.py --exp-id "$exp_id"; then
        log_success "推理完成"
    else
        log_error "推理失败"
        return 1
    fi
}

# 下载实验文件
download_experiment() {
    local exp_id="$1"
    
    log_info "下载实验文件: $exp_id"
    
    if download_config "$exp_id" && download_model "$exp_id"; then
        log_success "实验文件下载完成"
    else
        log_error "实验文件下载失败"
        return 1
    fi
}

# 交互式选择模式
interactive_mode() {
    log_info "交互模式启动"
    
    echo "请选择运行模式:"
    echo "1) 环境检查"
    echo "2) 训练模型"
    echo "3) 推理预测"
    echo "4) 下载模型"
    echo "5) 退出"
    
    read -p "请输入选择 (1-5): " choice
    
    case $choice in
        1) MODE="check" ;;
        2) MODE="train" ;;
        3) MODE="predict" ;;
        4) MODE="download" ;;
        5) exit 0 ;;
        *) log_error "无效选择"; exit 1 ;;
    esac
    
    if [[ "$MODE" != "check" ]]; then
        read -p "请输入实验ID: " EXP_ID
        if [[ -z "$EXP_ID" ]]; then
            log_error "必须提供实验ID"
            exit 1
        fi
    fi
}

# 主函数
main() {
    log_info "深度学习复现脚本启动"
    log_info "项目根目录: $PROJECT_ROOT"
    log_info "实验目录: $EXPERIMENTS_DIR"
    
    # 如果没有指定模式，进入交互模式
    if [[ -z "$MODE" ]]; then
        interactive_mode
    else
        check_required_params
    fi
    
    # 创建实验目录
    mkdir -p "$EXPERIMENTS_DIR"
    
    # 根据模式执行相应操作
    case "$MODE" in
        "check")
            check_environment
            ;;
        "train")
            train_model "$EXP_ID"
            ;;
        "predict")
            predict_model "$EXP_ID"
            ;;
        "download")
            download_experiment "$EXP_ID"
            ;;
        *)
            log_error "未知模式: $MODE"
            show_help
            exit 1
            ;;
    esac
    
    log_success "脚本执行完成"
}

# 运行主函数
main "$@"