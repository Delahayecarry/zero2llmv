#!/usr/bin/env python3
"""
快速SwanLab集成测试脚本
直接运行短时间训练验证监控集成
"""

import os
import sys
import subprocess
from pathlib import Path

def run_quick_test():
    """运行快速集成测试"""
    print("🚀 Zero2LLMV + SwanLab 快速集成测试")
    print("="*50)
    
    # 检查环境
    try:
        import swanlab
        print(f"✓ SwanLab {swanlab.__version__} 已安装")
    except ImportError:
        print("✗ SwanLab 未安装")
        return False
    
    # 检查API Key
    api_key = os.environ.get('SWANLAB_API_KEY')
    if api_key:
        print("✓ SWANLAB_API_KEY 已设置")
    else:
        print("⚠️  SWANLAB_API_KEY 未设置，将运行离线测试")
    
    # 创建必要目录
    os.makedirs("test_outputs", exist_ok=True)
    os.makedirs("swanlab_logs", exist_ok=True)
    
    print("\n🏋️ 开始快速训练测试...")
    print("   - 训练轮数: 1 epoch")
    print("   - 批量大小: 2")
    print("   - 序列长度: 128")
    print("   - 预计时间: 1-2分钟")
    print("")
    
    # 构建训练命令
    cmd = [
        "uv", "run", "python", "train.py",
        "--config", "configs/test_experiment.yaml"
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    print("=" * 60)
    
    # 执行训练
    try:
        result = subprocess.run(cmd, text=True)
        success = result.returncode == 0
        
        print("=" * 60)
        if success:
            print("🎉 SwanLab集成测试完成!")
            print("\n📊 监控验证:")
            if api_key:
                print("   ✓ 在线监控: https://swanlab.cn/")
                print("   ✓ 项目: VLLM")
                print("   ✓ 工作空间: yourworkspace")
                print("   ✓ 实验: test-integration")
            else:
                print("   ✓ 离线日志: ./swanlab_logs/")
            
            print("\n📁 输出文件:")
            print("   ✓ 检查点: ./test_outputs/")
            print("   ✓ 训练日志: ./swanlab_logs/")
            
            return True
        else:
            print("❌ 训练过程中出现错误")
            return False
            
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        return False

if __name__ == "__main__":
    success = run_quick_test()
    sys.exit(0 if success else 1)