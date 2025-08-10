#!/usr/bin/env python3
"""
测试运行脚本 - 提供不同的测试运行模式
"""
import argparse
import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description=""):
    """运行命令并处理结果"""
    print(f"\n{'='*60}")
    if description:
        print(f"运行: {description}")
    print(f"命令: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("stderr:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"错误: 命令失败，返回码 {e.returncode}")
        print("stdout:", e.stdout)
        print("stderr:", e.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Zero2LLMV 测试运行器")
    parser.add_argument(
        '--mode', 
        choices=['quick', 'unit', 'integration', 'performance', 'all', 'coverage'],
        default='quick',
        help='测试模式'
    )
    parser.add_argument(
        '--verbose', '-v', 
        action='store_true',
        help='详细输出'
    )
    parser.add_argument(
        '--parallel', '-p',
        action='store_true', 
        help='并行运行测试'
    )
    parser.add_argument(
        '--pattern',
        help='测试文件模式匹配'
    )
    parser.add_argument(
        '--markers', '-m',
        help='运行特定标记的测试'
    )
    
    args = parser.parse_args()
    
    # 确保在项目根目录运行
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # 基础pytest命令
    base_cmd = ['uv', 'run', 'pytest']
    
    if args.verbose:
        base_cmd.append('-v')
    
    if args.parallel:
        base_cmd.extend(['-n', 'auto'])
    
    if args.pattern:
        base_cmd.extend(['-k', args.pattern])
    
    if args.markers:
        base_cmd.extend(['-m', args.markers])
    
    success = True
    
    if args.mode == 'quick':
        # 快速测试：只运行最关键的单元测试
        cmd = base_cmd + [
            'tests/unit/test_config.py',
            'tests/unit/test_components.py::TestRMSNorm::test_rms_norm_forward',
            'tests/unit/test_attention.py::TestAttention::test_attention_forward_basic',
            'tests/unit/test_feedforward.py::TestFeedForward::test_feedforward_forward',
            '--tb=short'
        ]
        success &= run_command(cmd, "快速测试")
        
    elif args.mode == 'unit':
        # 所有单元测试
        cmd = base_cmd + ['tests/unit/', '--tb=short']
        success &= run_command(cmd, "单元测试")
        
    elif args.mode == 'integration':
        # 集成测试
        cmd = base_cmd + ['tests/integration/', '--tb=short']
        success &= run_command(cmd, "集成测试")
        
    elif args.mode == 'performance':
        # 性能测试
        cmd = base_cmd + ['tests/performance/', '--tb=short']
        success &= run_command(cmd, "性能测试")
        
    elif args.mode == 'coverage':
        # 覆盖率测试
        cmd = base_cmd + [
            'tests/',
            '--cov=models',
            '--cov=configs',
            '--cov-report=html',
            '--cov-report=term-missing',
            '--tb=short'
        ]
        success &= run_command(cmd, "覆盖率测试")
        
    elif args.mode == 'all':
        # 运行所有测试
        test_dirs = ['unit', 'integration', 'performance']
        for test_dir in test_dirs:
            cmd = base_cmd + [f'tests/{test_dir}/', '--tb=short']
            success &= run_command(cmd, f"{test_dir.title()}测试")
            if not success:
                break
    
    if success:
        print(f"\n✅ 所有测试通过！")
        sys.exit(0)
    else:
        print(f"\n❌ 测试失败！")
        sys.exit(1)


if __name__ == '__main__':
    main()