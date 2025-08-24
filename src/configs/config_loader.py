"""
Zero2LLMV 训练框架的配置加载和合并系统。

本模块提供全面的 YAML 配置加载功能，具有基于优先级的
参数合并（默认值 < YAML < 环境变量 < CLI 参数）。
包含错误处理、验证和配置持久化功能。
"""

import os
import sys
import yaml
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import asdict

from pydantic import ValidationError

from .training_models import TrainingConfigModel


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    从文件加载 YAML 配置，提供全面的错误处理。
    
    Args:
        config_path: YAML 配置文件路径
        
    Returns:
        包含加载配置的字典
        
    Raises:
        FileNotFoundError: 如果配置文件不存在
        yaml.YAMLError: 如果 YAML 解析失败
        ValueError: 如果配置格式无效
    """
    config_path = Path(config_path)
    
    # 检查文件是否存在
    if not config_path.exists():
        raise FileNotFoundError(
            f"配置文件未找到: {config_path}\n"
            f"建议: 检查文件路径或使用默认配置如 'configs/config.yaml'"
        )
    
    # 检查文件是否可读
    if not config_path.is_file():
        raise ValueError(
            f"配置路径不是文件: {config_path}\n"
            f"Expected: A valid YAML configuration file"
        )
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        # Handle empty file
        if config_dict is None:
            config_dict = {}
            
        # Ensure we have a dictionary
        if not isinstance(config_dict, dict):
            raise ValueError(
                f"Configuration file must contain a YAML dictionary, got {type(config_dict).__name__}\n"
                f"File: {config_path}\n"
                f"Expected format: key-value pairs organized in sections"
            )
            
        print(f"Successfully loaded configuration from: {config_path}")
        return config_dict
        
    except yaml.YAMLError as e:
        # Detailed YAML parsing error
        error_msg = f"YAML parsing error in {config_path}:\n"
        if hasattr(e, 'problem_mark'):
            mark = e.problem_mark
            error_msg += f"  Line {mark.line + 1}, Column {mark.column + 1}\n"
        error_msg += f"  Error: {str(e)}\n"
        error_msg += "  Suggestion: Check YAML syntax, indentation, and quotes"
        raise yaml.YAMLError(error_msg) from e
        
    except UnicodeDecodeError as e:
        raise ValueError(
            f"Configuration file encoding error: {config_path}\n"
            f"Error: {str(e)}\n"
            f"Suggestion: Ensure file is saved in UTF-8 encoding"
        ) from e
        
    except Exception as e:
        raise ValueError(
            f"Unexpected error loading configuration: {config_path}\n"
            f"Error: {str(e)}\n"
            f"Suggestion: Check file permissions and format"
        ) from e


def merge_config_with_args(yaml_config: Dict[str, Any], cli_args: argparse.Namespace) -> Dict[str, Any]:
    """
    Merge YAML configuration with CLI arguments using priority system.
    
    Priority (low to high):
    1. YAML configuration
    2. Environment variables 
    3. CLI arguments (highest priority)
    
    Args:
        yaml_config: Configuration loaded from YAML file
        cli_args: Parsed command line arguments
        
    Returns:
        Merged configuration dictionary
    """
    merged_config = {}
    
    # Start with YAML config (lowest priority)
    merged_config.update(yaml_config)
    
    # Map CLI arguments to configuration structure
    cli_mappings = {
        # Model configuration
        'model_type': ('model', 'model_type'),
        'model_config_path': ('model', 'model_config_path'),
        
        # Data configuration
        'data_path': ('data', 'data_path'),
        'max_seq_length': ('data', 'max_seq_length'),
        'batch_size': ('data', 'batch_size'),
        'num_workers': ('data', 'num_workers'),
        
        # Training configuration
        'num_epochs': ('training', 'num_epochs'),
        'learning_rate': ('training', 'learning_rate'),
        'weight_decay': ('training', 'weight_decay'),
        'warmup_steps': ('training', 'warmup_steps'),
        'gradient_accumulation_steps': ('training', 'gradient_accumulation_steps'),
        'max_grad_norm': ('training', 'max_grad_norm'),
        'use_amp': ('training', 'use_amp'),
        
        # Checkpoint configuration
        'output_dir': ('checkpoints', 'output_dir'),
        'save_steps': ('checkpoints', 'save_steps'),
        'logging_steps': ('checkpoints', 'logging_steps'),
        'eval_steps': ('checkpoints', 'eval_steps'),
        
        # SwanLab configuration
        'swanlab_project': ('swanlab', 'project'),
        'swanlab_workspace': ('swanlab', 'workspace'),
        'swanlab_experiment_name': ('swanlab', 'experiment_name'),
        'swanlab_description': ('swanlab', 'description'),
        'swanlab_logdir': ('swanlab', 'logdir'),
        'swanlab_api_key': ('swanlab', 'api_key'),    # Will be overridden by env var
    }
    
    # Apply CLI arguments (higher priority than YAML)
    cli_dict = vars(cli_args)
    for cli_key, (section, config_key) in cli_mappings.items():
        if cli_key in cli_dict and cli_dict[cli_key] is not None:
            # Handle special case for boolean arguments
            if cli_key == 'use_amp' and hasattr(cli_args, 'use_amp'):
                # argparse store_true actions need special handling
                value = getattr(cli_args, 'use_amp', False)
            else:
                value = cli_dict[cli_key]
            
            # Only override if the value was explicitly set (not default)
            # We check this by comparing to parser defaults when possible
            if section not in merged_config:
                merged_config[section] = {}
            merged_config[section][config_key] = value
    
    # Apply environment variables (highest priority for sensitive data)
    env_mappings = {
        'SWANLAB_API_KEY': ('swanlab', 'api_key'),
    }
    
    for env_var, (section, config_key) in env_mappings.items():
        env_value = os.environ.get(env_var)
        if env_value:
            if section not in merged_config:
                merged_config[section] = {}
            merged_config[section][config_key] = env_value
    
    return merged_config


def validate_config(config_dict: Dict[str, Any], allow_defaults: bool = False) -> TrainingConfigModel:
    """
    Validate merged configuration using pydantic models.
    
    Args:
        config_dict: Configuration dictionary to validate
        allow_defaults: Whether to use defaults for missing/invalid values
        
    Returns:
        Validated TrainingConfigModel instance
        
    Raises:
        ValidationError: If configuration is invalid and allow_defaults=False
    """
    try:
        # Create model from configuration dictionary
        validated_config = TrainingConfigModel(**config_dict)
        print("Configuration validation successful")
        return validated_config
        
    except ValidationError as e:
        # Format detailed error message
        error_msg = "Configuration validation failed:\n\n"
        
        for error in e.errors():
            field_path = " -> ".join(str(loc) for loc in error['loc'])
            error_type = error['type']
            error_msg_detail = error['msg']
            
            error_msg += f"Field: {field_path}\n"
            error_msg += f"Error: {error_msg_detail}\n"
            
            # Add specific suggestions based on error type
            if error_type == 'value_error.missing':
                error_msg += "Suggestion: Add this required field to your configuration\n"
            elif error_type in ['type_error.integer', 'type_error.float', 'type_error.bool']:
                expected_type = error_type.split('.')[-1]
                error_msg += f"Suggestion: Ensure value is a valid {expected_type}\n"
            elif 'greater' in error_type:
                error_msg += "Suggestion: Use a larger positive value\n"
            elif 'less' in error_type:
                error_msg += "Suggestion: Use a smaller value within the valid range\n"
            elif error_type == 'value_error':
                error_msg += "Suggestion: Check the value format and valid options\n"
            
            error_msg += "\n"
        
        # Add examples for common configurations
        error_msg += "Example valid configuration sections:\n"
        error_msg += "model:\n"
        error_msg += "  model_type: \"vlm\"  # or \"llm\"\n"
        error_msg += "data:\n"
        error_msg += "  batch_size: 8  # positive integer\n"
        error_msg += "  learning_rate: 2.0e-5  # positive float\n"
        error_msg += "training:\n"
        error_msg += "  num_epochs: 3  # positive integer\n"
        
        if allow_defaults:
            print(f"Warning: Using default configuration due to validation errors:\n{error_msg}")
            return TrainingConfigModel()  # Return model with all defaults
        else:
            raise ValueError(error_msg) from e
    
    except Exception as e:
        error_msg = f"Unexpected validation error: {str(e)}\n"
        error_msg += "Suggestion: Check configuration format and data types"
        
        if allow_defaults:
            print(f"Warning: Using default configuration due to unexpected error:\n{error_msg}")
            return TrainingConfigModel()
        else:
            raise ValueError(error_msg) from e


def save_merged_config(config_dict: Dict[str, Any], output_dir: str, 
                      filename: str = "merged_config.yaml") -> None:
    """
    Save merged configuration to experiment directory for reproducibility.
    
    Args:
        config_dict: Configuration dictionary to save
        output_dir: Output directory for experiment
        filename: Name of configuration file to save
    """
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        config_file = output_path / filename
        
        # Create a clean config for saving (remove None values and empty strings)
        clean_config = _clean_config_for_saving(config_dict)
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(
                clean_config,
                f,
                default_flow_style=False,
                indent=2,
                sort_keys=True,
                allow_unicode=True
            )
        
        print(f"Merged configuration saved to: {config_file}")
        
        # Also save as JSON for programmatic access
        json_file = output_path / filename.replace('.yaml', '.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(clean_config, f, indent=2, ensure_ascii=False)
        
        print(f"Configuration also saved as JSON: {json_file}")
        
    except Exception as e:
        print(f"Warning: Could not save merged configuration: {str(e)}")
        # Don't raise exception - this is not critical for training


def _clean_config_for_saving(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean configuration dictionary for saving by removing empty values.
    
    Args:
        config_dict: Raw configuration dictionary
        
    Returns:
        Cleaned configuration dictionary
    """
    cleaned = {}
    
    for key, value in config_dict.items():
        if isinstance(value, dict):
            # Recursively clean nested dictionaries
            cleaned_nested = _clean_config_for_saving(value)
            if cleaned_nested:  # Only include non-empty sections
                cleaned[key] = cleaned_nested
        elif value is not None and value != "" and value != []:
            # Include non-empty values
            cleaned[key] = value
    
    return cleaned


def create_experiment_config(base_config_path: str, overrides: Dict[str, Any], 
                           output_path: str) -> None:
    """
    Create experiment-specific configuration file with overrides.
    
    Args:
        base_config_path: Path to base configuration file
        overrides: Dictionary of configuration overrides
        output_path: Path for new experiment configuration file
    """
    try:
        # Load base configuration
        base_config = load_yaml_config(base_config_path)
        
        # Apply overrides using deep merge
        experiment_config = _deep_merge(base_config, overrides)
        
        # Save experiment configuration
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(
                experiment_config,
                f,
                default_flow_style=False,
                indent=2,
                sort_keys=True,
                allow_unicode=True
            )
        
        print(f"Experiment configuration created: {output_path}")
        
    except Exception as e:
        raise ValueError(f"Failed to create experiment configuration: {str(e)}") from e


def _deep_merge(base_dict: Dict[str, Any], override_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries, with override_dict taking precedence.
    
    Args:
        base_dict: Base dictionary
        override_dict: Override dictionary
        
    Returns:
        Merged dictionary
    """
    merged = base_dict.copy()
    
    for key, value in override_dict.items():
        if (key in merged and 
            isinstance(merged[key], dict) and 
            isinstance(value, dict)):
            # Recursively merge nested dictionaries
            merged[key] = _deep_merge(merged[key], value)
        else:
            # Override value
            merged[key] = value
    
    return merged


def load_and_validate_config(config_path: Optional[str] = None,
                           cli_args: Optional[argparse.Namespace] = None,
                           allow_defaults: bool = False) -> TrainingConfigModel:
    """
    Complete configuration loading pipeline with validation.
    
    This is the main entry point for configuration loading that combines
    YAML loading, CLI merging, and validation in a single call.
    
    Args:
        config_path: Path to YAML configuration file (optional)
        cli_args: CLI arguments namespace (optional)
        allow_defaults: Whether to use defaults on validation errors
        
    Returns:
        Validated TrainingConfigModel instance
    """
    # Load YAML config if provided
    yaml_config = {}
    if config_path:
        yaml_config = load_yaml_config(config_path)
    
    # Merge with CLI args if provided
    if cli_args:
        merged_config = merge_config_with_args(yaml_config, cli_args)
    else:
        merged_config = yaml_config
    
    # Validate merged configuration
    validated_config = validate_config(merged_config, allow_defaults)
    
    return validated_config