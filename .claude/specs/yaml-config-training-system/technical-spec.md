# Technical Specification: YAML Configuration Training System

## Problem Statement
- **Business Issue**: The current Zero2LLMV training framework uses command-line arguments exclusively, making experiments difficult to reproduce, track, and manage at scale. Complex parameter combinations are error-prone and difficult to version control.
- **Current State**: All training parameters must be passed via CLI args in train.py, with no centralized configuration management or validation beyond basic type checking.
- **Expected Outcome**: A YAML-based configuration system that maintains full backward compatibility while enabling reproducible, trackable, and validatable training configurations with priority-based parameter overrides.

## Solution Overview
- **Approach**: Implement PyYAML + pydantic based configuration system that initializes existing TrainingConfig dataclass from YAML files, with CLI override capabilities and comprehensive validation.
- **Core Changes**: Add configuration loading/merging system, pydantic models for validation, sample YAML files, and enhanced CLI argument handling with backward compatibility.
- **Success Criteria**: Complete YAML configuration support, maintained CLI compatibility, comprehensive validation, experiment reproducibility, and clear error messaging.

## Technical Implementation

### Database Changes
None required - this is a configuration system change only.

### Code Changes

#### Files to Modify
- **`/home/dela/projects/zero2LLMV/train.py`**: Add YAML config loading, CLI merging, validation integration
- **`/home/dela/projects/zero2LLMV/configs/llmconfig.py`**: Enhance with pydantic validation (optional)

#### New Files to Create
- **`/home/dela/projects/zero2LLMV/configs/config_loader.py`**: Configuration loading and merging system
- **`/home/dela/projects/zero2LLMV/configs/training_models.py`**: Pydantic configuration models
- **`/home/dela/projects/zero2LLMV/configs/config.yaml`**: Default training configuration template
- **`/home/dela/projects/zero2LLMV/configs/experiments/`**: Directory for experiment-specific configs
- **`/home/dela/projects/zero2LLMV/requirements.txt`**: Updated dependencies (if missing)

#### Function Signatures to Implement

**`configs/config_loader.py`**:
```python
def load_yaml_config(config_path: str) -> Dict[str, Any]
def merge_config_with_args(yaml_config: Dict, cli_args: argparse.Namespace) -> Dict[str, Any]
def validate_config(config_dict: Dict) -> TrainingConfigModel
def save_merged_config(config: TrainingConfig, output_dir: str) -> None
```

**`configs/training_models.py`**:
```python
class ModelConfigModel(BaseModel)
class DataConfigModel(BaseModel) 
class TrainingHyperparamsModel(BaseModel)
class WandbConfigModel(BaseModel)
class TrainingConfigModel(BaseModel)
```

### API Changes
No REST API changes - this is a CLI/configuration enhancement.

### Configuration Changes

#### New Configuration Structure
```yaml
# configs/config.yaml
model:
  model_type: "vlm"
  model_config_path: ""

data:
  data_path: "data/processed"
  max_seq_length: 512
  batch_size: 8
  num_workers: 4

training:
  num_epochs: 3
  learning_rate: 2.0e-5
  weight_decay: 0.01
  warmup_steps: 500
  gradient_accumulation_steps: 4
  max_grad_norm: 1.0
  use_amp: true

checkpoints:
  output_dir: "outputs"
  save_steps: 1000
  logging_steps: 100
  eval_steps: 500

wandb:
  project: "zero2llmv"
  name: ""
  notes: ""
  tags: []

# Environment variables (not in YAML):
# WANDB_API_KEY, WANDB_BASE_URL
```

#### Environment Variables
- **`WANDB_API_KEY`**: WandB authentication key (excluded from YAML for security)
- **`WANDB_BASE_URL`**: Self-hosted WandB server URL (excluded from YAML for security)

#### Feature Flags
- **`--config`**: Path to YAML configuration file
- **`--allow-default-fallback`**: Allow fallback to defaults when validation fails
- **`--save-merged-config`**: Save final merged configuration to experiment directory

## Implementation Sequence

### Phase 1: Core Configuration Models
- **Task 1.1**: Create `configs/training_models.py` with comprehensive pydantic models
  - ModelConfigModel with model_type validation
  - DataConfigModel with path validation and positive integer constraints  
  - TrainingHyperparamsModel with learning rate/batch size validation
  - WandbConfigModel for experiment tracking parameters
  - TrainingConfigModel as root validator combining all sections
- **Task 1.2**: Create `configs/config.yaml` default configuration template
  - All parameters from existing TrainingConfig dataclass
  - Properly structured sections matching pydantic models
  - Comments explaining parameter meanings and valid ranges

### Phase 2: Configuration Loading System  
- **Task 2.1**: Create `configs/config_loader.py` with core functions
  - `load_yaml_config()`: YAML file parsing with error handling
  - `merge_config_with_args()`: Priority-based parameter merging
  - `validate_config()`: Pydantic validation with detailed error messages
  - `save_merged_config()`: Save final config to experiment directory
- **Task 2.2**: Create experiment configuration directory structure
  - `configs/experiments/` directory
  - Sample experiment configs demonstrating overrides

### Phase 3: Integration with train.py
- **Task 3.1**: Modify `parse_args()` function in train.py
  - Add `--config` argument for YAML file path
  - Add `--allow-default-fallback` and `--save-merged-config` flags
  - Maintain all existing CLI arguments for backward compatibility
- **Task 3.2**: Modify `main()` function in train.py
  - Integrate YAML config loading before TrainingConfig instantiation
  - Add configuration merging and validation
  - Add merged config saving to output directory
  - Enhanced error handling with actionable error messages

## Validation Plan

### Unit Tests
- **`test_config_loading.py`**: YAML parsing, invalid file handling, missing file scenarios
- **`test_config_validation.py`**: Pydantic model validation, range checks, type validation
- **`test_config_merging.py`**: CLI override priority, parameter precedence, edge cases
- **`test_training_integration.py`**: End-to-end config loading in training workflow

### Integration Tests
- **`test_backward_compatibility.py`**: Existing CLI commands continue to work unchanged
- **`test_yaml_workflow.py`**: Complete YAML-driven training configuration and execution  
- **`test_mixed_configs.py`**: YAML base config with CLI overrides
- **`test_experiment_reproducibility.py`**: Saved configurations enable exact reproduction

### Business Logic Verification
- **Reproducibility Test**: Same YAML config produces identical training setup
- **Override Test**: CLI arguments correctly override YAML parameters with proper precedence
- **Validation Test**: Invalid configurations fail fast with clear error messages
- **Compatibility Test**: Existing training scripts run without modification

## Detailed Technical Requirements

### Pydantic Configuration Models

#### ModelConfigModel
```python
class ModelConfigModel(BaseModel):
    model_type: Literal["llm", "vlm"] = "vlm"
    model_config_path: str = ""
    
    @validator('model_type')
    def validate_model_type(cls, v):
        if v not in ["llm", "vlm"]:
            raise ValueError('model_type must be "llm" or "vlm"')
        return v
```

#### DataConfigModel  
```python
class DataConfigModel(BaseModel):
    data_path: str = "data/processed"
    max_seq_length: int = Field(default=512, gt=0, le=8192)
    batch_size: int = Field(default=8, gt=0, le=1024)
    num_workers: int = Field(default=4, ge=0, le=64)
    
    @validator('data_path')
    def validate_data_path(cls, v):
        if not v.strip():
            raise ValueError('data_path cannot be empty')
        return v
```

#### TrainingHyperparamsModel
```python
class TrainingHyperparamsModel(BaseModel):
    num_epochs: int = Field(default=3, ge=1, le=1000)
    learning_rate: float = Field(default=2e-5, gt=0, le=1)
    weight_decay: float = Field(default=0.01, ge=0, le=1)
    warmup_steps: int = Field(default=500, ge=0)
    gradient_accumulation_steps: int = Field(default=4, ge=1, le=128)
    max_grad_norm: float = Field(default=1.0, gt=0, le=10)
    use_amp: bool = True
```

#### WandbConfigModel
```python
class WandbConfigModel(BaseModel):
    project: str = "zero2llmv"
    name: str = ""
    notes: str = ""
    tags: List[str] = []
    
    @validator('project')
    def validate_project(cls, v):
        if not v.strip():
            raise ValueError('wandb project name cannot be empty')
        return v
```

#### CheckpointsConfigModel
```python
class CheckpointsConfigModel(BaseModel):
    output_dir: str = "outputs" 
    save_steps: int = Field(default=1000, gt=0)
    logging_steps: int = Field(default=100, gt=0)
    eval_steps: int = Field(default=500, gt=0)
    
    @validator('output_dir')
    def validate_output_dir(cls, v):
        if not v.strip():
            raise ValueError('output_dir cannot be empty')
        return v
```

### Configuration Loading Implementation

#### Priority System (Low to High)
1. **Hardcoded defaults** in pydantic models
2. **YAML configuration** from `--config` file
3. **Environment variables** for sensitive data (WANDB_API_KEY, WANDB_BASE_URL)
4. **CLI arguments** for debugging/override (highest priority)

#### Error Handling Strategy
- **Fail-fast approach**: Configuration errors cause immediate exit with detailed messages
- **Detailed error messages**: Include field name, expected type/range, provided value, and example
- **File-based errors**: Clear indication of YAML parsing issues with line numbers
- **Validation errors**: Specific pydantic validation errors with context

#### Example Error Messages
```
Configuration Error: Invalid learning_rate
  Field: training.learning_rate
  Provided: -0.001
  Expected: Positive float between 0 and 1
  Example: learning_rate: 2.0e-5
  
Configuration Error: Missing required file
  File: configs/my_experiment.yaml
  Error: File not found
  Suggestion: Check file path or use --config configs/config.yaml
```

### Integration Points

#### Backward Compatibility Strategy
- **Preserve existing CLI interface**: All current command-line arguments continue to work
- **TrainingConfig dataclass unchanged**: Internal structure remains identical
- **Default behavior preserved**: Running without --config works exactly as before  
- **Gradual migration path**: Users can adopt YAML configs incrementally

#### CLI Integration Enhancement
```python
def parse_args():
    parser = argparse.ArgumentParser(description="Zero2LLMV Training with YAML Config Support")
    
    # New YAML-specific arguments
    parser.add_argument("--config", type=str, default="", help="Path to YAML configuration file")
    parser.add_argument("--allow-default-fallback", action="store_true", 
                       help="Allow fallback to defaults on validation errors")
    parser.add_argument("--save-merged-config", action="store_true",
                       help="Save final merged configuration to output directory")
    
    # All existing arguments preserved...
    
    return parser.parse_args()
```

#### Enhanced main() Function Logic
```python
def main():
    args = parse_args()
    
    # Load YAML config if provided
    yaml_config = {}
    if args.config:
        yaml_config = load_yaml_config(args.config)
    
    # Merge with CLI args (CLI takes precedence)  
    merged_config = merge_config_with_args(yaml_config, args)
    
    # Validate merged configuration
    try:
        validated_config = validate_config(merged_config)
    except ValidationError as e:
        if args.allow_default_fallback:
            print("Warning: Using defaults due to validation errors")
            validated_config = TrainingConfigModel()
        else:
            print(f"Configuration validation failed: {e}")
            sys.exit(1)
    
    # Create TrainingConfig from validated data
    config = TrainingConfig(**validated_config.dict())
    
    # Save merged config if requested
    if args.save_merged_config:
        save_merged_config(config, config.output_dir)
    
    # Continue with existing training logic...
```

### File Structure Implementation

```
configs/
├── config.yaml              # Default configuration template
├── config_loader.py          # Configuration loading and merging system  
├── training_models.py        # Pydantic validation models
├── llmconfig.py             # Existing model config (unchanged)
├── experiments/             # Experiment-specific configurations
│   ├── high_lr_experiment.yaml
│   ├── large_batch.yaml
│   └── moe_training.yaml
└── models/                  # Future model-specific configs
    ├── llm.yaml
    └── vlm.yaml
```

### Dependencies Required

#### requirements.txt additions:
```
pyyaml>=6.0
pydantic>=1.10.0,<2.0.0
```

#### Import additions to train.py:
```python
import yaml
from pathlib import Path
from pydantic import ValidationError
from configs.config_loader import load_yaml_config, merge_config_with_args, validate_config, save_merged_config
from configs.training_models import TrainingConfigModel
```

This technical specification provides a complete blueprint for implementing the YAML-based configuration system while maintaining full backward compatibility and enabling comprehensive validation and reproducibility features.