#!/usr/bin/env python3
"""
Simple validation script to test the YAML configuration system functionality.

This script performs basic functional validation of the YAML configuration system
without requiring pytest or other test dependencies.
"""

import os
import sys
import tempfile
import traceback
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import configuration modules
try:
    from configs.config_loader import (
        load_yaml_config,
        merge_config_with_args,
        validate_config,
        save_merged_config,
        load_and_validate_config
    )
    from configs.training_models import TrainingConfigModel
    print("✓ Successfully imported configuration modules")
except Exception as e:
    print(f"✗ Failed to import configuration modules: {e}")
    sys.exit(1)


def test_yaml_loading():
    """Test basic YAML configuration loading."""
    print("\n" + "=" * 50)
    print("Testing YAML Configuration Loading")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Test 1: Valid YAML config
        print("Test 1: Valid YAML configuration...")
        
        valid_config = {
            "model": {
                "model_type": "vlm",
                "model_config_path": ""
            },
            "data": {
                "data_path": "data/test",
                "max_seq_length": 512,
                "batch_size": 8,
                "num_workers": 4
            },
            "training": {
                "num_epochs": 3,
                "learning_rate": 2e-5,
                "weight_decay": 0.01,
                "use_amp": True
            },
            "swanlab": {
                "project": "VLLM",
                "workspace": "delahayecarry"
            }
        }
        
        config_file = temp_path / "valid_config.yaml"
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(valid_config, f)
        
        try:
            loaded_config = load_yaml_config(str(config_file))
            assert loaded_config["model"]["model_type"] == "vlm"
            assert loaded_config["data"]["batch_size"] == 8
            assert loaded_config["training"]["learning_rate"] == 2e-5
            print("✓ Valid YAML configuration loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load valid YAML config: {e}")
            return False
        
        # Test 2: Empty YAML config
        print("Test 2: Empty YAML configuration...")
        
        empty_config_file = temp_path / "empty_config.yaml"
        empty_config_file.write_text("")
        
        try:
            empty_config = load_yaml_config(str(empty_config_file))
            assert empty_config == {}
            print("✓ Empty YAML configuration handled correctly")
        except Exception as e:
            print(f"✗ Failed to handle empty YAML config: {e}")
            return False
        
        # Test 3: Malformed YAML
        print("Test 3: Malformed YAML error handling...")
        
        malformed_config_file = temp_path / "malformed_config.yaml"
        malformed_config_file.write_text("""
        model:
          model_type: vlm
        data:
          batch_size: 8
          invalid_yaml: [unclosed list
        """)
        
        try:
            load_yaml_config(str(malformed_config_file))
            print("✗ Should have raised error for malformed YAML")
            return False
        except yaml.YAMLError:
            print("✓ Malformed YAML error handled correctly")
        except Exception as e:
            print(f"✗ Unexpected error for malformed YAML: {e}")
            return False
    
    return True


def test_pydantic_validation():
    """Test Pydantic configuration validation."""
    print("\n" + "=" * 50)
    print("Testing Pydantic Validation")
    print("=" * 50)
    
    # Test 1: Valid configuration
    print("Test 1: Valid configuration validation...")
    
    valid_config = {
        "model": {"model_type": "vlm"},
        "data": {
            "data_path": "data/test",
            "batch_size": 8,
            "max_seq_length": 512
        },
        "training": {
            "num_epochs": 3,
            "learning_rate": 2e-5,
            "use_amp": True
        },
        "swanlab": {
            "project": "VLLM",
            "workspace": "delahayecarry"
        }
    }
    
    try:
        validated_config = validate_config(valid_config, allow_defaults=False)
        assert validated_config.model.model_type == "vlm"
        assert validated_config.data.batch_size == 8
        assert validated_config.training.learning_rate == 2e-5
        print("✓ Valid configuration validated successfully")
    except Exception as e:
        print(f"✗ Failed to validate valid config: {e}")
        return False
    
    # Test 2: Invalid configuration (with defaults fallback)
    print("Test 2: Invalid configuration with defaults fallback...")
    
    invalid_config = {
        "model": {"model_type": "invalid_type"},
        "data": {"batch_size": -1},
        "training": {"learning_rate": -0.1}
    }
    
    try:
        validated_config = validate_config(invalid_config, allow_defaults=True)
        # Should use defaults
        assert validated_config.model.model_type == "vlm"  # Default
        assert validated_config.data.batch_size == 8  # Default
        assert validated_config.training.learning_rate == 2e-5  # Default
        print("✓ Invalid configuration handled with defaults fallback")
    except Exception as e:
        print(f"✗ Failed to handle invalid config with defaults: {e}")
        return False
    
    # Test 3: Invalid configuration (strict mode)
    print("Test 3: Invalid configuration in strict mode...")
    
    try:
        validate_config(invalid_config, allow_defaults=False)
        print("✗ Should have raised ValidationError in strict mode")
        return False
    except Exception:  # Should raise ValidationError (pydantic)
        print("✓ Invalid configuration rejected in strict mode")
    
    return True


def test_configuration_merging():
    """Test configuration merging with CLI arguments."""
    print("\n" + "=" * 50)
    print("Testing Configuration Merging")
    print("=" * 50)
    
    import argparse
    
    # Test 1: Basic YAML + CLI merging
    print("Test 1: Basic YAML and CLI argument merging...")
    
    yaml_config = {
        "model": {"model_type": "vlm"},
        "data": {"batch_size": 8, "max_seq_length": 512},
        "training": {"learning_rate": 2e-5, "use_amp": True},
        "swanlab": {"project": "VLLM", "workspace": "delahayecarry"}
    }
    
    # Create CLI args that override some values
    cli_args = argparse.Namespace()
    cli_args.model_type = "llm"  # Override
    cli_args.batch_size = 16  # Override
    cli_args.swanlab_experiment_name = "cli-experiment"  # New value
    cli_args.max_seq_length = None  # Don't override
    cli_args.learning_rate = None  # Don't override
    # Add all other possible CLI attributes as None to avoid AttributeError
    for attr in ['data_path', 'num_workers', 'num_epochs', 'weight_decay', 'warmup_steps', 
                 'gradient_accumulation_steps', 'max_grad_norm', 'use_amp', 'output_dir',
                 'save_steps', 'logging_steps', 'eval_steps', 'swanlab_project', 
                 'swanlab_workspace', 'swanlab_description', 'swanlab_logdir', 'swanlab_api_key']:
        if not hasattr(cli_args, attr):
            setattr(cli_args, attr, None)
    
    try:
        merged_config = merge_config_with_args(yaml_config, cli_args)
        
        # Check CLI overrides
        assert merged_config["model"]["model_type"] == "llm"
        assert merged_config["data"]["batch_size"] == 16
        assert merged_config["swanlab"]["experiment_name"] == "cli-experiment"
        
        # Check preserved YAML values
        assert merged_config["data"]["max_seq_length"] == 512
        assert merged_config["training"]["learning_rate"] == 2e-5
        assert merged_config["swanlab"]["project"] == "VLLM"
        
        print("✓ YAML and CLI configuration merged successfully")
    except Exception as e:
        print(f"✗ Failed to merge configurations: {e}")
        return False
    
    # Test 2: Environment variable integration
    print("Test 2: Environment variable integration...")
    
    import os
    from unittest.mock import patch
    
    with patch.dict(os.environ, {
        'SWANLAB_API_KEY': 'test-env-key'
    }):
        try:
            merged_config = merge_config_with_args(yaml_config, cli_args)
            
            # Environment variables should be added
            assert merged_config["swanlab"]["api_key"] == "test-env-key"
            
            print("✓ Environment variables integrated successfully")
        except Exception as e:
            print(f"✗ Failed to integrate environment variables: {e}")
            return False
    
    return True


def test_complete_pipeline():
    """Test the complete configuration loading and validation pipeline."""
    print("\n" + "=" * 50)
    print("Testing Complete Configuration Pipeline")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test config file
        config_data = {
            "model": {"model_type": "llm"},
            "data": {
                "data_path": "data/pipeline_test",
                "batch_size": 12,
                "max_seq_length": 1024
            },
            "training": {
                "num_epochs": 5,
                "learning_rate": 5e-5,
                "use_amp": False
            },
            "swanlab": {
                "project": "pipeline-test",
                "workspace": "delahayecarry",
                "experiment_name": "complete-pipeline-test"
            }
        }
        
        config_file = temp_path / "pipeline_test.yaml"
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        try:
            # Test complete pipeline
            validated_config = load_and_validate_config(
                config_path=str(config_file),
                cli_args=None,
                allow_defaults=False
            )
            
            # Verify configuration
            assert validated_config.model.model_type == "llm"
            assert validated_config.data.batch_size == 12
            assert validated_config.training.learning_rate == 5e-5
            assert validated_config.swanlab.project == "pipeline-test"
            
            # Test conversion to dict
            config_dict = validated_config.to_training_config_dict()
            assert config_dict["model_type"] == "llm"
            assert config_dict["batch_size"] == 12
            assert config_dict["swanlab_project"] == "pipeline-test"
            
            print("✓ Complete configuration pipeline working correctly")
            
        except Exception as e:
            print(f"✗ Complete pipeline test failed: {e}")
            traceback.print_exc()
            return False
    
    return True


def test_configuration_saving():
    """Test configuration saving functionality."""
    print("\n" + "=" * 50)
    print("Testing Configuration Saving")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Test configuration with empty values to be cleaned
        config_to_save = {
            "model": {
                "model_type": "vlm",
                "model_config_path": "",  # Should be cleaned
                "valid_param": "value"
            },
            "data": {
                "data_path": "data/save_test",
                "batch_size": 16,
                "empty_list": [],  # Should be cleaned
                "none_value": None  # Should be cleaned
            },
            "training": {
                "learning_rate": 5e-5,
                "use_amp": True
            }
        }
        
        try:
            save_merged_config(config_to_save, str(temp_path), "test_save.yaml")
            
            # Check that files were created
            yaml_file = temp_path / "test_save.yaml"
            json_file = temp_path / "test_save.json"
            
            assert yaml_file.exists(), "YAML file should be created"
            assert json_file.exists(), "JSON file should be created"
            
            # Verify YAML content
            import yaml
            with open(yaml_file, 'r') as f:
                saved_config = yaml.safe_load(f)
            
            # Should have cleaned empty values
            assert "model_config_path" not in saved_config["model"]
            assert "empty_list" not in saved_config["data"]
            assert "none_value" not in saved_config["data"]
            
            # Should preserve valid values
            assert saved_config["model"]["model_type"] == "vlm"
            assert saved_config["data"]["batch_size"] == 16
            assert saved_config["training"]["learning_rate"] == 5e-5
            
            print("✓ Configuration saving and cleaning working correctly")
            
        except Exception as e:
            print(f"✗ Configuration saving test failed: {e}")
            return False
    
    return True


def main():
    """Run all validation tests."""
    print("YAML Configuration System - Functional Validation")
    print("=" * 60)
    print("Running basic functionality tests without pytest...")
    
    tests = [
        ("YAML Loading", test_yaml_loading),
        ("Pydantic Validation", test_pydantic_validation), 
        ("Configuration Merging", test_configuration_merging),
        ("Complete Pipeline", test_complete_pipeline),
        ("Configuration Saving", test_configuration_saving)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n✓ {test_name}: PASSED")
            else:
                failed += 1
                print(f"\n✗ {test_name}: FAILED")
        except Exception as e:
            failed += 1
            print(f"\n✗ {test_name}: ERROR - {e}")
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {passed + failed}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 All validation tests passed! YAML configuration system is working correctly.")
        return 0
    else:
        print(f"\n❌ {failed} test(s) failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())