"""
Zero2LLMV 训练框架的 Pydantic 配置模型。

本模块使用 pydantic 模型为所有训练参数提供全面的验证和类型检查。
这些模型镜像了现有的 TrainingConfig 数据类，但添加了强大的验证规则、
类型检查和参数依赖验证功能。
"""

from typing import List, Literal, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class ModelConfigModel(BaseModel):
    """模型配置，包含类型验证。"""
    
    model_type: Literal["llm", "vlm"] = Field(
        default="vlm",
        description="模型类型：'llm' 为纯语言模型，'vlm' 为视觉语言模型"
    )
    model_config_path: str = Field(
        default="",
        description="模型配置文件路径（可选）"
    )
    
    @field_validator('model_type')
    @classmethod
    def validate_model_type(cls, v):
        """确保 model_type 是支持的类型之一。"""
        if v not in ["llm", "vlm"]:
            raise ValueError('model_type 必须是 "llm" 或 "vlm"')
        return v


class DataConfigModel(BaseModel):
    """数据配置，包含路径和大小验证。"""
    
    data_path: str = Field(
        default="../dataset/pretrain_data.jsonl",
        description="训练数据目录路径"
    )
    max_seq_length: int = Field(
        default=512,
        gt=0,
        le=8192,
        description="分词的最大序列长度 (1-8192)"
    )
    batch_size: int = Field(
        default=8,
        gt=0,
        le=1024,
        description="训练批次大小 (1-1024)"
    )
    num_workers: int = Field(
        default=4,
        ge=0,
        le=64,
        description="数据加载器工作进程数 (0-64)"
    )
    
    @field_validator('data_path')
    @classmethod
    def validate_data_path(cls, v):
        """确保 data_path 不为空。"""
        if not v.strip():
            raise ValueError('data_path 不能为空')
        return v.strip()


class TrainingHyperparamsModel(BaseModel):
    """训练超参数，包含范围验证和依赖检查。"""
    
    num_epochs: int = Field(
        default=3,
        ge=1,
        le=1000,
        description="训练轮数 (1-1000)"
    )
    learning_rate: float = Field(
        default=2e-5,
        gt=0,
        le=1,
        description="初始学习率 (0-1)"
    )
    weight_decay: float = Field(
        default=0.01,
        ge=0,
        le=1,
        description="权重衰减系数 (0-1)"
    )
    warmup_steps: int = Field(
        default=500,
        ge=0,
        description="学习率调度器的预热步数"
    )
    gradient_accumulation_steps: int = Field(
        default=4,
        ge=1,
        le=128,
        description="梯度累积步数 (1-128)"
    )
    max_grad_norm: float = Field(
        default=1.0,
        gt=0,
        le=10,
        description="梯度裁剪的最大梯度范数 (0-10)"
    )
    use_amp: bool = Field(
        default=True,
        description="是否使用自动混合精度训练"
    )
    
    @field_validator('learning_rate')
    @classmethod
    def validate_learning_rate(cls, v):
        """确保学习率在典型训练范围内。"""
        if v > 0.1:
            # 对异常高的学习率发出警告
            import warnings
            warnings.warn(f"学习率 {v} 异常高 (>0.1)。建议使用更小的值。")
        return v
    
    @model_validator(mode='after')
    def validate_training_params(self):
        """训练超参数的交叉验证。"""
        # 验证梯度累积不会让有效批量过大
        batch_size = getattr(self, 'batch_size', 8)  # 从上下文中，我们知道这个存在
        grad_accum = self.gradient_accumulation_steps
        
        effective_batch_size = batch_size * grad_accum
        if effective_batch_size > 512:
            import warnings
            warnings.warn(
                f"有效批量大小 ({effective_batch_size}) 非常大。"
                f"建议减小 batch_size 或 gradient_accumulation_steps。"
            )
        
        return self


class CheckpointsConfigModel(BaseModel):
    """检查点和日志配置，包含步数验证。"""
    
    output_dir: str = Field(
        default="outputs",
        description="保存模型输出和检查点的目录"
    )
    save_steps: int = Field(
        default=1000,
        gt=0,
        description="检查点保存间隔步数"
    )
    logging_steps: int = Field(
        default=100,
        gt=0,
        description="日志记录间隔步数"
    )
    eval_steps: int = Field(
        default=500,
        gt=0,
        description="评估运行间隔步数"
    )
    
    @field_validator('output_dir')
    @classmethod
    def validate_output_dir(cls, v):
        """确保输出目录路径不为空。"""
        if not v.strip():
            raise ValueError('output_dir 不能为空')
        return v.strip()
    
    @model_validator(mode='after')
    def validate_step_intervals(self):
        """确保日志/评估步数相对于保存步数是合理的。"""
        save_steps = self.save_steps
        logging_steps = self.logging_steps
        eval_steps = self.eval_steps
        
        # 确保日志记录比保存更频繁
        if logging_steps >= save_steps:
            import warnings
            warnings.warn(
                f"logging_steps ({logging_steps}) 应该小于 save_steps ({save_steps}) "
                f"以便进行适当的监控"
            )
        
        # 确保评估不会太频繁以免拖慢训练
        if eval_steps < logging_steps:
            import warnings
            warnings.warn(
                f"eval_steps ({eval_steps}) 比 logging_steps ({logging_steps}) 更频繁。"
                f"这可能会显著拖慢训练。"
            )
        
        return self


class SwanLabConfigModel(BaseModel):
    """SwanLab 实验跟踪配置，包含验证。"""
    
    project: str = Field(
        default="VLLM",
        description="SwanLab 项目名称，用于实验跟踪"
    )
    workspace: str = Field(
        default="delahayecarry",
        description="SwanLab 工作空间"
    )
    experiment_name: str = Field(
        default="",
        description="实验名称（可选，空时自动生成）"
    )
    description: str = Field(
        default="",
        description="实验描述"
    )
    logdir: str = Field(
        default="",
        description="日志目录（可选）"
    )
    
    @field_validator('project')
    @classmethod
    def validate_project(cls, v):
        """确保 swanlab 项目名称不为空。"""
        if not v.strip():
            raise ValueError('swanlab 项目名称不能为空')
        return v.strip()
        
    @field_validator('workspace')
    @classmethod
    def validate_workspace(cls, v):
        """确保 swanlab 工作空间不为空。"""
        if not v.strip():
            raise ValueError('swanlab 工作空间不能为空')
        return v.strip()


class TrainingConfigModel(BaseModel):
    """根配置模型，组合所有训练参数。"""
    
    # 配置章节
    model: ModelConfigModel = Field(
        default_factory=ModelConfigModel,
        description="模型配置参数"
    )
    data: DataConfigModel = Field(
        default_factory=DataConfigModel,
        description="数据加载和预处理配置"
    )
    training: TrainingHyperparamsModel = Field(
        default_factory=TrainingHyperparamsModel,
        description="训练超参数和优化设置"
    )
    checkpoints: CheckpointsConfigModel = Field(
        default_factory=CheckpointsConfigModel,
        description="检查点保存和日志配置"
    )
    swanlab: SwanLabConfigModel = Field(
        default_factory=SwanLabConfigModel,
        description="SwanLab 实验跟踪配置"
    )
    
    class Config:
        """Pydantic 模型配置。"""
        # 使用枚举值进行序列化
        use_enum_values = True
        # JSON schema 配置
        json_schema_extra = {
            "example": {
                "model": {
                    "model_type": "vlm",
                    "model_config_path": ""
                },
                "data": {
                    "data_path": "data/processed",
                    "max_seq_length": 512,
                    "batch_size": 8,
                    "num_workers": 4
                },
                "training": {
                    "num_epochs": 3,
                    "learning_rate": 2e-5,
                    "weight_decay": 0.01,
                    "warmup_steps": 500,
                    "gradient_accumulation_steps": 4,
                    "max_grad_norm": 1.0,
                    "use_amp": True
                },
                "checkpoints": {
                    "output_dir": "outputs",
                    "save_steps": 1000,
                    "logging_steps": 100,
                    "eval_steps": 500
                },
                "swanlab": {
                    "project": "VLLM",
                    "workspace": "delahayecarry",
                    "experiment_name": "experiment-1",
                    "description": "使用默认配置进行训练",
                    "logdir": ""
                }
            }
        }
    
    @model_validator(mode='after')
    def validate_cross_config_dependencies(self):
        """验证不同配置章节间的依赖关系。"""
        model_config = self.model
        data_config = self.data
        training_config = self.training
        checkpoints_config = self.checkpoints
        
        # 验证 VLM 有合理的序列长度用于视觉输入
        if hasattr(model_config, 'model_type') and model_config.model_type == "vlm":
            if hasattr(data_config, 'max_seq_length') and data_config.max_seq_length < 256:
                import warnings
                warnings.warn(
                    f"VLM 模型的 max_seq_length={data_config.max_seq_length} 可能对于"
                    f"视觉语言任务来说太短。建议至少使用 256。"
                )
        
        # 验证检查点保存相对于训练长度是合理的
        if (hasattr(data_config, 'batch_size') and 
            hasattr(training_config, 'num_epochs') and 
            hasattr(training_config, 'gradient_accumulation_steps') and
            hasattr(checkpoints_config, 'save_steps')):
            
            # 估计总步数（粗略计算）
            # 这是近似的，因为我们不知道数据集大小
            estimated_steps_per_epoch = 1000  # 保守估计
            total_steps = (estimated_steps_per_epoch * training_config.num_epochs 
                          // training_config.gradient_accumulation_steps)
            
            if checkpoints_config.save_steps >= total_steps:
                import warnings
                warnings.warn(
                    f"save_steps ({checkpoints_config.save_steps}) >= 估计的总步数 "
                    f"({total_steps})。您可能只能在最后得到一个检查点。"
                )
        
        return self
    
    def to_training_config_dict(self) -> Dict[str, Any]:
        """
        将 pydantic 模型转换为与现有 TrainingConfig 兼容的平均字典。
        
        此方法将层级配置平均化为现有 TrainingConfig 数据类
        期望的平均结构，以保持向后兼容性。
        """
        config_dict = {}
        
        # 模型配置
        config_dict['model_type'] = self.model.model_type
        config_dict['model_config_path'] = self.model.model_config_path
        
        # 数据配置
        config_dict['data_path'] = self.data.data_path
        config_dict['max_seq_length'] = self.data.max_seq_length
        config_dict['batch_size'] = self.data.batch_size
        config_dict['num_workers'] = self.data.num_workers
        
        # 训练配置
        config_dict['num_epochs'] = self.training.num_epochs
        config_dict['learning_rate'] = self.training.learning_rate
        config_dict['weight_decay'] = self.training.weight_decay
        config_dict['warmup_steps'] = self.training.warmup_steps
        config_dict['gradient_accumulation_steps'] = self.training.gradient_accumulation_steps
        config_dict['max_grad_norm'] = self.training.max_grad_norm
        config_dict['use_amp'] = self.training.use_amp
        
        # 检查点配置
        config_dict['output_dir'] = self.checkpoints.output_dir
        config_dict['save_steps'] = self.checkpoints.save_steps
        config_dict['logging_steps'] = self.checkpoints.logging_steps
        config_dict['eval_steps'] = self.checkpoints.eval_steps
        
        # SwanLab 配置（为兼容性添加前缀）
        config_dict['swanlab_project'] = self.swanlab.project
        config_dict['swanlab_workspace'] = self.swanlab.workspace
        config_dict['swanlab_experiment_name'] = self.swanlab.experiment_name
        config_dict['swanlab_description'] = self.swanlab.description
        config_dict['swanlab_logdir'] = self.swanlab.logdir
        
        # 环境变量将单独处理
        config_dict['swanlab_api_key'] = ""   # 来自环境变量
        
        return config_dict