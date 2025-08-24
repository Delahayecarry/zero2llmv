from transformers import PretrainedConfig

class llmconfig(PretrainedConfig):
    """
    Zero2LLMV 模型配置类
    
    继承自transformers的PretrainedConfig，包含模型的所有超参数配置。
    支持标准的Transformer架构以及可选的专家混合(MoE)扩展。
    
    主要特性：
    - 支持RMSNorm归一化和旋转位置编码(RoPE)
    - 可配置的注意力机制，支持分组查询注意力(GQA)
    - 可选的专家混合(MoE)架构提升模型容量
    - Flash Attention支持加速训练和推理
    """
    model_type = "zero2llm-v"

    def __init__(
            self,
            # 基础模型参数
            dropout: float = 0.1,  # Dropout概率，用于防止过拟合，通常0.1-0.3
            bos_token_id: int = 0,  # 序列开始标记ID，用于生成任务
            eos_token_id: int = 1,  # 序列结束标记ID，用于生成任务
            hidden_act: str = "gelu",  # 激活函数类型：gelu/swish/relu等
            hidden_size: int = 768,  # 模型隐藏层维度，决定模型大小
            intermediate_size: int = None,  # FFN中间层维度，通常为hidden_size的2.7-4倍
            max_positions_embeddings: int = 512,  # 支持的最大序列长度
            
            # 注意力机制参数
            num_attention_heads: int = 12,  # 多头注意力的头数，通常为8/12/16/32
            num_key_value_heads: int = None,  # KV头数，用于分组查询注意力(GQA)，如果为None则等于query头数
            
            # 词汇和位置编码参数  
            vocab_size: int = 30000,  # 词汇表大小，影响embedding层参数量
            rms_norm_eps: float = 1e-5,  # RMS归一化的epsilon，防止除零
            rope_theta: int = 100000.0,  # RoPE位置编码基频，影响位置编码的周期性
            flash_attn: bool = True,  # 是否启用Flash Attention优化

            # MOE（专家混合）相关参数
            use_moe: bool = False,  # 是否启用专家混合模型，提高模型容量而不增加推理成本
            num_experts_per_token: int = 1,  # 每个token激活的专家数量，通常为1-8个
            n_routed_experts: int = 2,  # 可路由专家的总数量，通常为8-64个
            n_shared_experts: int = 2,  # 共享专家数量，所有token都会使用这些专家
            scoring_function: str = "softmax",  # 专家评分函数：softmax/gumbel_softmax等
            aux_loss_alpha: float = 0.0,  # 辅助损失权重，用于负载均衡，防止专家使用不均
            seq_aux: bool = True,  # 是否使用序列级辅助损失而非全局损失
            norm_topk_prob: bool = True,  # 是否对top-k专家概率进行归一化
            **kwargs
    ):
        super().__init__(**kwargs)
        
        # 基础模型参数赋值
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size  # 注意：某些地方会用hidden_dim作为别名
        self.hidden_dim = hidden_size   # 为了兼容性添加的别名
        self.intermediate_size = intermediate_size
        self.max_positions_embeddings = max_positions_embeddings
        self.max_position_embeddings = max_positions_embeddings  # 添加标准名称兼容性
        self.num_hidden_layers = getattr(kwargs, 'num_hidden_layers', 12)  # 添加缺失的层数配置
        
        # 注意力机制参数
        self.num_attention_heads = num_attention_heads
        # 如果没有指定KV头数，则设置为与query头数相同
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        
        # 词汇和编码参数

        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.flash_attn = flash_attn
        # MOE参数赋值
        self.use_moe = use_moe
        self.num_experts_per_token = num_experts_per_token
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.scoring_func = scoring_function  # 注意：这里用scoring_func而不是scoring_function
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob


