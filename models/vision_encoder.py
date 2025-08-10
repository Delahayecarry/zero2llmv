import os
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor

from configs.llmconfig import llmconfig
from models.llm import CausalLM, MOEFeedForward


class VLLMconfig(llmconfig):
    """
    视觉语言模型配置类 - 扩展基础LLM配置以支持视觉输入
    
    继承自llmconfig，添加视觉相关的特殊配置参数：
    - 图像特殊token的定义
    - 图像token ID的映射
    - 视觉编码器相关参数
    """
    model_type = 'vision'  # 模型类型标识，用于区分纯文本模型

    def __init__(
            self,
            image_special_token: str = '@' * 196,  # 图像占位符token，默认196个@符号对应ViT的196个patch
            image_ids: List = [34] * 196,          # 图像token对应的ID列表，用于在token序列中标识图像位置
            **kwargs,                               # 其他继承自llmconfig的参数
    ):
        """
        初始化视觉语言模型配置
        
        Args:
            image_special_token: 图像在文本中的占位符，长度对应图像patch数量
            image_ids: 图像token的ID列表，用于在输入序列中识别图像位置
            **kwargs: 传递给父类llmconfig的其他配置参数
        """
        self.image_special_token = image_special_token  # 存储图像特殊token
        self.image_ids = image_ids                      # 存储图像token ID列表
        super().__init__(**kwargs)                      # 调用父类初始化方法

class VisionEncoder(torch.nn.Module):
    """
    视觉特征投影器 - 将视觉编码器的特征映射到语言模型的隐藏空间
    
    主要功能：
    1. 接收来自CLIP等视觉编码器的特征向量
    2. 通过线性变换将视觉特征维度对齐到语言模型的隐藏维度  
    3. 确保视觉和文本特征能够在同一语义空间中进行交互
    """
    
    def __init__(
            self,
            ve_hidden_size: int = 512,  # 视觉编码器输出的特征维度（如CLIP-ViT的输出维度）
            hidden_size: int = 512,     # 语言模型的隐藏层维度，需要与LLM配置一致
    ):
        """
        初始化视觉投影器
        
        Args:
            ve_hidden_size: 视觉编码器输出特征的维度
            hidden_size: 目标语言模型隐藏层的维度
        """
        super().__init__()
        self.ve_hidden_size = ve_hidden_size  # 保存视觉编码器特征维度
        self.hidden_size = hidden_size        # 保存目标隐藏层维度
        
        # 视觉特征投影层 - 简单的线性变换
        # 将视觉特征从ve_hidden_size维度映射到hidden_size维度
        self.vision_proj = nn.Sequential(
            nn.Linear(
                self.ve_hidden_size,  # 输入维度：视觉编码器特征维度
                self.hidden_size,     # 输出维度：语言模型隐藏层维度
            )
            # 注：这里可以扩展为更复杂的MLP结构，如添加激活函数、dropout等
            # 例如：nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden_size, hidden_size)
        )

    def forward(
            self,
            image_encoders: torch.Tensor  # 输入的图像编码特征
    ) -> torch.Tensor:
        """
        前向传播：将图像特征投影到语言模型空间
        
        Args:
            image_encoders: 图像编码器输出的特征张量
                          形状: [batch_size, num_patches, ve_hidden_size]
                          或: [num_patches, ve_hidden_size] (单张图像)
        
        Returns:
            vision_proj: 投影后的视觉特征，维度对齐到语言模型
                        形状: [batch_size, num_patches, hidden_size]
                        或: [num_patches, hidden_size]
        """
        # 通过线性投影层变换视觉特征维度
        # 这是多模态融合的关键步骤：将视觉信息映射到文本语义空间
        vision_proj = self.vision_proj(image_encoders)
        return vision_proj  
    
    
class VLM(CausalLM):
    """
    视觉语言模型 - 基于CausalLM的多模态大语言模型
    
    架构设计：
    1. 继承CausalLM的完整语言建模能力（包括MoE、GQA等高级特性）
    2. 集成CLIP视觉编码器进行图像理解
    3. 通过视觉投影器实现视觉-语言特征融合
    4. 支持图文混合输入的端到端训练和推理
    
    核心特性：
    - 🖼️ 多模态输入：同时处理文本和图像
    - 🧠 统一架构：共享语言模型的推理能力
    - ⚡ 高效推理：支持KV缓存加速生成
    - 🎯 灵活配置：支持不同规模的视觉编码器
    """
    config_class = VLLMconfig  # 指定使用的配置类

    def __init__(self, 
                 params: VLLMconfig = None, 
                 vision_model_path: str = "./models/vision_model/clip-vit-base-patch16"):
        """
        初始化VLLVLM模型
        
        Args:
            params: VLLMconfig配置对象，包含模型的所有超参数
            vision_model_path: CLIP视觉模型的本地路径或HuggingFace模型名
        """
        # 初始化语言模型部分（继承CausalLM的所有功能）
        super().__init__(params)
        
        # 设置默认配置
        if not params: 
            params = VLLMconfig()
        self.params = params
        
        # 加载并初始化视觉组件
        # 1. 加载预训练的CLIP视觉编码器和处理器
        self.vision_encoder, self.processor = self.__class__.get_vision_model(vision_model_path)
        
        # 2. 初始化视觉投影器，将视觉特征维度对齐到语言模型
        self.vision_proj = VisionEncoder(hidden_size=params.hidden_size)
        # 注：这里可以根据具体的CLIP模型调整ve_hidden_size参数

    @staticmethod
    def get_vision_model(model_path: str) -> Tuple[Optional[CLIPModel], Optional[CLIPProcessor]]:
        """
        加载预训练的CLIP视觉模型和处理器
        
        采用冻结预训练模型的策略，只训练视觉投影器部分：
        1. 保持CLIP视觉编码器的预训练知识
        2. 减少训练参数量，提高训练效率
        3. 避免视觉编码能力在多模态训练中退化
        
        Args:
            model_path: CLIP模型路径（本地目录或HuggingFace模型名）
            
        Returns:
            model: 冻结参数的CLIP模型（评估模式）
            processor: CLIP图像预处理器
            如果模型路径不存在，返回(None, None)
        """
        # 抑制HuggingFace的详细日志输出
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()
        
        # 检查模型路径是否存在
        if not os.path.exists(model_path):
            print(f"警告：视觉模型路径 {model_path} 不存在，将跳过视觉功能")
            return None, None
            
        # 从预训练模型加载CLIP模型和处理器
        model = CLIPModel.from_pretrained(model_path)
        processor = CLIPProcessor.from_pretrained(model_path)
        
        # 🔒 冻结视觉编码器的所有参数
        # 这是多模态训练的常见策略，避免破坏预训练的视觉表示能力
        for param in model.parameters():
            param.requires_grad = False
            
        # 设置为评估模式，禁用dropout等训练特定的层
        return model.eval(), processor

    @staticmethod
    def image2tensor(image, processor) -> torch.Tensor:
        """
        将PIL图像转换为模型可处理的张量格式
        
        2. 应用CLIP预处理（resize、normalize等）
        3. 转换为PyTorch张量格式
        
        Args:
            image: PIL图像对象
            processor: CLIP图像处理器
            
        Returns:
            inputs: 预处理后的图像张量
                   形状: [1, 3, 224, 224] (标准CLIP输入)
        """
        # 📸 图像格式标准化
        # 将RGBA、LA等格式转换为标准的RGB格式，确保兼容性
        if image.mode in ['RGBA', 'LA']: 
            image = image.convert('RGB')
            
        # 🔄 应用CLIP预处理管道
        # 包括：resize到224x224、归一化、tensor转换等
        inputs = processor(images=image, return_tensors="pt")['pixel_values']
        return inputs

    @staticmethod
    def get_image_embeddings(image_tensors: torch.Tensor, vision_model: CLIPModel) -> torch.Tensor:
        """
        使用CLIP视觉编码器提取图像特征
        
        特征提取细节：
        1. 使用无梯度模式，确保视觉编码器参数不被更新
        2. 获取ViT的patch级特征（排除[CLS] token）
        3. 每张图像产生196个patch特征（14x14网格）
        
        Args:
            image_tensors: 预处理后的图像张量 [batch_size, 3, 224, 224]
            vision_model: 冻结的CLIP模型
            
        Returns:
            img_embedding: 图像patch特征
                          形状: [196, hidden_size] 或 [batch_size, 196, hidden_size]
        """
        # 禁用梯度计算，节省内存并确保视觉编码器参数不被更新
        with torch.no_grad():
            # 通过CLIP视觉编码器处理图像
            outputs = vision_model.vision_model(pixel_values=image_tensors)
            
        # 提取patch特征（排除[CLS] token）
        # last_hidden_state形状: [batch_size, 197, hidden_size] (1个[CLS] + 196个patch)
        # [:, 1:, :] 表示跳过第一个[CLS] token，只保留196个patch特征
        img_embedding = outputs.last_hidden_state[:, 1:, :].squeeze()
        
        return img_embedding

    def count_vision_proj(self, tokens: torch.Tensor, h: torch.Tensor, 
                         vision_tensors: Optional[torch.Tensor] = None, 
                         seqlen: int = 512) -> torch.Tensor:
        """
        视觉特征融合核心函数 - 将图像特征嵌入到文本序列中
        
        实现多模态融合的关键算法：
        1. 在token序列中定位图像占位符位置
        2. 将对应的视觉特征投影后替换占位符
        3. 构建图文融合的统一表示序列
        
        Args:
            tokens: 输入的token序列 [batch_size, seq_len]
            h: 文本嵌入特征 [batch_size, seq_len, hidden_size]
            vision_tensors: 图像特征张量 [batch_size, num_images, 196, vision_hidden_size]
            seqlen: 序列最大长度限制
            
        Returns:
            融合视觉特征后的隐藏状态 [batch_size, seq_len, hidden_size]
        """
        
        def find_indices(tokens: torch.Tensor, image_ids: List[int]) -> Optional[dict]:
            """
            在token序列中查找图像占位符的位置索引
            
            使用滑动窗口算法匹配image_ids序列：
            1. 创建长度为len(image_ids)的滑动窗口
            2. 逐位置比较是否完全匹配image_ids模式
            3. 返回每个batch中所有匹配位置的起止索引
            
            Args:
                tokens: token序列 [batch_size, seq_len]
                image_ids: 图像占位符ID列表
                
            Returns:
                匹配结果字典 {batch_idx: [(start_idx, end_idx), ...]}
                如果没有匹配，返回None
            """
            # 将image_ids转换为tensor并移到相同设备
            image_ids_tensor = torch.tensor(image_ids).to(tokens.device)
            len_image_ids = len(image_ids)
            
            # 序列长度检查：如果image_ids比token序列还长，无法匹配
            if len_image_ids > tokens.size(1):
                return None
                
            # 🔍 滑动窗口匹配算法
            # unfold创建滑动窗口视图：[batch_size, num_windows, window_size]
            tokens_view = tokens.unfold(1, len_image_ids, 1)
            
            # 逐窗口比较，找出完全匹配image_ids的位置
            matches = (tokens_view == image_ids_tensor).all(dim=2)
            
            # 构建匹配结果字典
            return {
                batch_idx: [(idx.item(), idx.item() + len_image_ids - 1) for idx in
                            matches[batch_idx].nonzero(as_tuple=True)[0]]
                for batch_idx in range(tokens.size(0)) if matches[batch_idx].any()
            } or None

        # 🔍 定位图像占位符在token序列中的位置
        image_indices = find_indices(tokens, self.params.image_ids)
        
        # 🖼️ 如果存在图像数据和匹配位置，执行视觉特征融合
        if vision_tensors is not None and image_indices:
            # 通过视觉投影器将视觉特征映射到语言模型空间
            vision_proj = self.vision_proj(vision_tensors)
            
            # 确保视觉特征有正确的批次维度
            if len(vision_proj.shape) == 3:
                vision_proj = vision_proj.unsqueeze(0)
                
            # 🔄 逐batch处理特征融合
            new_h = []
            for i in range(h.size(0)):
                if i in image_indices:  # 当前batch包含图像
                    h_i = h[i]  # 获取当前batch的文本特征
                    img_idx = 0  # 图像索引计数器
                    
                    # 🎯 替换每个图像占位符位置
                    for start_idx, end_idx in image_indices[i]:
                        if img_idx < vision_proj.size(1):  # 确保不超出图像数量
                            # 🧩 特征拼接：文本前缀 + 视觉特征 + 文本后缀
                            h_i = torch.cat((
                                h_i[:start_idx],              # 图像位置之前的文本特征
                                vision_proj[i][img_idx],      # 投影后的视觉特征
                                h_i[end_idx + 1:]             # 图像位置之后的文本特征
                            ), dim=0)[:seqlen]  # 截断到最大序列长度
                            img_idx += 1
                    
                    new_h.append(h_i)
                else:  # 当前batch不包含图像，保持原文本特征
                    new_h.append(h[i])
                    
            # 重新堆叠为批次张量
            return torch.stack(new_h, dim=0)
            
        # 如果没有图像数据，直接返回原始文本特征
        return h

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                pixel_values: Optional[torch.FloatTensor] = None,
                **args):
        """
        VLLVLM前向传播 - 多模态语言建模的核心流程
        
        实现端到端的视觉语言理解与生成：
        1. 📝 文本编码：将input_ids转换为文本嵌入
        2. 🖼️ 视觉编码：处理图像并提取视觉特征  
        3. 🔗 多模态融合：将视觉特征嵌入文本序列
        4. 🧠 Transformer处理：通过多层注意力机制建模
        5. 📊 输出生成：产生下一token的概率分布
        
        Args:
            input_ids: 输入token序列 [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            past_key_values: KV缓存，用于加速推理生成
            use_cache: 是否启用KV缓存
            logits_to_keep: 保留的logits数量（用于内存优化）
            pixel_values: 图像像素值 [batch_size, num_images, channels, height, width]
            
        Returns:
            包含logits、hidden_states、aux_loss等的输出对象
        """
        # 获取输入维度信息
        batch_size, seq_length = input_ids.shape
        
        # 初始化或获取KV缓存
        past_key_values = past_key_values or [None] * len(self.model.layers)
        
        # 计算当前生成位置（用于位置编码和KV缓存）
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        # Step 1: 文本嵌入
        # 将token IDs转换为密集的向量表示
        hidden_states = self.model.dropout(self.model.embed_tokens(input_ids))

        # Step 2 & 3: 视觉处理与多模态融合
        # 只在初始输入时处理图像（start_pos == 0），生成阶段跳过
        if pixel_values is not None and start_pos == 0:
            # 处理图像张量维度
            # 如果输入是6维张量，去掉多余的维度
            if len(pixel_values.shape) == 6:
                pixel_values = pixel_values.squeeze(2)
            
            # 解析图像张量维度：[batch_size, num_images, channels, height, width]
            bs, num, c, im_h, im_w = pixel_values.shape
            
            # 🎯 批量处理图像特征提取
            # 根据批次大小决定堆叠维度
            stack_dim = 1 if bs > 1 else 0
            
            # 🔄 逐张图像提取特征并堆叠
            vision_tensors = torch.stack([
                VLM.get_image_embeddings(
                    pixel_values[:, i, :, :, :],  # 第i张图像
                    self.vision_encoder             # CLIP视觉编码器
                )
                for i in range(num)  # 遍历所有图像
            ], dim=stack_dim)
            
            # 🔗 执行视觉-文本特征融合
            # 将投影后的视觉特征替换文本序列中的图像占位符
            hidden_states = self.count_vision_proj(
                tokens=input_ids, 
                h=hidden_states, 
                vision_tensors=vision_tensors,
                seqlen=input_ids.shape[1]
            )

        # 🎯 Step 4: 位置编码准备
        # 获取对应当前序列位置的RoPE编码
        position_embeddings = (
            self.model.freqs_cos[start_pos:start_pos + seq_length],  # 余弦位置编码
            self.model.freqs_sin[start_pos:start_pos + seq_length]   # 正弦位置编码
        )

        # 🏗️ Step 5: Transformer层级联处理
        presents = []  # 存储新的KV缓存
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.model.layers, past_key_values)):
            # 🧠 通过Transformer层处理（包括注意力机制、FFN/MoE等）
            hidden_states, present = layer(
                hidden_states,         # 输入隐藏状态
                position_embeddings,   # RoPE位置编码
                past_key_value=past_key_value,  # 历史KV缓存
                use_cache=use_cache,    # 是否生成新的KV缓存
                attention_mask=attention_mask   # 注意力掩码
            )
            presents.append(present)  # 收集KV缓存

        # 🔧 Step 6: 最终层归一化
        hidden_states = self.model.norm(hidden_states)

        # 📊 计算MoE辅助损失（如果使用MoE架构）
        aux_loss = sum(
            layer.mlp.aux_loss  # 每层MoE的负载均衡损失
            for layer in self.model.layers
            if isinstance(layer.mlp, MOEFeedForward)  # 只统计MoE层
        )
        
        # 🎯 Step 7: 语言建模头输出
        # 根据logits_to_keep参数决定输出范围（内存优化）
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        
        # 📦 构建输出对象
        self.OUT.__setitem__('last_hidden_state', hidden_states)  # 最终隐藏状态
        self.OUT.__setitem__('logits', logits)                   # 词汇表上的概率分布
        self.OUT.__setitem__('aux_loss', aux_loss)               # MoE辅助损失
        self.OUT.__setitem__('past_key_values', presents)        # 更新的KV缓存
        
        return self.OUT