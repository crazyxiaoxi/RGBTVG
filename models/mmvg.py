import torch
import torch.nn as nn
import torch.nn.functional as F
from .vl_transformer import build_vl_transformer
import pdb
from .clip import *
from torchvision.transforms import Resize
from utils.misc import (NestedTensor, nested_tensor_from_tensor_list)
from collections import OrderedDict

# import bitsandbytes as bnb
from transformers import CLIPModel, CLIPConfig, CLIPTextConfig, CLIPTextModel, CLIPVisionConfig, CLIPVisionModel
from transformers import CLIPTokenizer, AutoTokenizer, CLIPImageProcessor
from peft import get_peft_config, PeftModel, get_peft_model, LoraConfig, TaskType,AdaLoraConfig
from torch.nn.parameter import Parameter
from typing import Any, Optional, Tuple, Union
import math


class Modified_CLIPVisionEmbeddings(nn.Module):
    def __init__(self,args, clip_embed):
        super().__init__()
        self.args = args
        self.config = clip_embed.config
        self.embed_dim = clip_embed.embed_dim
        self.image_size = clip_embed.image_size
        self.patch_size = clip_embed.patch_size
        self.class_embedding = clip_embed.class_embedding  # 768
        self.patch_embedding = clip_embed.patch_embedding
        self.num_patches = clip_embed.num_patches
        self.num_positions = clip_embed.num_positions  # 197
        self.position_embedding = clip_embed.position_embedding  # 197 * 768
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)))

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:

        batch_size = pixel_values.shape[0]  # B C H W 
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid], B 768 H/16 W/16
        h, w = patch_embeds.shape[2], patch_embeds.shape[3]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)  # B L H
        class_embeds = self.class_embedding.expand(batch_size, 1, -1)  # B * 1 * 768
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)

        cls_pos = self.position_embedding.weight[0:1, :]
        abs_pos = self.position_embedding.weight[1:, :]  # 196 * 768
        xy_num = abs_pos.shape[0]
        assert xy_num == self.num_patches  # 196
        size = int(math.sqrt(xy_num))  # 14
        assert size * size == xy_num

        if size != h or size != w:
            new_abs_pos = F.interpolate(  # 1 14 14 768 --> 1 768 14 14 --> 1 768 40 40
                abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2),
                size=(h, w),
                mode="bicubic",
                antialias=True,
                align_corners=False,
            )
            new_abs_pos = new_abs_pos.permute(0, 2, 3, 1).reshape(1, h * w, -1)
            position_embedding = torch.cat([cls_pos.unsqueeze(0), new_abs_pos], dim=1)  # 1 1601 768
            embeddings = embeddings + position_embedding.repeat(batch_size, 1, 1)
        else:  # 14 == 14
            embeddings = embeddings + self.position_embedding(self.position_ids)

        return embeddings


class VisionEmbeddings(nn.Module):
    def __init__(self, clip_embed):
        super().__init__()
        self.config = clip_embed.config
        self.embed_dim = clip_embed.embed_dim
        self.image_size = clip_embed.image_size
        self.patch_size = clip_embed.patch_size
        self.class_embedding = clip_embed.class_embedding  # 768
        self.patch_embedding = clip_embed.patch_embedding
        self.num_patches = clip_embed.num_patches
        self.num_positions = clip_embed.num_positions  # 此时是197
        self.position_embedding = clip_embed.position_embedding  # 197 * 768
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)))

    def forward(self, pixel_values: torch.FloatTensor, position_embedding) -> torch.Tensor:
        batch_size = pixel_values.shape[0]  # B C H W
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid], B 768 H/16 W/16
        h, w = patch_embeds.shape[2], patch_embeds.shape[3]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)  # B L H
        class_embeds = self.class_embedding.expand(batch_size, 1, -1)  # B * 1 * 768
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)

        embeddings = embeddings + position_embedding(self.position_ids)

        return embeddings


class ClassInstantier(OrderedDict):
    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)

#
# ACT2CLS = {
#     "gelu": GELUActivation,
#     "gelu_10": (ClippedGELUActivation, {"min": -10, "max": 10}),
#     "gelu_fast": FastGELUActivation,
#     "gelu_new": NewGELUActivation,
#     "gelu_python": (GELUActivation, {"use_gelu_python": True}),
#     "gelu_pytorch_tanh": PytorchGELUTanh,
#     "gelu_accurate": AccurateGELUActivation,
#     "laplace": LaplaceActivation,
#     "linear": LinearActivation,
#     "mish": MishActivation,
#     "quick_gelu": QuickGELUActivation,
#     "relu": nn.ReLU,
#     "relu2": ReLUSquaredActivation,
#     "relu6": nn.ReLU6,
#     "sigmoid": nn.Sigmoid,
#     "silu": SiLUActivation,
#     "swish": SiLUActivation,
#     "tanh": nn.Tanh,
# }
# ACT2FN = ClassInstantier(ACT2CLS)


class CLIP_Cross_Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)    
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim) 
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):

        # print(torch.equal(aa.flatten(), tensor.flatten()))
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    

    def forward(
        self,
        hidden_states: torch.Tensor,
        hidden_states_2: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj

        query_states = self.q_proj(hidden_states) * self.scale
        query_states = self._shape(query_states, -1, bsz)  #torch.Size([32, 12, 197, 64])

        key_states = self._shape(self.k_proj(hidden_states_2), -1, bsz) #torch.Size([77, 32, 768])->torch.Size([32, 12, 77, 64])
        # todo检查self._shape
        value_states = self._shape(self.v_proj(hidden_states_2), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = query_states.view(*proj_shape) #torch.Size([32*12, 197, 64])
        key_states = key_states.view(*proj_shape)     #torch.Size([32*12, 77, 64])
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {causal_attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len) 
        # import pdb; pdb.set_trace()
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
   
        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) 
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)
   
        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)


        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights_reshaped

class CLIP_Cross_Attention_VS(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)    
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim) 
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):

        # print(torch.equal(aa.flatten(), tensor.flatten()))
        return tensor.contiguous().view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,  # 作为 Key/Value 的来源（原逻辑中是 Query 来源）
            text_states: torch.Tensor,  # 作为 Query 的来源（原逻辑中是 Key/Value 来源）
            attention_mask: Optional[torch.Tensor] = None,
            causal_attention_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        输入形状：
        - hidden_states: (bsz, src_len, embed_dim)  # 作为 Key/Value
        - text_states: (bsz, tgt_len, embed_dim)    # 作为 Query
        """

        # 获取文本序列长度（Query 长度）和批次大小
        text_len,bsz,  embed_dim = text_states.size()  # 这里改为从 text_states 取形状（因为它是 Query）
        # 获取图像/隐藏状态序列长度（Key/Value 长度）
        tgt_len = hidden_states.size(1)  # hidden_states 形状：(bsz, tgt_len, embed_dim)

        # --------------------------
        # 1. 生成 Query（来自 text_states）
        # --------------------------

        query_states = self.q_proj(text_states) * self.scale  # text 作为 Query
        query_states = self._shape(query_states, text_len, bsz)  # 形状：(bsz, num_heads, text_len, head_dim)

        # --------------------------
        # 2. 生成 Key 和 Value（来自 hidden_states）
        # --------------------------
        # Key 投影 + 形状调整：(bsz, tgt_len, embed_dim) → (bsz, num_heads, tgt_len, head_dim)
        key_states = self._shape(self.k_proj(hidden_states), tgt_len, bsz)
        # Value 投影 + 形状调整：同上
        value_states = self._shape(self.v_proj(hidden_states), tgt_len, bsz)

        # --------------------------
        # 3. 调整形状以进行批量矩阵乘法
        # --------------------------
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = query_states.view(*proj_shape)  # (bsz*num_heads, text_len, head_dim)
        key_states = key_states.view(*proj_shape)  # (bsz*num_heads, tgt_len, head_dim)
        value_states = value_states.view(*proj_shape)  # (bsz*num_heads,tgt_len, head_dim)

        # --------------------------
        # 4. 计算注意力权重
        # --------------------------
        # 注意力分数：(bsz*num_heads, text_len,tgt_len)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        # 检查注意力权重形状是否正确
        if attn_weights.size() != (bsz * self.num_heads,text_len,tgt_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, text_len,tgt_len)}, but is"
                f" {attn_weights.size()}"
            )

        # --------------------------
        # 5. 应用掩码（若有）
        # --------------------------
        # 因果掩码（若需要，例如文本生成时的自注意力）
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1,text_len,tgt_len):
                raise ValueError(
                    f"Causal attention mask should be of size {(bsz, 1, text_len,tgt_len)}, but is"
                    f" {causal_attention_mask.size()}"
                )
            # 重塑后加掩码
            attn_weights = attn_weights.view(bsz, self.num_heads, text_len, tgt_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, text_len, tgt_len)

        # 普通注意力掩码（例如padding掩码）
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1,text_len, tgt_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, text_len,tgt_len)}, but is {attention_mask.size()}"
                )
            # 重塑后加掩码
            attn_weights = attn_weights.view(bsz, self.num_heads, text_len, tgt_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, text_len, tgt_len)

        # --------------------------
        # 6. 注意力归一化与 dropout
        # --------------------------
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        # --------------------------
        # 7. 计算注意力输出
        # --------------------------
        text_guided_vision= torch.bmm(attn_probs, value_states)  # (bsz*num_heads, text_len, head_dim)
        attn_weights_t = attn_weights.transpose(1, 2)
        # 聚合文本信息到视觉维度：(16*num_heads, 197, 64) → 与原始视觉长度一致
        attn_output = torch.bmm(attn_weights_t, text_guided_vision)
        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        # --------------------------
        # 8. 重塑输出并投影
        # --------------------------
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)  # 恢复多头维度
        attn_output = attn_output.transpose(1, 2)  # (bsz, tgt_len, num_heads, head_dim)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)  # 合并多头：(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)  # 最终线性投影

        # 输出注意力权重（若需要）
        if output_attentions:
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, text_len, tgt_len)
        else:
            attn_weights_reshaped = None

        return attn_output, attn_weights_reshaped


class CLIPAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):

        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {causal_attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class CLIPMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # self.activation_fn = ACT2FN[config.hidden_act]
        # self.activation_fn = nn.ReLU
        self.activation_fn = QuickGELU()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)  # 768, 3072
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class TOKEN_MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.activation_fn = QuickGELU()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class CLIPEncoderLayer_with_Crossmodal_Text_Guided_Fusion(nn.Module):

    def __init__(self, args, i, clip_encoder_layer, config: CLIPConfig, adapt_layer, extract_text_layer, text_config):
        super().__init__()
        self.args = args
        # 是否启用 Language-Aware Visual Synergy（LAVS）
        # lavs_mode: 'lavs' | 'iwm' | 'cmx'
        self.lavs_mode = getattr(args, "lavs_mode", "lavs")
        self.enable_text_guided_fusion = (self.lavs_mode == "lavs")
        self.embed_dim = clip_encoder_layer.embed_dim
        self.self_attn = clip_encoder_layer.self_attn

        """ Multi-layer Adaptive Cross-modal Text_Guided_Fusion """
        # self.enable_adaptive_weights = args.enable_adaptive_weights
        if i in adapt_layer:
            if self.args.modality == 'rgb':
                self.cross_norm_sv = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)  # eps=1e-05
                self.cross_attn_sv = CLIP_Cross_Attention_VS(config)
                self.cross_mlp_sv = CLIPMLP(config)
            elif self.args.modality == 'rgbt':
                self.cross_norm_st = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)  # eps=1e-05
                self.cross_norm_sv = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)  # eps=1e-05
                self.cross_attn_st = CLIP_Cross_Attention_VS(config)
                self.cross_attn_sv = CLIP_Cross_Attention_VS(config)
                self.cross_mlp_st = CLIPMLP(config)
                self.cross_mlp_sv = CLIPMLP(config)


        text_embed_dim = text_config.hidden_size  # 512 for base model, 768 for Large model
        # self.cross_gate = nn.Linear(text_embed_dim * len(extract_text_layer), self.embed_dim)  # clip vision 768
        # self.cross_adaptive_weights = nn.ModuleList([nn.Embedding(77, text_embed_dim) for i in range(len(extract_text_layer))])

        self.layer_norm1 = clip_encoder_layer.layer_norm1
        self.mlp = clip_encoder_layer.mlp
        self.layer_norm2 = clip_encoder_layer.layer_norm2

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer,
        adapt_layer,
        text_states,
        cur_modality,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)

        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states
        """ Multi-layer Adaptive Cross-modal Text_Guided_Fusion (LAVS) """
        # 只有在 lavs_mode == 'lavs' 时才启用文本引导的融合
        if self.enable_text_guided_fusion and (layer in adapt_layer):
            text_guided_fusion = True
            if text_guided_fusion == True:

                if cur_modality=='ir':
                    residual = hidden_states
                    hidden_states = self.cross_norm_st(hidden_states)
                    text_states = text_states[-1].to(hidden_states.dtype).permute(1, 0, 2)
                    
                    hidden_states, attn_weights = self.cross_attn_st(
                        hidden_states=hidden_states,
                        text_states=text_states,
                        attention_mask=attention_mask,
                        causal_attention_mask=causal_attention_mask,
                        output_attentions=output_attentions,
                    )
                    hidden_states = self.cross_mlp_st(hidden_states)
                else:
                    residual = hidden_states
                    hidden_states = self.cross_norm_sv(hidden_states)
                    text_states = text_states[-1].to(hidden_states.dtype).permute(1, 0, 2)
                    hidden_states, attn_weights = self.cross_attn_sv(
                        hidden_states=hidden_states,
                        text_states=text_states,
                        attention_mask=attention_mask,
                        causal_attention_mask=causal_attention_mask,
                        output_attentions=output_attentions,
                    )
                    hidden_states = self.cross_mlp_sv(hidden_states)
            hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)


        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class CLIPEncoder_with_Crossmodal_Text_Guided_Fusion(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`CLIPEncoderLayer`].

    Args:
        config: CLIPConfig
    """

    def __init__(self, args, clip_encoder, adapt_layer, extract_text_layer, text_config):
        super().__init__()
        self.config = clip_encoder.config
        self.layers = nn.ModuleList([CLIPEncoderLayer_with_Crossmodal_Text_Guided_Fusion(args, i, clip_encoder.layers[i], self.config,
                                                                             adapt_layer, extract_text_layer,
                                                                             text_config)
                                     for i in range(self.config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        adapt_layer,
        text_states,
        cur_modality,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            causal_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Causal mask for the text model. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(encoder_layer),
                    hidden_states=hidden_states,
                    layer = idx,
                    adapt_layer = adapt_layer,
                    text_states = text_states,
                    cur_modality = cur_modality,
                    attention_mask=attention_mask,
                    causal_attention_mask=causal_attention_mask,
                    output_attentions=output_attentions,
                    )

            else:
                layer_outputs = encoder_layer(

                    hidden_states=hidden_states,
                    layer = idx,
                    adapt_layer = adapt_layer,
                    text_states = text_states,
                    cur_modality=cur_modality,
                    attention_mask=attention_mask,
                    causal_attention_mask=causal_attention_mask,
                    output_attentions=output_attentions,

                )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)

        return {"last_hidden_state": hidden_states, "hidden_states": encoder_states, "attentions": all_attentions}


class CLIP_Vision_Model_with_Crossmodal_Text_Guided_Fusion(nn.Module):
    def __init__(self, args, clip_visu_model, adapt_layer, extract_text_layer, text_config):
        super().__init__()
        self.config = clip_visu_model.config
        self.embeddings = Modified_CLIPVisionEmbeddings(args, clip_visu_model.embeddings)
        self.pre_layrnorm = clip_visu_model.pre_layrnorm  # 原版代码拼错了
        self.encoder = CLIPEncoder_with_Crossmodal_Text_Guided_Fusion(args, clip_visu_model.encoder, adapt_layer, extract_text_layer,
                                                          text_config)
        self.post_layernorm = clip_visu_model.post_layernorm

    # @add_start_docstrings_to_model_forward(CLIP_VISION_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPVisionConfig)
    def forward(
        self,
        adapt_layer,
        text_states,
        reg_src,
        cur_modality,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            adapt_layer=adapt_layer,
            text_states=text_states,
            cur_modality=cur_modality,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs["last_hidden_state"]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return {
            "last_hidden_state": last_hidden_state,
            "pooler_output": pooled_output,
            "hidden_states": encoder_outputs["hidden_states"],
            "attentions": encoder_outputs["attentions"],
        }


"""
   MMVG is implemented on the basis of CLIP-VG, Github: https://github.com/linhuixiao/CLIP-VG
"""


class MMVG(nn.Module):
    def __init__(self, args):
        super(MMVG, self).__init__()
        self.args = args
        # LAVS 融合方式：'lavs'（默认原始 LAVS）、'iwm'、'cmx'
        # 只在 rgbt 模态下起作用
        self.lavs_mode = getattr(args, "lavs_mode", "lavs")
        print("init MMVG model...")
        if (args.model == "ViT-L/14-336"):
            print("init CLIP ViT-L/14-336")
            self.clip = CLIPModel.from_pretrained("/home/shared/pretrain_model/pretrained_weights/CLIP/clip-vit-large-patch14-336")
            self.extract_vision_layer = [12, 16, 20, 24]  # v4
            self.adapt_layer = [11, 15, 19, 23]
            self.patch_size = 14
        elif (args.model == "ViT-L/14"):  # main large model
            print("init CLIP ViT-L/14")
            self.clip = CLIPModel.from_pretrained("/home/shared/pretrain_model/pretrained_weights/CLIP/clip-vit-large-patch14")
            self.extract_vision_layer = [6, 12, 18, 24]  # final 版本
            self.adapt_layer = [] if args.warmup is True else [4, 10, 16, 22]  # large model is trained on two phrases
            self.patch_size = 14
        elif (args.model == "ViT-B/32"):
            print("init CLIP ViT-B/32")
            self.clip = CLIPModel.from_pretrained("/home/shared/pretrain_model/pretrained_weights/CLIP/clip-vit-base-patch32")
            self.extract_vision_layer = [1, 4, 8, 12]
            self.adapt_layer = [0, 3, 7, 11]
            self.patch_size = 32
        else:  # default base model
            print("init CLIP ViT-B/16")
            # self.clip = CLIPModel.from_pretrained("/home/shared/pretrain_model/pretrained_weights/CLIP/clip-vit-base-patch16")
            self.clip = CLIPModel.from_pretrained("../dataset_and_pretrain_model/pretrain_model/pretrained_weights/CLIP/clip-vit-base-patch16")
            """
             Note that there is no mistake here. Note that [1, 4, 8, 12], [0, 3, 7, 11] are the same layer.
             In the internal implementation of transformers, the index at vision branch [0] is the original
             image embedding. 
            """
            self.extract_vision_layer = [1, 4, 8, 12]
            self.adapt_layer = [0, 3, 7, 11]
            self.patch_size = 16
        # set extract_text_layer
        self.mixup_pretrain = args.mixup_pretrain
        if self.mixup_pretrain:
            self.extract_text_layer = [12]
        else:
            if args.dataset == "gref_umd" or args.dataset == "gref":
                self.extract_text_layer = [i+1 for i in range(12)]
            elif args.dataset == "unc+":
                self.extract_text_layer = [6, 12]
            elif args.dataset == "unc":
                self.extract_text_layer = [12]
            elif args.dataset == "referit":
                self.extract_text_layer = [6, 12]
            else:
                self.extract_text_layer = [12]

        print("\nextract vision layer: ", self.extract_vision_layer)
        print("extract text layer: ", self.extract_text_layer)
        print("image size: ", args.imsize, " * ", args.imsize)

        print("adapt_layer: ", self.adapt_layer)

        self.clip.vision_model = CLIP_Vision_Model_with_Crossmodal_Text_Guided_Fusion(args, self.clip.vision_model,
                                                                          self.adapt_layer, self.extract_text_layer,
                                                                          self.clip.text_model.config)
        # 语言感知视觉协同（LAVS）或替代模块（IWM/CMX/AVG）
        # 注意：IWM/CMX 依赖 self.hidden_dim，因此初始化放在 self.hidden_dim 确定之后
        self.cross_fusion_layers_vt = nn.ModuleList()
        self.cross_fusion_layers_tv = nn.ModuleList()
        self.iwm = None
        self.cmx = None

        """
            srameter in self.clip.parameters():
                        parameter.requires_grad_(False)elf.clip.print_trainable_parameters()
            Note that the essence of the HiLoRA mechanism is a process of decomposing parameter learning, and its
            effectiveness is influenced by the learning rate and the number of epochs. Therefore, HiLoRA requires
            different learning rates and numbers of epochs at various stages for specific model configurations.
            If you do not need to enable HiLoRA, simply leave args.hi_lora_stage=0 as the default.
        """
        open_lora = True
        self.open_lora = open_lora
        self.set_HiLoRA(args,open_lora)

        self.hidden_dim = self.clip.projection_dim  # base model 512，large model 768
        self.imsize = args.imsize
        clip_visu_hidden_dim = self.clip.vision_model.config.hidden_size  # 768
        self.visu_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.condition_text_proj = nn.Linear(self.hidden_dim, clip_visu_hidden_dim)  # clip vision 768
        self.ml_text_feat_perceiver = nn.Linear(self.clip.text_embed_dim * len(self.extract_text_layer), clip_visu_hidden_dim)
        self.text_proj = nn.Linear(self.hidden_dim, self.hidden_dim)  # clip vision 768
        self.reg_token = nn.Embedding(1, self.hidden_dim)

        # 在 hidden_dim 确定后，根据 lavs_mode 初始化融合模块
        if self.args.modality == "rgbt":
            if self.lavs_mode == "lavs":
                for _ in self.extract_vision_layer:
                    cross_attn_vt = CLIP_Cross_Attention(self.clip.vision_model.encoder.config)
                    cross_attn_tv = CLIP_Cross_Attention(self.clip.vision_model.encoder.config)
                    self.cross_fusion_layers_vt.append(cross_attn_vt)
                    self.cross_fusion_layers_tv.append(cross_attn_tv)
            elif self.lavs_mode == "iwm":
                self.iwm = IWM(k=1, alpha=0.8)
            elif self.lavs_mode == "cmx":
                self.cmx = CMX(dim=self.hidden_dim)
            elif self.lavs_mode == "avg":
                # 简单 0.5 / 0.5 加权相加，不需要额外模块
                pass

        # divisor = 16
        self.num_visu_token = int((args.imsize / self.patch_size) ** 2)
        self.num_text_token = args.max_query_len
        num_total = self.num_visu_token + 1 + self.num_text_token + 1  # v token + [cls]token + t token + [REG]token
        if args.modality=='rgbt':
            num_total = self.num_visu_token*2 + 1 + self.num_text_token + 1  # v token + [cls]token + t token + [REG]token
        self.vl_pos_embed = nn.Embedding(num_total, self.hidden_dim)

        self.vl_transformer = build_vl_transformer(args)

        self.bbox_embed = MLP(self.hidden_dim, self.hidden_dim, 4, 3)
        self.reg_pos_embed = nn.Embedding(1, self.hidden_dim)
        self.condition_text_pos_embed = nn.Embedding(self.num_text_token, clip_visu_hidden_dim)
        self.ml_visual_projection = nn.Linear(len(self.extract_vision_layer) * self.clip.vision_model.config.hidden_size,
                                            self.hidden_dim)
        self.ml_visual_projection.weight = nn.Parameter(torch.cat([self.clip.visual_projection.weight for i
                                                                in range(len(self.extract_vision_layer))], dim=1))
        # if args.modality=='rgbt':
        #     self.ml_visual_projection_ir = nn.Linear(len(self.extract_vision_layer) * self.clip.vision_model.config.hidden_size,
        #                                       self.hidden_dim)
        #     self.ml_visual_projection_ir.weight = nn.Parameter(self.ml_visual_projection.weight.clone())
              

        self.visu_token_norm = nn.LayerNorm(self.hidden_dim, eps=1e-05)  # 512, eps=1e-05
        self.visu_token_mlp = TOKEN_MLP(self.hidden_dim, 3072)  # 3072

        # TODO：Segmentation head for Referring Image Segmentation task. RIS works only when the seg mask is used.
        #  seg conv, 10GB, 14*14 --> 28*28 --> 56*56 --> 112*112
        hidden_dim = self.hidden_dim
        self.seg_conv1 = nn.ConvTranspose2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(2, 2), stride=(2, 2),
                                            padding=(0, 0), output_padding=(0, 0), bias=False)  # bias=False
        self.seg_conv2 = nn.ConvTranspose2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(2, 2), stride=(2, 2),
                                            padding=(0, 0), output_padding=(0, 0), bias=False)  # bias=False
        self.seg_conv3 = nn.ConvTranspose2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(2, 2), stride=(2, 2),
                                            padding=(0, 0), output_padding=(0, 0), bias=False)  # bias=False
    def set_HiLoRA(self, args,open_lora):
        
        open_text_text_guided_fusion = True
        close_lora_parameter_update = False
        close_lora_vision_parameter_update = False
        close_lora_text_parameter_update = False

        if open_lora:
            target_modules = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj",
                                                            "fc_in", "fc_out", "wte"]
            #peft_config = LoraConfig(target_modules=target_modules, inference_mode=False, r=32, lora_alpha=16,
            #                         lora_dropout=0.1, bias='none')
            # LoRA rank 可通过外部参数控制
            lora_r_rgb = int(getattr(args, "lora_r_rgb", 16))
            lora_r_ir = int(getattr(args, "lora_r_ir", 48))
            print(f"LoRA ranks: rgb={lora_r_rgb}, ir={lora_r_ir}")
            peft_config_rgb = LoraConfig(task_type="FEATURE_EXTRACTION", target_modules=target_modules, inference_mode=False,
                                         r=lora_r_rgb, lora_alpha=16, lora_dropout=0.1, bias='none')
            peft_config_ir = LoraConfig(task_type="FEATURE_EXTRACTION", target_modules=target_modules, inference_mode=False,
                                        r=lora_r_ir, lora_alpha=16, lora_dropout=0.1, bias='none')

            # peft_config = AdaLoraConfig(target_modules=target_modules, inference_mode=False, r=8, lora_alpha=8,
            #                          lora_dropout=0, bias='none')
            self.clip = get_peft_model(self.clip, peft_config_rgb,"lora_rgb")
            self.clip.add_adapter("lora_ir",peft_config_ir)
            for parameter in self.clip.parameters():
                parameter.requires_grad_(False)
            self.clip.print_trainable_parameters()

            if args.hi_lora_stage == 1:
                print("open lora stage 1")
                #self.clip = get_peft_model(self.clip, peft_config)
                #self.clip.print_trainable_parameters()
                #for parameter in self.clip.vision_model.parameters():
                #    parameter.requires_grad_(False)
                #import pdb;pdb.set_trace()
                for name, param in self.clip.named_parameters():
                    if "lora_A" in str(name).split(".") or "lora_B" in str(name).split("."):
                        if "0" in str(name).split(".") or "1" in str(name).split(".") or "2" in str(name).split(".") \
                                or "3" in str(name).split(".") or "4" in str(name).split("."):
                            print("param name: ", name)
                            param.requires_grad_(True)
                    else:
                        param.requires_grad_(False)
            if args.hi_lora_stage == 2:
                # stage 1:
                print("open lora stage 1")
                # self.clip = get_peft_model(self.clip, peft_config)
                #self.clip.print_trainable_parameters()
                #for parameter in self.clip.parameters():
                #    parameter.requires_grad_(False)
                #self.clip.print_trainable_parameters()

                # stage 2:
                print("open lora stage 2")
                # self.clip = get_peft_model(self.clip, peft_config)
                #self.clip.print_trainable_parameters()
                #for parameter in self.clip.vision_model.parameters():
                #    parameter.requires_grad_(False)

                for name, param in self.clip.named_parameters():
                    if "lora_A" in str(name).split(".") or "lora_B" in str(name).split("."):
                        if "0" in str(name).split(".") or "1" in str(name).split(".") or "2" in str(name).split(".") \
                                or "3" in str(name).split(".") or "4" in str(name).split(".") \
                                or "5" in str(name).split(".") or "6" in str(name).split(".") \
                                or "7" in str(name).split("."):
                            print("param name: ", name)
                            param.requires_grad_(True)
            
            if args.hi_lora_stage == 3:
            # if args.hi_lora_stage == 3 or args.hi_lora_stage == 1 or args.hi_lora_stage == 2 :
                # stage 1:
                print("open lora stage 1")
                # self.clip = get_peft_model(self.clip, peft_config)
                # self.clip.print_trainable_parameters()
                # for parameter in self.clip.parameters():
                #     parameter.requires_grad_(False)
                # self.clip.print_trainable_parameters()

                # stage 2:
                print("open lora stage 2")
                # self.clip = get_peft_model(self.clip, peft_config)
                #self.clip.print_trainable_parameters()
                #for parameter in self.clip.parameters():
                #    parameter.requires_grad_(False)
                #self.clip.print_trainable_parameters()

                # stage 3:
                print("open lora stage 3")
                # self.clip = get_peft_model(self.clip, peft_config)
                #self.clip.print_trainable_parameters()

                #for parameter in self.clip.vision_model.parameters():
                #    parameter.requires_grad_(False)

                for name, param in self.clip.named_parameters():
                    if "lora_A" in str(name).split(".") or "lora_B" in str(name).split("."):
                        if "0" in str(name).split(".") or "1" in str(name).split(".") or "2" in str(name).split(".") \
                                or "3" in str(name).split(".") or "4" in str(name).split(".") \
                                or "5" in str(name).split(".") or "6" in str(name).split(".") \
                                or "7" in str(name).split(".") \
                                or "8" in str(name).split(".") or "9" in str(name).split(".") \
                                or "10" in str(name).split(".") or "11" in str(name).split("."):
                            print("param name: ", name)
                            param.requires_grad_(True)
                    else:
                        param.requires_grad_(False)

            if close_lora_parameter_update:
                for parameter in self.clip.parameters():
                    parameter.requires_grad_(False)
                self.clip.print_trainable_parameters()
            else:
                if close_lora_vision_parameter_update:
                    for parameter in self.clip.vision_model.parameters():
                        parameter.requires_grad_(False)
                    self.clip.print_trainable_parameters()
                if close_lora_text_parameter_update:
                    for parameter in self.clip.text_model.parameters():
                        parameter.requires_grad_(False)
                    self.clip.print_trainable_parameters()
            if open_text_text_guided_fusion == True:
                print("Open Multi-layer Adaptive Cross-modal Text_Guided_Fusion parameters ...")
                for name, param in self.clip.vision_model.encoder.layers.named_parameters():
                    if "cross_attn_sv" in str(name).split(".") or "cross_attn_st" in str(name).split(".") or\
                            "cross_norm_sv" in str(name).split(".") or "cross_norm_st" in str(name).split(".") \
                      or "cross_mlp_sv" in str(name).split(".") or "cross_mlp_st" in str(name).split("."):
                        print("param name: ", name)
                        param.requires_grad_(True)
            self.clip.print_trainable_parameters()
    def tensorize_inputs(self, images: NestedTensor, texts: NestedTensor):
        image_tensors = images.tensors
        texts_tensors = texts.tensors

        return image_tensors, texts_tensors

    def get_masks(self, images: NestedTensor, texts: NestedTensor):
        # torch_resize = Resize([14, 14])
        torch_resize = Resize([int(self.imsize / self.patch_size), int(self.imsize / self.patch_size)])  # 14 * 14 = 196， or， 16 * 16 = 256
        visu_masks = torch_resize(images.mask)
        visu_masks = visu_masks.to(torch.bool)
        visu_masks = visu_masks.flatten(1)  # visu_mask：B*L, torch.Size([B, 196])
        # text mask follow bert process
        # text_masks = texts.mask.to(torch.bool)
        # text_masks = ~text_masks
        # text_masks = text_masks.flatten(1)
        # assert text_masks is not None

        return visu_masks

    def encode_text(self, text_data, device=None):
        text_tensors = clip.tokenize(text_data, context_length=77, truncate=True).to(device)  # 4 * 77
        text_mask = text_tensors.eq(0).bool()  # 4 * 77, The ones that need masking are 1.
        return text_tensors, text_mask

    def forward(self, img_data, text_data):
        if self.open_lora:
            self.clip.set_adapter("lora_rgb")
        if self.args.modality =="rgbt":
            image_tensors_ir=img_data.tensors[:,3:,:,:].repeat(1,3,1,1)
            img_data_ir_mask=img_data.mask
            img_data_ir = NestedTensor(image_tensors_ir,img_data_ir_mask)
            img_data.tensors=img_data.tensors[:,:3,:,:]
        batch_size = img_data.tensors.shape[0]  # 得到batch_size
        image_tensors = img_data.tensors
        
        text_tensors, text_mask = self.encode_text(text_data, img_data.tensors.device)

        clip_text_features = self.clip.text_model(text_tensors, output_attentions=True, output_hidden_states=True,
                                                  return_dict=True)  # B * 77 * 512
        text_features = self.clip.text_projection(clip_text_features.last_hidden_state)
        text_eos_embed = self.clip.text_projection(clip_text_features.pooler_output)  # torch.Size([64, 512])

        if self.mixup_pretrain:
            ml_text_features = [self.condition_text_proj(text_features.float())]
        else:
            ml_text_features = [clip_text_features.hidden_states[i] for i in self.extract_text_layer]

        visu_mask = self.get_masks(img_data, text_data)

        # target regression token
        reg_src = self.reg_token.weight.unsqueeze(0).repeat(batch_size, 1, 1)  # B * 1 * hidden_dim

        # for i in range(13):
        #     with open(f"./bs4_output_{i}.txt", "w") as f:
        #         print(clip_image_features["hidden_states"][i][0],file=f)
        #     f.close()
        if self.open_lora:
            self.clip.set_adapter("lora_rgb")        

        clip_image_features = self.clip.vision_model(adapt_layer = self.adapt_layer, 
                                                     text_states = ml_text_features, 
                                                     reg_src =reg_src, 
                                                     cur_modality = "rgb",
                                                     pixel_values = image_tensors, 
                                                     output_attentions=True, 
                                                     output_hidden_states=True,
                                                     return_dict=True)  # B * 197 * 512
        # attention_map = clip_image_features["attentions"]  # tuple, used for draw the attention map

        ml_image_features = [clip_image_features["hidden_states"][i] for i in self.extract_vision_layer]
        img_cls_embed = self.clip.visual_projection(clip_image_features["pooler_output"])  # torch.Size([64, 512])

        if self.args.modality =="rgbt":
            if self.open_lora:
                self.clip.set_adapter("lora_ir")
            clip_image_features_ir = self.clip.vision_model(adapt_layer = self.adapt_layer, 
                                                     text_states = ml_text_features, 
                                                     reg_src =reg_src, 
                                                     cur_modality = "ir",
                                                     pixel_values = image_tensors_ir, 
                                                     output_attentions=True, 
                                                     output_hidden_states=True,
                                                     return_dict=True)  # B * 197 * 512
            # attention_map_ir = clip_image_features["attentions"]  # tuple, used for draw the attention map

            ml_image_features_ir = [clip_image_features_ir["hidden_states"][i] for i in self.extract_vision_layer]
            img_cls_embed_ir = self.clip.visual_projection(clip_image_features_ir["pooler_output"])  # torch.Size([64, 512])

            # 只有在 LAVS 模式下才进行多层 Cross Attention 融合
            if self.lavs_mode == "lavs":
                for i in range(len(self.extract_vision_layer)):
                    ml_image_features_fusion_vt,_ = self.cross_fusion_layers_vt[i](ml_image_features[i], ml_image_features_ir[i],  attention_mask=None,causal_attention_mask=None,output_attentions=True)
                    ml_image_features_fusion_tv,_ = self.cross_fusion_layers_tv[i](ml_image_features_ir[i], ml_image_features[i],  attention_mask=None,causal_attention_mask=None,output_attentions=True)
                    ml_image_features[i] = ml_image_features[i]+ml_image_features_fusion_vt
                    ml_image_features_ir[i] =  ml_image_features_ir[i] +ml_image_features_fusion_tv
        ml_image_features = torch.cat(ml_image_features, dim=2)
        image_features = self.ml_visual_projection(ml_image_features)
        visu_src = self.visu_proj(image_features.float())  # (N*B)xC
        
        if self.args.modality =='rgbt':
            ml_image_features_ir = torch.cat(ml_image_features_ir, dim=2)
            image_features_ir = self.ml_visual_projection(ml_image_features_ir)

            visu_src_ir = self.visu_proj(image_features_ir.float())  # (N*B)xC
            visu_src_ir = visu_src_ir.permute(1, 0, 2)  # 197 * 4 * 512
            if self.open_lora:
                self.clip.set_adapter("lora_rgb")

        text_src = self.text_proj(text_features.float())  # B * 77 * 512

        # permute BxLenxC to LenxBxC
        visu_src = visu_src.permute(1, 0, 2)  # 197 * 4 * 512
        text_src = text_src.permute(1, 0, 2)  # 77 * 4 * 512
        reg_src = reg_src.permute(1, 0, 2)  # 1 * B * 512
        # mask
        reg_mask = torch.zeros((batch_size, 1)).to(reg_src.device).to(torch.bool)
        cls_mask = torch.zeros((batch_size, 1)).to(reg_src.device).to(torch.bool)
        #TODO ZTY
        if self.args.modality == 'rgbt':
            if self.lavs_mode == "lavs":
                # 原始：RGB / IR token 分别送入 VL Transformer
                vl_src = torch.cat([reg_src, visu_src, visu_src_ir[1:,:,:], text_src], dim=0)
                vl_mask = torch.cat([reg_mask, cls_mask, visu_mask, visu_mask, text_mask], dim=1)
            else:
                # 非 LAVS 模式：先将 RGB / IR visual token 融合为单一路径，再送入 VL Transformer
                # visu_src, visu_src_ir: Len x B x C
                if self.lavs_mode == "iwm" and self.iwm is not None:
                    # IWM 根据 RGB 图像估计权重，进行加权融合
                    w_vis = self.iwm(img_data.tensors[:,:3,:,:])  # B x 1
                    w_vis = w_vis.to(visu_src.device).view(batch_size, 1, 1)  # B x 1 x 1
                    rgb_feat = visu_src.permute(1, 0, 2)         # B x Len x C
                    ir_feat = visu_src_ir.permute(1, 0, 2)       # B x Len x C
                    fused_feat = w_vis * rgb_feat + (1.0 - w_vis) * ir_feat
                    fused_visu_src = fused_feat.permute(1, 0, 2)  # Len x B x C
                elif self.lavs_mode == "cmx" and self.cmx is not None:
                    # CMX：基于 token 还原为 feature map，做融合后再展平
                    rgb_feat = visu_src.permute(1, 0, 2)   # B x Len x C
                    ir_feat = visu_src_ir.permute(1, 0, 2) # B x Len x C
                    # 第一个 token 视为 cls，其余为 patch
                    rgb_cls, rgb_patch = rgb_feat[:, 0:1, :], rgb_feat[:, 1:, :]
                    ir_cls, ir_patch = ir_feat[:, 0:1, :], ir_feat[:, 1:, :]
                    patch_num = int(math.sqrt(self.num_visu_token))
                    assert patch_num * patch_num == self.num_visu_token
                    # B x L x C -> B x C x H x W
                    rgb_map = rgb_patch.permute(0, 2, 1).reshape(batch_size, self.hidden_dim, patch_num, patch_num)
                    ir_map = ir_patch.permute(0, 2, 1).reshape(batch_size, self.hidden_dim, patch_num, patch_num)
                    merged_map = self.cmx([rgb_map, ir_map])  # B x C x H x W
                    merged_patch = merged_map.flatten(2).permute(0, 2, 1)  # B x L x C
                    merged_cls = 0.5 * (rgb_cls + ir_cls)
                    fused_feat = torch.cat([merged_cls, merged_patch], dim=1)  # B x (1+L) x C
                    fused_visu_src = fused_feat.permute(1, 0, 2)  # Len x B x C
                elif self.lavs_mode == "avg":
                    # 简单 0.5 / 0.5 加权相加
                    fused_visu_src = 0.5 * (visu_src + visu_src_ir)
                else:
                    # 兜底：简单平均
                    fused_visu_src = 0.5 * (visu_src + visu_src_ir)

                vl_src = torch.cat([reg_src, fused_visu_src, text_src], dim=0)
                vl_mask = torch.cat([reg_mask, cls_mask, visu_mask, text_mask], dim=1)
        else:
            vl_src = torch.cat([reg_src, visu_src, text_src], dim=0)
            vl_mask = torch.cat([reg_mask, cls_mask, visu_mask, text_mask], dim=1)

        # 位置编码长度需要与 vl_src 的 token 数一致。
        # 在 rgbt + 非 LAVS 模式下，我们只使用一条 visual 流（fused_visu_src），
        # 此时 vl_src 的长度小于初始化时的 num_total，因此这里按实际长度截取。
        vl_len = vl_src.size(0)
        vl_pos = self.vl_pos_embed.weight[:vl_len].unsqueeze(1).repeat(1, batch_size, 1)

        vg_hs = self.vl_transformer(vl_src, vl_mask, vl_pos)  # (1+L+N)xBxC
        box_hs = vg_hs[0]
        pred_box = self.bbox_embed(box_hs).sigmoid()

        # normalized features
        img_cls_embed = img_cls_embed / img_cls_embed.norm(p=2, dim=-1, keepdim=True)
        text_eos_embed = text_eos_embed / text_eos_embed.norm(p=2, dim=-1, keepdim=True)
        if self.args.modality == 'rgbt':
            img_cls_embed_ir = img_cls_embed_ir / img_cls_embed_ir.norm(p=2, dim=-1, keepdim=True)
        # cosine similarity as logits
        logit_scale = self.clip.logit_scale.exp()
        logits_per_text = torch.matmul(text_eos_embed, img_cls_embed.t()) * logit_scale
        if self.args.modality == 'rgbt':
            logits_per_text_ir = torch.matmul(text_eos_embed, img_cls_embed_ir.t()) * logit_scale
            logits_per_image_ir = logits_per_text_ir.t()
        logits_per_image = logits_per_text.t()
        
        # visual token align
        vg_hs_visu_features = vg_hs[2: 2 + self.num_visu_token].permute(1, 0, 2)  # B L H
        clip_last_layer_features = self.visu_token_mlp(self.visu_token_norm(vg_hs_visu_features))

        vg_hs_text = vg_hs[2 + self.num_visu_token:].permute(1, 0, 2)
        vg_hs_text_eos_embed = vg_hs_text[torch.arange(vg_hs_text.shape[0]), text_tensors.argmax(dim=-1)]
        vg_hs_text_eos_embed = vg_hs_text_eos_embed / vg_hs_text_eos_embed.norm(p=2, dim=-1, keepdim=True)

        visu_token_similarity = torch.mul(vg_hs_text_eos_embed.unsqueeze(1).repeat(1, self.num_visu_token, 1),
                                          clip_last_layer_features)  # torch.Size([96, 196, 512])
        visu_token_similarity = visu_token_similarity.sum(axis=-1, keepdim=False)  # torch.Size([96, 196])

        patch_num = int(math.sqrt(vg_hs_visu_features.shape[1]))
        channel = vg_hs_visu_features.shape[2]
        assert patch_num * patch_num == vg_hs_visu_features.shape[1]
        seg_features = vg_hs_visu_features.permute(0, 2, 1).reshape(batch_size, channel, patch_num, patch_num)
        seg_features = self.seg_conv3(self.seg_conv2(self.seg_conv1(seg_features)))
        seg_features = seg_features.permute(0, 2, 3, 1)
        seg_mask = torch.mul(vg_hs_text_eos_embed.reshape(batch_size, 1, 1, vg_hs_text_eos_embed.shape[-1]).repeat(1, seg_features.shape[1], seg_features.shape[2], 1),
                             seg_features)
        seg_mask = seg_mask.sum(axis=-1, keepdim=False).unsqueeze(1)  # B 1 H W
        if self.args.modality == 'rgbt':
            return pred_box, logits_per_text, [logits_per_image,logits_per_image_ir] , visu_token_similarity, seg_mask
        else: 
            return pred_box, logits_per_text, logits_per_image , visu_token_similarity, seg_mask

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

###################### IWM ############################
###################### IWM ############################
###################### IWM ############################
class IWM(nn.Module):
    def __init__(self, k=1, alpha=0.8):
        super(IWM, self).__init__()

        # Convolutional layers with reduced dimensions
        # 使用标准的 3x3 卷积，padding=1，stride=1，保持特征图尺寸不变
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # MaxPool layers
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Reduced number of FC layers
        self.fc1 = nn.Linear(64 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, k)

        self.alpha = alpha
        self.k = k

    def forward(self, x):
        # Downsample
        x = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)

        # Conv + MaxPool layers
        x = self.maxpool(self.conv1(x))
        x = self.maxpool(self.conv2(x))

        # Flatten
        x = x.view(x.size(0), -1)

        # FC layers
        x = F.relu(self.fc1(x))
        illumination_level = self.fc2(x)

        # Transform
        weights = self.transform(illumination_level)
        weights = weights.to(torch.float16)
        return weights

    def transform(self, x):
        # x 的形状应该是 (batch_size, k)
        batch_size = x.shape[0]

        weights = torch.arange(0.5, 0.5 + 0.25 * self.k, 0.25, device=x.device)
        x_prime = x * weights

        p = F.softmax(x_prime, dim=1)

        w_vis = torch.zeros(batch_size, 1, device=x.device)
        if self.k == 1:
            w_vis[:, 0] = p[:, 0]
        else:
            sum1 = torch.sum(p[:, self.k // 2:self.k], dim=1)
            sum2 = torch.sum(p[:, :max(0, (self.k - 2) // 2)], dim=1)
            w_vis[:, 0] = (sum1 - sum2) / 2 * self.alpha + 0.5

        return w_vis


###################### IWM ############################
###################### IWM ############################
###################### IWM ############################

###################### CMX ############################
###################### CMX ############################
###################### CMX ############################
class CMX(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=8, norm_layer=nn.BatchNorm2d, lambda_c=0.5, lambda_s=0.5):
        super().__init__()
        # 特征校正模块
        self.frm = FeatureRectifyModule(dim, reduction, lambda_c, lambda_s)
        # 特征融合模块
        self.ffm = FeatureFusionModule(dim, reduction, num_heads, norm_layer)

    def forward(self, x):
        # 特征校正
        x1 = x[0]
        x2 = x[1]
        x1, x2 = self.frm(x1, x2)
        # 特征融合
        merged = self.ffm(x1, x2)
        return merged


class FeatureRectifyModule(nn.Module):
    def __init__(self, dim, reduction=1, lambda_c=.5, lambda_s=.5):
        super(FeatureRectifyModule, self).__init__()
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.channel_weights = ChannelWeights(dim=dim, reduction=reduction)
        self.spatial_weights = SpatialWeights(dim=dim, reduction=reduction)

    def forward(self, x1, x2):
        channel_weights = self.channel_weights(x1, x2)
        spatial_weights = self.spatial_weights(x1, x2)
        out_x1 = x1 + self.lambda_c * channel_weights[1] * x2 + self.lambda_s * spatial_weights[1] * x2
        out_x2 = x2 + self.lambda_c * channel_weights[0] * x1 + self.lambda_s * spatial_weights[0] * x1
        return out_x1, out_x2


class FeatureFusionModule(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.cross = CrossPath(dim=dim, reduction=reduction, num_heads=num_heads)
        self.channel_emb = ChannelEmbed(in_channels=dim * 2, out_channels=dim, reduction=reduction,
                                        norm_layer=norm_layer)

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        x1_flat = x1.flatten(2).transpose(1, 2)
        x2_flat = x2.flatten(2).transpose(1, 2)
        x1_cross, x2_cross = self.cross(x1_flat, x2_flat)
        merge = torch.cat((x1_cross, x2_cross), dim=-1)
        return self.channel_emb(merge, H, W)


class ChannelWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(ChannelWeights, self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim * 4, self.dim * 4 // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim * 4 // reduction, self.dim * 2),
            nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        avg = self.avg_pool(x).view(B, self.dim * 2)
        max = self.max_pool(x).view(B, self.dim * 2)
        y = torch.cat((avg, max), dim=1)  # B 4C
        y = self.mlp(y).view(B, self.dim * 2, 1)
        channel_weights = y.reshape(B, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4)  # 2 B C 1 1
        return channel_weights


class SpatialWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(SpatialWeights, self).__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Conv2d(self.dim * 2, self.dim // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // reduction, 2, kernel_size=1),
            nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)  # B 2C H W
        spatial_weights = self.mlp(x).reshape(B, 2, 1, H, W).permute(1, 0, 2, 3, 4)  # 2 B 1 H W
        return spatial_weights

    # Stage 1


class CrossAttentionCMX(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super(CrossAttentionCMX, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.kv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)

    def forward(self, x1, x2):
        B, N, C = x1.shape
        q1 = x1.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        q2 = x2.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        k1, v1 = self.kv1(x1).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k2, v2 = self.kv2(x2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()

        ctx1 = (k1.transpose(-2, -1) @ v1) * self.scale
        ctx1 = ctx1.softmax(dim=-2)
        ctx2 = (k2.transpose(-2, -1) @ v2) * self.scale
        ctx2 = ctx2.softmax(dim=-2)

        x1 = (q1 @ ctx2).permute(0, 2, 1, 3).reshape(B, N, C).contiguous()
        x2 = (q2 @ ctx1).permute(0, 2, 1, 3).reshape(B, N, C).contiguous()

        return x1, x2


class CrossPath(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=8, norm_layer=nn.LayerNorm):
        super().__init__()
        self.channel_proj1 = nn.Linear(dim, dim // reduction * 2)
        self.channel_proj2 = nn.Linear(dim, dim // reduction * 2)
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        self.cross_attn = CrossAttentionCMX(dim // reduction, num_heads=num_heads)
        self.end_proj1 = nn.Linear(dim // reduction * 2, dim)
        self.end_proj2 = nn.Linear(dim // reduction * 2, dim)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x1, x2):
        y1, u1 = self.act1(self.channel_proj1(x1)).chunk(2, dim=-1)
        y2, u2 = self.act2(self.channel_proj2(x2)).chunk(2, dim=-1)
        v1, v2 = self.cross_attn(u1, u2)
        y1 = torch.cat((y1, v1), dim=-1)
        y2 = torch.cat((y2, v2), dim=-1)
        out_x1 = self.norm1(x1 + self.end_proj1(y1))
        out_x2 = self.norm2(x2 + self.end_proj2(y2))
        return out_x1, out_x2


# Stage 2
class ChannelEmbed(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1, norm_layer=nn.BatchNorm2d):
        super(ChannelEmbed, self).__init__()
        self.out_channels = out_channels
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.channel_embed = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // reduction, kernel_size=1, bias=True),
            nn.Conv2d(out_channels // reduction, out_channels // reduction, kernel_size=3, stride=1, padding=1,
                      bias=True, groups=out_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // reduction, out_channels, kernel_size=1, bias=True),
            norm_layer(out_channels)
        )
        self.norm = norm_layer(out_channels)

    def forward(self, x, H, W):
        B, N, _C = x.shape
        x = x.permute(0, 2, 1).reshape(B, _C, H, W).contiguous()
        residual = self.residual(x)
        x = self.channel_embed(x)
        out = self.norm(residual + x)
        return out


###################### CMX ############################
###################### CMX ############################
###################### CMX ############################