import torch
import torch.nn.functional as F
from torch import nn, Tensor

class MM_adaption_linear(nn.Module):
    def __init__(self, d_model=256, d_model_visual=256, down_rate=4):
        super().__init__()
        self.visual_linear_static_down = nn.Linear(d_model_visual, d_model // down_rate, bias=False)
        self.gate_fusion = GF_block()
        self.text_linear_w = nn.Parameter(torch.zeros(d_model//2, d_model // down_rate))
        self.text_linear_b = nn.Parameter(torch.ones(d_model // down_rate))
        self.visual_linear_static_up = nn.Linear(d_model // down_rate, d_model_visual, bias=False)

    def forward(self, x, word_feat_embed):
        x_t = self.visual_linear_static_down(x)

        mm_embed = self.gate_fusion(x, word_feat_embed)

        Wx_t = torch.matmul(mm_embed, self.text_linear_w) + self.text_linear_b
        x_t = x_t * Wx_t
        x_t = self.visual_linear_static_up(x_t)
        return x_t

class MM_adaption_conv_2d(nn.Module):
    def __init__(self, d_model=256, d_model_visual=256, down_rate=4):
        super().__init__()
        self.visual_conv_static_down = nn.Conv2d(d_model_visual, d_model // down_rate, 3, 1, 1)
        self.gate_fusion = GF_block(d_model_visual=d_model_visual, isTR=False)
        self.text_linear_w = nn.Parameter(torch.zeros(d_model//2, d_model // down_rate))
        self.text_linear_b = nn.Parameter(torch.ones(d_model // down_rate))
        self.visual_conv_static_up = nn.Conv2d(d_model // down_rate, d_model_visual, 1)
        self.middle_channel = d_model // down_rate

    def forward(self, x, word_feat_embed):
        b,c,h,w = x.shape
        x_t = self.visual_conv_static_down(x)

        mm_embed = self.gate_fusion(x, word_feat_embed)

        Wx_t = torch.matmul(mm_embed, self.text_linear_w) + self.text_linear_b
        x_t = x_t.reshape(b, self.middle_channel, -1).permute(2, 0, 1)
        x_t = x_t * Wx_t
        x_t = x_t.permute(1, 2, 0).reshape(b, self.middle_channel, h, w)
        x_t = self.visual_conv_static_up(x_t)
        return x_t

class GF_block(nn.Module):
    def __init__(self, d_model=128, d_model_visual=256, isTR=True):
        super().__init__()
        self.isTR = isTR
        if self.isTR:
            self.visual_linear_static_down = nn.Linear(d_model_visual, d_model, bias=False)
        else:
            self.visual_linear_static_down = nn.Conv2d(d_model_visual, d_model, 1, bias=False)
        self.textual_linear_static_down = nn.Linear(256, d_model, bias=False)
        self.mm_feat_embed_transform = nn.Sequential(
          nn.Linear(d_model, d_model // 4),
          nn.ReLU(),
          nn.Linear(d_model // 4, d_model)
        )


    def forward(self, x, word_feat_embed):
        x = self.visual_linear_static_down(x)
        word_feat_embed = self.textual_linear_static_down(word_feat_embed)
        if self.isTR:
            x = F.avg_pool1d(x.transpose(0, 2), 400).transpose(0, 2)
        else:
            b, c, h, w = x.shape
            x = x.reshape(b, c, -1).permute(2, 0, 1)
            x = F.avg_pool1d(x.transpose(0, 2), h*w).transpose(0, 2)
        mm_embed_lamda = (self.mm_feat_embed_transform(x + word_feat_embed)).sigmoid()
        mm_embed = word_feat_embed + mm_embed_lamda * x
        return mm_embed