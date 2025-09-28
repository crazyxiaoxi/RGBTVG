# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from utils.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding
from .text_guide_linear import MM_adaption_conv_2d


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, name:str, backbone: nn.Module, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.return_layers_fuse = return_layers
        self.num_channels = num_channels
        self.multimodal_layers_index = ["layer2", "layer3", "layer4"]
        self.multimodal_layers_d_model = [512, 1024, 2048]
        self.multimodal_layers = nn.ModuleList()
        for i in range(len(self.multimodal_layers_index)):
            self.multimodal_layers.append(MM_adaption_conv_2d(256, self.multimodal_layers_d_model[i], down_rate=4))

    def forward(self, tensor_list: NestedTensor, word_feat_embed):
        # xs = self.body(tensor_list.tensors)
        xs = OrderedDict()
        x = tensor_list.tensors
        for name, module in self.body.items():
            if name not in self.multimodal_layers_index:
                x = module(x)
            else:
                for i in range(len(module)):
                    if i == len(module) - 1:
                        x_ori = x
                        x = module[i](x)
                        x_t = self.multimodal_layers[self.multimodal_layers_index.index(name)](x_ori, word_feat_embed)
                        x = x + x_t
                    else:
                        x = module[i](x)
            if name in self.return_layers_fuse:
                out_name = self.body.return_layers[name]
                xs[out_name] = x

        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 return_interm_layers: bool,
                 dilation: bool):

        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=False, norm_layer=FrozenBatchNorm2d)
            # pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        assert name in ('resnet50', 'resnet101')
        num_channels = 2048
        super().__init__(name, backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor, word_feat_embed):
        xs = self[0](tensor_list, word_feat_embed)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    # train_backbone = args.lr_detr > 0
    return_interm_layers = False
    backbone = Backbone(args.backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
