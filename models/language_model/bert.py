# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F

from torch import nn
from typing import Dict, List

from utils.misc import NestedTensor, is_main_process

from transformers import BertModel, BertConfig


class BERT(nn.Module):
    def __init__(self, name: str, train_bert: bool, hidden_dim: int, max_len: int, enc_num):
        super().__init__()
        if name == 'bert-base-uncased':
            self.num_channels = 768
        else:
            self.num_channels = 1024
        self.enc_num = enc_num

        # 新版 transformers BertModel
        self.bert = BertModel.from_pretrained(name, output_hidden_states=True)

        if not train_bert:
            for parameter in self.bert.parameters():
                parameter.requires_grad_(False)

    def forward(self, tensor_list: NestedTensor):
        """
        tensor_list.tensors: input_ids (B, L)
        tensor_list.mask: attention_mask (B, L)
        """
        # 新版返回值
        outputs = self.bert(input_ids=tensor_list.tensors,
                            attention_mask=tensor_list.mask,
                            token_type_ids=None)
        
        hidden_states = outputs.hidden_states  # tuple, 每层输出

        if self.enc_num > 0:
            xs = hidden_states[self.enc_num - 1]  # 第 enc_num 层输出
        else:
            xs = self.bert.embeddings.word_embeddings(tensor_list.tensors)  # embedding 输出

        mask = tensor_list.mask.to(torch.bool)
        mask = ~mask  # 取反
        out = NestedTensor(xs, mask)

        return out


def build_bert(args):
    train_bert = args.lr_bert > 0
    bert = BERT(args.bert_model, train_bert, args.hidden_dim, args.max_query_len, args.bert_enc_num)
    return bert
