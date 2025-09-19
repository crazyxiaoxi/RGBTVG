

import torch
import os
pt_model_root='../dataset_and_pretrain_model/pretrain_model/pretrained_weights/HiVG/mixup_pretraining_base/mixup'
pt_model = os.path.join(pt_model_root,'fixed_best_checkpoint_2.pth')
fixed_pt_model = os.path.join(pt_model_root,'fixed_best_checkpoint_peft0111.pth')
# 1. 加载原始文件
checkpoint = torch.load(pt_model, map_location="cpu")
state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

# 2. 修复键名
new_state_dict = {}
import pdb
pdb.set_trace()
for key, value in state_dict.items():
    if 'clip.base_model' in key:
        if ('_attn.k_proj' in key or '_attn.q_proj' in key or '_attn.v_proj' in key or '_attn.out_proj' in key) and ('lora' not in key):
            print(key)
            new_key = key.replace("_proj", "_proj.base_layer")
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    else:
        new_state_dict[key] = value 
import pdb
pdb.set_trace()
# 3. 保存
if "model" in checkpoint:
    checkpoint["model"] = new_state_dict
else:
    checkpoint = new_state_dict

torch.save(checkpoint, fixed_pt_model )
print("修复完成！已保存为 fixed_model.pt")