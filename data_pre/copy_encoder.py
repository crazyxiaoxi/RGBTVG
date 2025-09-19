import torch
import os
pt_model_root='../dataset_and_pretrain_model/pretrain_model/pretrained_weights/MMVG/mixup_pretraining_base/mixup'
pt_model_output='../dataset_and_pretrain_model/pretrain_model/pretrained_weights/MMVG_te/mixup_pretraining_base/mixup'
ori_model = os.path.join(pt_model_root,'merged_fixed_best_checkpoint.pth')
two_encoder_model = os.path.join(pt_model_output,'two_encoder_best_checkpoint.pth')
# 1. 加载原始文件
checkpoint = torch.load(ori_model, map_location="cpu")
state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

# 2. 修复键名
new_state_dict = {}

for key, value in state_dict.items():
    new_state_dict[key] = value
    if 'clip.base_model.model.vision_model' in key:
        new_key = key.replace("clip.base_model.model.vision_model", "clip.base_model.model.vision_model_ir")
        new_state_dict[new_key] = value.clone()
    if 'visu_proj' in key:
        new_key = key.replace("visu_proj", "visu_proj_ir")
        new_state_dict[new_key] = value.clone()

# 3. 保存
if "model" in checkpoint:
    checkpoint["model"] = new_state_dict
else:
    checkpoint = new_state_dict

torch.save(checkpoint, two_encoder_model )
print("修复完成！已保存为 two_encoder_best_checkpoint.pt")
