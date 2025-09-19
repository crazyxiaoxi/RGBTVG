import torch
import os
pt_model_root='../dataset_and_pretrain_model/pretrain_model/pretrained_weights/HiVG/mixup_pretraining_base/mixup'
pt_model = os.path.join(pt_model_root,'best_checkpoint.pth')
fixed_pt_model = os.path.join(pt_model_root,'fixed_best_checkpoint.pth')
# 1. 加载原始文件
checkpoint = torch.load(pt_model, map_location="cpu")
state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

# 2. 修复键名
new_state_dict = {}
import pdb
pdb.set_trace()
for key, value in state_dict.items():
    new_key = key.replace("base_model.model.base_model.model", "base_model.model")
    new_key = new_key.replace("base_model.model.base_model.model", "base_model.model")  # 二次替换确保完全修复
    new_state_dict[new_key] = value
import pdb
pdb.set_trace()
# 3. 保存
if "model" in checkpoint:
    checkpoint["model"] = new_state_dict
else:
    checkpoint = new_state_dict

torch.save(checkpoint, fixed_pt_model )
print("修复完成！已保存为 fixed_model.pt")