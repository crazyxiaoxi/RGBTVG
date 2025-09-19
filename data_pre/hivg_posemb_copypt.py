import torch
import os
pt_model_root='../dataset_and_pretrain_model/pretrain_model/pretrained_weights/HiVG/mixup_pretraining_base/mixup'
pt_model_output='../dataset_and_pretrain_model/pretrain_model/pretrained_weights/HiVG/mixup_pretraining_base/mixup'
ori_model = os.path.join(pt_model_root,'fixed_best_checkpoint.pth')
two_encoder_model = os.path.join(pt_model_output,'fixed_best_checkpoint_2.pth')
# 1. 加载原始文件
checkpoint = torch.load(ori_model, map_location="cpu")
state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

# 2. 修复键名
new_state_dict = {}

for key, value in state_dict.items():
    new_state_dict[key] = value
    if 'vl_pos_embed' in key:
        import pdb
        pdb.set_trace()
        reg_cls_pos_embed= value[:2,:]
        visual_pos_embed = value[2:198,:]
        text_pos_embed = value[198:,:]
        new_value = torch.cat([reg_cls_pos_embed,visual_pos_embed ,visual_pos_embed ,text_pos_embed ],dim=0)
       
        new_state_dict[key] = new_value
        ##实现插值
            # new_key = key.replace("clip.base_model.model.vision_model", "clip.base_model.model.vision_model_ir")
            # new_state_dict[new_key] = value.clone()

# 3. 保存
if "model" in checkpoint:
    checkpoint["model"] = new_state_dict
else:
    checkpoint = new_state_dict

torch.save(checkpoint, two_encoder_model )
print("修复完成！已保存为 fixed_best_checkpoint_2.pth")
