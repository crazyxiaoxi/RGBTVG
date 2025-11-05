import torch
import os

pt_model_root = './mixup_pretraining_large/mixup'
ori_model = os.path.join(pt_model_root, 'best_checkpoint.pth')

# 输出文件（覆盖原文件）
two_encoder_model = os.path.join(pt_model_root, 'best_checkpoint_rgbt.pth')

# 1. 加载模型权重
checkpoint = torch.load(ori_model, map_location="cpu")
state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

# 2. 查找并修改 vl_pos_embed
for key, value in state_dict.items():
    if 'vl_pos_embed' in key:
        print(f"找到权重: {key}, 原shape={value.shape}")

        # 拆分 + 拼接
        reg_cls_pos_embed = value[:2, :]
        visual_pos_embed = value[2:258, :]
        text_pos_embed = value[258:, :]

        new_value = torch.cat([reg_cls_pos_embed, visual_pos_embed, visual_pos_embed, text_pos_embed], dim=0)
        print(f"修改后shape={new_value.shape}")

        # 覆盖原始权重
        state_dict[key] = new_value

# 3. 保存修改结果
if "model" in checkpoint:
    checkpoint["model"] = state_dict
else:
    checkpoint = state_dict

torch.save(checkpoint, two_encoder_model)
print("✅ 修改完成并已覆盖原文件！")
