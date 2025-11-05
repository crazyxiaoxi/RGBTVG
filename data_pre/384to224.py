import torch
import torch.nn.functional as F
import os

pt_model_root = './rec_mixup_grounding_pretraining_base/mixup'
ori_model = os.path.join(pt_model_root, 'best_checkpoint.pth')
out_model = './b_rec_224.pth'

# 目标视觉token数量（14x14）
new_grid_size = 14

# 加载
checkpoint = torch.load(ori_model, map_location='cpu')
state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

key = 'beit3.encoder.embed_positions.A.weight'
if key in state_dict:
    pos_embed = state_dict[key]
    print(f"找到位置嵌入: {key}, 原始shape={pos_embed.shape}")

    # 拆分特殊token与grid部分
    num_special_tokens = 3
    special_tokens = pos_embed[:num_special_tokens, :]           # (3, 768)
    grid_tokens = pos_embed[num_special_tokens:, :]              # (576, 768)

    # reshape为2D grid (1, C, H, W)
    old_grid_size = int(grid_tokens.shape[0] ** 0.5)
    grid_tokens = grid_tokens.reshape(1, old_grid_size, old_grid_size, -1).permute(0, 3, 1, 2)  # (1, 768, 24, 24)
    print(f"重塑后: {grid_tokens.shape}")

    # 插值缩小到 14x14
    resized_grid = F.interpolate(grid_tokens, size=(new_grid_size, new_grid_size), mode='bilinear', align_corners=False)
    print(f"插值后: {resized_grid.shape}")

    # 展平 + 拼接
    resized_grid = resized_grid.permute(0, 2, 3, 1).reshape(-1, resized_grid.shape[1])  # (196, 768)
    new_pos_embed = torch.cat([special_tokens, resized_grid], dim=0)
    print(f"新位置嵌入shape={new_pos_embed.shape}")

    # 替换
    state_dict[key] = new_pos_embed
else:
    raise KeyError(f"未找到 {key}，请确认权重文件中包含该参数。")

# 保存
if "model" in checkpoint:
    checkpoint["model"] = state_dict
else:
    checkpoint = state_dict

torch.save(checkpoint, out_model)
print("✅ 修改完成并已保存到:", out_model)
