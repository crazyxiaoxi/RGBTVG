import torch

clip_model = "../dataset_and_pretrain_model/pretrain_model/pretrained_weights/CLIP/clip_b_ml_cascade_maskrcnn_model_224.pth"
out_clip_model = "../dataset_and_pretrain_model/pretrain_model/pretrained_weights/CLIP/clip_b_ml_cascade_maskrcnn_model_224_peft0111_nolora.pth"
# checkpoint = torch.load(out_clip_model)
# import pdb
# pdb.set_trace()
checkpoint = torch.load(clip_model)
new_state_dict = {}

for key, value in checkpoint['model'].items():
    # import pdb
    # pdb.set_trace
    if 'lora' in key:
        continue
    modified_key = key

    if 'self_attn' in key:
        # 示例：将 "self_atten" 改为 "self_attention"
        if 'weight' in key: 
            modified_key = key.replace('weight', 'base_layer.weight')
        elif 'bias' in key:
            modified_key = key.replace('.bias', '.base_layer.bias')
    new_state_dict[modified_key] = value

# 更新 checkpoint 并保存
checkpoint['model'] = new_state_dict
torch.save(checkpoint, out_clip_model)