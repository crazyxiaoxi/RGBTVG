import torch
ckpt = torch.load('./mixup_pretraining_large/mixup/best_checkpoint_rgbt.pth', map_location='cpu')
for k, v in ckpt['model'].items():
    if 'vl_pos_embed' in k:
        print(k, v.shape)
