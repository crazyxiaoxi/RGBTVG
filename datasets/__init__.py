# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from torchvision.transforms import Compose, ToTensor, Normalize

import datasets.transforms as T
from .data_loader import TransVGDataset

""""CLIP's default transform"""
# def _transform(n_px):
#     return Compose([
#         Resize(n_px, interpolation=BICUBIC),
#         CenterCrop(n_px),
#         _convert_image_to_rgb,
#         ToTensor(),
#         Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
#     ])


def make_transforms(args, image_set, is_onestage=False):
    # COCO mean=[0.485, 0.456, 0.406,0.1], std=[0.229, 0.224, 0.225,0.2]
    if args.modality=='rgbt':
        if args.dataset == 'rgbtvg_flir':
            mean,std = [0.631, 0.6401, 0.632, 0.5337], [0.2152, 0.227, 0.2439, 0.2562]#RGBT channel
        elif args.dataset == 'rgbtvg_m3fd':
           mean,std = [0.5013, 0.5067, 0.4923, 0.3264], [0.1948, 0.1989, 0.2117, 0.199]
        elif args.dataset == 'rgbtvg_mfad':
            mean,std = [0.4733, 0.4695, 0.4622, 0.3393], [0.1654, 0.1646, 0.1749, 0.2063]
        elif args.dataset == 'rgbtvg_mixup':

            mean,std = [0.5103, 0.5111, 0.502, 0.3735], [0.1926, 0.1973, 0.2091, 0.2289]
    elif args.modality=='rgb':
        if args.dataset == 'rgbtvg_flir':
            mean,std = [0.631, 0.6401, 0.632], [0.2152, 0.227, 0.2439]
        elif args.dataset == 'rgbtvg_m3fd':
            mean,std = [0.5013, 0.5067, 0.4923], [0.1948, 0.1989, 0.2117]
        elif args.dataset == 'rgbtvg_mfad':
            mean,std = [0.4733, 0.4695, 0.4622], [0.1654, 0.1646, 0.1749]
        else:
            mean,std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    elif args.modality=='ir':
        if args.dataset == 'rgbtvg_flir':
            mean,std = [0.5337, 0.5337, 0.5337], [0.2562, 0.2562, 0.2562]
        elif args.dataset == 'rgbtvg_m3fd':
            mean,std = [0.3264, 0.3264, 0.3264], [0.199, 0.199, 0.199]
        elif args.dataset == 'rgbtvg_mfad':
            mean,std = [0.3393, 0.3393, 0.3393], [0.2063, 0.2063, 0.2063]
    else:
        mean,std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    if is_onestage:
        normalize = Compose([
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return normalize

    imsize = args.imsize

    if image_set in ['train', 'train_pseudo']:
        scales = []
        if args.aug_scale:
            stride = int(imsize/8)
            for i in range(7):
                scales.append(imsize - stride * i)
        else:
            scales = [imsize]

        if args.aug_crop:
            crop_prob = 0.5
        else:
            crop_prob = 0.

        # By default, RandomResize sets with_long_side = True. The difference lies in whether the resizing is based on
        # the long side or the short side. Ultimately, the entire image needs to be compressed.
        return T.Compose([
            T.RandomSelect(
                T.RandomResize(scales),
                T.Compose([
                    T.RandomResize([400, 500, 600], with_long_side=False),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales),
                ]),
                p=crop_prob
            ),
            T.ColorJitter(0.4, 0.4, 0.4),
            T.GaussianBlur(aug_blur=args.aug_blur),
            # T.RandomHorizontalFlip(),  # It has a certain impact on grounding performance and needs to be turned off
            T.ToTensor(),
            T.NormalizeAndPad(mean=mean, std=std ,size=imsize, aug_translate=args.aug_translate)
        ])

    if image_set in ['val', 'test', 'testA', 'testB', 'testC',"flir_val", "flir_test", "flir_testA", "flir_testB", "flir_testC", "m3fd_val", "m3fd_test", "m3fd_testA", "m3fd_testB", "m3fd_testC", "mfad_val", "mfad_test", "mfad_testA", "mfad_testB", "mfad_testC"]:
        return T.Compose([
            T.RandomResize([imsize]),
            T.ToTensor(),
            T.NormalizeAndPad(mean=mean,std=std ,size=imsize),
        ])

    raise ValueError(f'unknown {image_set}')


# args.data_root default='./ln_data/', args.split_root default='data', '--dataset', default='referit'
# split = test, testA, val, args.max_query_len = 20
def build_dataset(split, args):
    return TransVGDataset(args,
                          data_root=args.data_root,
                          split_root=args.split_root,
                          dataset=args.dataset,
                          split=split,
                          transform=make_transforms(args, split),
                          max_query_len=args.max_query_len,
                          prompt_template=args.prompt,
                          bert_model='../dataset_and_pretrain_model/pretrain_model/pretrained_weights/Bert')
