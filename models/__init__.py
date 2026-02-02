from .HiVG import HiVG
from .clip_vg import CLIP_VG
from .trans_vg import TransVG, TransVGSwin
from .mmca_vg import MMCA
from .dynamic_mdetr_resnet import DynamicMDETR as DynamicMDETR_ResNet
from .dynamic_mdetr_clip import DynamicMDETR as DynamicMDETR_CLIP
from .OneRef_model import beit3_base_patch16_224_grounding, beit3_large_patch16_384_grounding
from .clip_vg_deprecated import ML_CLIP_VG
def build_model(args):
    if args.model_name=='MMVG':
        from .mmvg import MMVG
        print('Building MMVG model...')
        return MMVG(args)
    if args.model_name=='MMVG_te':
        from .mmvg_twoenc import MMVG
        print('Building MMVG two encoder model...')
        return MMVG(args)
    elif args.model_name=='HiVG':
        print('Building HiVG model...')
        return HiVG(args)
    elif args.model_name=='CLIP_VG':
        # if hasattr(args, 'eval_model') and args.eval_model:
        #     print('Building ML_CLIP_VG model...')
        #     return ML_CLIP_VG(args)
        # else: 
        print('Building CLIP_VG model...')
        return CLIP_VG(args)
    elif args.model_name=='TransVG':
        print('Building TransVG model...')
        return TransVG(args)
    elif args.model_name=='QRNet':
        print('Building QRNet model...')
        return TransVGSwin(args)
    elif args.model_name=='MMCA':
        print('Building MMCA model...')
        return MMCA(args)
    elif args.model_name=='MDETR':
        print('Building MDETR model...')
        if args.model_type == 'ResNet':
            return DynamicMDETR_ResNet(args)
        elif args.model_type == 'CLIP':
            return DynamicMDETR_CLIP(args)
    elif args.model_name=='OneRef':
        print('Building OnRef model...')
        if args.model == 'beit3_base_patch16_224':
            return beit3_base_patch16_224_grounding(args)
        elif args.model == 'beit3_large_patch16_384':
            return beit3_large_patch16_384_grounding(args)
