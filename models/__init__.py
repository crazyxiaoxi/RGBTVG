from .HiVG import HiVG
from .clip_vg import CLIP_VG


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
        print('Building CLIP_VG model...')
        return CLIP_VG(args)



