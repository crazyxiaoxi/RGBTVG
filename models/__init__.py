from .HiVG import HiVG
from .clip_vg import CLIP_VG
from .trans_vg import TransVG, TransVGSwin
from .mmca_vg import MMCA
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
    elif args.model_name=='TransVG':
        print('Building TransVG model...')
        return TransVG(args)
    elif args.model_name=='QRNet':
        print('Building QRNet model...')
        return TransVGSwin(args)
    elif args.model_name=='MMCA':
        print('Building MMCA model...')
        return MMCA(args)

