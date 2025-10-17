import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.box_utils import  xywh2xyxy
import os 

def visualization(args, img_data,text_data,pred_bboxes,gt_bboxes,ori_size=(640,512)):
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

    #######可视化
    for i, pred_bbox in enumerate(pred_bboxes):
        img=img_data.tensors[i].cpu().numpy()
        text=text_data[i]
        pred_bbox = pred_bbox.cpu()
        gt_bbox = gt_bboxes[i].cpu()
        img = img.transpose(1, 2, 0)  # 调整通道顺序
        img = (img * std + mean) * 255  # 恢复像素值到 [0, 255]
        img = np.clip(img, 0, 255).astype(np.uint8)  # 确保在 0-255 范围内
        img_with_bbox = img.copy()

        imsize = args.imsize
        pred_x_min, pred_y_min, pred_x_max, pred_y_max = (imsize*xywh2xyxy(pred_bbox).numpy()).astype(int)
        gt_x_min, gt_y_min, gt_x_max, gt_y_max = (imsize*xywh2xyxy(gt_bbox).numpy()).astype(int)
        # import pdb
        # pdb.set_trace()
        cv2.rectangle(img_with_bbox, (pred_x_min, pred_y_min), (pred_x_max, pred_y_max), (0, 255, 0), 2)  # 预测 (绿色)
        cv2.rectangle(img_with_bbox, (gt_x_min, gt_y_min), (gt_x_max, gt_y_max), (255, 0, 0), 2)  # 真实 (红色)

        # 添加文本
        cv2.putText(img_with_bbox, f"Pred: {text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # cv2.putText(img_with_bbox, f"GT: {text}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        output_dir = f"./output_visualization"
        output_path = os.path.join(output_dir, f"result_{i}.jpg")
        cv2.imwrite(output_path, cv2.cvtColor(img_with_bbox, cv2.COLOR_RGB2BGR))  #


