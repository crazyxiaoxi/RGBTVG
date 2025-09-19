import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def calculate_image_stats(image_roots='.'):
    """计算指定目录下所有图片的通道均值和标准差
    
    Args:
        image_root (str): 图片目录路径，默认为当前目录
    Returns:
        tuple: (mean, std) 各通道的均值和标准差
    """
    # 初始化统计变量
    pixel_sum = np.zeros(3)      # [R_sum, G_sum, B_sum]
    pixel_sq_sum = np.zeros(3)   # [R²_sum, G²_sum, B²_sum]
    pixel_count = 0              # 总像素数

    # 获取目录下所有图片文件（支持常见格式）
    img_files = []
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    for image_root in image_roots:
        for f in os.listdir(image_root):
            if f.lower().endswith(valid_extensions):
                img_files.append(os.path.join(image_root, f))
        
        if not img_files:
            raise FileNotFoundError(f"目录 {image_root} 下未找到图片文件（支持格式：{valid_extensions}）")

        # 遍历图片并计算统计量
        for img_path in tqdm(img_files, desc='Processing Images'):
            try:
                with Image.open(img_path) as img:
                    # 统一转为RGB格式并归一化到[0,1]
                    img_array = np.array(img.convert('RGB')) / 255.0
                    
                    # 累加统计量（优化计算效率）
                    pixel_sum += img_array.sum(axis=(0, 1))          # 各通道总和
                    pixel_sq_sum += (img_array ** 2).sum(axis=(0, 1)) # 各通道平方和
                    pixel_count += img_array.size // 3  # 总像素数 = 数组元素数 / 3通道
            except Exception as e:
                print(f"\n警告：跳过损坏图片 {os.path.basename(img_path)} - {str(e)}")
                continue

    if pixel_count == 0:
        raise ValueError("没有有效的图片数据可供计算")

    # 计算均值和标准差
    mean = pixel_sum / pixel_count
    std = np.sqrt((pixel_sq_sum / pixel_count) - mean ** 2)

    return mean, std

if __name__ == "__main__":
    # 使用示例：可以修改为你的实际图片路径
    image_roots = ['/home/ma-user/work/test/dataset_and_pretrain_model/datasets/VG/image_data/rgbtvg/rgbtvg-images/flir/rgb',
                  '/home/ma-user/work/test/dataset_and_pretrain_model/datasets/VG/image_data/rgbtvg/rgbtvg-images/m3fd/rgb',
                  '/home/ma-user/work/test/dataset_and_pretrain_model/datasets/VG/image_data/rgbtvg/rgbtvg-images/mfad/rgb']    
    try:
        print(f"正在计算目录 [{image_roots}] 的图片统计量...")
        mean, std = calculate_image_stats(image_roots)
        
        print("\n=== 计算结果 ===")
        print(f"RGB通道均值 (Mean): {np.round(mean, 4)}")
        print(f"RGB通道标准差 (Std): {np.round(std, 4)}")
        print("=" * 20)
        
        # 输出可直接粘贴到PyTorch transforms的格式
        print("\nPyTorch transforms.Normalize 格式：")
        print(f"transforms.Normalize(mean={list(mean.round(4))}, std={list(std.round(4))})")
        
    except Exception as e:
        print(f"\n错误: {str(e)}")