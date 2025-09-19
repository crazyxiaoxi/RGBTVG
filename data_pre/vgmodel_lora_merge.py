# import torch
# import torch.nn as nn
# from transformers import CLIPModel, CLIPProcessor, AutoConfig
# from peft import LoraConfig, get_peft_model, TaskType, PeftModel
# import os
# import sys
# sys.path.append("/home/ma-user/work/test/RGBTVG-benchmark")

# # --- 1. 导入您的 HiVG 模型 (重要：请确保 HiVG.py 在可导入路径中) ---
# # 请将 'HiVG' 替换为您实际的模块名和类名
# try:
#     from models.HiVG import HiVG # 假设您的模型类名为 HiVG，且在 HiVG.py 中
#     print("成功导入 HiVG 模型类。")
# except ImportError:
#     print("错误：无法导入 HiVG 模型类。请确保 'HiVG.py' 文件存在且在 Python 路径中。")
#     print("您可能需要将 'HiVG.py' 放在与此脚本相同的目录，或将其目录添加到 PYTHONPATH。")
#     exit()

# # --- 2. 配置路径 ---
# # 替换为您的原始 CLIP 模型名称或本地路径
# # 即使不加载其权重，也需要它来获取 CLIP 模型的架构和 Processor
# BASE_MODEL_NAME_OR_PATH = "/home/ma-user/work/test/dataset_and_pretrain_model/pretrain_model/pretrained_weights/CLIP/clip-vit-base-patch16"

# # 替换为您的 LoRA 权重 state_dict (.pth 文件) 的完整路径
# # 这个文件包含了您训练后保存的整个模型（或PeftModel）的权重
# LORA_STATE_DICT_PATH = "/home/ma-user/work/test/dataset_and_pretrain_model/pretrain_model/pretrained_weights/HiVG/mixup_pretraining_base/mixup/fixed_best_checkpoint.pth" # 假设您之前将LoRA适配器保存到这里

# # 定义最终保存的完整 VG 模型 .pth 文件的路径
# FINAL_VG_MODEL_SAVE_PATH = "./final_merged_vg_model.pth"

# # --- 3. 加载完整的 state_dict 并检查其内容 ---
# print(f"--- 正在从 {LORA_STATE_DICT_PATH} 加载完整的 state_dict ---")
# try:
#     full_state_dict = torch.load(LORA_STATE_DICT_PATH, map_location="cpu")
#     print("State_dict 已加载。")
#     print(f"State_dict 包含 {len(full_state_dict['model'])} 个键。")
#     print("State_dict 键的示例 (前10个):", list(full_state_dict['model'].keys())[:10])

#     # 检查 state_dict 中是否包含 LoRA 相关的键，特别是针对 CLIP 部分
#     # 假设您的 CLIP 模型在 VG 模型中名为 'clip_model'
#     # 键可能像 'clip_model.base_model.model.text_model...' 或 'clip_model.lora_A'
#     is_peft_state_dict_for_clip = any("clip_model.lora_" in key or "clip_model.base_model.model." in key for key in full_state_dict['model'].keys())
#     print(f"State_dict 是否包含 LoRA 适配器键 (针对 vg_model.clip_model): {is_peft_state_dict_for_clip}")

# except Exception as e:
#     print(f"加载 state_dict 失败: {e}")
#     exit()

# print("-" * 50)

# # --- 4. 手动定义 LoRA 配置 (必须与您训练时的配置完全一致) ---
# print("--- 正在手动定义 LoRA 配置 (请务必与您训练时一致) ---")
# lora_config = LoraConfig(
#     r=32,
#     lora_alpha=16, # 请务必根据您实际训练时的配置调整此值
#     target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc_in", "fc_out", "wte"],
#     lora_dropout=0.1,
#     bias="none",
# )
# print("LoRA 配置定义完成。")
# print("-" * 50)

# # --- 5. 实例化您的 HiVG 模型并加载完整的 state_dict ---
# print(f"--- 正在实例化您的 HiVG 模型并加载 state_dict ---")

# # 根据 state_dict 是否包含 LoRA 键来决定 HiVG 的初始化方式
# # 如果包含 LoRA 键，HiVG 内部的 clip_model 应该被包装成 PeftModel
# # vg_model = HiVG(BASE_MODEL_NAME_OR_PATH, lora_config=lora_config if is_peft_state_dict_for_clip else None)
# vg_model = HiVG(BASE_MODEL_NAME_OR_PATH)

# # 将完整的 state_dict 载入 HiVG 模型
# print("正在将完整的 state_dict 载入 HiVG 模型...")
# # strict=False 允许 state_dict 中包含 HiVG 模型架构中没有的键（例如优化器状态或其他训练元数据）
# # 或者 HiVG 模型中包含 state_dict 中没有的键（如果 state_dict 仅包含部分权重）
# vg_model.load_state_dict(full_state_dict, strict=False)
# print("完整的 state_dict 已成功载入 HiVG 模型。")

# # 将模型移动到适当的设备
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# vg_model.to(device)
# vg_model.eval() # 设置为评估模式，确保合并过程的确定性

# print("-" * 50)

# # --- 6. 执行 LoRA 合并 (如果需要) ---
# if is_peft_state_dict_for_clip:
#     print("\n--- 检测到 HiVG 模型内部 CLIP 部分是 PeftModel，正在进行 LoRA 合并 ---")
#     # 访问 HiVG 模型内部的 PeftModel 实例
#     # 这假设您的 HiVG 有一个名为 'clip_model' 的属性，它是一个 PeftModel
#     if hasattr(vg_model, 'clip_model') and isinstance(vg_model.clip_model, PeftModel):
#         print(f"HiVG 模型内部的 CLIP 部分是 PeftModel。类型: {type(vg_model.clip_model)}")
#         # 调用 merge_and_unload() 方法。
#         # 这将修改 vg_model.clip_model.base_model.model (即原始 CLIPModel) 的权重，
#         # 并将 vg_model.clip_model 的引用更新为这个合并后的 CLIPModel。
#         merged_clip_part = vg_model.clip_model.merge_and_unload()
        
#         print(f"LoRA 权重已成功合并到 HiVG 模型内部的 CLIP 部分。")
#         print(f"HiVG 模型内部 CLIP 部分的新类型: {type(vg_model.clip_model)}")
#         assert not isinstance(vg_model.clip_model, PeftModel), "合并后 CLIP 部分仍是PeftModel类型，合并失败！"
#     else:
#         print("警告：state_dict 包含 LoRA 键，但 vg_model.clip_model 不是 PeftModel 或不存在。无法执行合并。")
#         print("请检查您的 HiVG 构造函数是否正确地将 CLIP 模型包装为 PeftModel，以及其属性名是否为 'clip_model'。")
# else:
#     print("\n--- State_dict 不包含 LoRA 适配器键，假定 CLIP 部分已合并或从未进行 LoRA 微调。跳过合并。---")

# print("-" * 50)

# # --- 7. 保存整个 HiVG 模型的 state_dict 到 .pth 文件 ---
# print(f"--- 正在保存完整的 HiVG 模型 state_dict 到 {FINAL_VG_MODEL_SAVE_PATH} ---")
# torch.save(vg_model.state_dict(), FINAL_VG_MODEL_SAVE_PATH)
# print("完整的 HiVG 模型 state_dict 已成功保存。")
# print("-" * 50)

# # --- 8. (可选) 保存 CLIP Processor ---
# # 即使保存整个 VG 模型的 .pth，通常也需要单独的 Processor 来处理输入

# print("-" * 50)

# # --- 9. (可选) 验证通过加载保存的 .pth 模型并进行推理 ---
# print(f"--- 正在验证加载保存的 .pth 模型并进行推理 ---")


# vg_model.eval() 
# merged_clip_part = vg_model.clip_model.merge_and_unload()
# FINAL_VG_MODEL_SAVE_PATH = "./final_merged_vg_model.pth"
# torch.save(vg_model.state_dict(), FINAL_VG_MODEL_SAVE_PATH)