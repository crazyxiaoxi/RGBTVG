import torch
from transformers import CLIPModel, CLIPProcessor, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import os
# 假设您已经通过某种方式加载或创建了您的 peft_model 对象
# 例如，如果您是从训练脚本中获得的，或者从保存的LoRA适配器加载的：

# --- 1. 假设您已经有了一个在内存中的 peft_model 对象 ---
# 为了让这个示例可运行，我将模拟加载一个 peft_model。
# 在您的实际代码中，您会直接使用您训练好的 `peft_model` 变量。

# 模拟加载原始CLIP模型（这是您训练LoRA时的基模型）
# 替换为您实际使用的CLIP模型名称或路径
BASE_MODEL_NAME_OR_PATH = "/home/ma-user/work/test/dataset_and_pretrain_model/pretrain_model/pretrained_weights/CLIP/clip-vit-base-patch16"

# 替换为您的 LoRA 权重 state_dict (.pth 文件) 的完整路径
# 这个文件包含了您训练后保存的整个模型（或PeftModel）的权重
LORA_STATE_DICT_PATH = "/home/ma-user/work/test/dataset_and_pretrain_model/pretrain_model/pretrained_weights/HiVG/mixup_pretraining_base/mixup/fixed_best_checkpoint.pth" # 假设您之前将LoRA适配器保存到这里

# 定义合并后模型的保存路径
SAVE_PATH = "./final_merged_clip_model"

# --- 2. 加载 state_dict 并检查其内容 ---
print(f"--- 正在从 {LORA_STATE_DICT_PATH} 加载 state_dict 进行检查 ---")
try:
    # 加载整个 state_dict 到 CPU，以避免立即占用大量 GPU 内存
    full_state_dict = torch.load(LORA_STATE_DICT_PATH, map_location="cpu")
    print("State_dict 已加载。")
    print(f"State_dict 包含 {len(full_state_dict['model'])} 个键。")
    print("State_dict 键的示例 (前10个):", list(full_state_dict['model'].keys())[:10])

    # 检查 state_dict 中是否包含 LoRA 相关的键
    is_peft_state_dict = any("lora_" in key for key in full_state_dict['model'].keys())
    print(f"State_dict 是否包含 LoRA 适配器键 ('lora_'): {is_peft_state_dict}")

except Exception as e:
    print(f"加载 state_dict 失败: {e}")
    exit() # 如果加载失败，程序无法继续

print("-" * 50)

# --- 3. 根据 state_dict 内容决定加载和合并策略 ---
merged_model = None

if is_peft_state_dict:
    print("\n--- 检测到 state_dict 包含 LoRA 适配器键，将加载为 PeftModel 并进行合并 ---")

    # 手动定义 LoRA 配置 (必须与您训练时的配置完全一致)
    # 根据您提供的模型结构：r=32, lora_dropout=0.1, bias="none"
    # target_modules: k_proj, v_proj, q_proj, out_proj
    # lora_alpha: 通常与 r 相同或 r*2，这里假设为 r

    lora_config = LoraConfig(
        r=32,
        lora_alpha=16, # 请务必根据您实际训练时的配置调整此值
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc_in", "fc_out", "wte"],
        lora_dropout=0.1,
        bias="none",
    )
    print("LoRA 配置已定义。")

    # 加载原始 CLIP 模型的架构 (不加载权重)
    print(f"正在加载原始 CLIP 模型架构: {BASE_MODEL_NAME_OR_PATH}...")
    base_model = CLIPModel.from_pretrained(
        BASE_MODEL_NAME_OR_PATH,
        torch_dtype=torch.float16, # 建议使用fp16减少内存占用
        device_map="cuda", # 假设有GPU，如果无GPU或内存不足，请改为 "cpu"
        low_cpu_mem_usage=True # 减少CPU内存使用
    )
    print("原始 CLIP 模型架构加载完成。")

    # 将原始模型包装成 PeftModel 结构
    peft_model = get_peft_model(base_model, lora_config)
    print(f"PeftModel 结构已创建。类型: {type(peft_model)}")
    peft_model.print_trainable_parameters() # 此时会显示 LoRA 参数是可训练的

    # 将加载的 state_dict 载入 PeftModel
    # strict=False 允许 state_dict 中包含 PeftModel 中没有的键（如优化器状态）
    # 或 PeftModel 中包含 state_dict 中没有的键（如果 state_dict 只包含 LoRA 部分）
    print("正在将 state_dict 载入 PeftModel...")
    peft_model.load_state_dict(full_state_dict, strict=False)
    print("State_dict 已成功载入 PeftModel。")

    # 合并 LoRA 权重并卸载适配器
    print("正在合并 LoRA 权重并卸载适配器...")
    peft_model.eval() # 建议在合并前将模型设置为评估模式
    merged_model = peft_model.merge_and_unload()
    print("LoRA 权重已成功合并到原始模型中，适配器已卸载。")

else:
    print("\n--- 检测到 state_dict 不包含 LoRA 适配器键，假定已合并。直接加载为标准 CLIPModel ---")

    # 加载原始 CLIP 模型的架构
    print(f"正在加载原始 CLIP 模型架构: {BASE_MODEL_NAME_OR_PATH}...")
    merged_model = CLIPModel.from_pretrained(
        BASE_MODEL_NAME_OR_PATH,
        torch_dtype=torch.float16,
        device_map="cuda", # 假设有GPU，如果无GPU或内存不足，请改为 "cpu"
        low_cpu_mem_usage=True
    )
    print("原始 CLIP 模型架构加载完成。")

    # 直接将 state_dict 载入模型
    print("正在将 state_dict 载入标准 CLIPModel...")
    # strict=True 确保所有键都匹配，如果您的 state_dict 是一个完整的合并模型，这通常是安全的
    merged_model.load_state_dict(full_state_dict, strict=True)
    print("State_dict 已成功载入标准 CLIPModel。")

print(f"\n最终模型类型: {type(merged_model)}")
assert not isinstance(merged_model, PeftModel), "最终模型仍是PeftModel类型，处理失败！"
print("验证成功：最终模型是标准的 Hugging Face CLIP 模型。")
print("-" * 50)

# --- 4. 保存合并后的模型和Processor/Tokenizer ---
os.makedirs(SAVE_PATH, exist_ok=True)

print(f"--- 正在保存合并后的模型到: {SAVE_PATH} ---")
# 保存合并后的模型
merged_model.save_pretrained(SAVE_PATH)

# 对于CLIP模型，通常需要保存 CLIPProcessor (它包含了 tokenizer 和 feature_extractor)
# 这里仍然需要原始模型的名称来加载并保存Processor，因为它不包含在您的 state_dict 中
processor = CLIPProcessor.from_pretrained(BASE_MODEL_NAME_OR_PATH)
processor.save_pretrained(SAVE_PATH)

print(f"合并后的CLIP模型和Processor已成功保存到 {SAVE_PATH}")
print("-" * 50)