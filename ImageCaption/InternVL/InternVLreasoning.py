import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from modelscope import AutoModel, AutoTokenizer
from peft import PeftModel,LoraConfig
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values
new_lora_config = LoraConfig(r=8,lora_alpha=32,target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], lora_dropout=0.05,bias="none",task_type="CAUSAL_LM")
# If you want to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.
path = "/InternVL2_5"
lora_path = "/checkpoint-xxx/"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval().cuda()
#model = PeftModel.from_pretrained(model, lora_path ,config=new_lora_config).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

generation_config = dict(max_new_tokens=1024, do_sample=True)


from tqdm import tqdm
import json
import csv
import os
# 加载 JSON 文件
json_path = "/test.json"  # 替换成你的 JSON 文件路径
with open(json_path, "r") as f:
    data = json.load(f)
# 统一格式：列表
records = data if isinstance(data, list) else [data]
# 创建输出 CSV 文件
output_csv = "base_singleCaption.csv"
with open(output_csv, mode="w", newline='') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=["image_path", "reference", "prediction"])
    writer.writeheader()

    # 遍历每个样本，显示进度条
    for item in tqdm(records, desc="Generating captions"):
        image_path = item["images"][0]
        reference = item["caption"] if isinstance(item["caption"], list) else item["caption"]

        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        try:
            pixel_values = load_image(image_path, max_num=12).to(torch.float16).cuda()
            generation_config = dict(max_new_tokens=1024, do_sample=True)

            question = "<image>\nDescribe this image: "
            prediction = model.chat(tokenizer, pixel_values, question, generation_config)

            writer.writerow({
                "image_path": image_path,
                "reference": reference,
                "prediction": prediction
            })
        except Exception as e:
            print(f"Failed to process {image_path}: {e}")

