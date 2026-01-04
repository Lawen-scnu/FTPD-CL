import os
import json
import csv
from tqdm import tqdm
import torch
from peft import LoraConfig, PeftModel, PeftConfig
from modelscope import AutoModelForCausalLM
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images
new_lora_config = LoraConfig(r=8,lora_alpha=32,target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], lora_dropout=0.05,bias="none",task_type="CAUSAL_LM")
# 设置模型路径
model_path = "/deepseek-vl-7b-chat/"
lora_ckpt_path = "/checkpoint-xxx/"
# 初始化处理器和模型
#vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

#vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
vl_gpt = MultiModalityCausalLM.from_pretrained(model_path, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.float16).cuda()

vl_gpt = PeftModel.from_pretrained(vl_gpt, lora_ckpt_path,config=new_lora_config)
vl_gpt = vl_gpt.eval()
# 加载数据集 JSON 文件
json_path = "test.json"
with open(json_path, "r") as f:
    data = json.load(f)

# 确保是列表格式
records = data if isinstance(data, list) else [data]

# 设置输出 CSV 文件
output_csv = "ft_singleCaption.csv"
with open(output_csv, mode="w", newline='') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=["image_path", "reference", "prediction"])
    writer.writeheader()

    for item in tqdm(records, desc="Generating captions"):
        image_path = item["images"][0]
        reference = item["caption"] if isinstance(item["caption"], list) else item["caption"]

        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        try:
            # 构造对话内容
            conversation = [
                {
                    "role": "User",
                    "content": "<image_placeholder>Describe this image:",
                    "images": [image_path]
                },
                {
                    "role": "Assistant",
                    "content": ""
                }
            ]

            # 加载图片
            pil_images = load_pil_images(conversation)
            prepare_inputs = vl_chat_processor(
                conversations=conversation,
                images=pil_images,
                force_batchify=True
            ).to(vl_gpt.device)
            # 保证 prepare_inputs 中所有 tensor 的 dtype 与模型一致（float16）
            for k, v in vars(prepare_inputs).items():
                if isinstance(v, torch.Tensor) and v.dtype == torch.bfloat16:
                    setattr(prepare_inputs, k, v.to(torch.float16))



            # 获取图像嵌入
            inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
            


            # 生成描述
            outputs = vl_gpt.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=False,
                use_cache=True
            )

            prediction = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)

            # 写入结果
            writer.writerow({
                "image_path": image_path,
                "reference": reference,
                "prediction": prediction
            })

        except Exception as e:
            print(f"Failed to process {image_path}: {e}")
