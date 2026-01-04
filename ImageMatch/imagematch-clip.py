import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset,random_split
from torchvision import transforms
from loguru import logger
import torch.optim as optim
import pandas as pd
import os
import clip
from PIL import Image
from torch.optim import lr_scheduler
import torch.nn as nn

# 定义数据集类
class ImageTextDataset(Dataset):
    def __init__(self, image_paths, text_paths, preprocess):
        self.image_paths = image_paths
        self.text_paths = text_paths  # 存储文本文件路
        self.preprocess = preprocess
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 加载图像并进行预处理
        image = Image.open(self.image_paths[idx])
        image = self.preprocess(image).to(device)
        
        # 获取文本路径并加载文本
        text_path = self.text_paths[idx]
        with open(text_path, 'r') as file:
            text = file.read().strip()  # 读取文本文件内
        
        # 对文本进行编码
        text_tokens = clip.tokenize([text], truncate=True).squeeze(0).to(device)  # 确保维度正确
        
        return image, text_tokens  # 返回索引以便打印路径


def get_image_and_caption_paths(root_folder):
    image_paths = []
    texts = []
    a = 0
    
    for c_folder in os.listdir(root_folder):
        c_folder_path = os.path.join(root_folder, c_folder)
        
        if os.path.isdir(c_folder_path):
            
            image_path = None
            for file in os.listdir(c_folder_path):
                file_path = os.path.join(c_folder_path, file)
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  # 常见图片文件格式
                    image_path = file_path
                    break
            
            # 获取caption.txt文件路径
            caption_path = os.path.join(c_folder_path, 'caption.txt')
            
            # 只有在image_path和caption_path都存在时才添加
            if image_path and os.path.exists(caption_path):
                image_paths.append(image_path)
                texts.append(caption_path)
                a = a+1
        # if a > 90000:
        #     break
    
    return image_paths, texts
# 创建模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net, preprocess = clip.load("ViT-B/32",device=device,jit=False)

optimizer = optim.Adam(net.parameters(), lr=1e-6,betas=(0.9,0.98),eps=1e-6,weight_decay=0.001)
scheduler = lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.1)

# 创建损失函数
loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()


# Usage
root_folder = 'path1'  # root_folder路径

image_paths, texts = get_image_and_caption_paths(root_folder)

print(f"Total dataset size before sampling: {len(image_paths)}")

# 确保数据够多
if len(image_paths) < 90000:
    raise ValueError(f"总数据量不足90000条，只有 {len(image_paths)} 条")

# 截取前 90000 条，保持顺序一致
image_paths_trimmed = image_paths[:90000]
texts_trimmed = texts[:90000]
# 使用自定义数据集
batch_size = 16
dataset = ImageTextDataset(image_paths_trimmed, texts_trimmed, preprocess)
dataset_size = len(dataset)
# train_ratio = 0.9
# train_size = int(train_ratio * len(dataset))
# test_size = len(dataset) - train_size
# train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # 训练集shuffle
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)   # 测试集不shuffle

# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)  # 设定批次大小为2

phase = "train"
model_name = "clip-finetune"
ckt_gap = 4
epoches = 20
for epoch in range(epoches):
    # scheduler.step()
    total_loss = 0
    batch_num = 0
    # 使用混合精度，占用显存更小
    with torch.cuda.amp.autocast(enabled=True):
        for images,label_tokens in train_loader:
            # 将图片和标签token转移到device设备
            images = images.to(device)
            label_tokens = label_tokens.to(device)
            batch_num += 1
            # 优化器梯度清零
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == "train"):
                logits_per_image, logits_per_text = net(images, label_tokens)
                ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
                cur_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
                total_loss += cur_loss
                if phase == "train":
                    cur_loss.backward()
                    if device == "cpu":
                        optimizer.step()
                    else:
                        optimizer.step()
                        clip.model.convert_weights(net) 
            # if batch_num % 16 == 0:
            #     logger.info('{} epoch:{} loss:{}'.format(phase,epoch,cur_loss))
        epoch_loss = total_loss / dataset_size
        logger.info('{} epoch:{} loss:{}'.format(phase,epoch,epoch_loss))
        torch.save(net.state_dict(),f"{model_name}_epoch_{epoch}.pth")
        logger.info(f"weights_{epoch} saved")
        if epoch % ckt_gap == 0:
            checkpoint_path = f"{model_name}_ckt.pth"
            checkpoint = {
                'it': epoch,
                'network': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()}
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"checkpoint_{epoch} saved")
        logger.info('{} Loss: {:.4f}'.format(
            phase, epoch_loss))
        scheduler.step()
