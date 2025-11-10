#!/usr/bin/env python
# coding: utf-8

# In[1]:


#引入需要的包

import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import warnings
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
warnings.filterwarnings("ignore")


# In[2]:


#下载训练集和测试集

train_set = datasets.MNIST("data",train=True,download=True, transform=transforms.ToTensor(),)
test_set = datasets.MNIST("data",train=False,download=True, transform=transforms.ToTensor(),)


# In[3]:


#计算像素均值和标准差

all_pixels = []

for img, _ in train_set:
    all_pixels.append(img.view(-1))

all_pixels = torch.cat(all_pixels, dim=0)

mean = torch.mean(all_pixels).item()
std = torch.std(all_pixels).item()

print(f"MNIST训练集像素均值：{mean:.4f}")
print(f"MNIST训练集像素标准差：{std:.4f}")


# In[4]:


#将像素转化为[C, H, W]的浮点型张量，并进行归一化

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307,), std=(0.3081,))
])


# In[5]:


#重新导入处理之后的数据

train_dataset = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=False, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=0)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False,num_workers=0)


# In[6]:


#展示12个手写数字样本

fig = plt.figure()
for i in range(12):
    plt.subplot(3, 4, i+1)
    plt.tight_layout()
    plt.imshow(train_dataset.train_data[i], cmap='gray', interpolation='none')
    plt.title("Labels: {}".format(train_dataset.train_labels[i]))
    plt.xticks([])
    plt.yticks([])
plt.show()


# In[7]:


from collections import Counter

def count_digit_distribution(dataset, dataset_name):
    """统计数据集中每个数字（0-9）的样本数量"""
    labels = [dataset[i][1] for i in range(len(dataset))]
    # 初始化字典时用整数作为key，避免类型不统一
    count_result = {d: 0 for d in range(10)}  # 关键修改：去掉str()
    count_result.update(Counter(labels))  # 现在key都是整数，会覆盖初始0值
    print(f"\n===== {dataset_name} 数字分布 =====")
    for digit, count in count_result.items():
        print(f"数字 {digit}: {count} 个样本")
    print(f"总样本数: {len(dataset)}")
    return

count_digit_distribution(train_dataset, "训练集")
count_digit_distribution(test_dataset, "测试集")


# In[8]:


class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        # 第一卷积模块
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # 第二卷积模块
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # 全连接模块
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MNIST_CNN().to(device)

from torchinfo import summary
summary(model, (64, 1, 28, 28)) 


# In[9]:


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)


# In[10]:


def train(model, train_loader, optimizer, epoch):
    model.train()
    running_loss = 0.0
    for i in range(epoch):
        for batch_idx, data in enumerate(train_loader):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f'Epoch [{i+1}/{epoch}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')
                running_loss = 0.0

train(model, train_loader, optimizer, 5)


# In[11]:


classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# 1) 在 test 函数里收集
y_trues, y_preds = [], []
model.eval()
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1).cpu()
        y_preds.extend(preds.numpy())
        y_trues.extend(y.numpy())

# 2) 计算并打印报告
print(classification_report(y_trues, y_preds, target_names=classes))

# 3) 混淆矩阵
cm = confusion_matrix(y_trues, y_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=classes)
disp.plot(cmap=plt.cm.Blues)
plt.xticks(rotation=45)
plt.title("Confusion Matrix")
plt.show()


# In[12]:


from sklearn.metrics import precision_score, recall_score, f1_score

# 计算每个类别的指标
precisions = precision_score(y_trues, y_preds, average=None)
recalls = recall_score(y_trues, y_preds, average=None)
f1_scores = f1_score(y_trues, y_preds, average=None)

# 绘制柱状图
x = np.arange(len(classes))
width = 0.25  # 柱子宽度

fig, ax = plt.subplots(figsize=(8, 4))
rects1 = ax.bar(x - width, precisions, width, label='Precision', color='#3498db')
rects2 = ax.bar(x, recalls, width, label='Recall', color='#e74c3c')
rects3 = ax.bar(x + width, f1_scores, width, label='F1-Score', color='#2ecc71')

# 添加标签和标题
ax.set_xlabel('Digit Class')
ax.set_ylabel('Score')
ax.set_title('Class-wise Precision, Recall and F1-Score')
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.legend()
ax.set_ylim(0.95, 1.0)  # 聚焦高分区间，突出差异

# 在柱子上添加数值标签
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3点垂直偏移
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.tight_layout()
plt.show()


# In[13]:


# 收集错误案例

wrong_imgs, wrong_preds, wrong_trues = [], [], []
model.eval()
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1).cpu()
        wrong_mask = (preds != y)
        if wrong_mask.any():
            wrong_imgs.extend(x[wrong_mask].cpu().squeeze(1).numpy())
            wrong_preds.extend(preds[wrong_mask].numpy())
            wrong_trues.extend(y[wrong_mask].numpy())
        if len(wrong_imgs) >= 16:
            wrong_imgs = wrong_imgs[:16]
            wrong_preds = wrong_preds[:16]
            wrong_trues = wrong_trues[:16]
            break

fig, axes = plt.subplots(4, 4, figsize=(8,8))
axes = axes.flatten()

for idx, (img, pred, true) in enumerate(zip(wrong_imgs, wrong_preds, wrong_trues)):
    axes[idx].imshow(img, cmap='gray')
    axes[idx].set_title(f'True: {true} | Pred: {pred}', 
                       color='red' if pred != true else 'green')
    axes[idx].axis('off')

plt.suptitle('Wrong Classification Examples (Red = Misclassification)', fontsize=14)
plt.tight_layout()
plt.show()

