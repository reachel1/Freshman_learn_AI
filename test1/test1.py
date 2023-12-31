# -*- coding: utf-8 -*-
"""test1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1FESZY3rSehmdAip_X-W5tjH_QJ6qLrqy
"""




import gradio as gr

import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np

from PIL import Image

# pytorch
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR

num_classes = 10
# 超参数设置
num_epochs = 2
batch_size = 5
learning_rate = 0.001

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

LOAD_CIFAR = True
DOWNLOAD_CIFAR = False
# 从data继承读取数据集的类
from torch.utils.data import Dataset, DataLoader

# 训练数据集
train_data = torchvision.datasets.CIFAR10(
    root='/home/crq2/AI/dataset',
    train=True,
    transform=transform,
    download=DOWNLOAD_CIFAR,
)

# 测试数据集
test_data = torchvision.datasets.CIFAR10(
    root='/home/crq2/AI/dataset',
    train=False,
    transform=transform
)

# 训练数据加载器
train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=batch_size,
                                           shuffle=True)
# 测试数据加载器
test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                          batch_size=batch_size,
                                          shuffle=False)

# 查看数据,取一组batch
data_iter = iter(test_loader)

images, labels = next(data_iter)
# 取batch中的一张图像
idx = 2
image = images[idx].numpy()
image = np.transpose(image, (1,2,0))
plt.imshow(image)
print(classes[labels[idx].numpy()])

# 搭建卷积神经网络模型
# 三个卷积层
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
            # 32*32*3
        self.conv1 = nn.Sequential(
            # 卷积层计算
            nn.Conv2d(3, 5, kernel_size=5, stride=1, padding=2),
            #  批归一化
            nn.BatchNorm2d(5),
            #ReLU激活函数
            nn.ReLU(),
            # 池化层：最大池化
            nn.MaxPool2d(kernel_size=2, stride=1))
            # 31*31*5

        self.conv2 = nn.Sequential(
            nn.Conv2d(5, 8, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1))# 搭建卷积神经网络模型
            # 30*30*8

        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))# 搭建卷积神经网络模型
            # 15*15*16

        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 24, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1))# 搭建卷积神经网络模型
            # 14*14*24

        self.conv5 = nn.Sequential(
            nn.Conv2d(24, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))# 搭建卷积神经网络模型
            # 7*7*32

        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(7*7*32, 400),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(400, 120),
            nn.ReLU(),
            nn.Linear(120, num_classes),
        )

    # 定义前向传播顺序
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

# 实例化一个模型
model = ConvNet(num_classes)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# 自动调整学习率
#scheduler = StepLR(optimizer, step_size=10, gamma=0.9)

# 设置cuda-gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# 开始训练
model = model.cuda()
# 存储损失与精度
loss_history = []
accuracy_history = []
loss_times_history = []

total_step = len(train_loader)
for epoch in range(num_epochs):
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for i, (images, labels) in enumerate(train_loader):

        images = images.cuda()
        labels = labels.cuda()
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #累计损失
        running_loss += loss.item()
        loss_times_history.append(loss.item())
        # 计算准确率
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

        if (i+1) % 1000 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
    #scheduler.step()
    #计算各epoch中的平均损失值和准确率
    avg_loss = running_loss / len(train_loader)
    accuracy = 100.0 * correct_train / total_train
    #存储平均损失值和准确率
    loss_history.append(avg_loss)
    accuracy_history.append(accuracy)
    #输出学习率
    #print(f"Epoch [{epoch+1}/{num_epochs}], Learning Rate: {scheduler.get_lr()[0]}")

# 输出测试集精度
#总精度及各类精度相关参数定义
correct = 0
total = 0
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}
with torch.no_grad():
    for images, labels in test_loader:
        images = images.cuda()
        labels = labels.cuda()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        for label, prediction in zip(labels, predicted):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

print('Accuracy of the net on the train iamges is {:.2f} %'.format(accuracy_history[-1]))
print('Accuracy of the net on the test iamges is {:.2f} %'.format(100 * correct / total))
print('\n')
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

#  保存模型
torch.save(model.state_dict(), 'model.ckpt')

# 可视化数据查看
import itertools
def print_label(input_dex):
  plt.figure(figsize=(10, 4))
  plt.subplot(1, 2, 1)
  plt.plot(range(num_epochs), loss_history)
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.title('Loss Curve')

  plt.subplot(1, 2, 2)
  plt.plot(range(num_epochs), accuracy_history)
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.title('Accuracy Curve')

  curve_fig = "curve.png"
  plt.savefig(curve_fig)
  curve = Image.open(curve_fig)
  k = int(int(input_dex)/batch_size)
  m = int(int(input_dex)%batch_size)
  data_iter = iter(test_loader)
  images, labels = next(itertools.islice(data_iter, k, k+1))
  image = images[m].numpy()
  image = np.transpose(image, (1,2,0))
  show_img = image
  imagebatch = image.reshape(-1,3,32,32)
  # 转换为torch tensor
  image_tensor = torch.from_numpy(imagebatch)
  image_tensor = image_tensor.cuda()
  # 调用模型进行评估
  model.eval()
  output = model(image_tensor)
  precise, predicted = torch.max(output.data, 1)
  pre = predicted.cpu().numpy()

  return curve,classes[pre[0]],classes[labels[m].numpy()],show_img


demo = gr.Interface(fn=print_label,
                    inputs=gr.Textbox(label="Input 0-9999"),
                    outputs=[gr.outputs.Image(type="pil",label="loss and acc"),
                    gr.Textbox(label="predict label"),
                    gr.Textbox(label="true label"),
                    gr.outputs.Image(type="pil",label="Image")]
                    )
demo.launch()