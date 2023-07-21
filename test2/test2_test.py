import gradio as gr

import matplotlib.pyplot as plt # plt 用于显示图片
plt.switch_backend('agg')
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np

from PIL import Image

# pytorch
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

import pickle

num_classes = 10
# 超参数设置
num_epochs = 100
batch_size = 32
learning_rate = 0.005

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


transform0 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

# 测试数据集
test_data = torchvision.datasets.CIFAR10(
    root='/home/crq2/AI/dataset',
    train=False,
    transform=transform0
)
# 测试数据加载器
test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                          batch_size=batch_size,
                                          shuffle=False)

#定义残差块
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1,Resblock=False):  #需要判断是否需要1×1的卷积
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                    kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                    kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                        kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.resblock = Resblock

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        if self.resblock:
            Y += X                                                 ###############################################可以去掉 不存在残差
        return F.relu(Y)


def resnet_block(input_channels, num_channels, num_residuals, first_block=False,res_block=False):
      blk = []
      for i in range(num_residuals):
          if i == 0 and not first_block:
              blk.append(Residual(input_channels, num_channels,use_1x1conv=True, strides=2,Resblock=res_block))
          else:
              blk.append(Residual(num_channels, num_channels,Resblock=res_block))
      return blk

# 生成Resnet
class ResNet(nn.Module):
    def __init__(self,resblock):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True,res_block=resblock))
        self.b3 = nn.Sequential(*resnet_block(64, 128, 2,res_block=resblock))
        self.b4 = nn.Sequential(*resnet_block(128, 256, 2,res_block=resblock))
        self.b5 = nn.Sequential(*resnet_block(256, 512, 2,res_block=resblock))
        self.linear = nn.Linear(512, 10)
        self.Aavgpool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.Aavgpool(x)
        x = torch.flatten(x,1)
        x = self.linear(x)

        return x

# 实例化一个模型
model = ResNet(False)
model_resblock = ResNet(True)
model.load_state_dict(torch.load('test2/modelres.ckpt'))
model_resblock.load_state_dict(torch.load('test2/modelres_block.ckpt'))
model = model.cuda()
model_resblock = model_resblock.cuda()

# 读入train过程的结果
with open('test2/accres_block.pickle', 'rb') as file:   #用with的优点是可以不用写关闭文件操作
    accuracy_history = pickle.load(file)
with open('test2/lossres_block.pickle', 'rb') as file:   #用with的优点是可以不用写关闭文件操作
    loss_history = pickle.load(file)
with open('test2/accres.pickle', 'rb') as file:   #用with的优点是可以不用写关闭文件操作
    accuracy_history0 = pickle.load(file)
with open('test2/lossres.pickle', 'rb') as file:   #用with的优点是可以不用写关闭文件操作
    loss_history0 = pickle.load(file)


# 输出测试集精度
model.eval()
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

print('Accuracy of the net without resblock on the train iamges is {:.2f} %'.format(accuracy_history[-1]))
print('Accuracy of the net without resblock on the test iamges is {:.2f} %'.format(100 * correct / total))
print('\n')
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

# 输出测试集精度
#总精度及各类精度相关参数定义
model_resblock.eval()
correct = 0
total = 0
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}
with torch.no_grad():
    for images, labels in test_loader:
        images = images.cuda()
        labels = labels.cuda()
        outputs = model_resblock(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        for label, prediction in zip(labels, predicted):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

print('Accuracy of the net with resblock on the train iamges is {:.2f} %'.format(accuracy_history[-1]))
print('Accuracy of the net with resblock on the test iamges is {:.2f} %'.format(100 * correct / total))
print('\n')
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(range(100), loss_history0,loss_history)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend(labels=["no_resblock","resblock"],loc="best")

plt.subplot(1, 2, 2)
plt.plot(range(100), accuracy_history0,accuracy_history)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend(labels=["no_resblock","resblock"],loc="best")
plt.show()
curve_fig = "test2/curve.png"
plt.savefig(curve_fig)
# 可视化数据查看
import itertools
def print_label(input_dex):
  
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

  return classes[pre[0]],classes[labels[m].numpy()],show_img


demo = gr.Interface(fn=print_label,
                    inputs=gr.Textbox(label="Input 0-9999"),
                    outputs=[
                    gr.Textbox(label="predict label"),
                    gr.Textbox(label="true label"),
                    gr.outputs.Image(type="pil",label="Image")]
                    )
demo.launch()