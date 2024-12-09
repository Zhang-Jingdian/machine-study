import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import torch

# 模型类
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 定义网络层
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)  # 输入1个通道，输出32个通道，卷积核大小为3x3
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # 输入32个通道，输出64个通道，卷积核大小为3x3
        self.fc1 = nn.Linear(7 * 7 * 64, 128)  # 全连接层，输入大小为 7x7x64（经过池化后的图像大小）
        self.fc2 = nn.Linear(128, 10)  # 输出10个类别（对应0-9）

    def forward(self, x):
        """ 前向传播 """
        x = F.relu(self.conv1(x))  # 卷积层 + 激活函数
        x = F.max_pool2d(x, 2)  # 最大池化
        x = F.relu(self.conv2(x))  # 卷积层 + 激活函数
        x = F.max_pool2d(x, 2)  # 最大池化
        x = x.view(-1, 7 * 7 * 64)  # 展平操作，将图像从二维变为一维向量
        x = F.relu(self.fc1(x))  # 全连接层 + 激活函数
        x = self.fc2(x)  # 输出层
        return x

    # 训练函数
    def train_model(self, trainloader, epochs_num, learning_rate):
        """ 训练模型 """
        self.trainloader = trainloader #训练集
        self.epochs_num= epochs_num # 设置训练轮数
        self.learning_rate = learning_rate # 设置学习率
        self.to(self.device)  # 将模型移动到设备
        self.train()  # 设置模型为训练模式
        
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()   # 使用交叉熵损失函数
        optimizer = optim.Adam(self.parameters(), lr=learning_rate) # 使用Adam优化器
        
        train_losses = []
        train_accuracies = []
        
        for epoch in range(epochs_num):
            running_loss = 0.0
            correct = 0
            total = 0

            # 使用 tqdm 包装 trainloader 来显示进度条
            for inputs, labels in tqdm(trainloader, desc=f'Epoch {epoch + 1}/{epochs_num}', unit='batch'):
                inputs, labels = inputs.to(self.device), labels.to(self.device) # 将数据移到合适的设备
                
                # 前向传播
                outputs = self(inputs) 
                loss = criterion(outputs, labels)  # 计算损失
                
                # 反向传播和优化
                optimizer.zero_grad()  # 梯度清零
                loss.backward()  # 反向传播
                optimizer.step()   # 更新权重
                
                running_loss += loss.item() # 统计损失和准确率
                _, predicted = torch.max(outputs.data, 1)  # 获取预测的类别
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            epoch_loss = running_loss / len(trainloader)
            epoch_accuracy = 100 * correct / total
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_accuracy)
            
            print(f"Epoch {epoch + 1}/{epochs_num}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
            
        return train_losses, train_accuracies

    # 测试函数
    def test_model(self, testloader):
        """ 测试模型 """
        self.testloader = testloader
        self.to(self.device)  # 将模型移动到设备
        self.eval()  # 设置模型为测试模式

        correct = 0
        total = 0

        with torch.no_grad():  # 在测试时不计算梯度
            for inputs, labels in tqdm(testloader, desc=f'Test', unit='batch'):
                inputs, labels = inputs.to(self.device), labels.to(self.device)  # 将数据移到合适的设备

                # 前向传播
                outputs = self(inputs)
                
                # 统计准确率
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_accuracy = 100 * correct / total
        print(f"Test Accuracy: {test_accuracy}%")
                
        return test_accuracy