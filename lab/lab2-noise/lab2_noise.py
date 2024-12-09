import torch
import torchvision
import torchvision.transforms as transforms
import random

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# 数据集类-----------------------------------------------------------------------------------------------------
class Dataset:

    def __init__(self, batch_size, noise_rate,noise_option, root, transform,):
        self.batch_size = batch_size 
        self.noise_rate = noise_rate 
        self.root = root
        self.transform = transform 
        self.noise_option = noise_option

        # 加载数据集
        self.trainset = torchvision.datasets.MNIST(root=self.root, train=True, download=True, transform=self.transform)
        self.testset = torchvision.datasets.MNIST(root=self.root, train=False, download=True, transform=self.transform)
        
        # 添加噪声到训练数据
        if self.noise_option == 1:
            self._add_noise2()
        if self.noise_option == 2:
            self._add_noise2()
        
        # 使用 DataLoader 加载数据集
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.batch_size, shuffle=False)

    def _add_noise1(self):
        """ 对训练数据进行标签对称噪声处理 """
        labels = self.trainset.targets
        num_samples = len(labels)
        num_replace = int(num_samples * self.noise_rate)
        symmetry_sample = random.sample(range(num_samples), num_replace)

        # 对称噪声标签
        for idx in symmetry_sample:
            original_label = labels[idx]
            new_label = random.randint(0, 9)
            while new_label == original_label:  # 确保噪声标签与原标签不同
                new_label = random.randint(0, 9)
            labels[idx] = new_label

        # 使用 TensorDataset 更新数据集
        self.trainset.targets = labels  # 直接修改 trainset 的标签

    def _add_noise2(self):
        """ 对训练数据进行标签非对称噪声处理 """
        labels = self.trainset.targets
        num_samples = len(labels)
        num_replace = int(num_samples * self.noise_rate)
        asymmetric_sample = random.sample(range(num_samples), num_replace)

        # 非对称噪声标签
        for idx in asymmetric_sample:
            original_label = labels[idx]
            new_label=int((original_label+1)%10)
            labels[idx] = new_label

        # 使用 TensorDataset 更新数据集
        self.trainset.targets = labels  # 直接修改 trainset 的标签

    

# 模型类-----------------------------------------------------------------------------------------------------
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
        x = x.view(x.size(0), -1)  # 动态展平
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
    


# 绘图类-----------------------------------------------------------------------------------------------------
class Plotter:
    @staticmethod
    def noise_accuracy(noise_rates, test_accuracies, average_accuracies):
        """ 噪声率与准确率的绘图 """
        # 设置 seaborn 样式
        sns.set_palette("pastel")  # 使用柔和的颜色调色板
        sns.set_context("talk")    # 提升字体大小
        # 创建图形和子图，调整图形大小
        fig, ax1 = plt.subplots(figsize=(8, 6))  # 缩小图形大小
        
        # 启用网格线，并自定义网格线样式
        plt.grid(True, which='both', axis='both', linestyle='--', color='gray', alpha=0.7)  # 设置网格线样式

        # 绘制曲线图（左 y 轴）
        ax1.set_xlabel("Noise Rate (%)", fontsize=14)

        # 定义渐变色
        cmap = LinearSegmentedColormap.from_list("grad_orange", ["lightcoral", "orange"])  # 渐变色从浅珊瑚到橙色

        # 设定颜色条的范围
        norm = plt.Normalize(min(average_accuracies), max(average_accuracies))

        # 使用渐变色填充条形图
        bars = ax1.bar(noise_rates, average_accuracies, color=cmap(norm(average_accuracies)), width=4, alpha=0.7)

        # 给条形图添加统一的标签
        ax1.set_ylabel("Average Train Accuracy (%)", fontsize=14, color='orange')
        ax1.tick_params(axis='y', labelcolor='orange')

        # 添加标题
        plt.title('Noise Rate and Accuracy', fontsize=16)

        # 创建第二个 y 轴（右 y 轴）
        ax2 = ax1.twinx()
        
        # 绘制曲线图（右 y 轴）
        ax2.plot(noise_rates, test_accuracies, color='blue', label='Test Accuracy', marker='o')

        ax2.set_ylabel("Test Accuracy (%)", fontsize=14, color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')

        # 在每个点上添加标签
        for i, (x, y) in enumerate(zip(noise_rates, test_accuracies)):
            ax2.text(x, y+1, f'{y:.2f}%', color='blue', ha='center', va='bottom', fontsize=10) 

        # 获取 y 轴的最大最小值
        y_min = min(min(test_accuracies), min(average_accuracies))
        y_max = max(max(test_accuracies), max(average_accuracies))
        
        # 统一设置两个 y 轴的范围
        ax1.set_ylim(y_min - 5, y_max + 5)  # 给 y 轴留一点间隙
        ax2.set_ylim(y_min - 5, y_max + 5)

        # 显示图例
        ax2.legend(loc='upper right')

        # 显示图形
        plt.show()

# 主程序
def main():
    # 初始化参数
    """
    batch: 批大小
    noise_rate: 标签噪声率
    root: 数据集路径
    transform: 数据预处理
    epochs_num: 训练轮数
    learning_rate: 学习率
    noise_option: 噪声选项
    """
    #-----------------------------------------------------------------------------------------
    batch=64
    noise_rate=0
    root='./data'
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    epochs_num=5
    learning_rate=0.001
    noise_option=1
    #-----------------------------------------------------------------------------------------

    noise_rates=[]
    accuracies_test=[]
    accuracies_train=[]


    # 遍历噪声率
    for noise_rate in range(0,110,10):
        # 转换噪声率为小数
        noise_rate=noise_rate/100.0

        # 打印噪声率
        print(f"Noise Rate: {noise_rate*100}%")

        # 初始化数据集
        dataset = Dataset(batch, noise_rate, noise_option, root,transform)
        
        # 初始化模型
        model = CNNModel()

        # 训练模型
        train_losses, train_accuracies=model.train_model(dataset.trainloader, epochs_num, learning_rate)
        print()
        
        # 测试模型
        test_accuracy= model.test_model(dataset.testloader)
        print()

        # 记录噪声率,测试准确率,平均训练准确率
        noise_rates.append(noise_rate*100)
        accuracies_test.append(test_accuracy)
        accuracies_train.append(sum(train_accuracies)/epochs_num)

    # 绘制噪声率和准确率曲线
    Plotter.noise_accuracy(noise_rates, accuracies_test,accuracies_train)


if __name__ == "__main__":
    main()