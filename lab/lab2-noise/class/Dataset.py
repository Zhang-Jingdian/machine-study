import torch
import torchvision
import torchvision.transforms as transforms
import random

# 数据集类
class Dataset:

    def __init__(self, batch_size, noise_rate, root, transform):
        self.batch_size = batch_size 
        self.noise_rate = noise_rate 
        self.root = root
        self.transform = transform 

        # 加载数据集
        self.trainset = torchvision.datasets.MNIST(root=self.root, train=True, download=True, transform=self.transform)
        self.testset = torchvision.datasets.MNIST(root=self.root, train=False, download=True, transform=self.transform)
        
        # 添加噪声到训练数据
        self._add_noise()
        
        # 使用 DataLoader 加载数据集
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.batch_size, shuffle=False)

    def _add_noise(self):
        """ 对训练数据进行标签噪声处理 """
        labels = self.trainset.targets
        
        num_samples = len(labels)
        num_replace = int(num_samples * self.noise_rate)
        random_sample = random.sample(range(num_samples), num_replace)

        # 随机替换噪声标签
        for idx in random_sample:
            original_label = labels[idx]
            new_label = random.randint(0, 9)
            # 确保噪声标签与原标签不同
            while new_label == original_label:
                new_label = random.randint(0, 9)
            labels[idx] = new_label

        self.trainset.targets = labels
