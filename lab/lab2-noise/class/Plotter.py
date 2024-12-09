import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

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

        # 设定颜色条的范围为 [0, 100]，无论实际数据如何
        norm = plt.Normalize(0, 100)

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