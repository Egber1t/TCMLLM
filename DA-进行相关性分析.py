import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# 设置matplotlib配置，使用支持中文的字体
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 定义一个函数，用于绘制相关性热图
def plot_correlation_heatmap(corr_matrix, title, figsize=(16, 14), fontsize=8):
    plt.figure(figsize=figsize)  # 增大画布大小
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": fontsize})
    plt.xticks(rotation=90)  # X轴标签旋转90度以更好地展示
    plt.yticks(rotation=0)
    plt.title(title)
    plt.tight_layout()  # 调整布局
    plt.show()

# 加载数据
df = pd.read_csv("filtered_data.csv")

# 第二列是标签列，特征从第三列开始
label_column = df.iloc[:, 1]
features = df.iloc[:, 2:]

# 将标签列添加到特征DataFrame中
features['label'] = label_column

# 定义一个函数，用于找出相关系数大于0.5的特征
def find_high_correlations(corr_matrix, threshold=0.5):
    # 将对角线置为NaN，忽略自己与自己的相关性
    np.fill_diagonal(corr_matrix.values, np.nan)
    # 找到相关性的绝对值大于阈值的特征
    high_corr = corr_matrix.abs().unstack()
    high_corr = high_corr[high_corr > threshold]
    return high_corr.sort_values(ascending=False)

# 设置pandas显示选项，以便输出完整的DataFrame内容到文本文件
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# 打开文件以保存相关性结果
with open('correlation_results.txt', 'w', encoding='utf-8') as file:
    for method in ['pearson', 'spearman', 'kendall']:
        corr = features.corr(method=method)
        high_corr = find_high_correlations(corr, 0.5)
        file.write(f"使用{method}方法，相关系数大于0.5的特征:\n{high_corr.to_string()}\n\n")

# 重新计算相关系数并绘制热图
for method, title in [('pearson', "特征与标签的皮尔森相关性热图"), ('spearman', "特征与标签的斯皮尔曼相关性热图"), ('kendall', "特征与标签的肯德尔相关性热图")]:
    corr = features.corr(method=method)
    plot_correlation_heatmap(corr, title, figsize=(20, 18), fontsize=6)
