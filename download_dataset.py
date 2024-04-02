from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# 加载AG新闻数据集
dataset = load_dataset('ag_news')

# 将数据集转换为pandas DataFrame
df = pd.DataFrame(dataset['train'])

# 划分训练集和验证集
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# 将测试集转换为pandas DataFrame
test_df = pd.DataFrame(dataset['test'])

# 创建数据目录
data_dir = 'data/ag'
os.makedirs(data_dir, exist_ok=True)

# 保存数据集为csv格式
train_df.to_csv(os.path.join(data_dir, 'train.csv'), index=False)
val_df.to_csv(os.path.join(data_dir, 'val.csv'), index=False)
test_df.to_csv(os.path.join(data_dir, 'test.csv'), index=False)

print("数据集已保存到'data/ag'目录下。")
