# 导入必要的库
import torch
import torchvision
from sklearn.metrics import roc_curve, auc
import torchvision.transforms as transforms
import os
import pandas as pd
import numpy as np

from torch import nn

# 定义超参数
num_epochs = 10  # 训练的轮数
batch_size = 64  # 每个批次的图像数量
learning_rate = 0.01  # 学习率
num_of_batch_to_show = 100

from utils import preprocess_csv, preprocess_csv_ori, get_mean_std
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sys

# 要添加的路径
new_path = "/home/ubuntu/Desktop/ly/astronomy/BiFormer-public_release/"

# 检查路径是否已经在sys.path中
if new_path not in sys.path:
    sys.path.append(new_path)
from models.biformer import biformer_base, biformer_small, biformer_tiny

device1 = torch.device('cuda:3')


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, preprocess=False):
        self.root_dir = root_dir
        self.files = []
        self.labels = []

        # 遍历所有子目录并收集文件
        idx = 0
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.csv'):
                    idx += 1
                    self.files.append(os.path.join(subdir, file))
                    # 假设目录名就是类别标签
                    self.labels.append(subdir.split('/')[-1])

                    # 将标签编码为二分类（例如，class1为0，class2为1）
                # if idx == 128:
                #     break
            idx = 0
        self.labels = [0 if label == 'p' else 1 for label in self.labels]
        self.transform = transform
        self.preprocess = preprocess

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]
        if not self.preprocess:
            # 读取CSV文件
            data = pd.read_csv(file_path, header=None)
            # 假设CSV文件有多列特征数据，取所有列作为特征
            feature = data.values.astype(np.float32)
            # 如果需要，应用转换（例如，转换为Tensor）
        else:
            feature = preprocess_csv_ori(file_path, device1)
        if self.transform:
            feature = self.transform(feature).float()
        else:
            feature = torch.Tensor(feature).unsqueeze(0).float()
        # print('feature_shape',feature.shape)
        return feature, label


# 实例化Dataset
base_path = "/home/ubuntu/Desktop/ly/astronomy/data/NEAtoTrain0222"
# test_set = CustomDataset(root_dir=os.path.join(base_path, 'val'), transform=None)
# test_ms = get_mean_std(test_set,device1)
# transform = transforms.Compose(
#     [
#         # transforms.ToTensor(),
#         transforms.ToPILImage(),
#         transforms.Resize((224, 224)),  # 缩放到 224x224
#         # transforms.RandomHorizontalFlip(),  # 随机水平翻转
#         transforms.ToTensor(),
#         transforms.Normalize(mean=test_ms[0], std=test_ms[1]),
#     ])
# test_set = CustomDataset(root_dir=os.path.join(base_path, 'val'), transform=transform)
# test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

transform = transforms.Compose(
    [
        # transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  # 缩放到 224x224
        # transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
test_set = CustomDataset(root_dir=os.path.join(base_path, 'test'), transform=transform)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# 定义验证函数
def test(dataloader, model, loss_fn):
    # 初始化验证的统计量
    test_loss = 0.0
    test_acc = 0.0
    test_steps = 0
    # 用于计算ROC的指标
    y_true = []  # 真实标签列表
    y_scores = []  # 模型输出的概率分数列表

    # 切换到评估模式
    model.eval()
    with torch.no_grad():
        vbar = tqdm(total=len(test_set))  # 初始化进度条
        for inputs, labels in dataloader:
            # 将数据移动到设备上
            inputs = inputs.repeat(1, 3, 1, 1).to(device1)
            labels = torch.unsqueeze(labels, 1).to(device1)

            # 前向传播
            outputs = model(inputs)

            # 计算损失
            loss = loss_fn(outputs, labels.float())

            # 累加损失和准确率
            test_loss += loss.item()
            test_acc += torch.sum(torch.round(outputs) == labels).item() / len(labels)
            test_steps += 1

            # 收集真实标签和模型输出的概率分数
            y_true.extend(labels.cpu().numpy())
            y_scores.extend(outputs.cpu().numpy())

            vbar.update(inputs.size(0))
        vbar.close()  # 关闭进度条

    # 计算验证的平均损失和准确率
    test_loss = test_loss / test_steps
    test_acc = test_acc / test_steps
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

    # 计算FPR, TPR, 和 ROC AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    print('FPR:', np.mean(fpr))
    print('TPR:', np.mean(tpr))
    print(f'ROC AUC: {roc_auc:.4f}')


model = torchvision.models.resnet101(pretrained=True)  # 加载预训练的模型
model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 100),
                         nn.Dropout(0.5),
                         nn.Linear(100, 1),
                         nn.Sigmoid())  # 修改最后一层为二分类
model.to(device1)  # 将模型移动到GPU上
weight_path="resnet101_ast128.pth"
model.load_state_dict(torch.load(weight_path))
# 定义优化器和损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 使用Adam优化器
criterion = torch.nn.BCELoss()  # 使用交叉熵损失函数
print(weight_path)
test(test_loader, model, criterion)  # 验证模型
