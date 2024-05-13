# 导入必要的库
from unet.first import ICC
# 导入必要的库
import torch
import torchvision.transforms as transforms
import os
import pandas as pd
import numpy as np

from torch import nn

from models.biformer import biformer_base, biformer_small, biformer_tiny
from utils import preprocess_csv, preprocess_csv_ori,preprocess_csv_bin
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc

device0 = torch.device('cuda:1')
device1 = torch.device('cuda:0')

# 定义超参数
num_epochs = 5  # 训练的轮数
batch_size = 64  # 每个批次的图像数量
learning_rate = 0.001  # 学习率
weight_decay = 0.01
num_of_batch_to_show = 50
base_path = "/home/ubuntu/Desktop/ly/astronomy/data/train0202"
weight_path = 'icc_biformer_n_bin_' + str(batch_size) + '_0202' + '.pth'
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
        self.labels = [1 if label == 'p' else 0 for label in self.labels]
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
            feature = preprocess_csv_bin(file_path, device1)
        if self.transform:
            feature = self.transform(feature).float()
        else:
            feature = torch.Tensor(feature).unsqueeze(0).float()
        # print('feature_shape',feature.shape)
        return feature, label


# 实例化Dataset
transform = transforms.Compose(
    [
        # transforms.ToTensor(),
        transforms.ToPILImage(),
        # transforms.Resize((224, 224)),  # 缩放到 224x224
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ToTensor(),
    ])


# 定义模型，包含三个部分：net, biformer_b, classifier
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net = ICC().to(device1)  # 将net放在device1上
        self.biformer_b = biformer_base(pretrained=True, num_classes=1000).to(device0)  # 将biformer_b放在device0上
        # for param in self.biformer_b.parameters():
        #     param.requires_grad = False  # 冻结biformer_b的参数
        self.resize=transforms.Resize((224,224))
        self.classifier = nn.Sequential(nn.Linear(1000, 100),
                                        nn.Dropout(0.5),
                                        nn.Linear(100, 1),
                                        nn.Sigmoid()).to(device1)  # 将classifier放在device1上

    def forward(self, x):
        x = self.net(x.to(device1))  # 将输入移动到device1，并进行net的计算
        x = self.resize(x.repeat(1, 3, 1, 1))
        x = self.biformer_b(x.to(device0))  # 将中间输出移动到device0，并进行biformer_b的计算
        x = self.classifier(x.to(device1))  # 将中间输出移动到device1，并进行classifier的计算
        return x


# 实例化模型
model = Model()

# 定义优化器和损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)  # 使用Adam优化器
criterion = torch.nn.BCELoss()  # 使用交叉熵损失函数


# 定义训练函数
def train(model,train_set, train_loader, optimizer, criterion):
    model.train()  # 将模型设置为训练模式
    train_loss = 0  # 记录训练损失
    train_acc = 0  # 记录训练准确率
    pbar = tqdm(total=len(train_set))
    for i, (images, labels) in enumerate(train_loader):  # 遍历训练数据
        images = images.to(device1)  # 将图像移动到GPU上
        labels = torch.unsqueeze(labels, 1).to(device1)  # 将标签移动到GPU上
        optimizer.zero_grad()  # 清空梯度
        outputs = model(images)  # 前向传播
        loss = criterion(outputs, labels.float())  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        # 累加损失和准确率
        train_loss += loss.item()
        train_acc += torch.sum(torch.round(outputs) == labels).item() / len(labels)
        if (i + 1) % num_of_batch_to_show == 0:  # 每num_of_batch_to_show个批次打印一次信息
            print(
                f'Batch {i + 1}, Loss: {loss.item():.4f}, Accuracy: {(torch.sum(torch.round(outputs) == labels).item() / len(labels))*100:.2f}%')
        pbar.update(images.size(0))
    pbar.close()  # 关闭进度条
    train_loss = train_loss / len(train_loader)  # 计算平均训练损失
    train_acc = train_acc / len(train_loader)  # 计算平均训练准确率
    return train_loss, train_acc


# 定义验证函数
def validate(model, val_set, val_loader, loss_fn):
    val_loss = 0.0
    val_acc = 0.0
    val_steps = 0
    # 用于计算ROC的指标
    y_true = []  # 真实标签列表
    y_scores = []  # 模型输出的概率分数列表
    # 切换到评估模式
    model.eval()
    with torch.no_grad():
        vbar = tqdm(total=len(val_set))  # 初始化进度条
        for inputs, labels in val_loader:
            # 将数据移动到设备上
            inputs = inputs.to(device1)
            labels = torch.unsqueeze(labels, 1).to(device1)

            # 前向传播
            outputs = model(inputs)

            # 计算损失
            loss = loss_fn(outputs, labels.float())

            # 累加损失和准确率
            val_loss += loss.item()
            val_acc += torch.sum(torch.round(outputs) == labels).item() / len(labels)
            val_steps += 1

            # 收集真实标签和模型输出的概率分数
            y_true.extend(labels.cpu().numpy())
            y_scores.extend(outputs.cpu().numpy())

            vbar.update(inputs.size(0))
        vbar.close()  # 关闭进度条

    # 计算验证的平均损失和准确率
    val_loss = val_loss / val_steps
    val_acc = val_acc / val_steps
    print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

    # 计算FPR, TPR, 和 ROC AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    print('FPR:', np.mean(fpr))
    print('TPR:', np.mean(tpr))
    print(f'ROC AUC: {roc_auc:.4f}')
    return val_loss, val_acc


def main():
    train_set = CustomDataset(root_dir=os.path.join(base_path, 'train'), transform=transform,
                              preprocess=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    val_set = CustomDataset(root_dir=os.path.join(base_path, 'val'), transform=transform,
                            preprocess=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    # 记录训练过程中的损失和准确率
    train_losses = []  # 记录每个epoch的训练损失
    train_accs = []  # 记录每个epoch的训练准确率
    val_losses = []  # 记录每个epoch的测试损失
    val_accs = []  # 记录每个epoch的测试准确率
    best_val_acc = 0
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path))
    # 开始训练
    for epoch in range(num_epochs):  # 遍历每个epoch
        print(f'Epoch {epoch + 1}')
        train_loss, train_acc = train(model,train_set, train_loader, optimizer, criterion)  # 训练模型
        val_loss, val_acc = validate(model,val_set, val_loader, criterion)  # 验证模型
        train_losses.append(train_loss)  # 记录训练损失
        train_accs.append(train_acc)  # 记录训练准确率
        val_losses.append(val_loss)  # 记录测试损失
        val_accs.append(val_acc)  # 记录测试准确率
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc * 100:.2f}%')
        print(f'val Loss: {val_loss:.4f}, val Accuracy: {val_acc * 100:.2f}%')
        if best_val_acc < val_acc:
            # 保存模型
            torch.save(model.state_dict(), weight_path)
        print('-' * 20)


def test(model, loss_fn):
    print(weight_path)
    model.load_state_dict(torch.load(weight_path))
    # 初始化验证的统计量
    test_loss = 0.0
    test_acc = 0.0
    test_steps = 0
    # 用于计算ROC的指标
    y_true = []  # 真实标签列表
    y_scores = []  # 模型输出的概率分数列表
    test_set = CustomDataset(root_dir=os.path.join(base_path, 'test'), transform=transform,
                             preprocess=True)
    dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    # 切换到评估模式
    model.eval()
    with torch.no_grad():
        vbar = tqdm(total=len(test_set))  # 初始化进度条
        for inputs, labels in dataloader:
            # 将数据移动到设备上
            inputs = inputs.to(device1)
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


main()
test(model, criterion)  # 验证模型
