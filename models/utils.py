import numpy as np
import torch
from PIL import Image
import pandas as pd
# 导入PyTorch和其他必要的库
import torch
from PIL import Image

device0 = torch.device('cuda:2')


def preprocess_csv_an(csv_path):
    # 将灰度图像转换为PyTorch张量，形状为(H, W)，H和W是图像的高度和宽度
    gray_tensor = torch.Tensor(pd.read_csv(csv_path, header=None).values).to(device0)

    # 计算图像的总像素数量
    total_pixels = gray_tensor.numel()
    # 计算需要保留的像素数量（10% - 20%）
    num_pixels_to_keep = int(total_pixels * 0.1)
    # 对图像的像素值进行排序，得到一个一维的张量，形状为(total_pixels,)，和一个一维的索引张量，形状相同
    sorted_pixels, _ = torch.sort(gray_tensor.flatten())
    # 获取最暗的80%的像素值，形状为(int(total_pixels * 0.8),)
    darkest_pixels = sorted_pixels[:int(total_pixels * 0.8)]
    # 获取最亮的10%的像素值，形状为(num_pixels_to_keep,)
    brightest_pixels = sorted_pixels[-num_pixels_to_keep:]
    # 对最暗的80%的像素值设为1，使用torch.clamp函数，将图像的像素值限制在1和darkest_pixels[-1]之间，然后赋值为1
    # gray_tensor = torch.clamp(gray_tensor, 1, darkest_pixels[-1]).to(device0)
    gray_tensor[gray_tensor <= darkest_pixels[-1]] = 1
    # 对最亮的10%的像素值设为255，使用torch.clamp函数，将图像的像素值限制在brightest_pixels[0]和255之间，然后赋值为255
    # gray_tensor = torch.clamp(gray_tensor, brightest_pixels[0], 255).to(device0)
    gray_tensor[gray_tensor >= brightest_pixels[0]] = 255
    # 对剩余的10% - 20%的像素值进行线性变换到0--255，使用torch.where函数，根据条件选择不同的值，使用torch.linspace函数，生成一个等差数列，用来作为线性变换的参数
    # 使用torch.floor函数，将用作索引的张量向下取整
    index = (gray_tensor - brightest_pixels[0] - 1).long()
    # 使用torch.linspace函数，生成一个等差数列，用来作为线性变换的参数
    linear = torch.linspace(1, 254, num_pixels_to_keep, device=gray_tensor.device)
    # 使用torch.where函数，根据条件选择不同的值
    gray_tensor = torch.where(
        (gray_tensor > darkest_pixels[-1]) & (gray_tensor < brightest_pixels[0]),  # 条件
        linear[index],  # 如果条件为真，选择这个值
        gray_tensor  # 如果条件为假，选择原来的值
    )
    # 在第一个维度上增加一个维度，使得张量的形状变为(1, H, W)，表示一个批次的灰度图像
    img_tensor = gray_tensor.unsqueeze(0)
    # 返回img_tensor
    return img_tensor


def preprocess_csv_ori(csv_path, device):
    """
    按照比例,直接转换
    :param csv_path
    :param device
    :return:
    """
    # 将灰度图像转换为PyTorch张量，形状为(H, W)，H和W是图像的高度和宽度
    gray_tensor_ori = torch.Tensor(np.int32(pd.read_csv(csv_path, header=None).values)).to(device)

    # 计算图像的总像素数量
    sorted_pixels, _ = torch.sort(gray_tensor_ori.flatten())
    chayi = sorted_pixels[-1] - sorted_pixels[0]
    # 使用torch.where函数，根据条件选择不同的值
    gray_tensor = (gray_tensor_ori - sorted_pixels[0]) / chayi
    # 在第一个维度上增加一个维度，使得张量的形状变为(1, H, W)，表示一个批次的灰度图像
    img_tensor = gray_tensor.unsqueeze(0)
    # 返回img_tensor
    return img_tensor

def preprocess_csv_bin(csv_path,device):
    """
    二值化
    :param csv_path:
    :param device:
    :return:
    """
    # 将灰度图像转换为PyTorch张量，形状为(H, W)，H和W是图像的高度和宽度
    gray_tensor_ori = torch.Tensor(np.int32(pd.read_csv(csv_path, header=None).values)).to(device)

    # 计算图像的总像素数量
    total_pixels = gray_tensor_ori.numel()
    sorted_pixels, _ = torch.sort(gray_tensor_ori.flatten())
    # 获取最暗的80%的像素值，形状为(int(total_pixels * 0.8),)
    darkest_pixels = sorted_pixels[:int(total_pixels * 0.98)]
    # 获取最亮的10%的像素值，形状为(num_pixels_to_keep,)
    brightest_pixels = sorted_pixels[-int(total_pixels * 0.02):]
    # 对最暗的80%的像素值设为0，使用torch.clamp函数，将图像的像素值限制在0和darkest_pixels[-1]之间，然后赋值为0
    # gray_tensor=torch.randn(gray_tensor_ori.shape)
    gray_tensor = gray_tensor_ori.clone()
    gray_tensor[gray_tensor <= darkest_pixels[-1]] = 0
    # 对最亮的10%的像素值设为1，使用torch.clamp函数，将图像的像素值限制在brightest_pixels[0]和1之间，然后赋值为1
    gray_tensor[gray_tensor >= brightest_pixels[0]] = 1
    img_tensor = gray_tensor.unsqueeze(0)
    # 返回img_tensor
    return img_tensor


def preprocess_csv(csv_path,device):
    """
    按照比例
    :param csv_path:
    :return:
    """
    # 将灰度图像转换为PyTorch张量，形状为(H, W)，H和W是图像的高度和宽度
    gray_tensor_ori = torch.Tensor(np.int32(pd.read_csv(csv_path, header=None).values)).to(device)

    # 计算图像的总像素数量
    total_pixels = gray_tensor_ori.numel()
    # 计算需要保留的像素数量（10% - 20%）
    # 对图像的像素值进行排序，得到一个一维的张量，形状为(total_pixels,)，和一个一维的索引张量，形状相同
    sorted_pixels, _ = torch.sort(gray_tensor_ori.flatten())
    # sorted_pixels = torch.Tensor(sorted_pixels).unsqueeze(1)
    # 获取最暗的80%的像素值，形状为(int(total_pixels * 0.8),)
    darkest_pixels = sorted_pixels[:int(total_pixels * 0.85)]
    # 获取最亮的10%的像素值，形状为(num_pixels_to_keep,)
    brightest_pixels = sorted_pixels[-int(total_pixels * 0.05):]
    # 对最暗的80%的像素值设为0，使用torch.clamp函数，将图像的像素值限制在0和darkest_pixels[-1]之间，然后赋值为0
    # gray_tensor=torch.randn(gray_tensor_ori.shape)
    gray_tensor = gray_tensor_ori
    gray_tensor[gray_tensor <= darkest_pixels[-1]] = 0
    # 对最亮的10%的像素值设为1，使用torch.clamp函数，将图像的像素值限制在brightest_pixels[0]和1之间，然后赋值为1
    gray_tensor[gray_tensor >= brightest_pixels[0]] = 1
    # 对剩余的10% - 20%的像素值进行线性变换到0--1，使用torch.where函数，根据条件选择不同的值，使用torch.linspace函数，生成一个等差数列，用来作为线性变换的参数
    # 线性插值
    mid_pixels = sorted_pixels[(sorted_pixels > darkest_pixels[-1]) & (sorted_pixels < brightest_pixels[0])]
    chayi = mid_pixels[-1] - mid_pixels[0]
    # 使用torch.where函数，根据条件选择不同的值
    gray_tensor = torch.where(
        (gray_tensor > darkest_pixels[-1]) & (gray_tensor < brightest_pixels[0]),  # 条件
        (gray_tensor - mid_pixels[0]) / chayi,  # 如果条件为真，选择这个值
        gray_tensor  # 如果条件为假，选择原来的值
    )
    # 在第一个维度上增加一个维度，使得张量的形状变为(1, H, W)，表示一个批次的灰度图像
    img_tensor = gray_tensor.unsqueeze(0)
    # 返回img_tensor
    return img_tensor


def np_preprocess_csv(csv_path):
    # 将灰度图像转换为numpy数组
    gray_array = pd.read_csv(csv_path, header=None).values
    # 计算图像的总像素数量
    total_pixels = gray_array.size

    # 计算需要保留的像素数量（10% - 20%）
    num_pixels_to_keep = int(total_pixels * 0.1)

    # 对图像的像素值进行排序
    sorted_pixels = np.sort(gray_array, axis=None)

    # 获取最暗的80%的像素值
    darkest_pixels = sorted_pixels[:int(total_pixels * 0.8)]

    # 获取最亮的10%的像素值
    brightest_pixels = sorted_pixels[-num_pixels_to_keep:]

    # 对最暗的80%的像素值设为1
    gray_array[gray_array <= darkest_pixels[-1]] = 1

    # 对最亮的10%的像素值设为255
    gray_array[gray_array >= brightest_pixels[0]] = 255

    # 对剩余的10% - 20%的像素值进行线性变换到0--255
    for i in range(num_pixels_to_keep):
        # 计算当前像素值在排序后的像素值中的位置比例
        position_ratio = (i + 1) / num_pixels_to_keep

        # 根据位置比例计算线性变换后的像素值
        transformed_pixel = int((position_ratio * 254) + 1)

        # 将当前像素值设为线性变换后的像素值
        gray_array[gray_array == brightest_pixels[i]] = transformed_pixel
    img_tensor = torch.Tensor(gray_array).unsqueeze(0)
    # print(img_tensor.shape)
    return img_tensor

def get_mean_std(dataset,device):
    # 初始化均值和标准差为0
    mean = torch.zeros(1).to(device)
    std = torch.zeros(1).to(device)

    # 遍历数据集中的所有图像
    for data in dataset:
        images, _ = data  # 只关心图像数据，忽略标签
        # 将图像数据展平，并累加到均值和标准差中
        batch_mean = images.mean(dim=(0, 1, 2)).to(device)
        batch_std = images.std(dim=(0, 1, 2)).to(device)

        # 更新全局的均值和标准差
        mean += batch_mean.to(mean.device)
        std += batch_std.to(std.device)

        # 计算整个数据集的均值和标准差
    mean /= len(dataset)
    std /= len(dataset)

    # print("Mean:", mean)
    # print("Std:", std)
    return mean,std

if __name__ == '__main__':
    # 将处理后的numpy数组转换回PIL图像
    # gray_array = np.array(preprocess_csv('/home/ubuntu/Desktop/ly/astronomy/data/train0202/train/p/rs20000.csv'))
    # print(gray_array)
    # processed_image = Image.fromarray(gray_array).convert('L')
    # processed_image.save("output1.jpg")
    # # with open('./tt.png','wb') as f:
    # #     processed_image.save(f)
    """torch"""
    # 将处理后的PyTorch张量转换回PIL图像，使用torch.squeeze函数，去掉多余的维度，使用torch.clamp函数，将张量的元素限制在0到255之间，使用torch.uint8类型，表示无符号的8位整数，使用Image.fromarray函数，将张量转换为图像
    img_tensor = preprocess_csv('/home/ubuntu/Desktop/ly/astronomy/data/train0202/train/p/rs20000.csv').cpu()
    print(img_tensor.shape)
    img_tensor = img_tensor.repeat(3, 1, 1)
    print(img_tensor.shape)
    # processed_image = Image.fromarray(torch.squeeze(torch.clamp(img_tensor, 0, 255)).type(torch.uint8).numpy())
    # print(processed_image)
    # # 保存处理后的图像，假设路径为./output2.jpg
    # processed_image.save("./output100.jpg")
