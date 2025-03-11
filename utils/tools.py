import json
import os
import random
import sys

from matplotlib import pyplot as plt

import torch
from torch.nn import CrossEntropyLoss
import tqdm

from utils.global_var import get_global_logger
from utils.settings import CLASS_INDICES_PATH, IMAGES_PATH


# 如果给定目录不存在，则创建目录
def create_dir(dir_path: str):
    if os.path.exists(dir_path) is False:
        os.makedirs(dir_path)


def read_split_data(data_path: str, val_rate: float = 0.2, plot_image: bool = True):

    random.seed(0)  # 固定随机种子，保证随机结果可复现

    logger = get_global_logger()
    
    if os.path.exists(data_path) == False:
        logger.error("dataset path: {} does not exist.".format(data_path))
        sys.exit()

    # 获取所有类别：遍历文件夹，一个文件夹对应一个类别
    dataset_class = [cla for cla in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, cla))]

    # 排序
    dataset_class.sort()

    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(dataset_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)

    with open(CLASS_INDICES_PATH, 'w') as json_file:
        json_file.write(json_str)

    train_data_path = []  # 存储训练集的所有图片路径
    train_data_label = [] # 存储训练集图片对应的索引信息

    val_data_path = []   # 存储验证集的所有图片路径
    val_data_label = []   # 存储验证集图片对应的索引信息

    every_class_num = []    # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]

    # 遍历每个文件夹下的文件
    for cla in dataset_class:
        
        cla_path = os.path.join(data_path, cla)

        # 遍历获取支持的所有文件的路径
        images = [os.path.join(data_path, cla, i) for i in os.listdir(cla_path) if os.path.splitext(i)[-1] in supported]

        # 排序
        images.sort()

        # 样本数量
        class_num = len(images)

        # 记录样本数量
        every_class_num.append(class_num)

        # 按比例随机采样验证样本
        # random.sample 函数能够确保抽取的元素是唯一的，即不会有重复的元素。
        val_path = random.sample(images, k=int(class_num * val_rate))

        # 获取该类别对应的索引
        image_class = class_indices[cla]

        # 分离训练集和验证集
        for img_path in images:
            if img_path in val_path:
                # 将路径存入验证集
                val_data_path.append(img_path)
                val_data_label.append(image_class)
            else: # 存入训练集
                train_data_path.append(img_path)
                train_data_label.append(image_class)
        
    logger.info("{} images were found in the dataset.".format(sum(every_class_num)))
    logger.info("{} images for training.".format(len(train_data_path)))
    logger.info("{} images for validation.".format(len(val_data_path)))

    if len(train_data_path) <= 0:
        logger.error("number of training images must greater than 0.")
        sys.exit()

    if len(val_data_path) <= 0:
        logger.error("number of validation images must greater than 0.")
        sys.exit()

    if plot_image:
        # 绘制每个中的类别个数柱状图
        plt.bar(range(len(dataset_class)), every_class_num, align='center')

        # 将横坐标设置成相应类别的名称
        plt.xticks(range(len(dataset_class)), dataset_class)

        # 在柱状图上加上数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v+5, s=str(v), ha='center')
        
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')

        # 设置柱状图的标题
        plt.title('dataset class distribution')
        
        # 创建图像输出目录
        create_dir(IMAGES_PATH)

        plt.savefig(IMAGES_PATH + "/dataset_class_distribution.png")

    return train_data_path, train_data_label, val_data_path, val_data_label


def train_model(model, optimizer, data_loader, device, epoch):

    logger = get_global_logger()

    model.train()

    loss_fun = CrossEntropyLoss()

    # 累计损失
    accuracy_loss = torch.zeros(1).to(device) 
    accuracy_num = torch.zeros(1).to(device)

    optimizer.zero_grad()


    sample_num = 0
    data_loader = tqdm.tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):

        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]

        accuracy_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_fun(pred, labels.to(device))
        loss.backward()

        accuracy_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, accuracy: {:.3f}".format(epoch, accuracy_loss.item() / (step + 1), accuracy_num.item() / sample_num)

        if not torch.isfinite(loss):
            logger.warning('WARNING: non-finite loss, ending training', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accuracy_loss.item() / (step + 1), accuracy_num.item() / sample_num


@torch.no_grad()
def evaluate_model(model, data_loader, device, epoch):

    loss_fun = CrossEntropyLoss()

    model.eval()

    accuracy_num = torch.zeros(1).to(device) # 累计预测正确的样本数
    accuracy_loss = torch.zeros(1).to(device) # 累计损失

    sample_num = 0
    data_loader = tqdm.tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):

        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]

        accuracy_num += torch.eq(pred_classes, labels.to(device)).sum()
        loss = loss_fun(pred, labels.to(device))

        accuracy_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, accuracy: {:.3f}".format(epoch, accuracy_loss.item() / (step + 1), accuracy_num.item() / sample_num)

    return accuracy_loss.item() / (step + 1), accuracy_num.item() / sample_num











