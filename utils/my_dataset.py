from PIL import Image
import torch
from torch.utils.data import Dataset

from utils.global_var import get_global_logger

logger = get_global_logger()

# 继承自 Dataset 类是为了创建一个自定义的数据集类 MyDataSet，以便与深度学习框架（如 PyTorch）中的数据加载机制兼容。
class MyDataset(Dataset):

    def __init__(self, images_path: list, images_class: list, transform=None):
        super().__init__()
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):

        img = Image.open(self.images_path[item])

        if img.mode != 'RGB':
            logger.error("iamge: {} isn't RGB mode.".format(self.images_path[item]))
            raise
        
        label = self.images_class[item]

        if self.transform != None:
            img = self.transform(img)
            
        return img, label 
    
    
    # 这个方法的主要作用是将一个批次的数据（图像和标签）整理成适合模型输入的张量格式。这样可以方便地进行批量训练和推理。
    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels