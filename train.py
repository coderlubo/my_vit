import argparse
import math
import os

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


from torchvision import transforms

from model import vit_base_patch16_224_in21k as create_model

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from utils.global_var import get_global_logger, set_global_logger
from utils.my_dataset import MyDataset
from utils.settings import *
from utils.tools import create_dir, evaluate_model, read_split_data, train_model

def main(args):

    logger = get_global_logger()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # 预训练权重路径
    if os.path.exists(args.pre_weight_path) is False:
        os.makedirs(args.pre_weight_path)

    # TensorBoard 日志路径
    create_dir(TENSORBOARD_PATH)

    tb_writer = SummaryWriter(TENSORBOARD_PATH)

    train_data_path, train_data_lable, val_data_path, val_data_lable = read_split_data(args.data_path)

    data_transform = {
        # 训练数据的预处理（'train'）
        # transforms.RandomResizedCrop(224)：随机裁剪图像，并调整到 224x224 的大小。
        # transforms.RandomHorizontalFlip()：随机水平翻转图像。
        # transforms.ToTensor()：将图像转换为张量，并将像素值缩放到 [0, 1] 之间。
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])：使用均值 [0.5, 0.5, 0.5] 和标准差 [0.5, 0.5, 0.5] 对图像进行标准化。
        'train': transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.5, 0.5, 0.5],
                    [0.5, 0.5, 0.5]
                )
            ]
        ),
        # 验证数据的预处理（'val'）
        # transforms.Resize(256)：将图像调整到 256 像素的大小。
        # transforms.CenterCrop(224)：从图像中心裁剪出 224x224 的区域。
        # transforms.ToTensor()：将图像转换为张量，并将像素值缩放到 [0, 1] 之间。
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])：使用均值 [0.5, 0.5, 0.5] 和标准差 [0.5, 0.5, 0.5] 对图像进行标准化。
        'val': transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.5, 0.5, 0.5],
                    [0.5, 0.5, 0.5]
                )
            ]
        )
    }

    # 训练数据集
    train_dataset = MyDataset(train_data_path, train_data_lable, data_transform['train'])

    # 验证数据集
    val_dataset = MyDataset(val_data_path, val_data_lable, data_transform['val'])
    
    logger.info('Using {} dataloader workers every process'.format(NUM_WORKER))

    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, pin_memory=True, num_workers=NUM_WORKER, collate_fn=train_dataset.collate_fn)

    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=True, pin_memory=True, num_workers=NUM_WORKER, collate_fn=train_dataset.collate_fn)

    model = create_model(num_classes=args.num_classes, has_logits=False).to(device)

    if args.pre_weight_path != "":

        if os.path.exists(args.pre_weight_path) == False:

            logger.error("weights file: {} not exist.".format(args.pre_weight_path))
        
        weights_dict = torch.load(args.pre_weight_path, map_location=device)

        # 删除不需要的权重
        del_keys = ['head.weight', 'head.bias'] if model.has_logits else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head_weight', 'head_bias']

        for k in del_keys:

            del weights_dict[k]

        logger.info(model.load_state_dict(weights_dict, strict=False))

        if args.freeze_layers:
            for name, para in model.named_parameters():

                # 除head, pre_logits外，其他权重全部冻结
                if 'head' not in name and 'pre_logits' not in name:
                    para.requires_grad_(False)

                else:
                    print('training {}'.format(name))
        
        pg = [p for p in model.parameters() if p.requires_grad]

        optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)

        lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf

        scheduler = lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lf)
        
        for epoch in range(args.epochs):
            
            # train
            train_loss, train_accuracy = train_model(model=model, optimizer=optimizer, data_loader=train_loader, device=device, epoch=epoch)

            scheduler.step()

            # validate
            val_loss, val_accuracy = evaluate_model(model=model, data_loader=val_loader, device=device, epoch=epoch)

            tags = ['train_loss', 'train_accurcay', 'val_loss', 'val_accuracy', 'learning_rate']

            tb_writer.add_scalar(tags[0], train_loss, epoch)
            tb_writer.add_scalar(tags[1], train_accuracy, epoch)
            tb_writer.add_scalar(tags[2], val_loss, epoch)
            tb_writer.add_scalar(tags[3], val_accuracy, epoch)
            tb_writer.add_scalar(tags[4], optimizer.param_groups[0]['lr'], epoch)
            
        # 创建模型存储目录
        create_dir(WEIGHT_STORAGE_DIR)
        torch.save(model.state_dict(), WEIGHT_STORAGE_PATH)


if __name__ == "__main__":
    # 设置相关的全局变量参数
    set_global_logger()

    
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_classes', type=int, default=NUM_CLASSES)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--lrf', type=float, default=LRF)

    parser.add_argument("--data-path", type=str, default=DATA_PATH)
    parser.add_argument('--model-name', type=str, default=MODEL_NAME)

    parser.add_argument('--pre_weight_path', type=str, default=PRE_WEIGHT_PATH)
    parser.add_argument('--freeze-layers', type=bool, default=FREEZE_LAYERS)

    parser.add_argument('--device', default=DEVICE)

    args = parser.parse_args()

    main(args)



