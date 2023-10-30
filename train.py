import logging
import os.path
import warnings

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional as F
import torchvision.utils as vutils


import numpy as np
import matplotlib.pyplot as plt
from datasets import CarvanaSegmentation
from models import fcn_resnet50
from utils import transforms as T
from utils import create_lr_scheduler, train_one_epoch, evaluate

class SegmentationPresetTrain:

    def __init__(self, hfilp_prob=0.5, vfilp_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        trans = [T.Resize(256, 256)]
        if hfilp_prob > 0:
            trans.append(T.RandomHorizontalFlip(hfilp_prob))
        if vfilp_prob > 0:
            trans.append(T.RandomVerticalFlip(vfilp_prob))
        trans.extend([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetValid:

    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.Resize(256, 256),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transforms(train: bool):
    return SegmentationPresetTrain(hfilp_prob=0, vfilp_prob=0) if train else SegmentationPresetValid()


def plot_pred_img(samples, pred):
    image = samples[0].to('cpu')
    mask = samples[1].to('cpu')
    pred = pred.to('cpu')
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(12,6))
    fig.tight_layout() # tight_layout 函数会自动调整子图的参数

    ax1.axis('off')
    ax1.set_title('image')
    ax1.imshow(np.transpose(vutils.make_grid(image, padding=2).numpy(), (1, 2, 0)))

    ax2.axis('off')
    ax2.set_title('ground trues')
    ax2.imshow(np.transpose(vutils.make_grid(mask, padding=2).numpy(), (1, 2, 0)), cmap='gray')

    ax3.axis('off')
    ax3.set_title('pred masks')
    ax3.imshow(np.transpose(255 - vutils.make_grid(pred, padding=2).numpy().astype('uint8'), (1, 2, 0)), cmap='gray')
    
    plt.show()


def plot_train_progress(model: nn.Module, dataloader: DataLoader, device: str):
        samples = next(iter(dataloader))
        val_imgs = samples[0].to(device)
        val_targets = samples[1].to(device)
        pred = model(val_imgs)
        pred = pred['out']
        plot_pred_img(samples, pred.detach())


def create_model(aux: bool, num_classes: int, pretrain_backbone: bool=False, backbone_dir=None,pretrain_model: bool=False, checkpoint=None):
    model = fcn_resnet50(aux=aux, num_classes=num_classes, pretrain_backbone=pretrain_backbone, pretrain_dir=backbone_dir)

    if pretrain_model and checkpoint is not None:
        weight_dict = torch.load(checkpoint, map_location='cpu')

        missing_keys, unexpected_keys = model.load_state_dict(weight_dict, strict=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)

    return model


def create_logger():
    # 设置训练logger
    logger = logging.getLogger(name='training_log')
    logger.setLevel(logging.INFO)

    # 输出控制台的处理器
    consoleHandler = logging.StreamHandler()
    # 输出到文件中的处理器
    fileHandler = logging.FileHandler(filename='Carvana.log', mode='w')

    standard_formatter = logging.Formatter('%(asctime)s %(name)s [%(pathname)s line:(lineno)d] %(levelname)s %(message)s]')
    simple_formatter = logging.Formatter('%(levelname)s %(message)s')

    consoleHandler.setFormatter(standard_formatter)
    fileHandler.setFormatter(simple_formatter)

    logger.addHandler(consoleHandler)
    logger.addHandler(fileHandler)

    return logger


def main(args):
    # torch.device 可以之指明创建那个设备对象
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 设置打印日志
    logger = create_logger()
    print(logger)

    batch_size = args.batch_size
    num_classes = 1 if args.num_classes == 1 else args.num_classes + 1


    train_dataset = CarvanaSegmentation(args.data_path, get_transforms(train=True))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)

    valid_dataset = CarvanaSegmentation(args.data_path, get_transforms(train=False), 'valid', 'valid.txt')
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

    model = create_model(aux=True, num_classes=num_classes, pretrain_backbone=True, backbone_dir=args.backbone_path, pretrain_model=False)
    model.to(device)

    params_to_optimize = [
        {"params": [p for p in model.backbone.parameters() if p.requires_grad]},
        {"params": [p for p in model.classifier.parameters() if p.requires_grad]}
    ]

    if args.aux:
        params = [p for p in model.aux_classifier.parameters() if p.requires_grad]
        params_to_optimize.append({"params": params, "lr": args.lr * 10})

    optimizer = torch.optim.Adam(params_to_optimize, lr=args.lr, weight_decay=args.weight_decay)

    criterion = nn.BCEWithLogitsLoss()

    # 自动混合精度，用于加速运算，节约显存
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    lr_scheduler = create_lr_scheduler(optimizer, len(train_dataloader), args.num_epochs, warmup=True)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    # 训练设置
    train_losses = []
    val_losses = []
    lr_rates = []

    min_loss = 0x3f3f3f3f
    max_score = 0.

    for epoch in range(args.start_epoch, args.num_epochs):
        logger.info("epoch : {}".format(epoch + 1))
        # 训练一个epoch
        train_epoch_loss = train_one_epoch(model, optimizer, criterion, train_dataloader, device, lr_scheduler, scaler)
        train_losses.append(train_epoch_loss)

        # 验证一个epoch
        val_epoch_loss, dice_epoch_score = evaluate(model, criterion, valid_dataloader, device)
        val_losses.append(val_epoch_loss)

        lr_rates.append(optimizer.param_groups[0]['lr'])

        logger.info(f"epoch - {epoch + 1}/{args.num_epochs}:\n"
                    f"{' ' * 5}dice score - {dice_epoch_score}\n"
                    f"{' ' * 5}train loss - {train_epoch_loss}\n"
                    f"{' ' * 5}val loss - {val_epoch_loss}\n"
                    f"{' ' * 5}LR rate - {lr_scheduler.get_last_lr()}")

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}

        if (val_epoch_loss <= min_loss and dice_epoch_score >= max_score):
            min_loss = val_epoch_loss
            max_score = dice_epoch_score
            torch.save(save_file, args.save_weight_path + "model_weight.pth")
            logger.info(f"epoch - {epoch + 1} save the model which dice score : {dice_epoch_score} and val loss : {val_epoch_loss}")

    return {
        'lr': lr_rates,
        'train_loss': train_losses,
        'valid_loss': val_losses,
    }

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="pytorch fcn training")

    parser.add_argument("--data-path", default="./data/", help="Data root")
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--aux", default=True, type=bool, help="auxilier loss")
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=12, type=int)
    parser.add_argument("--num-epochs", default=30, type=int, metavar="N",
                        help="number of total epochs to train")
    parser.add_argument("--num-workers", default=8, type=int, help='num of worker')

    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start_epoch')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")
    parser.add_argument("--backbone-path", default='./weight/resnet50.pth', help='backbone pretrain weight path')
    parser.add_argument("--save-weight-path", default='./weight/', help='save weight path')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    args = parse_args()

    if not os.path.exists("./weight"):
        os.mkdir("./weight")

    main(args)
