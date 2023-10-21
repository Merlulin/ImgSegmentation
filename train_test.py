import datetime
import os
import time
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional as F
import torchvision.utils as vutils


import numpy as np
import matplotlib.pyplot as plt
from my_dataset import CarvanaSegmentation
from src import fcn_resnet50
import transforms as T
from train_utils import create_lr_scheduler, train_one_epoch

class pretrain_transforms:

    def __init__(self, hfilp_prob=0.5, vfilp_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        trans = [T.Resize(256, 256)]
        if hfilp_prob > 0:
            trans.append(T.RandomHorizontalFlip(hfilp_prob))
        if vfilp_prob > 0:
            trans.append(T.RandomVerticalFlip(vfilp_prob))
        trans.extend([
            T.ToTensor(),
            # T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class prevalid_transforms:

    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.Resize(256, 256),
            T.ToTensor(),
            # T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transforms(train: bool):
    return pretrain_transforms(hfilp_prob=0, vfilp_prob=0) if train else prevalid_transforms()

def dice_score(preds, targets):
    preds = F.sigmoid(preds)
    preds = (preds > 0.5).float()
    score = (2. * (preds * targets).sum()) / (preds + targets).sum()
    return torch.mean(score).item()

def get_loss(preds, targets):
    loss_fn = nn.BCEWithLogitsLoss()
    loss = loss_fn(preds, targets)
    return loss

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

def plot_train_progress(model: nn.Module, dataloader: DataLoader, device):
        samples = next(iter(dataloader))
        val_imgs = samples[0].to(device)
        val_targets = samples[1].to(device)
        pred = model(val_imgs)
        pred = pred['out']
        plot_pred_img(samples, pred.detach())


if __name__ == '__main__':

    config = {'num_epochs': 30,
              'num_classes': 1,
              'num_workers': 4,
              'batch_size': 16,
              'device': 'cuda' if torch.cuda.is_available() else 'cpu',
              'root_dir': './data/',
              'aux': True,
              'learning_rate': 0.0001,
              'weight_decay': 1e-4,
              'save_weight_path': './model_weight.pth',
              }

    train_dataset = CarvanaSegmentation(config['root_dir'], get_transforms(train=True), 'train.txt')
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    valid_dataset = CarvanaSegmentation(config['root_dir'], get_transforms(train=False), 'valid.txt')
    valid_dataloader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

    device = config['device']

    model = fcn_resnet50(aux=True, num_classes=config['num_classes'], pretrain_backbone=True)
    model.to(device)

    params_to_optimizer = [
        {'params': [p for p in model.backbone.parameters() if p.requires_grad]},
        {'params': [p for p in model.classifier.parameters() if p.requires_grad]}
    ]
    if config['aux']:
        params = [p for p in model.aux_classifier.parameters() if p.requires_grad]
        params_to_optimizer.append({'params': params, 'lr': config['learning_rate'] * 0.1})

    optimizer = torch.optim.Adam(params_to_optimizer, lr=config['learning_rate'], weight_decay=config['weight_decay'])

    lr_scheduler = create_lr_scheduler(optimizer, len(train_dataloader), config['num_epochs'], warmup=True)

    criterion = nn.BCEWithLogitsLoss()

    # 展示以下图片
    # samples = next(iter(train_dataloader))
    #
    # fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 4))
    # fig.tight_layout()
    #
    # ax1.axis('off')
    # ax1.set_title('input image')
    # ax1.imshow(np.transpose(vutils.make_grid(samples[0], padding=2).numpy(),
    #                         (1, 2, 0)))
    #
    # ax2.axis('off')
    # ax2.set_title('input mask')
    # ax2.imshow(np.transpose(vutils.make_grid(samples[1], padding=2).numpy(),
    #                         (1, 2, 0)), cmap='gray')
    #
    # plt.show()

    def train(model, optimizer, criterion, scheduler=None):
        train_losses = []
        val_losses = []
        lr_rates = []

        min_loss = 0x3f3f3f3f
        max_score = 0.

        for epoch in range(config['num_epochs']):

            # 训练一个epoch
            model.train()
            train_total_loss = 0
            train_iterations = 0

            for idx, data in enumerate(tqdm(train_dataloader)):
                train_iterations += 1
                train_img = data[0].to(device)
                train_mask = data[1].to(device)

                optimizer.zero_grad()

                train_output_mask = model(train_img)
                train_loss = criterion(train_output_mask['out'], train_mask)
                train_total_loss += train_loss.item()

                train_loss.backward()
                optimizer.step()

            train_epoch_loss = train_total_loss / train_iterations
            train_losses.append(train_epoch_loss)

            # 验证一个epoch
            model.eval()
            with torch.no_grad():
                valid_total_loss = 0
                valid_iterations = 0
                scores = 0

                for vidx, val_data in enumerate(tqdm(valid_dataloader)):
                    valid_iterations += 1
                    val_img, val_mask = val_data[0].to(device), val_data[1].to(device)
                    pred = model(val_img)
                    val_loss = criterion(pred['out'], val_mask)
                    valid_total_loss += val_loss.item()
                    scores += dice_score(pred['out'], val_mask)
            val_epoch_loss = valid_total_loss / valid_iterations
            dice_epoch_score = scores / valid_iterations

            val_losses.append(val_epoch_loss)

            print('epochs - {}/{} [{}/{}], dice score: {}, train loss: {}, val loss: {}'.format(
                epoch + 1, config['num_epochs'],
                idx + 1, len(train_dataloader),
               dice_epoch_score, train_epoch_loss, val_epoch_loss
            ))

            lr_rates.append(optimizer.param_groups[0]['lr'])
            if scheduler:
                scheduler.step()  # decay learning rate
                print('LR rate:', scheduler.get_last_lr())

            if(val_epoch_loss <= min_loss and dice_epoch_score >= max_score):
                min_loss = val_epoch_loss
                max_score = dice_epoch_score
                print(f'epochs - {epoch + 1} save the model which dice score: {dice_epoch_score}, val loss: {val_epoch_loss}')
                torch.save(model.state_dict(), config['save_weight_path'])

        return {
            'lr': lr_rates,
            'train_loss': train_losses,
            'valid_loss': val_losses,
        }

    train(model, optimizer, criterion)
