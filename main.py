import os

import torch
from torch.utils.data import DataLoader
from PIL import Image
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from my_dataset import CarvanaSegmentation
import transforms as T
from src import fcn_resnet50

class SegmentationPresetTrainTransformer:
    def __init__(self, hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        trans = []
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        trans.extend([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)

def im_covert(image, isMask = False):
    image = image.cpu().clone().detach().numpy()
    if not isMask:
        image = image.transpose(1, 2, 0)
        image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
        image = image.clip(0, 1)
    else:
        print(image.shape)
    return image

if __name__ == '__main__':
    # 查看一下数据集的一些信息
    '''
    meta 数据：
        总共有6572种可能在训练和测试集当中出现的车的类型
    train 文件夹：
        总共有5088个图片
        需要进一步分割成训练和验证集
    '''
    data_dir = Path('./data')

    train_dir = data_dir / 'train'
    train_mask_dir = data_dir / 'train_mask'
    metadata_dir = data_dir / 'metadata.csv'

    metadata = pd.read_csv(metadata_dir, sep=',')

    # print("metadata size if : ", len(metadata))

    # print("train imgs numbers : ", len(os.listdir(train_dir)))

    train_image_paths = os.listdir(train_dir)
    train_image_mask_path = os.listdir(train_mask_dir)
    # 查看第一张图片
    # first_img_path = train_dir / train_image_paths[0]
    # first_img_mask_path = train_mask_dir / train_image_mask_path[0]
    # first_img = Image.open(first_img_path)
    # first_img_mask = Image.open(first_img_mask_path)
    # fig, ax = plt.subplots(1, 2, figsize=(50, 50))
    # ax[0].set_title('car_image', fontsize=50)
    # ax[0].imshow(first_img)
    # ax[1].set_title('car_segment_mask', fontsize=50)
    # ax[1].imshow(first_img_mask)
    # plt.show()

    # 查看一下模型的网络结构
    model = fcn_resnet50(True, 2, False)
    print(model)


