import torch
from torch.utils.data import DataLoader
from torchvision import utils as vutils

from datasets.my_dataset import CarvanaSegmentation
from utils import transforms as T
from torchvision import transforms as TT
from torch.nn import functional as F

import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary

from models import fcn_resnet50

import time
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

def imshow_dataloader(dataloader):
    batch = next(iter(dataloader))
    imgs, masks = batch[0], batch[1]

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 4))
    fig.tight_layout()

    ax1.axis('off')
    ax1.set_title('images')
    ax1.imshow(np.transpose(vutils.make_grid(imgs, padding=2).numpy(), (1, 2, 0)))

    ax2.axis('off')
    ax2.set_title('masks')
    ax2.imshow(np.transpose(vutils.make_grid(masks, padding=2).numpy(), (1, 2, 0)), cmap='gray')
    plt.show()

def test_rle_score():
    train_masks = pd.read_csv('./data/train_masks.csv')
    print(train_masks.head())

    # show_mask_image('00087a6bd4dc', '01')

    def rle_encode(mask_image):
        pixels = mask_image.flatten()
        # We avoid issues with '1' at the start or end (at the corners of
        # the original image) by setting those pixels to '0' explicitly.
        # We do not expect these to be non-zero for an accurate mask,
        # so this should not harm the score.
        pixels[0] = 0
        pixels[-1] = 0
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
        runs[1::2] = runs[1::2] - runs[:-1:2]
        return ' '.join(str(x) for x in runs)

    img = Image.open('./data/train/00087a6bd4dc_01.jpg')
    img_mask = Image.open('./data/train_mask/00087a6bd4dc_01_mask.gif')

    transform = TT.Compose([
        TT.Resize((256, 256)),
        TT.ToTensor(),
        TT.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)

    model = fcn_resnet50(True, 1, False)
    model_weight = torch.load('./weight/model_weight.pth')['model']
    model.load_state_dict(model_weight)

    with torch.no_grad():
        pred = model(img_tensor)['out']
        pred = F.interpolate(pred, (1280, 1918), mode='bilinear', align_corners=False)
        pred = torch.sigmoid(pred)
        pred = pred > 0.5
        pred = pred.cpu().detach().numpy().astype('uint8')
    pred_rle = rle_encode(pred)
    print(pred_rle)
    print(train_masks[train_masks['img'] == '00087a6bd4dc_01.jpg'].rle_mask)

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(14, 6))
    fig.tight_layout()
    ax1.imshow(img)
    ax2.imshow(img_mask)
    pred = np.squeeze(pred)
    ax3.imshow(pred)
    plt.show()

if __name__ == '__main__':

    # transforms = T.Compose([T.ToTensor(),])
    #
    # train_dataset = CarvanaSegmentation(root_dir='./data/', transforms=transforms)
    # valid_dataset = CarvanaSegmentation(root_dir='./data/', transforms=transforms, mode='valid', txt_name='valid.txt')
    #
    # train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    # valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False)

    # imshow_dataloader(train_loader)
    # imshow_dataloader(valid_loader)

    # 展示模型的网络结构和参数量, 注意需要使用新版的torch-summary
    # net = fcn_resnet50(True, 1, False)
    # net.to('cuda')
    # print(summary(net, (3, 224, 224)))

    test_rle_score()