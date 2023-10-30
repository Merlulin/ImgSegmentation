import csv
import os

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from torchvision import transforms as T

from models import fcn_resnet50
from datasets import CarvanaSegmentation
from utils import mask2rle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

def plot_pred_segment(img, mask = None, pred = None, epoch = 0, root_dir = './data/pred/'):

    if not os.path.exists(root_dir):
        os.mkdir(root_dir)

    img = img.to('cpu')
    if mask is not None:
        mask = mask.to('cpu')
    pred = pred.to('cpu')

    if mask is None:
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(14, 4))
        fig.tight_layout()

        ax1.axis('off')
        ax1.set_title('images')
        ax1.imshow(np.transpose(vutils.make_grid(img, padding=2).numpy(), (1, 2, 0)))

        ax2.axis('off')
        ax2.set_title('pred masks')
        pred = torch.sigmoid(pred)
        mask_np = vutils.make_grid(pred, padding=2).numpy()
        ax2.imshow(np.transpose(mask_np, (1, 2, 0)), cmap='gray')
        # ax2.imshow(np.transpose(255 - vutils.make_grid(pred, padding=2).numpy().astype('uint8'), (1, 2, 0)), cmap='gray')
        plt.savefig(root_dir + f'epoch_{epoch}_pred.jpg')
        plt.show()
        plt.close(fig)
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(14, 6))
        fig.tight_layout()

        ax1.axis('off')
        ax1.set_title('images')
        ax1.imshow(np.transpose( vutils.make_grid(img, padding=2).numpy(), (1, 2, 0)))

        ax2.axis('off')
        ax2.set_title('ground trues')
        ax2.imshow(np.transpose(vutils.make_grid(mask, padding=2).numpy(), (1, 2, 0)), cmap='gray')

        ax3.axis('off')
        ax3.set_title('pred masks')
        pred = torch.sigmoid(pred)
        mask_np = vutils.make_grid(pred, padding=2).numpy()
        ax3.imshow(np.transpose(mask_np, (1, 2, 0)), cmap='gray')
        # ax3.imshow(np.transpose(255 - vutils.make_grid(pred, padding=2).numpy().astype('uint8'), (1, 2, 0)), cmap='gray')
        plt.savefig(root_dir + f'epoch_{epoch}_pred.jpg')
        plt.show()
        plt.close(fig)


def main():
    
    batch_size = 4
    submission_path = './data/submission.csv'
    weight_path = './weight/model_weight.pth'
    root_dir = './data/'
    assert os.path.exists(weight_path), f"weights {weight_path} not found"
    assert os.path.exists(root_dir), f"img root {root_dir} not found"

    model = fcn_resnet50(aux=True, num_classes=1, pretrain_backbone=False)

    model_weight = torch.load(weight_path)['model']
    model.load_state_dict(model_weight)

    transforms = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    test_dataset = CarvanaSegmentation(root_dir=root_dir, transforms=transforms, mode='test', txt_name='test.txt')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    model = model.to(device)
    print("生成csv文件")
    with open(submission_path, encoding="utf-8", mode='w', newline='') as f:
        csv_f = csv.writer(f)
        csv_f.writerow(['rle_mask'])
        with torch.no_grad():
            for idx, imgs in enumerate(tqdm(test_dataloader)):
                imgs = imgs.to(device)
                pred = model(imgs)
                pred = pred['out']
                # plot_pred_segment(imgs, pred=pred, epoch=idx, root_dir='./data/pred/')
                pred = torch.sigmoid(F.interpolate(pred, (1280, 1918), mode='bilinear', align_corners=False))
                pred = pred > 0.5
                pred_np = pred.cpu().detach().numpy().astype('uint8')
                masks = [[mask2rle(item)] for item in pred_np]
                csv_f.writerows(masks)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    print(f"生成结束，共生成有{batch_size * (idx + 1)}行数据")
    mask = pd.read_csv(submission_path)
    with open('./data/test.txt', 'r') as f:
        mask.insert(0, 'img', f.readlines())
        mask['img'] = mask['img'].apply(lambda x: x[:-1])
    mask.to_csv(submission_path, index=False)


if __name__ == '__main__':
    main()

