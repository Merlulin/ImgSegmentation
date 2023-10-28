import os

import torch
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from torchvision import transforms as T

from models import fcn_resnet50
from datasets.my_dataset import CarvanaSegmentation

import matplotlib.pyplot as plt
import numpy as np
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
        ax3.imshow(np.transpose(255 - vutils.make_grid(pred, padding=2).numpy().astype('uint8'), (1, 2, 0)), cmap='gray')
        plt.savefig(root_dir + f'epoch_{epoch}_pred.jpg')
        plt.show()
        plt.close(fig)


def main():

    weight_path = '../weight/model_weight.pth'
    root_dir = '../data/'
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
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    model = model.to(device)
    with torch.no_grad():
        for idx, imgs in enumerate(tqdm(test_dataloader)):
            imgs = imgs.to(device)
            pred = model(imgs)
            pred = pred['out']
            plot_pred_segment(imgs, pred=pred, epoch=idx, root_dir='../data/pred/')


if __name__ == '__main__':
    main()
