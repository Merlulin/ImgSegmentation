from torch.utils.data import DataLoader
from torchvision import utils as vutils

from datasets.my_dataset import CarvanaSegmentation
from utils import transforms as T

import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary

from models import fcn_resnet50

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



if __name__ == '__main__':

    transforms = T.Compose([T.ToTensor(),])

    train_dataset = CarvanaSegmentation(root_dir='./data/', transforms=transforms)
    valid_dataset = CarvanaSegmentation(root_dir='./data/', transforms=transforms, mode='valid', txt_name='valid.txt')

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False)

    # imshow_dataloader(train_loader)
    # imshow_dataloader(valid_loader)

    # 展示模型的网络结构和参数量, 注意需要使用新版的torch-summary
    net = fcn_resnet50(True, 1, False)
    net.to('cuda')
    print(summary(net, (3, 224, 224)))