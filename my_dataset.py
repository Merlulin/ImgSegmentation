import os

import torch
from torch.utils.data import Dataset
from PIL import Image

class CarvanaSegmentation(Dataset):
    def __init__(self, root_dir, transforms=None, txt_name: str = 'train.txt'):
        super(CarvanaSegmentation, self).__init__()
        # 确认一下root路径是否存在
        assert os.path.exists(root_dir), "path '{}' dose not exist.".format(root_dir)
        img_dir = os.path.join(root_dir, 'train')
        mask_dir = os.path.join(root_dir, 'train_mask')
        txt_path = os.path.join(root_dir, txt_name)
        assert os.path.exists(txt_path), "txt_name '{}' dose not exist.".format(txt_name)

        with open(txt_path, 'r') as f:
            file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
        self.images = [os.path.join(img_dir + '/' + file_name) for file_name in file_names]
        self.masks = [os.path.join(mask_dir + '/' + file_name[:-4] + '_mask.gif') for file_name in file_names]
        assert (len(self.masks) == len(self.images))
        self.transforms = transforms

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        target = Image.open(self.masks[idx]).convert('L')

        if self.transforms is not None:
            # 如果有数据增强，必须连带着mask一起增强，不然标签对应不上
            img, target = self.transforms(img, target)

        return img, target


    def __len__(self):
        return len(self.images)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        return images, targets


# def cat_list(images, fill_value=0):
#     # 计算该batch数据中，channel, h, w的最大值
#     max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
#     batch_shape = (len(images),) + max_size # 元组加法
#     batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
#     for img, pad_img in zip(images, batched_imgs):
#         # 实际就是拿到每一个Image，然后切片成同一个尺寸大小的图片
#         # 因为验证集的图片大小很有可能尺寸不同，需要保证tensor中的尺寸相同，所以会选择最大的size
#         pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
#     return batched_imgs


class CarvanaSegmentation_Predict(Dataset):
    def __init__(self, root_dir, transforms=None, txt_name: str = 'train.txt'):
        super(CarvanaSegmentation_Predict, self).__init__()
        # 确认一下root路径是否存在
        assert os.path.exists(root_dir), "path '{}' dose not exist.".format(root_dir)
        img_dir = os.path.join(root_dir, 'test')
        txt_path = os.path.join(root_dir, txt_name)
        assert os.path.exists(txt_path), "txt_name '{}' dose not exist.".format(txt_name)

        with open(txt_path, 'r') as f:
            file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
        self.images = [os.path.join(img_dir + '/' + file_name) for file_name in file_names]
        self.transforms = transforms

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')

        if self.transforms is not None:
            # 如果有数据增强，必须连带着mask一起增强，不然标签对应不上
            img = self.transforms(img)

        return img


    def __len__(self):
        return len(self.images)