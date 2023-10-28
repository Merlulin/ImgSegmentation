import os

from torch.utils.data import Dataset
from PIL import Image

class CarvanaSegmentation(Dataset):
    def __init__(self, root_dir, transforms=None, mode='train', txt_name: str = 'train.txt'):
        super(CarvanaSegmentation, self).__init__()
        # 确认一下root路径是否存在
        assert os.path.exists(root_dir), "path '{}' dose not exist.".format(root_dir)
        self.mode = 'train' if mode == 'valid' else mode
        img_dir = os.path.join(root_dir, self.mode)
        mask_dir = os.path.join(root_dir, self.mode + '_mask')
        txt_path = os.path.join(root_dir, txt_name)
        assert os.path.exists(txt_path), "txt_name '{}' dose not exist.".format(txt_name)

        with open(txt_path, 'r') as f:
            file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
        self.images = [os.path.join(img_dir + '/' + file_name) for file_name in file_names]
        if self.mode != 'test':
            self.masks = [os.path.join(mask_dir + '/' + file_name[:-4] + '_mask.gif') for file_name in file_names]
            assert (len(self.masks) == len(self.images))
        self.transforms = transforms

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        # 如果是训练集和验证集
        if self.mode != 'test':
            mask = Image.open(self.masks[idx]).convert('L')
            if self.transforms is not None:
                # 如果有数据增强，必须连带着mask一起增强，不然标签对应不上
                img, mask = self.transforms(img, mask)
            return img, mask
        else:
            # 否则作为测试集只有image
            if self.transforms is not None:
                img = self.transforms(img)
            return img

    def __len__(self):
        return len(self.images)
