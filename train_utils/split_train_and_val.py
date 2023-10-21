import os
from pathlib import Path
import random

def split_data(data_dir, label_dir, ratio: float = 0.2):
    '''将所有的图片按照图片名字按比例随机划分成训练集和验证集文档'''
    data_paths = os.listdir(data_dir)
    data_len = len(data_paths)
    valid_data_size = int(data_len * ratio)
    train_data_size = data_len - valid_data_size
    train_data_paths = random.sample(data_paths, train_data_size)
    valid_data_paths = list(set(data_paths).difference(set(train_data_paths)))
    with open('../data/train.txt', 'w') as f:
        for train_data_path in train_data_paths:
            f.write(train_data_path + '\n')
    with open('../data/valid.txt', 'w') as f:
        for valid_data_path in valid_data_paths:
            f.write(valid_data_path + '\n')



if __name__ == '__main__':
    root_dir = Path('../data/train')
    root_masks_dir = Path('../data/train_mask')
    random.seed(520)

    split_data(root_dir, root_masks_dir, ratio=0.2)