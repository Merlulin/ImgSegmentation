import torch

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main(args):
    device = torch.device('cude' if torch.cuda.is_available() else 'cpu')
    batch_size = args.batch_size
    # 加1加上的是背景
    num_classes = args.num_classes + 1




def parse_args():
    import argparse
    parse =argparse.ArgumentParser(description="pytorch for training")


if __name__ == '__main__':
    args = parse_args()
    main()