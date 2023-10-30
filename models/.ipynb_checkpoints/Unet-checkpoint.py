import torch
from torch import nn
from torch.nn import functional as F

from typing import Dict
from torchsummary import summary

class DoubleConv(nn.Sequential):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            # inplace = False(默认)时,不会修改输入对象的值,而是返回一个新创建的对象,所以打印出对象存储地址不同,类似于C语言的值传递
            # inplace = True 时,会修改输入对象的值,所以打印出对象存储地址相同,类似于C语言的址传递,节省反复申请与释放内存的空间与时间,只是将原来的地址传递,效率更好
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class Down(nn.Sequential):

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels),
        )


class Up(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear: bool=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        x1 = self.up(x1)
        # 得到两个拼接图片的高宽差值
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        # 对输出的x1进行填充，使得两个concat的图片大小相同
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        # 在通道数上拼接
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x



class Unet(nn.Module):

    def __init__(self, in_channels, num_classes, bilinear: bool=True, base_c=64):
        '''
        :param in_channels:
        :param num_classes:
        :param bilinear: 是否使用双线性插值代替转置卷积
        :param base_c: 基准channel数
        '''
        super(Unet, self).__init__()
        self.head = DoubleConv(in_channels, base_c)
        self.down_1 = Down(base_c, base_c * 2)
        self.down_2 = Down(base_c * 2, base_c * 4)
        self.down_3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down_4 = Down(base_c * 8, base_c * 16 // factor)

        self.up_1 = Up(base_c * 16, base_c * 8 // factor)
        self.up_2 = Up(base_c * 8, base_c * 4 // factor)
        self.up_3 = Up(base_c * 4, base_c * 2 // factor)
        self.up_4 = Up(base_c * 2, base_c)
        self.classify = nn.Conv2d(base_c, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.head(x)
        x2 = self.down_1(x1)
        x3 = self.down_2(x2)
        x4 = self.down_3(x3)
        x5 = self.down_4(x4)
        x = self.up_1(x5, x4)
        x = self.up_2(x, x3)
        x = self.up_3(x, x2)
        x = self.up_4(x, x1)
        logits = self.classify(x)
        return {'out': logits}


def cal_h_w(hin, win, k_size, padding, stride):
    hout = (hin + 2 * padding - k_size) / stride + 1
    wout = (win + 2 * padding - k_size) / stride + 1
    return hout, wout

if __name__ == '__main__':
    # hout, wout = cal_h_w(480, 480, 3, 1, 1)
    # print(hout, wout)

    net = Unet(in_channels=3, num_classes=1)
    net.to('cuda')
    print(summary(net, (3, 480, 480)))