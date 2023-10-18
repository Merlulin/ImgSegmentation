from collections import OrderedDict
from typing import Dict

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from .Backbone import resnet50

class IntermediateLayerGetter(nn.ModuleDict):

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        # 如果model当中没有return_layers所需要的层，则报错
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("模型中并没有返回层所需要的层结构")
        orig_return_layers = return_layers # 保存下原有返回层，不受后序使用干扰
        # 总有序字典顺序存储需要保留的层结构
        layers = OrderedDict()
        for name, layer in model.named_children():
            layers[name] = layer
            # 知道return_layers删空时，说明所有需要的层结构以及访问到了，直接break返回
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break
        # 直接调用父类的初始化函数，通过MoudleDict来构建一个Model
        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        # 因为需要最后返回的是一个字典，这样便于辅助分支所需要的数据
        out = OrderedDict()
        for name, layer in self.items():
            x = layer(x)
            if name in self.return_layers:
                out[self.return_layers[name]] = x
        return out


class FCN(nn.Module):

    def __init__(self, backbone, classifier, aux_classifier=None):
        super(FCN, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier

    def forward(self, x: Tensor):
        # 先记录下输入的分辨率
        input_shape = x.shape[-2:]
        feature = self.backbone(x)
        result = OrderedDict()
        x = feature['out']
        x = self.classifier(x)
        # 插值（mode调整为双线性插值）
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)

        result['out'] = x
        if self.aux_classifier is not None:
            x = feature['aux']
            x = self.aux_classifier(x)
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            result['aux'] = x

        return result

class FCNHead(nn.Sequential):
    '''结构比较简单的话，直接用Sequential，调用父类的初始化就行了'''
    def __init__(self, in_channels, channels):
        layers = []
        inter_channels = in_channels // 4
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=inter_channels, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(inter_channels))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.1))
        layers.append(nn.Conv2d(in_channels=inter_channels, out_channels=channels, kernel_size=1, stride=1))
        super(FCNHead, self).__init__(*layers)

def fcn_resnet50(aux, num_classes, pretrain_backbone=False):
    backbone = resnet50(replace_stride_with_dilation=[False, True, True])
    if pretrain_backbone:
        # map_location :
        # 具体来说，map_location参数是用于重定向，比如此前模型的参数是在cpu中的，我们希望将其加载到cuda:0中。
        # 或者我们有多张卡，那么我们就可以将卡1中训练好的模型加载到卡2中，这在数据并行的分布式深度学习中可能会用到。
        backbone.load_state_dict("resnet50.pth", map_location="cpu")

    # FCNhead 的输入通道数
    out_inplanes = 2048
    aux_inplanes= 1024

    layers = {'layer4': 'out'}
    if aux:
        layers['layer3'] = 'aux'

    # resnet50的原始结构中存在一些不必要的层次，比如在stage4之后的全局平均池化之类的
    backbone = IntermediateLayerGetter(backbone, return_layers = layers)
    aux_classifer = None
    if aux:
        aux_classifer = FCNHead(aux_inplanes, num_classes)
    classifier = FCNHead(out_inplanes, num_classes)

    model = FCN(backbone, classifier, aux_classifer)
    return model