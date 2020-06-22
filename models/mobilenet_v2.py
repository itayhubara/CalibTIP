import torch
import torch.nn as nn
from torch.nn.modules.utils import _single, _pair, _triple
import math
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import torchvision.transforms as transforms
from models.resnet import depBatchNorm2d
from .modules.quantize import QConv2d, QLinear, RangeBN
__all__ = ['mobilenet_v2']


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1,
                 batch_norm=True,num_bits=8, num_bits_weight=8, perC=True, measure=False, cal_qparams=False):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            QConv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False,
                      num_bits=num_bits, num_bits_weight=num_bits_weight, perC=perC, measure=measure, cal_qparams=cal_qparams),
            depBatchNorm2d(batch_norm, out_planes),
            nn.ReLU6(inplace=False)
        )


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, batch_norm=True,num_bits=8, num_bits_weight=8,
                 perC=True, measure=False, cal_qparams=False):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, batch_norm=batch_norm, num_bits=num_bits,
                                 num_bits_weight=num_bits_weight, perC=perC, measure=measure, cal_qparams=cal_qparams))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, batch_norm=batch_norm, num_bits=num_bits,
                                 num_bits_weight=num_bits_weight, perC=perC, measure=measure, cal_qparams=cal_qparams),
            # pw-linear
            QConv2d(hidden_dim, oup, 1, 1, 0, bias=False, num_bits=num_bits,
                                 num_bits_weight=num_bits_weight, perC=perC, measure=measure, cal_qparams=cal_qparams),
            depBatchNorm2d(batch_norm, oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None, batch_norm=True,num_bits=8, num_bits_weight=8, perC=True, measure=False, cal_qparams=False):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
        """
        super(MobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2, batch_norm=batch_norm, num_bits=num_bits,
                                 num_bits_weight=num_bits_weight, perC=perC, measure=measure, cal_qparams=cal_qparams)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, batch_norm=batch_norm, num_bits=num_bits,
                                 num_bits_weight=num_bits_weight, perC=perC, measure=measure, cal_qparams=cal_qparams))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, batch_norm=batch_norm, num_bits=num_bits,
                                 num_bits_weight=num_bits_weight, perC=perC, measure=measure, cal_qparams=cal_qparams))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            QLinear(self.last_channel, num_classes, num_bits=num_bits, num_bits_weight=num_bits_weight,
                                  perC=perC, measure=measure, cal_qparams=cal_qparams),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


def mobilenet_v2(**config):
    r"""MobileNet v2 model architecture from the `"MobileNetV2: Inverted Residuals and Linear Bottlenecks"
    <https://arxiv.org/abs/1801.04381>`_ paper.
    """
    dataset = config.pop('dataset', 'imagenet')
    assert dataset == 'imagenet'
    return MobileNetV2(**config)
