# torchvision-based DCNv2 compatibility shim (MambaCenter)
# 替换原 CharlesShang/DCNv2 CUDA 扩展方案: 新版 PyTorch 无法编译旧扩展。
# torchvision.ops.deform_conv2d (>=0.8) 提供等价的 modulated deformable conv,
# 语义与原 dcn_v2_conv 完全一致: offset (B, 2*dg*kh*kw, H, W), mask (B, dg*kh*kw, H, W)。
# 对外接口保持原样: DCN(in_channels, out_channels, kernel_size, stride, padding, dilation, deformable_groups)
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math

import torch
from torch import nn
from torch.nn.modules.utils import _pair
from torchvision.ops import deform_conv2d


class DCNv2(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, deformable_groups=1):
        super(DCNv2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels // groups, *self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()

    def forward(self, input, offset, mask):
        return deform_conv2d(
            input, offset, self.weight, self.bias,
            stride=self.stride, padding=self.padding, dilation=self.dilation,
            mask=mask)


class DCN(DCNv2):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1, padding=0,
                 dilation=1, deformable_groups=1):
        super(DCN, self).__init__(in_channels, out_channels,
                                  kernel_size, stride, padding,
                                  dilation, deformable_groups)

        self.conv_offset_mask = nn.Conv2d(self.in_channels,
                                          self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
                                          kernel_size=self.kernel_size,
                                          stride=(self.stride[0], self.stride[1]),
                                          padding=(self.padding[0], self.padding[1]),
                                          bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input):
        out = self.conv_offset_mask(input)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return deform_conv2d(
            input, offset, self.weight, self.bias,
            stride=self.stride, padding=self.padding, dilation=self.dilation,
            mask=mask)
