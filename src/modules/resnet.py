# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torch import nn
from torchvision.models.resnet import conv3x3, conv1x1

from src.modules.utils import Flatten


class Contiguousize(nn.Module):
    def forward(self, x):
        return x.contiguous()


class BasicBlock(nn.Module):
    def __init__(self, in_dim, out_dim, stride=1, downsample=None,
                 norm_layer=None, end_act=True):
        super(BasicBlock, self).__init__()
        assert isinstance(in_dim, (tuple, list))
        in_planes = in_dim[0]
        assert isinstance(out_dim, (tuple, list))
        out_planes = out_dim[0]
        # Both self.conv1 and self.downsample layers downsample
        # the input when stride != 1
        self.conv1 = conv3x3(in_planes, out_planes, stride)
        self.norm1 = get_norm_layer(norm_layer, out_dim)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(out_planes, out_planes)
        self.norm2 = get_norm_layer(norm_layer, out_dim)
        self.downsample = downsample
        self.stride = stride
        self.end_act = end_act

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        if self.end_act:
            out = self.relu(out)

        return out
    def extra_repr(self):
        return 'end_act={}'.format(self.end_act)


def get_norm_layer(norm_layer, in_size):
    assert len(in_size) == 3, 'Normalization should be applied on 3D ' \
                              'feature maps, got shape {}'.format(in_size)
    if norm_layer == 'batch':
        return nn.BatchNorm2d(num_features=in_size[0], affine=True)
    elif norm_layer == 'layer':
        return nn.LayerNorm(normalized_shape=in_size[1:],
                            elementwise_affine=True)
    elif norm_layer == 'instance':
        return nn.InstanceNorm2d(num_features=in_size[0], affine=True)
    else:
        raise ValueError('Unknown normalzation: {}'.format(norm_layer))
    

def _make_layer(block, in_dim, out_dim, blocks, stride=1, pool=None,
                pool_size=None, norm_layer=None, is_first=False,
                is_last=False, end_act=True, n_blocks=None):

    if n_blocks is None:
        n_blocks = 1 if is_first or is_last else blocks
    if stride is None:
        stride = []
    elif isinstance(stride, int):
        stride = [stride] + [1]*(n_blocks-1)
    if isinstance(in_dim[0], int):
        in_dim = [in_dim] + [out_dim] * (n_blocks-1)
    if isinstance(out_dim, int):
        out_dim = [out_dim]
    elif isinstance(out_dim[0], int):
        out_dim = [out_dim] * n_blocks
    if pool_size is None or isinstance(pool_size, int):
        pool_size = [None] * (n_blocks-1) + [pool_size]
    assert len(stride) == n_blocks or is_last and len(stride) == n_blocks - 1

    layers = []

    if is_first:
        in_size, in_dim = in_dim[0], in_dim[1:]
        in_planes = in_size[0]
        out_size, out_dim = out_dim[0], out_dim[1:]
        out_planes = out_size[0]
        s, stride = stride[0], stride[1:]
        p, pool_size = pool_size[0], pool_size[1:]
        layers.extend([conv3x3(in_planes, out_planes),
                get_norm_layer(norm_layer, out_size)])
        if end_act:
            layers.append(nn.ReLU())
    assert len(stride) <= len(in_dim) and len(stride) <= len(out_dim)
    for l in range(len(stride)):
        in_s, in_dim = in_dim[0], in_dim[1:]
        out_s, out_dim = out_dim[0], out_dim[1:]
        s, stride = stride[0], stride[1:]
        downsample = None
        if s != 1 or in_s[0] != out_s[0]:
            downsample = nn.Sequential(
                Contiguousize(),
                conv1x1(in_s[0], out_s[0], s),
                get_norm_layer(norm_layer, out_s),
            )

        layers.append(block(in_s, out_s, s, downsample, norm_layer,
                            end_act or l < blocks - 1))

        p, pool_size = pool_size[0], pool_size[1:]
        if p is not None:
            layers.append(nn.ReLU())
            layers.append(get_pool_l(pool, p))
            # layers.append(nn.ReLU())

    assert not stride
    assert not any(pool_size)
    # if stride:
    #     in_dim = in_dim[d:]
    #     out_dim = out_dim[d:]

        # part.remove(0)
    # in_planes = out_planes * block.expansion
    # for d in range(1, blocks):
    #     if d in part:
        # layers.append(block(out_dim, out_dim, norm_layer=norm_layer,
        #                      end_act=end_act or d < blocks - 1))

    # if any(pool_size):
    #
    #     layers.append(nn.ReLU())
    #     layers.append(pool_l(1))

    if is_last:
        in_size, in_dim = in_dim[0], in_dim[1:]
        in_planes = in_size[0]
        out_size, out_dim = out_dim[0], out_dim[1:]
        if not isinstance(out_size, int):
            assert len(out_size) == 1
            out_size = out_size[0]
        assert len(in_size) == 3
        layers.extend([
            Flatten(),
            nn.Linear(in_planes, out_size)])

    return layers


def get_pool_l(p_type, p_size):
    # assert not any(pool_size[:-1])
    if p_type == 'avgpool':
        # pool_l = nn.AvgPool2d
        pool_l = nn.AdaptiveAvgPool2d
    elif p_type == 'maxpool':
        pool_l = nn.AdaptiveMaxPool2d
    else:
        raise NotImplementedError()
    return pool_l(1)
