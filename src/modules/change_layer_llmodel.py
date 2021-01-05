# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

from torch import nn

from src.modules.ll_model import LifelongLearningModel
from src.modules.resnet import _make_layer, BasicBlock
from src.modules.utils import Flatten, count_params

logger = logging.getLogger(__name__)


def _conv_block(in_dim, out_dim, k, stride, pad, dropout_p, pool,
                pool_k, is_last):
    assert len(in_dim) == 3
    layers = [nn.Conv2d(in_dim[0], out_dim[0], k, stride, pad)]

    if not is_last:
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_p))

    if pool is not None:
        assert pool == 'maxpool', 'Only Maxpool supported for now'
        layers.append(nn.MaxPool2d(pool_k))
    return layers


def _lin_block(in_dim, out_dim, dropout_p, is_last):
    layers = []
    if not isinstance(in_dim, int):
        in_dim = in_dim[0] * in_dim[1] * in_dim[2]
        layers.append(Flatten())

    layers.append(nn.Linear(in_dim, out_dim))

    if not is_last:
        layers += [nn.ReLU(),
                   nn.Dropout(dropout_p)]

    return layers


def get_block(in_dim, out_dim, dropout_p, k, stride, pad, pool, pool_k,
              norm_layer, is_first, is_last, residual, block_depth):
    if residual:
        if isinstance(in_dim, (list, tuple)):
            assert len(in_dim) == 3
        layers = _make_layer(BasicBlock, in_dim, out_dim, block_depth, stride,
                       pool, pool_k, norm_layer=norm_layer, is_first=is_first,
                             is_last=is_last)
    elif isinstance(out_dim, int) or len(out_dim) == 1:
        # Linear layer
        layers = _lin_block(in_dim, out_dim, dropout_p, is_last)
    elif len(out_dim) == 3:
        layers = _conv_block(in_dim, out_dim, k, stride, pad, dropout_p,
                             pool, pool_k, is_last)
    else:
        raise ValueError('Don\'t know which kind of layer to use for input '
                         'size {} and output size {}.'.format(in_dim, out_dim))

    return nn.Sequential(*layers)


class ChangeLayerLLModel(LifelongLearningModel):
    def __init__(self, share_layer, k, pool, pool_k, padding, stride, residual,
                 norm_layer, block_depth, init, freeze_backbone,
                 *args, **kwargs):
        super(ChangeLayerLLModel, self).__init__(*args, **kwargs)
        self.common_model = None
        self.share_layer = share_layer
        self.common_layers = None
        self.task_specific_layers = None
        assert init in ['rand', 'prev']
        self.init = init
        self.freeze_backbone = freeze_backbone
        self.norm_layer = norm_layer

        self.pool = pool
        self.residual = residual
        self.block_depth = block_depth
        self.n_new_layers = 0

        if residual:
            assert k == 3, 'Resnet requires a kernel of size 3'
            self._k = k
            assert len(stride) == self.n_convs
            self._stride = stride
            self._pad = padding
            self._pool_k = pool_k
            assert len(self.hidden_size) == 1, 'Resnet requires only 1 ' \
                                                'hidden size'
            # self.hidden_size = [self.hidden_size[0]]
            n_full_layers = self.n_convs - self.n_new_layers
            for i in range(1, n_full_layers):
                last_size = self.hidden_size[-1]
                if self.channel_scaling:
                    new_size = last_size * self.get_stride(i)
                else:
                    new_size = last_size
                self.hidden_size.append(new_size)
            self.hidden_size.extend([self.hidden_size[-1]]*self.n_new_layers)
            assert not self.dropout_p
            self.dropout_p = [None] * self.n_convs
            self._pool_k = [None] * self.n_convs
            self._pool_k[-1] = pool_k
            # self.hidden_size = [first] + [first * 2**i for i in range(4)]
            # self.n_convs = 5

        else:
            if isinstance(stride, int):
                stride = [stride] * self.n_convs
            self.stride = stride
            if isinstance(k, int):
                k = [k] * self.n_convs
            self.k = k

            if isinstance(pool_k, int):
                pool_k = [pool_k] * self.n_convs
            self.pool_k = pool_k

            if isinstance(padding, int):
                padding = [padding] * self.n_convs
            self.padding = padding

        self.frozens = []

    def _new_model(self, x_dim, n_classes, task_id, **kwargs):
        model_dims = self.get_sizes(x_dim, n_classes)
        if self.common_layers is None:
            # Need to init the shared layers
            self._init_common_layers(model_dims)

        model = FrozenSequential()
        for i, (in_dim, out_dim) in enumerate(zip(model_dims, model_dims[1:])):
            layer = self._get_layer(i, in_dim, out_dim, task_id)
            model.add_module(str(i), layer)
            if task_id > 0 and self.share_layer[i]:
                model.frozen_modules_idx.append(i)
            if self.share_layer[i]:
                model.shared_modules_idx.append(i)
        model.n_out = 1
        return model

    def _init_common_layers(self, model_dims):
        self.common_layers = nn.ModuleDict()
        self.task_specific_layers = nn.ModuleDict()
        for i, (in_dim, out_dim) in enumerate(zip(model_dims, model_dims[1:])):
            if self.share_layer[i]:
                self.common_layers[str(i)] = self._get_block(i, in_dim, out_dim)
            else:
                self.task_specific_layers[str(i)] = nn.ModuleList()

    def _get_layer(self, i, in_dim, out_dim, task_id):
        if self.share_layer[i]:
            layer = self.common_layers[str(i)]
        elif len(self.task_specific_layers[str(i)]) > task_id:
            layer = self.task_specific_layers[str(i)][task_id]
        else:
            layer = self._get_block(i, in_dim, out_dim)
            self.task_specific_layers[str(i)].append(layer)
            if self.init == 'prev':
                return NotImplementedError()
        return layer

    def _get_block(self, depth, in_dim, out_dim):
        is_first = depth == 0
        is_last = depth == len(self.share_layer) - 1
        drop_p = self.dropout_p[depth] if depth < len(self.dropout_p) else None
        if self.residual:
            stride = self.get_stride(depth)
            k = self.get_k(depth)
            pad = self.get_pad(depth)
            pool_k = self.get_pool_k(depth)
        elif depth < self.n_convs:
            stride = self.stride[depth]
            k = self.k[depth]
            pad = self.padding[depth]
            pool_k = self.pool_k[depth] if self.pool else None
        else:
            stride = k = pad = pool_k = None

        return get_block(in_dim, out_dim, drop_p, k, stride, pad,
                          self.pool, pool_k, self.norm_layer, is_first,
                         is_last, self.residual, self.block_depth)

    # def finish_task(self, dataset, task_id, viz=None, path=None):
    def finish_task(self, ds, task_id, *args, **kwargs):
        if task_id in self.frozens:
            raise RuntimeError('Task {} has already been finished'
                               .format(task_id))
        if task_id == 0 and self.freeze_backbone:
            self.common_layers.requires_grad_(False)
            # for p in self.common_layers.parameters():
            #     p.require
        self.frozens.append(task_id)
        return {}


class FrozenSequential(nn.Sequential):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frozen_modules_idx = []
        self.shared_modules_idx = []

    def requires_grad(self):
        return any(map(lambda t: t.requires_grad, self.parameters()))

    def forward(self, *args, **kwargs):
        for i in self.frozen_modules_idx:
            self[i].eval()
        return super().forward(*args, **kwargs)

    def feats_forward(self, input):
        for i in self.frozen_modules_idx:
            self[i].eval()

        feats = self[:-1](input)
        return feats, self[-1](feats)

    def shared_named_parameters(self, *args, **kwargs):
        # named_params = [p[0] for p in self.named_parameters(*args, **kwargs)]
        filtered_modules = [(i, mod) for i, mod in enumerate(self) if i in self.shared_modules_idx]
        # a = self.frozen_modules_idx
        # b = self.shared_modules_idx
        fil_p = [(f'{i}.'+p[0], p[1]) for i, mod in filtered_modules for p in mod.named_parameters()]
        return fil_p

    def arch_repr(self):
        detail = {}
        for block in self:
            detail[len(detail)] = count_params(block)['trainable']
        return detail
