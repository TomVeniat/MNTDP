"""
Abstract class representing Modular approaches. Contains all methods allowing
to interact with a model as a combination of blocks.
/!\ Only works with Vision models in the current implementation, supporting
other modalities would require splitting this class to remove all of the CV
specific stuff.
"""
import abc
from functools import partial

from torch import nn
import numpy as np

from src.models.base import get_block_model
from src.models.utils import get_conv_out_size


class ModularModel(nn.Module, abc.ABC):
    def __init__(self, n_hidden, n_convs, hidden_size, dropout_p,
                 channel_scaling, base_model=get_block_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(hidden_size, list)
        if n_hidden is not None:
            assert len(hidden_size) == 1
            hidden_size = hidden_size * n_hidden
        else:
            assert n_hidden is None
        self.hidden_size = hidden_size
        self.channel_scaling = channel_scaling

        if isinstance(dropout_p, int):
            dropout_p = [dropout_p] * len(self.hidden_size)
        self._dropout_p = dropout_p

        # self.n_hidden = n_hidden
        self.n_convs = n_convs
        self.base_model_func = partial(base_model,
                                       dropout_p=dropout_p,
                                       n_convs=self.n_convs)

    def get_k(self, layer):
        if layer >= self.n_convs:
            return None
        elif isinstance(self._k, (tuple, list)):
            return self._k[layer]
        else:
            return self._k

    def get_stride(self, layer):
        if layer >= self.n_convs:
            return None
        elif isinstance(self._stride, (tuple, list)):
            return self._stride[layer]
        else:
            return self._stride

    def get_pad(self, layer):
        if layer >= self.n_convs:
            return None
        elif isinstance(self._pad, (tuple, list)):
            return self._pad[layer]
        else:
            return self._pad

    def get_dropout_p(self, layer):
        if isinstance(self._dropout_p, (tuple, list)):
            return self._dropout_p[layer]
        else:
            return self._dropout_p

    def get_pool_k(self, layer):
        if layer >= self.n_convs:
            return None
        if isinstance(self._pool_k, (tuple, list)):
            return self._pool_k[layer]
        else:
            return self._pool_k

    def get_sizes(self, x_dim, n_classes):
        if len(x_dim) == 1:
            x_dim = x_dim[0]
            assert self.n_convs == 0, 'Can\'t use convs on 1D inputs.'
        assert len(n_classes) == 1, 'Only supports single output'
        n_classes = n_classes[0]

        # Put all dimensions together for current model.
        model_dims = [x_dim, *self.hidden_size, n_classes]
        # if self.residual:
        #     return model_dims

        for i in range(self.n_convs):
            # Compute intermediate map sizes
            img_size = model_dims[i][1:]
            k = self.get_k(i)
            pad = self.get_pad(i)
            stride = self.get_stride(i)
            out_size = get_conv_out_size(img_size, k, pad, stride)
            pool_k = self.get_pool_k(i)
            if pool_k is not None:
                out_size = get_conv_out_size(out_size, pool_k, 0, pool_k)
            model_dims[i + 1] = [model_dims[i + 1], *out_size]
        return model_dims

    def get_res_sizes(self, x_dim, n_classes):
        if len(x_dim) == 1:
            x_dim = x_dim[0]
            assert self.n_convs == 0, 'Can\'t use convs on 1D inputs.'
        assert len(n_classes) == 1, 'Only supports single output'
        n_classes = n_classes[0]

        # Put all dimensions together for current model.
        model_dims = [x_dim, *self._res_hidden_size, n_classes]
        # if self.residual:
        #     return model_dims

        for i in range(self.n_res_blocks + 1):
            # Compute intermediate map sizes
            img_size = model_dims[i][1:]
            k = self._k
            pad = self._pad
            stride = self._res_stride[i]
            out_size = get_conv_out_size(img_size, k, pad, stride)
            pool_k = self._res_pool_k[i]
            if pool_k is not None:
                out_size = get_conv_out_size(out_size, pool_k, 0, pool_k)
            model_dims[i + 1] = [model_dims[i + 1], *out_size]
        return np.array(model_dims)
