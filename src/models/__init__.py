# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict


from . import _utils
from .PNN_llmodel import PNN_LLmodel
from .PSSN_llmodel import MNTDP
from .change_layer_llmodel import ChangeLayerLLModel
from .ewc_llmodel import EWCLLModel
from .experience_replay_llmodel import ExperienceReplayLLModel
from .hat_llmodel import HATLLModel
from .zoo_llmodel import ZooLLModel


def get_module_by_name(name):
    if name.startswith('change-layer'):
        return ChangeLayerLLModel
    if name.startswith('ewc'):
        return EWCLLModel
    if name.startswith('er'):
        return ExperienceReplayLLModel
    if name.startswith('hat'):
        return HATLLModel
    if name == 'pnn':
        return PNN_LLmodel
    if name.startswith('pssn'):
        return MNTDP
    if name == 'zoo':
        return ZooLLModel
    if name.endswith('-dict'):
        return OrderedDict
    raise NotImplementedError(name)


def init_module(**kwargs):
    for k, v in kwargs.items():
        if isinstance(v, dict):
                v = init_module(**v)
        kwargs[k] = v
    if '_name' in kwargs:
        return get_module_by_name(kwargs.pop('_name'))(**kwargs)
    else:
        return kwargs

