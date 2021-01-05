# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict

from src.modules.PNN_llmodel import PNN_LLmodel
from src.modules.PSSN_llmodel import ProgressiveSSN
from src.modules.SuperNet_llmodel import SuperNetLLModel
from src.modules.change_layer_llmodel import ChangeLayerLLModel
from src.modules.ewc_llmodel import EWCLLModel
from src.modules.experience_replay_llmodel import ExperienceReplayLLModel
from src.modules.fine_tune_head_llmodel import FineTuneHeadLLModel
from src.modules.fine_tune_leg_llmodel import FineTuneLegLLModel
from src.modules.multitask_head_llmodel import MultitaskHeadLLModel
from src.modules.multitask_leg_llmodel import MultitaskLegLLModel
from . import _utils
from .hat_llmodel import HATLLModel
from .zoo_llmodel import ZooLLModel


def get_module_by_name(name):
    if name.startswith('change-layer'):
        return ChangeLayerLLModel
    if name == 'multitask-head':
        return MultitaskHeadLLModel
    if name == 'multitask-leg':
        return MultitaskLegLLModel
    if name == 'finetune-head':
        return FineTuneHeadLLModel
    if name == 'finetune-leg':
        return FineTuneLegLLModel
    if name.startswith('ewc'):
        return EWCLLModel
    if name.startswith('er'):
        return ExperienceReplayLLModel
    if name.startswith('hat'):
        return HATLLModel
    if name == 'pnn':
        return PNN_LLmodel
    if name == 'ssn':
        return SuperNetLLModel
    if name.startswith('pssn'):
        return ProgressiveSSN
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

