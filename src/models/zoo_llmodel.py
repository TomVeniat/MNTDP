# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import src.models.zoo.CifarResnet as cifar_resnet
import src.models.zoo.ImageNetResnet as imgnet_resnet
from src.models.ll_model import LifelongLearningModel

logger = logging.getLogger(__name__)


MODELS = {
    'resnet-imgnet18': imgnet_resnet.resnet18,
    'resnet-imgnet34': imgnet_resnet.resnet34,
    'resnet-imgnet50': imgnet_resnet.resnet50,

    'resnet-cifar20': cifar_resnet.resnet20,
    'resnet-cifar32': cifar_resnet.resnet32,
    'resnet-cifar44': cifar_resnet.resnet44
}


class ZooLLModel(LifelongLearningModel):
    def __init__(self, model_name, pool, pool_k, padding, init,
                 model_params, *args, **kwargs):
        super(ZooLLModel, self).__init__(*args, **kwargs)
        self.model_name = model_name
        self.model = None
        if init != 'rand':
            raise NotImplementedError()
        self.init = init
        self.model_params = model_params or {}

    def _new_model(self, x_dim, n_classes, task_id, **kwargs):
        assert task_id == 0 and self.model is None

        model = MODELS[self.model_name]
        assert len(n_classes) == 1
        model = model(num_classes=n_classes[0], **self.model_params)
        model.n_out = 1
        return model

