import logging

from src.modules.ll_model import LifelongLearningModel
from src.modules.ssn_wrapper import SSNWrapper
from supernets.implementation.ThreeDimNeuralFabric import ThreeDimNeuralFabric

logger = logging.getLogger(__name__)


class SuperNetLLModel(LifelongLearningModel):
    def __init__(self, model_params, share, freeze, *args, **kwargs):
        super(SuperNetLLModel, self).__init__(*args, **kwargs)
        self.model_params = model_params
        self.share = share
        self.freeze = freeze
        assert not self.freeze

    def _new_model(self, x_dim, n_classes, **kwargs):
        assert len(n_classes) == 1
        n_classes = n_classes[0]
        if not self.models or not self.share:
            model = ThreeDimNeuralFabric(input_dim=x_dim, n_classes=n_classes,
                                         **self.model_params)
            cnf = SSNWrapper(model)
            cnf.n_out = 1
        else:
            cnf = self.models[-1]
        return cnf


