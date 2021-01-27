import numpy as np
import torch
import torch.nn.init as weight_init
from torch import nn
from torch.nn import Parameter

from src.models.samplers.arch_sampler import ArchSampler


class StaticArchGenerator(ArchSampler):
    def __init__(self, initial_p, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params = Parameter(torch.Tensor(1, self.distrib_dim))
        logit = np.log(
            initial_p / (1 - initial_p)) if initial_p < 1 else float('inf')
        weight_init.constant_(self.params, logit)

    def forward(self, z=None):
        if self.frozen and self.training:
            raise RuntimeError('Trying to sample from a frozen distrib gen in '
                               'training mode')
        return self.params.sigmoid()

    def entropy(self):
        distrib = torch.distributions.Bernoulli(self.params.sigmoid())
        return distrib.entropy()

    def remove_var(self, name):
        assert self.var_names
        self.distrib_dim -= 1
        remove_idx = self.var_names.index(name)
        self.var_names.remove(name)
        all_idx = torch.ones_like(self.params).bool()
        all_idx[0, remove_idx] = 0
        self.params = nn.Parameter(self.params[all_idx].unsqueeze(0))

    def is_deterministic(self):
        distrib = self()
        return torch.equal(distrib, distrib**2)

