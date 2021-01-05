import torch

from src.modules.samplers.arch_sampler import ArchSampler


class ConstantArchGenerator(ArchSampler):
    def __init__(self, initial_p, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_p = initial_p

    def forward(self, z=None):
        if self.frozen:
            raise RuntimeError('Trying to sample from a frozen distrib gen')
        arch_proba = torch.ones(1, self.distrib_dim) * self.initial_p
        return arch_proba
