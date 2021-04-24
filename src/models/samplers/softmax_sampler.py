from collections import defaultdict

import torch
import torch.nn.functional as f
from torch import nn
from torch.distributions import Categorical

from src.models.samplers.arch_sampler import ArchSampler


class SoftmaxArchGenerator(ArchSampler):
    def __init__(self, groups, graph, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.groups = groups
        id_gen = defaultdict()
        id_gen.default_factory = id_gen.__len__
        group_ids = torch.tensor([id_gen[grp] for grp in groups])
        self.group_ids_index = dict(id_gen.items())  # Freeze the id generator

        self.distrib_groups = group_ids
        self.graph = graph

        self.group_ids = self.distrib_groups.unique()
        self.group_masks = [self.distrib_groups.eq(id)
                            for id in self.group_ids]
        self.group_var_names = [[] for _ in range(self.group_ids.numel())]
        for grp, var in zip(self.distrib_groups.unbind(), self.var_names):
            self.group_var_names[grp.item()].append(var)
        self.group_sizes = [mask.sum().item() for mask in self.group_masks]

        self.params = [nn.Parameter(torch.zeros(size))
                       for size in self.group_sizes]

        self.params = nn.ParameterList(self.params)

    def forward(self, z=None):
        return self.get_distrib()

    def entropy(self):
        entropies = [Categorical(p.softmax(0)).entropy() for p in self.params]
        # if self.training:
        #     print('Training mode')

        return torch.stack(entropies).mean() if entropies else torch.Tensor()

    def remove_var(self, name):
        assert self.var_names
        self.distrib_dim -= 1
        remove_idx = self.var_names.index(name)
        grp_idx = self.distrib_groups[remove_idx]
        idx_in_group = self.group_masks[grp_idx][:remove_idx].sum().item()
        self.var_names.remove(name)

        self.group_var_names[grp_idx].remove(name)

        all_idx_mask = torch.ones_like(self.distrib_groups).bool()
        all_idx_mask[remove_idx] = 0
        self.distrib_groups = self.distrib_groups[all_idx_mask]
        self.group_masks = [m[all_idx_mask] for m in self.group_masks]

        param = self.params[grp_idx]
        group_idx_mask = torch.ones_like(param).bool()
        group_idx_mask[idx_in_group] = 0
        self.params[grp_idx] = nn.Parameter(param[group_idx_mask])

        self.group_sizes[grp_idx] -= 1

    def get_distrib(self):
        device = self.params[0].device if self.params else 'cpu'
        distrib = torch.zeros(self.distrib_dim).to(device)
        for p, mask in zip(self.params, self.group_masks):
            mask = mask.to(device)
            distrib[mask] = p.softmax(0)

        return distrib.unsqueeze(0)

    def is_deterministic(self):
        distrib = self.get_distrib()
        return torch.equal(distrib, distrib**2)

    def sample_archs(self, batch_size, probas, force_deterministic=False):
        """
        Hook called by pytorch before each forward
        :param _: Current module
        :param input: Input given to the module's forward
        :return:
        """
        deterministic = not self.training and self.deter_eval \
                        or force_deterministic
        self._check_probas(probas, self.all_same)

        # Check the compatibility with the batch_size
        if probas.size(0) != batch_size:
            if probas.size(0) != 1:
                raise ValueError('Sampling probabilities dimensions {} '
                                 'doesn\'t match with batch size {}.'
                                 .format(probas.size(), batch_size))
            # if not self.all_same:
            #     probas = probas.expand(batch_size, -1)

        samplings = torch.zeros_like(probas)
        entropy = torch.zeros(0).to(samplings.device)
        log_probs = torch.zeros(0).to(samplings.device)
        if not self.all_same:
            samplings = samplings.repeat(batch_size, 1)
            entropy = entropy.repeat(batch_size, 1)
            log_probs = log_probs.repeat(batch_size, 1)

        for mask in self.group_masks:
            if not any(mask):
                continue
            masked_probas = probas[mask.unsqueeze(0)].unsqueeze(0)
            if not self.all_same:
                masked_probas = masked_probas.repeat(batch_size, 1)
            distrib = Categorical(masked_probas)

            if deterministic:
                index = masked_probas.argmax(1)
            else:
                index = distrib.sample()

            new_sampling = f.one_hot(index, masked_probas.size(-1)).float()

            expanded_mask = mask.expand_as(samplings)
            samplings[expanded_mask] = new_sampling.flatten()

            entropy = torch.cat((entropy, distrib.entropy().unsqueeze(1)), 1)
            log_probs = torch.cat((log_probs, distrib.log_prob(index)
                                   .unsqueeze(1)), 1)

        if self.all_same:
            samplings = samplings.expand(batch_size, -1)

        # self._seq_probas.append(probas)
        self.distrib_entropies.append(entropy)
        self.log_probas.append(log_probs)
        return samplings

    def nodes_to_prune(self, *args, **kwargs):
        distrib = self()
        arch = self.sample_archs(1, distrib, True)
        nodes = self.var_names.copy()

        for var, sampled in zip(self.var_names, arch.unbind(-1)):
            if sampled[0]:
                nodes.remove(var)
            # keep_idx = params.argmax()
            # group_vars = group_vars.copy()
            # group_vars.pop(keep_idx)
            # nodes.extend(group_vars)
        return nodes
