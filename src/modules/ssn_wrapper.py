import logging
from collections import OrderedDict
from operator import itemgetter

import networkx as nx
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.modules.samplers.arch_sampler import ArchSampler
from src.modules.samplers.conditional_softmax_sampler import \
    CondiSoftmaxArchGenerator
from src.modules.samplers.softmax_sampler import SoftmaxArchGenerator
from src.modules.samplers.static_sampler import StaticArchGenerator
from src.modules.utils import graph_arch_details
from src.train.utils import strat_split_from_y, RoundRobinDataloader

logger = logging.getLogger(__name__)


class SSNWrapper(nn.Module):
    _SAMPLERS = dict(static=StaticArchGenerator,
                     layer_softmax=SoftmaxArchGenerator,
                     node_softmax=SoftmaxArchGenerator,
                     conditional_softmax=CondiSoftmaxArchGenerator)

    def __init__(self, ssn_model, initial_p, deter_eval, arch_loss_coef,
                 entropy_coef, split_training, t_id, all_same=False,
                 arch_sampler=None):
        super().__init__()
        self.ssn = ssn_model
        self.weight_mask = torch.zeros(len(self.ssn.stochastic_node_ids))
        for node, pos in self.ssn.stochastic_node_ids.items():
            if node[0] == t_id:
                # print('OUIOUIOUI ', node)
                self.weight_mask[pos] = 1
            # else:
            #     print('NONNONNON', node)

        # self.distrib_gen = ConstantDistribGenerator(n_vars, p=0.7)


        self.all_same = all_same
        self.initial_p = initial_p

        self.arch_cost_coef = arch_loss_coef
        self.entropy_coef = entropy_coef

        self.split_training = split_training
        self.cur_split = None
        self.frozen_arch = False
        # frozen model is a list to prevent it from appearing in the state_dict
        self.frozen_model = []
        self.t_id = t_id

        if isinstance(arch_sampler, str):
            self.arch_sampler = self.get_distrib_gen(deter_eval,
                                                     arch_sampler)
        elif isinstance(arch_sampler, ArchSampler):
            self.arch_sampler = arch_sampler
        else:
            raise ValueError('Unknown arch sampler type: {}'
                             .format(type(arch_sampler)))
        weights = self.arch_sampler().squeeze()
        # model.arch_sampler.freeze()
        nodes = list(self.ssn.stochastic_node_ids.keys())
        assert len(nodes) == weights.size(0) and weights.dim() == 1

    def get_distrib_gen(self, deter_eval, arch_sampler):
        n_vars = self.ssn.n_sampling_params
        if n_vars == 0:
            logger.warning('No sotchastic nodes are detected.')
            # logger.warning(f'Replacing {arch_sampler} with static sampler')
            # arch_sampler = 'static'
        var_names = list(self.ssn.stochastic_node_ids.keys())

        if arch_sampler == 'layer_softmax':
            groups = [name[1] for name in var_names]
        elif arch_sampler in ['node_softmax', 'conditional_softmax']:
            groups = []
            for var in var_names:
                preds = list(self.ssn.graph.predecessors(var))
                assert len(preds) == 1
                groups.append(preds[0])
        else:
            groups = None

        samp_cls = self._SAMPLERS[arch_sampler]
        return samp_cls(distrib_dim=n_vars,
                        initial_p=self.initial_p,
                        groups=groups,
                        deter_eval=deter_eval,
                        all_same=self.all_same,
                        var_names=var_names,
                        graph=self.ssn.graph)

    def forward(self, inputs, splits=None):
        if splits is None:
            assert not (self.split_training and self.training) or \
                   self.arch_sampler.is_deterministic()
            self.cur_split = None
        elif self.frozen_arch:
            self.cur_split = 0
        else:
            self.cur_split = splits.unique().item()
        if self.cur_split == 0:
            self.arch_sampler.eval()
        elif self.cur_split == 1:
            self.ssn.eval()

        if self.frozen_arch:
            return self.frozen_model[0](inputs)

        self.arch_sampler.start_new_sequence()
        arch_probas = self.arch_sampler()

        # Case of multiple input nodes where input is a list:

        b_size = inputs[0].size(0) if isinstance(inputs, list) \
            else inputs.size(0)
        arch_samplings = self.arch_sampler.sample_archs(b_size, arch_probas)
        self.ssn.samplings = arch_samplings

        # self.ssn.start_new_sequence()
        # self.ssn.set_probas()
        # print('Arch_ent: {} ({}), Train={}, split={}'.format(self.arch_sampler.distrib_entropies[0].mean(),
        #                                            self.frozen_arch,
        #                                            self.training,
        #                                            self.cur_split))
        # self.check_arch_freezing()
        return self.ssn(inputs)

    def check_arch_freezing(self, *args, ent=None, epoch=None):
        # print('called with epoch={}'.format(epoch))
        if ent is None:
            ent = self.arch_sampler.distrib_entropies[0].mean()
            # print('CALLED WITHOUT ENT')
        else:
            ent = ent
            # print('CALLED WITH ENT')
        # print('ENT={}, weights={}'.format(ent, weights))
        if ent < 0.001 or epoch > 0.5:
            self.freeze_arch()

    def get_frozen_model(self):
        if not self.frozen_arch:
            self.freeze_arch()
        return self.frozen_model[0]

    def freeze_arch(self):
        # print('FREEEEEZE')
        if self.frozen_arch:
            return
        self.frozen_arch = True
        weights = self.arch_sampler()
        arch_samplings = self.arch_sampler.sample_archs(1, weights).squeeze()

        self.frozen_model.append(self.ssn.get_pruned_model(arch_samplings))

    def requires_grad(self):
        if self.frozen_arch:
            req = self.frozen_model[0].requires_grad()
            # print('REQ:{}'.format(req))
            return req
        return True

    def loss_wrapper(self, loss_fn):
        def loss(*args, **kwargs):
            task_loss = loss_fn(*args, **kwargs)
            reward = -task_loss.detach()
            # samp = self.ssn.samplings
            # samp_size = samp.size()
            # a = self.weight_mask
            # print(samp_size)
            # print(a)

            # if self.frozen_arch:
            if self.frozen_arch or len(self.arch_sampler.log_probas) == 0:
                arch_loss = torch.zeros(1).to(task_loss.device)
                entropy_loss = torch.zeros(1).to(task_loss.device)
            else:
                arch_costs = self.ssn.samplings * self.weight_mask.to(self.ssn.samplings.device)
                arch_costs = arch_costs.sum(-1)
                reward -= self.arch_cost_coef * arch_costs
                reward -= reward.mean()
                log_p = self.arch_sampler.log_probas
                assert len(log_p) == 1
                log_p = log_p[0]
                assert task_loss.dim() == 1
                assert task_loss.size(0) == log_p.size(0) or log_p.size(0) == 1
                arch_loss = -(reward.unsqueeze(-1) * log_p).mean(1)
                assert arch_loss.dim() == 1

                entropy_loss = -self.entropy_coef * self.arch_sampler.entropy().mean()
            # ent_1 = self.arch_sampler.entropy()
            # ent_2 = [e.size() for e in self.arch_sampler.distrib_entropies]
            # ent_3 = [e.mean() for e in self.arch_sampler.distrib_entropies]
            # print(ent_1)
            # print(ent_2)
            # print(ent_3)
            # print()
            # if self.cur_split is None:
            losses = {'task all_loss': task_loss,
                      'arch all_loss': arch_loss,
                      'entropy all_loss': entropy_loss}
            return sum(losses.values()), losses
            # elif self.cur_split == 0:  # and self.t_id == 0:
            #     return task_loss
            # else:
            #     return arch_loss + entropy_loss

        return loss

    @property
    def in_node(self):
        return self.ssn.in_nodes[0]

    @property
    def out_node(self):
        return self.ssn.out_nodes[0]

    def nodes_to_prune(self, *args, **kwargs):
        return self.arch_sampler.nodes_to_prune(*args, **kwargs)

    def get_weights(self):
        return self.arch_sampler().squeeze()

    def get_stoch_nodes(self):
        return list(self.ssn.stochastic_node_ids.keys())

    def get_graph(self):
        return self.ssn.graph

    def sampled_entropy(self):
        if self.frozen_arch or len(self.arch_sampler.distrib_entropies) == 0:
            return torch.zeros(1)
        else:
            assert len(self.arch_sampler.distrib_entropies) == 1
            return self.arch_sampler.distrib_entropies[0].mean()

    def global_entropy(self):
        return self.arch_sampler.entropy().mean()

    def train_loader_wrapper(self, train_loader):
        if not self.split_training or self.arch_sampler.is_deterministic():
            return train_loader

        ds = train_loader.dataset
        splits = strat_split_from_y(ds)

        new_loaders = [DataLoader(split, train_loader.batch_size, shuffle=True,
                                  num_workers=train_loader.num_workers)
                        for split in splits]

        return RoundRobinDataloader(new_loaders)

    def param_groups(self, *args, **kwargs):
        return [
            {'params': self.ssn.parameters(*args, **kwargs)},
            {'params': self.arch_sampler.parameters(*args, **kwargs)}
        ]

    def set_h_params(self, arch_loss_coef, entropy_coef, split_training):
        self.arch_cost_coef = arch_loss_coef
        self.entropy_coef = entropy_coef
        self.split_training = split_training

    def get_top_archs(self, n=1):
        weights = self.get_weights()
        res = []
        for path in nx.all_simple_paths(self.ssn.graph, self.ssn.in_nodes[0],
                                        self.ssn.out_nodes[0]):
            p = 1
            for node in path:
                if node in self.ssn.stochastic_node_ids:
                    p *= weights[self.ssn.stochastic_node_ids[node]]
            res.append((tuple(path), p.item()))
        res = sorted(res, key=itemgetter(1), reverse=True)
        return OrderedDict(res[:n])

    def arch_repr(self):
        return graph_arch_details(self.ssn.graph)
