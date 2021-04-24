# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

###
# From https://github.com/kuc2477/pytorch-ewc
# Credits to Ha Junsoo - [kuc2477](https://github.com/kuc2477)
###
import itertools
from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F, init
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models import ChangeLayerLLModel


def get_data_loader(dataset):
    return DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

def xavier_initialize(model):
    modules = [
        m for n, m in model.named_modules() if
        'conv' in n or 'linear' in n
    ]

    parameters = [
        p for
        m in modules for
        p in m.parameters() if
        p.dim() >= 2
    ]

    for p in parameters:
        init.xavier_normal(p)


class EWCLLModel(ChangeLayerLLModel):
    def __init__(self, lamda, n_fisher_estimate_samples, classic, online,
                 gamma=None, share_head=True, *args, **kwargs):
        """
        The EWC architecture is the same as the other models, so we just
        have to use the ChangeLayer basis, changing only the last layer between
        tasks without freezing the backbone parameters and with the added
        FI wwight constraint.
        """
        super(EWCLLModel, self).__init__(freeze_backbone=False, init='rand',
                                         share_layer=None, *args, **kwargs)

        self.share_layer = [1] * len(self.hidden_size)
        self.share_layer += [1 if share_head else 0]
        # self.share_layer[0] = 0
        self.lamda = lamda
        self.n_fisher_estimate_samples = n_fisher_estimate_samples

        self.online = online
        self.gamma = gamma
        # Classic means using the ground truth to evaluate the FI matrix.
        # If classic=False, the predicted label will be used instead.
        self.classic = classic

        self.model = None
        self.tasks_consolidations = []

    def get_model(self, *args, **kwargs):
        model = super().get_model(*args, **kwargs)
        model.lamda = self.lamda

        return model

    def _new_model(self, *args, **kwargs):
        mod = super()._new_model(*args, **kwargs)
        self.model = EWCWrap(mod, self.lamda, self.classic)
        self.model.n_out = mod.n_out

        for i, consol in enumerate(self.tasks_consolidations):
            for k, v in consol.items():
                k = f'T{i}_{k}'
                self.model.register_buffer(k, v)
        self.model.t_id = len(self.tasks_consolidations)
        # self.model.t_id = 0
        return self.model

    def finish_task(self, dataset, *args, **kwargs):
        self.model.eval()
        fisher = self.model.estimate_fisher(dataset,
                                              self.n_fisher_estimate_samples)
        fisher = self.model.consolidate(fisher)
        # for i, consol in enumerate(self.tasks_consolidations):
        #     for k, v in consol.items():
        #         k = f'T{i}_{k}'
        #         # if hasattr(self.model, k):
        #         delattr(self.model, k)
        if self.online and self.tasks_consolidations:
            assert len(self.tasks_consolidations) == 1
            old = self.tasks_consolidations.pop()
            new_constraints = {}
            for (k_old, v_old), (k_t, v_t) in zip(old.items(), fisher.items()):
                assert k_old == k_t
                if 'fisher' in k_old:
                    new_constraints[k_old] = v_old * self.gamma + v_t.cpu()
                else:
                    assert 'mean' in k_old
                    new_constraints[k_old] = v_t.cpu()
            fisher = new_constraints
        self.tasks_consolidations.append(fisher)
        return {}

    def set_h_params(self, lamda, gamma=None):
        self.lamda = lamda
        if gamma is not None:
            self.gamma = gamma

    def n_params(self, t_id):
        if t_id < 0:
            return 0
        all_params = set()
        for i in range(t_id+1):
            model = self.get_model(i)
            all_params.update(model.parameters())
        n_weights = sum(map(torch.numel, all_params))
        # if t_id < 0:
        #     return 0
        # model = self.get_model(0)
        # n_weights = sum(map(torch.numel, set(model.parameters())))
        # n_buffs = sum(map(torch.numel, model.buffers()))
        n_constraints = sum(map(torch.numel, itertools.chain(
            *[itm.values() for itm in self.tasks_consolidations[:t_id+1]])))

        return n_weights + n_constraints


class EWCWrap(nn.Module):
    def __init__(self, model, lamda, classic):
        # Configurations.
        super().__init__()

        self.model = model
        self.lamda = lamda
        self.classic = classic

        self.t_id = None

        self.opt_params = [torch.Tensor() for _ in self.parameters()]
        self.fisher_diags = [torch.Tensor() for _ in self.parameters()]

    def forward(self, x):
        # print(x.size())
        return self.model(x)
        # x = x.view(x.size(0), -1)
        # return reduce(lambda x, l: l(x), self.layers, x)

    def estimate_fisher(self, dataset, sample_size):
        # sample loglikelihoods from the dataset
        device = self.device()

        data_loader = get_data_loader(dataset)
        fi_estimate = defaultdict(int)
        for i, (x, y) in enumerate(tqdm(data_loader, disable=True), start=1):
            x = x.to(device)
            y = y.to(device).squeeze()

            logits = self(x)
            y_hat = logits.argmax()
            if not self.classic:
                y = y_hat
            ll = F.log_softmax(logits, dim=1)[:, y]

            self.zero_grad()
            ll.backward()
            for n, p in self.model.shared_named_parameters():
                fi_estimate[n] += p.grad ** 2

            if i == sample_size:
                break
        fi_estimate = {n.replace('.', '__'):  (fi/i).detach().cpu()
                       for n, fi in fi_estimate.items()}

        return fi_estimate

    def consolidate(self, fisher):
        props = {}
        for i, (n, p) in enumerate(self.model.shared_named_parameters()):
            n = n.replace('.', '__')
            props['{}_mean'.format(n)] = p.detach().clone()
            props['{}_fisher'.format(n)] = fisher[n].detach().clone()
        return props

    def loss_wrapper(self, loss_fn):
        def loss(*args, **kwargs):
            task_loss = loss_fn(*args, **kwargs)
            # new_loss = lambda y_hat, y: F.cross_entropy(y_hat, y.squeeze()) + self.model.ewc_loss()
            ewc_loss = self.ewc_loss()
            # if self.training:
            #     print(ewc_loss)
            return task_loss + ewc_loss

        return loss

    def ewc_loss(self):
        if self.lamda == 0:
            return 0
        losses = []
        for i, (n, p) in enumerate(self.model.shared_named_parameters()):
            n = n.replace('.', '__')
            for t in range(self.t_id):
                # retrieve the consolidated mean and fisher information.
                mean = getattr(self, f'T{t}_{n}_mean')
                # if i == 0:
                #     print(((p-mean) ** 2).mean())
                    # print(mean)
                fisher = getattr(self, f'T{t}_{n}_fisher')
                # calculate a ewc loss. (assumes the parameter's prior as
                # gaussian distribution with the estimated mean and the
                # estimated cramer-rao lower bound variance, which is
                # equivalent to the inverse of fisher information)
                f_loss = (fisher * (p-mean) ** 2)
                # if i == 0:
                # print(f_loss.mean())
                # f_loss_2 = self.fisher_diags[i][-1] * (p - self.opt_params[i][-1])**2
                # eq = torch.equal(f_loss, f_loss_2)
                # assert eq
                losses.append(f_loss.sum())
        return (self.lamda/2)*sum(losses)

    def device(self):
        return next(self.parameters()).device

    def arch_repr(self):
        return self.model.arch_repr()
