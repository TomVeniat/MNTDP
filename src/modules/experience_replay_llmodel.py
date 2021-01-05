# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import operator
from functools import reduce

import numpy as np
import torch
from torch import nn

from src.modules.change_layer_llmodel import ChangeLayerLLModel

logger = logging.getLogger(__name__)


class ExperienceReplayLLModel(ChangeLayerLLModel):
    def __init__(self, mem_size_per_class, share_head, mode, *args, **kwargs):
        super().__init__(freeze_backbone=False, init='rand', share_layer=None,
                         *args, **kwargs)
        self.share_layer = [1] * len(self.hidden_size)
        self.share_layer += [1 if share_head else 0]
        self.share_head = share_head
        self.mem_size_per_class = mem_size_per_class
        self.mem = None
        self.model = None
        self.mode = mode
        self.mem_size_hisory = []

        # self.pool = pool
        # base_model = nn.Sequential(*base_model(self.input_dim, self.n_hidden, self.hidden_size))
        # self.model = ERLearner(base_model, self.mem)

    def _new_model(self, x_dim, n_classes, task_id, *args, **kwargs):
        mod = super()._new_model(x_dim, n_classes, task_id, *args, **kwargs)
        if self.mem is None:
            self.mem = Memory(self.mem_size_per_class, x_dim,
                              mode=self.mode)

        self.mem.add_classes(n_classes[0])
        # logger.error(len(self.mem))
        if self.share_head:
            heads = []
        else:
            assert len(self.task_specific_layers) == 1
            heads = next(iter(self.task_specific_layers.values()))[:-1]
        model = ERLearner(mod, heads, self.mem)
        model.n_out = 1

        if len(n_classes) != 1:
            raise NotImplementedError('ER not yet implemented for attributes')

        return model

    def finish_task(self, dataset, task_id, *args, **kwargs):
        ds_len = dataset.tensors[0].size(0)
        n_new_seen_samples = self.mem.n_examples_seen - self.mem.unique_seen_samples

        # Handle the case where we haven't seen the whole dataset
        self.mem.unique_seen_samples += min(ds_len, n_new_seen_samples)
        self.mem.n_examples_seen.copy_(self.mem.unique_seen_samples)
        self.mem_size_hisory.append(len(self.mem))
        return {}

    def n_params(self, t_id):
        if t_id < 0:
            return 0
        all_params = set()
        for i in range(t_id+1):
            model = self.get_model(i)
            all_params.update(model.parameters())

        n_weights = sum(map(torch.numel, all_params))

        mem_size = self.mem_size_hisory[t_id]
        if mem_size > 0:
            mem = model.memory
            x_size = reduce(operator.mul, mem.mem[0][0].size())
            y_size = reduce(operator.mul, mem.mem[1][0].size())
            mem_size = (x_size + y_size + 1) * mem_size

        return n_weights + mem_size


class ERLearner(nn.Module):
    def __init__(self, base_model, heads, memory):
        super(ERLearner, self).__init__()
        # assert base_model[-1] == heads[-1]
        self.model = base_model
        self.heads = heads
        self.memory = memory
        self.register_buffer('mem_x', self.memory.mem[0])
        self.register_buffer('mem_y', self.memory.mem[1])
        self.register_buffer('mem_t_id', self.memory.mem[2])
        self.register_buffer('n_examples_seen', self.memory.n_examples_seen)

        self.register_buffer('mem_class_counter', self.memory.class_counter)
        self.register_buffer('mem_filled_cells', self.memory.filled_cell)

    def forward(self, xs):
        """
        :param xs: Tuple containing the input features and the task id for each example.
        The format is (Bxn_features, B) format where le first element is the input data and the second contain the id
        of the task corresponding with each feature vector. The task id will be used to select the head corresponding to
        the task.
        :return:

        """
        if not isinstance(xs, (tuple, list)):
            return self.model(xs)

        xs, t_idx = xs
        if not self.heads:
            return self.model(xs)

        features, meh = self.model.feats_forward(xs)

        tasks_in_batch = sorted(t_idx.unique().tolist())

        y_hat = None
        for t_id in tasks_in_batch:
            mask = t_idx == t_id
            if t_id == len(self.heads):
                y = meh[mask]
            else:
                assert t_id < len(self.heads), t_id
                y = self.heads[t_id](features[mask])
            if y_hat is None:
                y_hat = torch.empty((xs.size(0), *y.size()[1:]),
                                    device=y.device)
            y_hat[mask] = y


        # n_classes = [self.n_classes[idx] for idx in tasks_in_batch]
        # sizes = torch.tensor(n_classes).sum(0)
        # y_hat = [torch.ones(x.size(0), size, device=features.device) * -float('inf') for size in sizes]
        #
        # #contains the offset for each attribute, this offset will be incremented after each task
        # task_cols = [0] * len(n_classes[0])
        # for i, t_id in enumerate(tasks_in_batch):
        #     mask = t_idx == t_id
        #     res = self.heads[t_id](features[mask])
        #     for j, (attr_y_hat, attr_res) in enumerate(zip(y_hat, res)):
        #         attr_y_hat[mask, task_cols[j]:task_cols[j]+attr_res.size(1)] = attr_res
        #         task_cols[j] += attr_res.size(1)
        #         assert n_classes[i][j] == attr_res.size(1)

        return y_hat

    def add_head(self, new_head, n_classes):
        self.heads.append(new_head)
        self.n_classes.append(n_classes)

    def prepare_batch_wrapper(self, func, task_id):
        def prepare_batch(batch, *args, **kwargs):
            x, y = batch
            if self.training:
                batch = self.memory.extend_batch(x, y, task_id)
                self.memory.update_memory(x, y, task_id)
            else:
                batch = (x, torch.ones(x.size(0)).long() * task_id), y
            return func(batch, *args, **kwargs)
        return prepare_batch

    def load_state_dict(self, state_dict, strict=True):
        self.mem_x.resize_as_(state_dict['mem_x'])
        self.mem_y.resize_as_(state_dict['mem_y'])
        self.mem_t_id.resize_as_(state_dict['mem_t_id'])
        self.mem_class_counter.resize_as_(state_dict['mem_class_counter'])
        self.mem_filled_cells.resize_as_(state_dict['mem_filled_cells'])
        super().load_state_dict(state_dict, strict)

    def arch_repr(self):
        return self.model.arch_repr()


class Memory(object):
    def __init__(self, mem_size_per_class, input_dim, total_n_classes=0,
                 mode='reservoir'):
        super(Memory, self).__init__()
        self.mem_size_per_class = mem_size_per_class
        self.total_n_classes = total_n_classes

        self.input_dim = input_dim
        self.output_dim = 1
        self.mem_size = self.mem_size_per_class * self.total_n_classes
        self.mem = [torch.empty(0, *self.input_dim),
                    torch.empty(0, self.output_dim).long(),
                    torch.empty(0).long()]

        self.n_examples_seen = torch.zeros(1)
        self.unique_seen_samples = torch.zeros(1)
        self.mode = mode

        # Only used by the ring buffer
        self.class_counter = torch.zeros(total_n_classes).long()
        self.classes_per_task = []
        self.filled_cell = torch.zeros(total_n_classes).bool()

    @property
    def n_items(self):
        if self.filled_cell.numel() == 0:
            return self.mem[0].size(0)
        else:
            return self.filled_cell.sum().item()

    def add_classes(self, n_classes):
        self.total_n_classes += n_classes
        self.mem_size = self.mem_size_per_class * self.total_n_classes
        if self.mode == 'ring':
            assert self.class_counter.size(0) + n_classes == self.total_n_classes, self.class_counter.size()
            # self.class_counter.resize_((self.total_n_classes,
            #                             *self.class_counter.size()[1:]))
            self.class_counter = torch.cat([self.class_counter,
                                           torch.zeros(n_classes).long()])
            self.class_counter[-n_classes:] = 0
            self.classes_per_task.append(n_classes)

            # self.mem[0].resize_(self.mem_size, *self.mem[0].size()[1:])
            # self.mem[1].resize_(self.mem_size, *self.mem[1].size()[1:])
            # self.mem[2].resize_(self.mem_size)
            new_items = n_classes * self.mem_size_per_class
            for i in range(3):
                pad = torch.zeros(new_items, *self.mem[i].size()[1:]).type_as(self.mem[i])
                self.mem[i] = torch.cat([self.mem[i], pad])

            # self.filled_cell.resize_(self.mem_size)
            self.filled_cell = torch.cat([self.filled_cell,
                                          torch.zeros(new_items).bool()])
            self.filled_cell[-self.mem_size_per_class*n_classes:] = False

    def extend_batch(self, x, y, task_id):
        """
        return a bunch of (xs, ys, tid)
        :param x:
        :param y:
        :param task_id:
        :return:
        """
        batch_size = x.size(0)

        x_candidates = self.mem[0]
        y_candidates = self.mem[1]
        t_candidates = self.mem[2]

        if self.filled_cell.numel() > 0:
            x_candidates = x_candidates[self.filled_cell]
            y_candidates = y_candidates[self.filled_cell]
            t_candidates = t_candidates[self.filled_cell]

        mask = t_candidates != task_id
        x_candidates = x_candidates[mask]
        y_candidates = y_candidates[mask]
        t_candidates = t_candidates[mask]

        available_items = x_candidates.size(0)
        n_mem_samples = min(available_items, batch_size)
        selected_idx = np.random.randint(available_items, size=n_mem_samples)

        mem_x_samples = x_candidates[selected_idx]
        mem_y_samples = y_candidates[selected_idx]
        mem_t_id_samples = t_candidates[selected_idx]

        ext_x = torch.cat([x, mem_x_samples])
        ext_y = torch.cat([y, mem_y_samples])

        cur_task_ids = torch.ones(x.size(0)).long() * task_id
        task_ids = torch.cat([cur_task_ids, mem_t_id_samples])

        # tasks_in_batch = sorted(task_ids.unique().tolist())
        # n_classes = [n_classes[idx] for idx in tasks_in_batch]
        # attr_offset = [0]*len(n_classes[0])
        # for t_id, task_classes in zip(tasks_in_batch, n_classes):
        #     for i, attr_classes in enumerate(task_classes):
        #         ext_y[task_ids==t_id, i] += attr_offset[i]
        #         attr_offset[i] += attr_classes

        return (ext_x, task_ids), ext_y

    def update_memory(self, *args, **kwargs):
        if self.mode == 'reservoir':
            self._update_reservoir(*args, **kwargs)
        elif self.mode == 'ring':
            self._update_ring(*args, **kwargs)
        else:
            raise ValueError(f'Unknown memory mode: {self.mode}')

    def _update_reservoir(self, xs, ys, task_id):
        if self.mem_size == 0:
            return
        batch_size = xs.size(0)

        # First, fill the memory with the first items of the batch if it
        # isn't full.
        n_new_items = min(self.mem_size - self.n_items, batch_size)
        assert n_new_items >= 0
        if n_new_items > 0:
            new_size = self.n_items + n_new_items
            print(type(self.mem[0]))
            print(self.mem[0].device)
            # self.mem[0].resize_(new_size, *xs.size()[1:])
            # self.mem[1].resize_(new_size, *ys.size()[1:])
            # self.mem[2].resize_(new_size)
            for i in range(3):
                pad = torch.zeros(n_new_items, *self.mem[i].size()[1:]).type_as(self.mem[i])
                self.mem[i] = torch.cat([self.mem[i], pad])

            self.mem[0][-n_new_items:] = xs[:n_new_items]
            self.mem[1][-n_new_items:] = ys[:n_new_items]
            self.mem[2][-n_new_items:] = task_id
            self.n_examples_seen += n_new_items

        # Then check if each remaining batch element should be placed in the
        # memory
        if n_new_items < batch_size:
            for x, y in zip(xs[n_new_items:].split(1), ys[n_new_items:].split(1)):
                i = np.random.randint(self.n_examples_seen)
                if i < self.mem_size:
                    self.mem[0][i] = x.squeeze(0)
                    self.mem[1][i] = y.squeeze(0)
                    self.mem[2][i] = task_id
                self.n_examples_seen += 1

    def _update_ring(self, xs, ys, task_id):
        offset = sum(self.classes_per_task[:task_id])
        for y, x in zip(ys.unbind(0), xs.unbind(0)):
            y = y.item()
            idx = (offset + y) * self.mem_size_per_class
            idx += self.class_counter[offset+y] % self.mem_size_per_class
            self.class_counter[offset + y] += 1
            self.mem[0][idx] = x
            self.mem[1][idx] = y
            self.mem[2][idx] = task_id
            self.filled_cell[idx] = True

    def __len__(self):
        return self.n_items
