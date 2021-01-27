# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import operator
import re
from concurrent.futures.process import ProcessPoolExecutor
from functools import reduce

import networkx as nx
import torch
from torch import nn

from src.utils.misc import count_params
from supernets.interface.NetworkBlock import DummyBlock, Add_Block


class MultiHead(nn.Module):
    def __init__(self, in_size, out_sizes, *args, **kwargs):
        super(MultiHead, self).__init__(*args, **kwargs)
        if isinstance(in_size, torch.Size):
            assert len(in_size) == 1, 'Multhihead expect 1d inputs, got {}'\
               .format(in_size)
            in_size = in_size[0]

        heads = [nn.Linear(in_size, out) for i, out in enumerate(out_sizes)]
        # heads = [nn.Linear(in_size, 1 if out in [1, 2] else out) for i, out in enumerate(out_sizes)]
        self.heads = nn.ModuleList(heads)
        self.n_out = len(out_sizes)

    def forward(self, input):
        return [head(input) for head in self.heads]


class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)


def flatten(x):
    x = x.contiguous()
    n_features = reduce(operator.mul, x.size()[1:])
    return x.view(x.size(0), n_features)


def make_model(*blocks, seq=nn.Sequential):
    blocks_list = []
    for block in blocks:
        if isinstance(block, nn.Module):
            block = [block]
        assert isinstance(block, list)
        blocks_list += block

    model = seq(*blocks_list)

    model.n_out = blocks_list[-1].n_out
    return model


def get_conv_out_size(in_size, kernel_size, padding, stride):
    return [(d + 2*padding - (kernel_size-1) - 1) // stride + 1
            for d in in_size]


def graph_arch_details(graph):
    detail = {}
    for node in nx.topological_sort(graph):
        node_props = graph.nodes[node]
        mod = node_props['module']
        n_params = count_params(mod)['trainable']
        det = []
        if n_params != 0:
            detail[len(detail)] = n_params
            for n in mod:
                if is_dummy_block(n):
                    continue
                if isinstance(n, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU,
                                  nn.AdaptiveAvgPool2d, Flatten, nn.Linear,
                                  nn.Dropout, nn.MaxPool2d)):
                    det.append(count_params(n)['trainable'])
                elif isinstance(n, nn.Sequential):
                    det.append([count_params(b) for b in n])
                else:
                    l = {
                        'c1': count_params(n.conv1)['trainable'],
                        'c2': count_params(n.conv2)['trainable'],
                        'down': 0
                    }
                    if n.downsample is not None:
                        l['down'] = count_params(n.downsample)['trainable']
                    det.append(l)
    return detail


def is_dummy_block(block):
    if isinstance(block, DummyBlock):
        return True
    if isinstance(block, Add_Block) and not block.activation:
        return True
    if isinstance(block, nn.Sequential):
        return all(is_dummy_block(b) for b in block)
    return False


def execute_step(step_calls, use_processes, max_workers=None, ctx=None):
    if not use_processes or len(step_calls) < 2:
        return [f() for f in step_calls]

    with ProcessPoolExecutor(max_workers, ctx) as executor:
        training_futures = [executor.submit(f) for f in step_calls]
        return [future.result() for future in training_futures]


def _get_used_nodes(graph, candidate_nodes, input, output):
    if input not in candidate_nodes:
        return set()
    # input = self.COL_IN_NODE.format(col)
    # candidate_nodes.append(self.IN_NODE)

    # candidate_nodes.append(input)

    candidate_graph = graph.subgraph(candidate_nodes)

    # last_node = (col, len(self.hidden_size)+1)
    used_nodes = set()
    for path in nx.all_simple_paths(candidate_graph, input, output):
        used_nodes.update(path)
    return used_nodes


def format_key(k):
    return tuple(int(itm) if itm.isnumeric() else itm for itm in k.split('.'))


PREFIX_TO_SKIP = {'arch_sampler', 'mem_', 'n_examples_seen', 'embeddings', }


def normalize_params_names(params, prefix_to_skip=None, prune_names=True):
    if prefix_to_skip is None:
        prefix_to_skip = PREFIX_TO_SKIP
    res = []
    prefix = None
    idx = None
    for k, p in params.items():
        if any(k.startswith(pref) for pref in prefix_to_skip):
            continue
        if prune_names:
            if prefix is None:
                idx = re.search(r"\d", k).start()
                prefix = k[:idx]
            else:
                assert k.startswith(prefix), k
            k = format_key(k[idx:])
        res.append((k, p))
    return sorted(res)
