import copy

import torch
import torch.nn as nn

from src.models.change_layer_llmodel import FrozenSequential
from src.models.utils import is_dummy_block, _get_used_nodes
from supernets.networks.StochasticSuperNetwork import StochasticSuperNetwork


class SPNN(StochasticSuperNetwork):
    # IN_NODE = 'IN'
    # OUT_NODE = 'OUT'

    def __init__(self, graph, tunable_modules, frozen_modules,
                 stochastic_nodes, in_node, out_node, single_stage):
        super(SPNN, self).__init__()
        self.tunable_modules = nn.ModuleList(tunable_modules)
        self.tunable_modules.requires_grad_(True)
        self.frozen_modules = nn.ModuleList(frozen_modules)
        self.frozen_modules.requires_grad_(False)
        self.graph = graph

        self.single_stage = single_stage
        self.block_inits = {}
        nodes_to_raise = []
        for node in graph.nodes:
            mod = graph.nodes[node]['module']
            if not (self.single_stage or mod in frozen_modules):
                self.block_inits[node] = copy.deepcopy(mod.state_dict())
            if node in stochastic_nodes:
                if node not in graph:
                    nodes_to_raise.append(node)
                self.register_stochastic_node(node)

        if nodes_to_raise:
            raise ValueError('Nodes to register not '
                             'in graph {}'.format(nodes_to_raise))

        self.set_graph(self.graph, [in_node], [out_node])

    def forward(self, inputs):
        return super(SPNN, self).forward(inputs)[0]

    def train(self, mode=True):
        super().train(mode)
        self.frozen_modules.train(False)
        return self

    def get_pruned_model(self, weights):
        assert weights.ndim == 1
        # print(weights)
        assert torch.equal(weights, weights ** 2)
        candidates = set()
        for node in self.graph.nodes:
            if node not in self.stochastic_node_ids:
                candidates.add(node)
            else:
                idx = self.stochastic_node_ids[node]
                if weights[idx] == 1:
                    candidates.add(node)
        used_nodes = _get_used_nodes(self.graph, candidates,
                                     self.in_nodes[0], self.out_nodes[0])

        new_model = FrozenSequential()
        last = None
        i = 0
        for node in self.traversal_order:
            if node not in used_nodes:
                continue
            assert node == self.in_nodes[0] \
                   or node in self.graph.successors(last)
            nn_module = self.graph.nodes[node]['module']
            last = node
            if is_dummy_block(nn_module):
                continue
            new_model.add_module(str(i), nn_module)
            if nn_module in self.frozen_modules:
                new_model.frozen_modules_idx.append(i)
                # else:
                #     nn_module.load_state_dict(self.block_inits[node])
            i += 1
        return new_model
