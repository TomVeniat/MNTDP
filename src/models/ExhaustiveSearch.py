import logging
from collections import OrderedDict
from functools import partial
from operator import itemgetter
from pathlib import Path

import networkx as nx
import torch
import torch.nn as nn

from src.models.change_layer_llmodel import FrozenSequential
from src.models.utils import is_dummy_block, execute_step, graph_arch_details
from src.train.training import get_classic_dataloaders, train
from src.train.utils import _load_datasets
from src.utils.misc import pad_seq

logger = logging.getLogger(__name__)


class ExhaustiveSearch(nn.Module):
    def __init__(self, graph, tunable_modules, frozen_modules,
                 stochastic_nodes, in_node, out_node, max_new_blocks):
        super(ExhaustiveSearch, self).__init__()
        self.tunable_modules = nn.ModuleList(tunable_modules)
        self.tunable_modules.requires_grad_(True)
        self.frozen_modules = nn.ModuleList(frozen_modules)
        self.frozen_modules.requires_grad_(False)
        self.stochastic_nodes = stochastic_nodes
        self.graph = graph
        self.in_node = in_node
        self.out_node = out_node
        self.models = nn.ModuleList()
        self.models_idx = {}
        self.res = {}

        self.max_new_blocks = max_new_blocks

    def init_models(self):
        archs = list(nx.all_simple_paths(self.graph, self.in_node,
                                         self.out_node))
        for path in archs:
            new_model = FrozenSequential()
            last = None
            i = 0
            n_new_blocks = 0
            for node in path:
                assert node == self.in_node \
                       or node in self.graph.successors(last)
                nn_module = self.graph.nodes[node]['module']
                last = node
                if is_dummy_block(nn_module):
                    continue
                new_model.add_module(str(i), nn_module)
                if nn_module in self.frozen_modules:
                    new_model.frozen_modules_idx.append(i)
                else:
                    n_new_blocks += 1
                    # else:
                    #     nn_module.load_state_dict(self.block_inits[node])
                i += 1

            if n_new_blocks > self.max_new_blocks and len(archs) > 1:
                # print('Skipping {}'.format(path))
                continue
            # print('Adding {}'.format(path))
            new_model.n_out = self.n_out
            self.models_idx[tuple(path)] = len(self.models_idx)
            self.models.append(new_model)

    def get_weights(self):
        weights = torch.tensor([1. if n in self._selected_path() else 0.
                                for n in self.stochastic_nodes])
        return weights

    def _selected_path(self):
        assert self.res
        all_res = map(lambda x: (x[1][2]['value'], x[0]), self.res.items())
        return max(all_res, key=itemgetter(0))[1]
        # return next(iter(self.res.keys()))

    def get_stoch_nodes(self):
        return self.stochastic_nodes

    def get_graph(self):
        return self.graph

    def nodes_to_prune(self, *args):
        return [n for n in self.stochastic_nodes
                if n not in self._selected_path()]

    # def parameters(self):
    #     if len(self.models) == 0:
    #         self.init_models()
    #     return self.models[self.models_idx[self._selected_path()]].parameters()

    def train_func(self, datasets_p, b_sizes, optim_fact, *args, **kwargs):
        # if datasets_p['task']['data_path'][0].startswith('/net/blackorpheus/veniat/lileb/datasets/1406'):
        if datasets_p['task']['data_path'][0].startswith( '/net/blackorpheus/veniat/lileb/datasets/2775'):
            kwargs['n_ep_max'] = 6
            kwargs['patience'] = 6
            p = Path('../../understood/')
            p.mkdir(parents=True, exist_ok=True)

        if not self.models:
            self.init_models()
        # uid = np.random.randint(0, 10000)
        # logger.warning(f'{uid}-Training all {len(self.models)} options')
        calls = []
        for path, idx in self.models_idx.items():
            model = self.models[idx]
            # print(path, id(next(iter(model.parameters()))))
            # print(path, [type(p.grad)for p in model.parameters()])
            # print(path, [p.grad for p in model.parameters()])
            # print(path, [p is None for p in model.parameters()])
            # print(path, all([p.grad is None for p in model.parameters()]))
            # model = deepcopy(model)
            calls.append(partial(wrap, model=model, idx=idx,
                                 optim_fact=optim_fact, datasets_p=datasets_p,
                                 b_sizes=b_sizes, *args, **kwargs))
        ctx = torch.multiprocessing.get_context('spawn')
        # ctx = None
        # torch.multiprocessing.set_sharing_strategy('file_system')
        all_res = execute_step(calls, True, 4, ctx=ctx)
        for path, res in zip(self.models_idx.keys(), all_res):
            self.res[path] = res
        all_res = list(map(lambda x: (x[1][2]['value'], x[0]), self.res.items()))
        best_path = max(all_res, key=itemgetter(0))[1]
        _, best_metrics, best_chkpt = self.res[best_path]
        total_t = 0
        cum_best_iter = 0

        splits = ['Trainer', 'Train', 'Val', 'Test']
        metrics = ['accuracy_0']

        all_xs = []
        best_accs = list(self.res[best_path][1]['Val accuracy_0'].values())

        for i, (path, (t, logs, best)) in enumerate(self.res.items()):
            total_t += t
            cum_best_iter += best['iter']
            for m in metrics:
                for s in splits:
                    best_metrics[f'{s} {i} {m}_all'] = logs[f'{s} {m}']
                    if s == 'Val':
                        all_xs.append(list(logs[f'{s} {m}'].keys()))
        # print()
        max_len = max(map(len, all_xs))
        # print(all_xs)
        all_xs = [pad_seq(xs, max_len, xs[-1]) for xs in all_xs]
        # print(all_xs)
        # print(best_accs)
        # print(list(zip(*all_xs)))
        # print(list(zip_longest(*all_xs, fillvalue=0)))
        scaled_xs = [sum(steps) for steps in zip(*all_xs)]
        scaled_ys = pad_seq(best_accs, len(scaled_xs), best_accs[-1])
        new_metric = dict(zip(scaled_xs, scaled_ys))
        # print(new_metric)
        # print()
        best_metrics['Val accuracy_0_rescaled'] = new_metric
                # best_metrics[f'Val {m}_scales'] = logs[f'{s} {m}']

        self.models[self.models_idx[best_path]].load_state_dict(best_chkpt['state_dict'])
        best_chkpt['cum_best_iter'] = cum_best_iter
        return total_t, best_metrics, best_chkpt

    def forward(self, input):
        if not self.models:
            self.init_models()
        # Assert that we are at test time
        assert len(self.models) == 1, len(self.models)
        return self.models[0](input)

    def get_frozen_model(self):
        if not self.models:
            self.init_models()
        assert len(self.models) == 1, len(self.models)
        return self.models[0]

    def get_top_archs(self, n=1):
        res = list(map(lambda x: (x[0], x[1][2]['value']), self.res.items()))
        res = sorted(res, key=itemgetter(1), reverse=True)
        return OrderedDict(res[:n])

    def arch_repr(self):
        return graph_arch_details(self.graph)


def wrap(*args, idx=None, uid=None, optim_fact, datasets_p, b_sizes, **kwargs):
    model = kwargs['model']
    optim = optim_fact(model=model)
    datasets = _load_datasets(**datasets_p)
    train_loader, eval_loaders = get_classic_dataloaders(datasets, b_sizes, 0)
    if hasattr(model, 'train_loader_wrapper'):
        train_loader = model.train_loader_wrapper(train_loader)

    res = train(*args, train_loader=train_loader, eval_loaders=eval_loaders,
                optimizer=optim, **kwargs)
    # logger.warning('{}=Received option {} results'.format(uid, idx))
    return res
