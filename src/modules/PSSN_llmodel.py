import logging
import os
from collections import defaultdict
from operator import itemgetter

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import KNeighborsClassifier

from src.modules.ExhaustiveSearch import ExhaustiveSearch
from src.modules.SPNN import SPNN
from src.modules.change_layer_llmodel import _lin_block, _conv_block
from src.modules.ll_model import LifelongLearningModel
from src.modules.resnet import _make_layer, BasicBlock
from src.modules.ssn_wrapper import SSNWrapper
from src.modules.utils import _get_used_nodes, flatten
from src.utils.plotting import graph_to_svg, plot_svg
from supernets.interface.NetworkBlock import Add_Block, DummyBlock

logger = logging.getLogger(__name__)


class ConstMult(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.Tensor(1))
        nn.init.constant_(self.alpha, alpha)

    def forward(self, x):
        return self.alpha * x


def get_aggreg_block(dropout_p, activation):
    mods = [Add_Block(activation)]
    if dropout_p is not None:
        mods.append(nn.Dropout(dropout_p))

    return nn.Sequential(*mods)


class ZeroModel(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

    def forward(self, x):
        return torch.zeros(x.size(0), self.n_classes).to(x.device)


class ProgressiveSSN(LifelongLearningModel):
    IN_NODE = 'INs'
    # COL_IN_NODE = 'IN{}'
    OUT_NODE = 'OUT'

    def __init__(self, deter_eval, use_adapters, residual, block_depth,
                 initial_p, stride, k, pool, pool_k, padding, pruning_treshold,
                 connections, store_graphs, learn_in_and_out, learn_all,
                 norm_layer, arch_sampler_type, split_training, single_stage,
                 strat, method='knn', block_id=None, n_neighbors=15,
                 max_new_blocks=float('inf'), arch_loss_coef=None,
                 n_source_models=-1, entropy_coef=None, *args, **kwargs):
        super(ProgressiveSSN, self).__init__(*args, **kwargs)
        self.columns = []

        self.column_repr_sizes = []
        self.fixed_columns = set()
        self.fixed_graphs = []

        self.graph = nx.DiGraph()
        self.deter_eval = deter_eval
        self.pruning_treshold = pruning_treshold
        self.connections = connections
        assert store_graphs, 'All graphs should be stored'
        self.learn_in_and_out = learn_in_and_out
        self.learn_all = learn_all
        self.norm_layer = norm_layer

        self.use_adapters = use_adapters
        self.residual = residual
        self.block_depth = block_depth
        self.initial_p = initial_p

        self.max_new_blocks = max_new_blocks
        self.n_source_models = n_source_models
        self.n_neighbors = n_neighbors

        self.pool = pool

        if self.residual:
            n_layers = (self.n_convs - 1) * self.block_depth * 2 + 2
            assert n_layers == 18
            self.n_res_blocks = (self.n_convs - 1) * self.block_depth
            # if self.split_last:
            #     self.n_new_layers = self.block_depth - 1
            #     self.n_convs += self.n_new_layers
            #     stride.extend([stride[-1]] * self.n_new_layers)
            assert k == 3, 'Resnet requires a kernel of size 3'
            self._k = k
            assert len(stride) == self.n_convs
            self._stride = stride
            self._res_stride = [stride[0]]
            for stride in stride[1:]:
                self._res_stride.append(stride)
                self._res_stride += [1] * (self.block_depth - 1)
            assert len(self._res_stride) == self.n_res_blocks + 1
            self._res_stride.append(None)
            self._res_stride = np.array(self._res_stride)

            self._pad = padding
            assert len(self.hidden_size) == 1, 'Resnet requires only 1 ' \
                                               'hidden size, others are ' \
                                               'computed using the first and' \
                                               ' the stride'
            self._res_hidden_size = [self.hidden_size[0]]
            for i in range(1, self.n_convs):
                last_size = self.hidden_size[-1]
                if self.channel_scaling:
                    new_size = last_size * self.get_stride(i)
                    new_res_size = last_size * self._res_stride[i]
                else:
                    new_size = last_size
                    new_res_size = last_size
                self.hidden_size.append(new_size)
                self._res_hidden_size.extend([new_res_size]*self.block_depth)
            assert not self.dropout_p
            self.dropout_p = [None] * (self.n_res_blocks + 2)
            self._pool_k = [None] * (self.n_convs)
            self._res_pool_k = np.array([None] * (self.n_res_blocks + 2))
            self._pool_k[-1] = pool_k
            self._res_pool_k[-2] = pool_k

        else:
            if isinstance(stride, int):
                stride = [stride] * self.n_convs
            self._stride = stride

            if isinstance(k, int):
                k = [k] * self.n_convs
            self._k = k

            if isinstance(pool_k, int):
                pool_k = [pool_k] * self.n_convs
            self._pool_k = pool_k

            if isinstance(padding, int):
                padding = [padding] * self.n_convs
            self._pad = padding

        # self.graph.add_node(self.IN_NODE, module=DummyBlock())
        # self.graph.add_node(self.OUT_NODE, module=Add_Block(False))

        if residual:
            if block_id is None:
                block_id = [0]
                for i in range(self.n_res_blocks):
                    resnet_block_ids = [i//self.block_depth+1]
                    block_id += resnet_block_ids
                block_id += [block_id[-1]+1]
            self.block_id = np.array(block_id)
            self.n_modules = self.block_id.max() + 1
            assert len(self.block_id) == self.n_res_blocks + 2
            assert 10 == len(self.block_id), "SHould be 10 for resnets"
        else:
            self.n_modules = len(self.hidden_size) + 1

        # if self.split_last:
        #     for i in range(len(self.block_id) - 2, len(block_id)):
        #         self.block_id[i] += 1

        self.arch_sampler_type = arch_sampler_type
        self.arch_samplers = []
        self.search_res = []
        self.knn_res = []
        self.method = method

        self.arch_loss_coef = arch_loss_coef
        self.entropy_coef = entropy_coef
        self.split_training = split_training
        self.single_stage = single_stage
        self.strat = strat

        self.arch_scores = defaultdict(dict)

    def get_candidate_models(self, task_id, descriptor, dataset, topk,
                             n_neighbors, max_samples=1000):
        if topk == -1:
            return list(range(task_id))
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        train, val = dataset
        knn_perfs = []
        model_perfs = []
        rand_dists = []
        for t_id in range(task_id):
            model = self.get_model(t_id)
            # model.freeze_arch()
            pruned_model = model.get_frozen_model()
            pruned_model.to(device)

            features, model_preds, labels = [], [], []
            with torch.no_grad():
                n_samples = 0
                for x, y in train:
                    x = x.to(device)
                    y = y.to(device)
                    feats, preds = pruned_model.feats_forward(x)
                    features.append(flatten(feats))
                    model_preds.append(preds)
                    labels.append(y)
                    n_samples += x.size(0)
                    if n_samples >= max_samples:
                        break
                features = torch.cat(features, 0)
                model_preds = torch.cat(model_preds, 0)
                labels = torch.cat(labels, 0).flatten()
            n_neighbors = min(n_neighbors, features.size(0))
            knn = KNeighborsClassifier(n_neighbors=n_neighbors)
            knn.fit(features.cpu(), labels.cpu())
            preds, gt = [], []
            with torch.no_grad():
                n_samples = 0
                for x, y in val:
                    x = x.to(device)
                    y = y.to(device)
                    feats, _ = pruned_model.feats_forward(x)
                    feats = flatten(feats).cpu()
                    preds.append(torch.tensor(knn.predict(feats)))
                    gt.append(y.flatten())
                    n_samples += x.size(0)
                    if n_samples >= max_samples:
                        break
                preds = torch.cat(preds, 0).to(device)
                gt = torch.cat(gt, 0)
                # print(preds.shape, gt.shape)
                # print(torch.stack((preds, gt)))
                # print(preds)
                knn_perfs.append((preds == gt).float().mean().item())
                # print(knn_perfs)
                acc = (model_preds.argmax(1) == labels).float().mean().item()
                model_perfs.append(acc)
                rand_dists.append(abs(1/model_preds.size(1) - acc))
        self.arch_scores[task_id]['task id'] = list(range(task_id))
        self.arch_scores[task_id]['knn'] = knn_perfs
        self.arch_scores[task_id]['accs'] = model_perfs
        self.arch_scores[task_id]['rand dist'] = rand_dists
        if not hasattr(self, 'method'):
            self.method = 'knn'
        if self.method == 'knn':
            arg_res = np.argsort(knn_perfs)
            self.knn_res.append({k: knn_perfs[k] for k in arg_res})
        elif self.method == 'best-rand':
            arg_res = np.argsort(rand_dists)
            self.knn_res.append({k: rand_dists[k] for k in arg_res})
        elif self.method == 'rand':
            arg_res = np.random.choice(task_id, task_id, replace=False)
            self.knn_res.append({k: knn_perfs[k] for k in arg_res})
        else:
            raise ValueError(f'unknown method {self.method}')
        res = arg_res[-topk:].tolist()
        return res

    def get_nodes_from_models(self, model_ids):
        nodes = set()
        for mod in model_ids:
            graph = self.fixed_graphs[mod]
            nodes.update(graph.nodes)
        return nodes

    def get_backward_lateral_connections(self, col_id, targ_depth,
                                         candidate_nodes, is_last):
        """
        Return the new connections for current depth:
        If current node is (2, 1), we will get all backward connections __for
        current depth__ (and not current node) i.e. the connections from
        (2, 0) to (0, 1) and from (2, 0) to (1, 1).
        We need additional check because we can still have bw modules in
        `self.columns` at position [col_id][targ_depth] even if the node
        (col_id, targ_depth) has already been removed.
        :param col_id:
        :param targ_depth:
        :param is_last:
        :return:
        """
        assert len(self.columns) == col_id + 1
        lateral_connections = {}
        if 'bw' not in self.connections or is_last:
            return lateral_connections

        for targ_col_id, targ_col in enumerate(self.columns[:col_id]):
            # if targ_col_id not in candidate_nodes:
            #     continue
            targ_layer = targ_col.get(targ_depth)
            if targ_layer and (targ_col_id, targ_depth) in targ_layer\
                and (targ_col_id, targ_depth) in candidate_nodes:
                repr_size = self.column_repr_sizes[targ_col_id][targ_depth]
                lateral_connections[targ_col_id] = repr_size
        return lateral_connections

    def get_forward_lateral_connections(self, col_id, depth, candidate_nodes):
        assert len(self.columns) == col_id + 1
        lateral_connections = {}
        if 'fw' not in self.connections or depth == 1:
            # This is the first "real" layer, forward connection would be
            # identical to the (col_id, depth, 'w') layer
            return lateral_connections
        for src_col_id, src_col in enumerate(self.columns[:col_id]):
            # if src_col_id not in columns:
            #     continue
            src_layer = src_col.get(depth - 1)
            if src_layer and (src_col_id, depth - 1) in src_layer and\
                (src_col_id, depth-1) in candidate_nodes:
                repr_size = self.column_repr_sizes[src_col_id]
                repr_size = repr_size[depth - 1] \
                    if len(repr_size) >= depth else -1
                lateral_connections[src_col_id] = repr_size
        return lateral_connections

    def add_column(self, sizes, candidate_nodes):
        new_col_id = len(self.columns)
        self.columns.append({})

        # Add input and first node
        in_nodes = self.connect_new_input(new_col_id, candidate_nodes)
        self.columns[new_col_id][self.IN_NODE] = in_nodes

        in_size = sizes[0]
        # for depth, out_size in enumerate(sizes[1:], start=1):
        for depth in range(1, self.n_modules+1):
            # Start from 1 because input (depth 0) is already connected.
            # assert out_size == sizes[depth]
            out_size = sizes[depth] if depth < len(sizes) else None
            f_connections = \
                self.get_forward_lateral_connections(new_col_id, depth,
                                                     candidate_nodes)
            is_last_layer = depth == self.n_modules
            backward_connections = \
                self.get_backward_lateral_connections(new_col_id, depth,
                                                      candidate_nodes,
                                                      is_last_layer)
            out_activations = not is_last_layer
            new_modules = self.add_layer(f_connections, backward_connections,
                                         depth, new_col_id, out_activations,
                                         in_size, out_size)
            in_size = out_size
            self.columns[-1][depth] = new_modules

        # Add output
        out_nodes = self.connect_new_output(new_col_id, candidate_nodes)
        self.columns[new_col_id][self.OUT_NODE] = out_nodes

    def connect_new_output(self, col_id, candidate_nodes):
        new_out = (col_id, self.OUT_NODE)
        mod = Add_Block(False)
        self.graph.add_node(new_out, module=mod)

        out_connections = {new_out: mod}
        for prev_col_id in range(col_id + 1):
            last_layer_depth = self.n_modules
            # last_layer_depth_old = len(self.hidden_size) + 1
            src_layer = self.columns[prev_col_id].get(last_layer_depth)
            last_node = (prev_col_id, last_layer_depth)
            if not src_layer or last_node not in src_layer\
                    or last_node not in \
                    candidate_nodes | {(col_id, last_layer_depth)}:
                    # or prev_col_id not in (columns + [col_id]):
                continue
            if self.learn_in_and_out:
                conn_name = (col_id, self.OUT_NODE, prev_col_id)
                mod = DummyBlock()
                out_connections[conn_name] = mod
                self.graph.add_node(conn_name, module=mod)
                self.graph.add_edge(last_node, conn_name)
                self.graph.add_edge(conn_name, new_out)
            else:
                self.graph.add_edge(last_node, new_out)
        return out_connections

    def connect_new_input(self, col_id, candidate_nodes):
        # new_input = self.COL_IN_NODE.format(col_id)
        new_input = (col_id, self.IN_NODE)
        # mod = InputBlock()
        mod = DummyBlock()
        self.graph.add_node(new_input, module=mod)

        mod = DummyBlock()
        self.graph.add_node((col_id, 0), module=mod)
        self.columns[col_id][0] = {(col_id, 0): mod}

        # input_connections = {new_input: mod}
        # real_in = (col_id, self.IN_NODE, self.REAL_IN_NODE)
        # rmod = DummyBlock()
        # self.graph.add_node(real_in, module=rmod)
        input_connections = {new_input: mod}
        for prev_col_id in range(col_id + 1):
            in_layer = self.columns[prev_col_id].get(0)
            target_node = (prev_col_id, 0)
            if not in_layer or target_node not in in_layer\
                    or target_node not in candidate_nodes | {(col_id, 0)}:
                    # or prev_col_id not in columns + [col_id]:
                continue

            if self.learn_in_and_out:
                conn_name = (col_id, self.IN_NODE, prev_col_id)
                mod = DummyBlock()
                input_connections[conn_name] = mod
                self.graph.add_node(conn_name, module=mod)
                self.graph.add_edge(new_input, conn_name)
                self.graph.add_edge(conn_name, target_node)
            else:
                self.graph.add_edge(new_input, target_node)
        return input_connections

    def add_layer(self, f_connections, b_connections, depth, col_id, out_act,
                  in_size, out_size):
        """ Add a layer to the 'col_id' column
        :returns the modules created for this block
        """
        # Create node corresponding to current depth and column id
        h_name = (col_id, depth)
        # h = Add_Block(activation=out_act)
        # drop_p = self.dropout_p[depth - 1] if out_act else None
        h = get_aggreg_block(None, out_act)
        self.graph.add_node(h_name, module=h)

        # Create the connection from same column
        w_name = (col_id, depth, 'w')
        w = self.get_module(in_size, out_size, depth - 1)
        self.graph.add_node(w_name, module=w)
        if depth == 0:
            src_node = (col_id, self.IN_NODE)
        else:
            src_node = (col_id, depth - 1)
        self.graph.add_edge(src_node, w_name)
        self.graph.add_edge(w_name, h_name)

        added_modules = {w_name: w, h_name: h}

        # Create forward lateral connections
        u_name = (col_id, depth, 'u')

        for source_column, size in f_connections.items():
            if self.use_adapters:
                # Add the V module
                raise NotImplementedError('Need to figure out what to do with'
                                          ' pooling')
                mod = nn.Sequential(ConstMult(),
                                    self.get_module(size, in_size, depth - 1,
                                                    True),
                                    nn.ReLU())
                lateral_out_node = u_name
            else:
                # Connect directly to the current col, depth out node
                mod = self.get_module(size, out_size, depth - 1)
                lateral_out_node = h_name

            proj_name = (col_id, depth, source_column, 'f')
            source = (source_column, depth - 1)
            self.graph.add_node(proj_name, module=mod)
            self.graph.add_edge(source, proj_name)
            self.graph.add_edge(proj_name, lateral_out_node)

            added_modules[proj_name] = mod

        if f_connections and self.use_adapters:
            # Add the second layer of the non linear lateral connection
            raise NotImplementedError('Need to figure out what to do with'
                                      ' pooling')
            u = nn.Sequential(Add_Block(), self.get_module(in_size, out_size,
                                                           depth - 1))
            self.graph.add_node(u_name, module=u)
            self.graph.add_edge(u_name, h_name)
            added_modules[u_name] = u

        # Create backward lateral connections
        r_name = (col_id, depth, 'r')

        for target_column, size in b_connections.items():
            if self.use_adapters:
                # Add the S module
                raise NotImplementedError('Need to figure out what to do with'
                                          ' pooling')
                mod = nn.Sequential(ConstMult(),
                                    self.get_module(out_size, size, depth - 1,
                                                    True),
                                    nn.ReLU())
                lateral_in_node = r_name
            else:
                # Connect directly to the current col, depth out node
                mod = self.get_module(in_size, size, depth - 1)
                lateral_in_node = src_node
            proj_name = (col_id, depth, target_column, 'b')
            target = (target_column, depth)
            self.graph.add_node(proj_name, module=mod)
            self.graph.add_edge(lateral_in_node, proj_name)
            self.graph.add_edge(proj_name, target)
            added_modules[proj_name] = mod

        if b_connections and self.use_adapters:
            # Add the second layer of the non linear lateral connection
            raise NotImplementedError('Need to figure out what to do with'
                                      ' pooling')
            r = nn.Sequential(Add_Block(), self.get_module(in_size, out_size,
                                                           depth - 1))
            self.graph.add_node(r_name, module=r)
            self.graph.add_edge(src_node, r_name)
            added_modules[r_name] = r

        return added_modules

    def get_module(self, in_size, out_size, depth, is_adapter=False):
        if self.residual:
            # assert in_size is None or len(in_size) == 3
            # in_size = in_size[0]
            is_first = depth == 0
            # is_last = depth == self.n_convs
            # stride = self.get_stride(depth)
            # pool_k = self.get_pool_k(depth)

            # if not is_first:
            #     assert in_size[0] == self.hidden_size[depth-1]

            # layers = _make_layer(BasicBlock, in_size, out_size,
            #                      self.block_depth, stride, self.pool,
            #                      pool_k, norm_layer=self.norm_layer,
            #                      is_first=is_first, is_last=is_last,
            #                      end_act=False)

            is_last2 = depth == np.max(self.block_id)
            resnet_block_ids_bis = self.block_id == depth
            in_size = self._res_sizes[:-1][resnet_block_ids_bis]
            out_size = self._res_sizes[1:][resnet_block_ids_bis]
            res_stride = self._res_stride[resnet_block_ids_bis]
            if is_last2:
                assert res_stride[-1] is None
                res_stride = res_stride[:-1]
            res_pool = self._res_pool_k[resnet_block_ids_bis]
            layers = _make_layer(BasicBlock, in_size, out_size,
                                 self.block_depth, res_stride, self.pool,
                                 res_pool, norm_layer=self.norm_layer,
                                 is_first=is_first, is_last=is_last2,
                                 end_act=True,
                                 n_blocks=resnet_block_ids_bis.sum())

        elif isinstance(out_size, int) or len(out_size) == 1:
            # Linear
            drop_p = self.get_dropout_p(depth)
            is_last = depth == self.n_modules - 1
            layers = _lin_block(in_size, out_size,
                                dropout_p=drop_p, is_last=is_last)

        elif len(out_size) == 3:
            assert depth < self.n_convs
            stride = 1 if is_adapter else self.get_stride(depth)
            k = 1 if is_adapter else self.get_k(depth)
            pad = self.get_pad(depth)
            pool_k = self.get_pool_k(depth)
            drop_p = self.get_dropout_p(depth)
            is_last = depth == self.n_modules - 1
            layers = _conv_block(in_size, out_size, k, stride, pad,
                                 dropout_p=drop_p, pool=self.pool,
                                 pool_k=pool_k, is_last=is_last)
        else:
            raise ValueError(
                'Don\'t know which kind of layer to use for input '
                'size {} and output size {}.'.format(in_size, out_size))
        # if not str(layers) == str(layers_new):
        #     print('yooo')
        return nn.Sequential(*layers)

    def get_model(self, task_id, **task_infos):
        new_task = task_id >= len(self.columns)
        if new_task:
            # this is a new task
            # New tasks should always give the x_dim and n_classes infos.
            assert 'x_dim' in task_infos and 'n_classes' in task_infos
            # self.models shouldn't be used in ProgressiveSSN, models must be
            # reconstructed before each usage to take the modifications that
            # occured in the main graph since last time we've seen this task
            assert len(self.models) == 0
            assert task_id == len(self.column_repr_sizes) == len(self.columns)

            sizes = self.get_sizes(task_infos['x_dim'],
                                   task_infos['n_classes'])
            if self.residual:
                self._res_sizes = self.get_res_sizes(
                    task_infos['x_dim'],
                    task_infos['n_classes'])
            # if self.split_last:
            #     sizes[-self.n_new_layers-1:-1] = \
            #         [sizes[-self.n_new_layers-2]] * self.n_new_layers
                # sizes[-2] = sizes[-3]
            self.column_repr_sizes.append(sizes)
            if task_id > 0:
                desrc = task_infos['descriptor']
                dataset = task_infos['dataset']
                n_sources = self.n_source_models
                n_neighbors = self.n_neighbors \
                    if hasattr(self, 'n_neighbors') else 15
                candidate_models = self.get_candidate_models(task_id, desrc,
                                                             dataset,
                                                             n_sources,
                                                             n_neighbors)
            else:
                candidate_models = []
            # logger.warning('##T{}#n_source:{} - {}'.format(task_id,
            #                                                self.n_source_models,
            #                                             candidate_models))
            candidate_nodes = self.get_nodes_from_models(candidate_models)
            self.add_column(sizes, candidate_nodes)

        if task_id < len(self.fixed_graphs):
            sub_graph = self.fixed_graphs[task_id]
        else:
            active_nodes = self.get_used_nodes(task_id)
            if active_nodes:
                sub_graph = self.graph.subgraph(active_nodes).copy()
            else:
                sub_graph = None

        if sub_graph:
            in_node = (task_id, self.IN_NODE)
            out_node = (task_id, self.OUT_NODE)
            assert nx.has_path(sub_graph, in_node, out_node)

            trainable_modules = []
            frozen_modules = []
            stoch_nodes = []

            for layer in self.columns[task_id].values():
                for node, mod in layer.items():
                    trainable_modules.append(mod)
                    if len(node) > 2:
                        stoch_nodes.append(node)

            for column in self.columns[:task_id]:
                for layer in column.values():
                    for node, mod in layer.items():
                        if node in sub_graph:
                            frozen_modules.append(mod)
                            if self.learn_all and len(node) > 2:
                                stoch_nodes.append(node)
            if self.strat in ['nas', 'full']:
                if self.strat == 'full':
                    stoch_nodes = []
                ssn = SPNN(sub_graph, trainable_modules, frozen_modules,
                           stoch_nodes, in_node, out_node, self.single_stage)
                if new_task:
                    arch_sampler = self.arch_sampler_type
                else:
                    arch_sampler = self.arch_samplers[task_id]

                model = SSNWrapper(ssn_model=ssn,
                                   initial_p=self.initial_p,
                                   deter_eval=self.deter_eval,
                                   arch_loss_coef=self.arch_loss_coef,
                                   entropy_coef=self.entropy_coef,
                                   split_training=self.split_training,
                                   t_id=task_id,
                                   all_same=False,
                                   arch_sampler=arch_sampler)
                if new_task:
                    self.arch_samplers.append(model.arch_sampler)
            elif self.strat == 'search_all':
                model = ExhaustiveSearch(sub_graph, trainable_modules,
                                         frozen_modules, stoch_nodes, in_node,
                                         out_node, self.max_new_blocks)
                if new_task:
                    self.search_res.append(model.res)
                else:
                    model.res = self.search_res[task_id]
            # elif self.strat == 'full':
            #     model = SPNN(sub_graph, trainable_modules, frozen_modules,
            #                [], in_node, out_node, self.single_stage)
        else:
            model = ZeroModel(self.column_repr_sizes[task_id][-1])

        model.n_out = 1
        # if task_id < len(self.fixed_graphs) and sub_graph:
        #     assert equal_nn_graphs(self.fixed_graphs[task_id], ssn.graph)
        #     assert equal_nn_graphs(self.fixed_graphs[task_id], sub_graph)

        # print('ID', id(next(iter(model.parameters()))))
        return model

    def remove_node(self, node):
        node_col = node[0]
        node_lay = node[1]
        logger.debug('Pruning {}'.format(node))
        self.graph.remove_node(node)
        del self.columns[node_col][node_lay][node]

        for i, arch_sampler in enumerate(self.arch_samplers[node_col:]):
            if node in arch_sampler.var_names:
                arch_sampler.remove_var(node)
            elif i == 0:
                is_aggreg_node = len(node) == 2
                is_dummy_node = node[1] in (self.IN_NODE, self.OUT_NODE)
                assert is_aggreg_node or \
                       not self.learn_in_and_out and is_dummy_node

    def clean_graph(self):
        for i in range(len(self.columns)):
            # We clean the graph column by column
            used_nodes = self.get_used_nodes(i)
            for lay in self.columns[i].values():
                for node in list(lay.keys()):
                    if node not in used_nodes:
                        self.remove_node(node)

    def get_used_nodes(self, col):
        candidate_nodes = []
        for column in self.columns[:col + 1]:
            for lay in column.values():
                candidate_nodes.extend(lay.keys())

        input = (col, self.IN_NODE)
        last_node = (col, self.OUT_NODE)

        return _get_used_nodes(self.graph, candidate_nodes, input, last_node)

    def _new_model(self, **kwargs):
        raise NotImplementedError

    def finish_task(self, dataset, task_id, viz=None, path=None):
        if task_id in self.fixed_columns:
            raise ValueError('Task {} is already finished'.format(task_id))
        assert set(range(task_id)) == self.fixed_columns
        self.fixed_columns.add(task_id)
        additional_results = {}

        model = self.get_model(task_id)
        weights = model.get_weights()
        stoch_nodes = model.get_stoch_nodes()
        if hasattr(model, 'arch_sampler'):
            model.arch_sampler.freeze()
            additional_results['entropy'] = \
                model.arch_sampler.entropy().mean().item()

        assert len(stoch_nodes) == weights.size(0) and weights.dim() == 1
        logger.debug('Removing unused nodes after T{}'.format(task_id))
        graph = model.get_graph()
        plot_graph = graph.copy()
        nodes_to_remove = model.nodes_to_prune(self.pruning_treshold)
        for node in stoch_nodes:
            if node in nodes_to_remove:
                if node[0] == task_id:
                    self.remove_node(node)
                    plot_graph.node[node]['color'] = 'red'
                    graph.remove_node(node)
                else:
                    logger.debug('Was supposed to remove {}, but no'
                                 .format(node))
                    plot_graph.node[node]['color'] = 'orange'
                    graph.remove_node(node)
                    if hasattr(model, 'arch_sampler') and \
                            node in model.arch_sampler.var_names:
                        model.arch_sampler.remove_var(node)

            else:
                plot_graph.node[node]['color'] = 'blue'

        # weights_dict = model.get_weights_dict()
        weights_dict = {n: '{:.2f}'.format(w) for n, w in
                        zip(stoch_nodes, weights.unbind())}
        svg = graph_to_svg(plot_graph, edge_labels=weights_dict)
        if path:
            file = os.path.join(path, 'model_T{}_trained.svg'.format(task_id))
            with open(file, 'wb') as f:
                f.write(svg)
        if viz:
            plot_svg(str(svg), 'Trained', viz)

        svg = graph_to_svg(graph.copy())
        if path:
            file = os.path.join(path, 'model_T{}_pruned.svg'.format(task_id))
            with open(file, 'wb') as f:
                f.write(svg)
        if viz:
            plot_svg(str(svg), 'Pruned', viz)

        node_to_remove = clean_graph(graph, model.in_node, model.out_node)
        for n in node_to_remove:
            graph.remove_node(n)
            if n[0] == task_id:
                self.remove_node(n)
            elif n in stoch_nodes and hasattr(model, 'arch_sampler'):
                model.arch_sampler.remove_var(n)
        logger.info('Cleaning graph')
        self.clean_graph()
        svg = graph_to_svg(graph.copy())
        file = os.path.join(path, 'model_T{}_cleaned.svg'.format(task_id))
        with open(file, 'wb') as f:
            f.write(svg)
        if viz:
            plot_svg(str(svg), 'Cleaned', viz)

        file = os.path.join(path, 'model_T{}_full.svg'.format(task_id))
        svg = graph_to_svg(self.graph.copy())
        with open(file, 'wb') as f:
            f.write(svg)
        if viz:
            plot_svg(str(svg), 'Full', viz)

        if graph:
            self.fixed_graphs.append(graph)
        else:
            self.fixed_graphs.append(None)
        pruned_model = self.get_model(task_id)
        if not isinstance(pruned_model, ZeroModel):
            graph = pruned_model.get_graph()
            file = os.path.join(path, 'model_T{}_newget.svg'.format(task_id))
            svg = graph_to_svg(graph.copy())
            with open(file, 'wb') as f:
                f.write(svg)
        if self.knn_res:
            print(self.knn_res)
            pres = '\n'.join(map(str, reversed(list(self.knn_res[-1].items()))))
            viz.text('<pre>{}</pre>'.format(pres))

        if self.search_res:
            all_res = map(lambda x: (x[1][2]['value'], x[0]),
                          self.search_res[-1].items())
            pres = '\n'.join(map(str, sorted(all_res, key=itemgetter(0),
                                             reverse=True)))
            viz.text('<pre>{}</pre>'.format(pres))
        return additional_results

    def set_h_params(self, arch_loss_coef, entropy_coef, split_training):
        self.arch_loss_coef = arch_loss_coef
        self.entropy_coef = entropy_coef
        self.split_training = split_training


def equal_nn_graphs(ga, gb):
    a_nodes = set((n, tuple(sorted(d.items()))) for n, d in ga.nodes(True))
    b_nodes = set((n, tuple(sorted(d.items()))) for n, d in gb.nodes(True))
    if ga == gb:
        return True
    elif a_nodes == b_nodes and ga.edges == gb.edges:
        return True
    else:
        if b_nodes != a_nodes:
            print('Nodes issues')
            print(a_nodes)
            print(b_nodes)
        elif ga.edges != gb.edges:
            print('Edges issues')
            print(ga.edges)
            print(gb.edges)
        else:
            print('Meta problem')
        return False


def clean_graph(g, in_node, out_node):
    # candidate_nodes = list(g.nodes)
    # for column in self.columns[:col+1]:
    #     for lay in column.values():
    #         candidate_nodes.extend(lay.keys())
    # input = (col, self.IN_NODE)
    # if in_node not in candidate_nodes:
    #     return set()
    # input = self.COL_IN_NODE.format(col)
    # candidate_nodes.append(self.IN_NODE)

    # candidate_nodes.append(input)

    # last_node = (col, len(self.hidden_size)+1)
    used_nodes = set()
    for path in nx.all_simple_paths(g, in_node, out_node):
        used_nodes.update(path)

    nodes_to_remove = set(g.nodes) - used_nodes
    # for node in list(g.nodes):
    #     if node not in used_nodes:
    #         g.remove_node(node)
    return nodes_to_remove
