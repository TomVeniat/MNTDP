# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import time
from collections import defaultdict

import torch
import torchvision
import torchvision.transforms.functional as tf
from ignite.engine import Events
from ignite.metrics import Accuracy, ConfusionMatrix
from more_itertools import roundrobin
from torch import nn
from torch.utils.data import ConcatDataset, TensorDataset, DataLoader
from tqdm import tqdm

from src.datasets.TensorDataset import MyTensorDataset
from src.train.ignite_utils import create_supervised_evaluator

logger = logging.getLogger(__name__)


class StopAfterIterations(object):
    def __init__(self, log_iterations):
        """
        Should be attached to an Ignite.Engine.
        Will stop the training immediately after the `n_iterations_max`
        iteration of the given engine.
        """
        self.log_iterations = log_iterations
        self.iteration = 0

    def __call__(self, engine):
        self.iteration += 1
        if self.iteration in self.log_iterations \
                or self.iteration % self.log_iterations[-1] == 0:
            engine.terminate()

    def attach(self, engine):
        engine.add_event_handler(Events.ITERATION_COMPLETED, self)


class RoundRobinDataloader(DataLoader):
    def __init__(self, dataloaders):
        assert all(isinstance(dl, DataLoader) for dl in dataloaders)
        self.dataloaders = dataloaders

    def __iter__(self):
        return roundrobin(*self.dataloaders)

    def __len__(self):
        return sum(len(dl) for dl in self.dataloaders)



def get_multitask_dataset(tasks, min_samples_per_class=10):
    all_ds = []
    normal_splits_len = [None, None, None]
    elems_per_task = []

    for i, (paths, loss) in enumerate(tasks):
        cur_ds = []
        for j, split_path in enumerate(paths):
            x, y = torch.load(split_path)
            if normal_splits_len[j] is None:
                normal_splits_len[j] = x.size(0)
                elems_per_task.append(int(x.size(0)/len(tasks)))
            assert x.size(0) == normal_splits_len[j], 'All split should have ' \
                                                      'the same size'
            if j == 0:
                # Keep all elements for the train set
                n_elems = x.size(0)
            else:
                n_elems = elems_per_task[j]
                n_classes = y.unique().size(0)
                if n_elems < min_samples_per_class * n_classes:
                    logger.warning('Not enough sample, will select {} elems'
                                     ' for {} classes when requiring at '
                                     'least {} samples per class'
                                     .format(n_elems, n_classes, min_samples_per_class))
                    n_elems = min_samples_per_class * n_classes
                selected_idx = torch.randint(x.size(0), (n_elems,))
                x = x[selected_idx]
                y = y[selected_idx]
            z = torch.ones(n_elems, dtype=torch.long)*i
            cur_ds.append(TensorDataset(x, y, z))
        all_ds.append(cur_ds)

    ds = [ConcatDataset(split_datasets) for split_datasets in zip(*all_ds)]
    return ds


def mytimeit(f):

    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()

        print('### func:{} took: {:2.4f} sec ###'.format(f.__name__, te - ts))
        return result

    return timed


def set_dropout(model, dropout_p):
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = dropout_p


def set_optim_params(optim_func, optim_params, model, split_optims):
    if isinstance(optim_params, dict):
        optim_params = [optim_params]

    if hasattr(model, 'param_groups'):
        param_groups = model.param_groups()
    else:
        param_groups = [{'params': model.parameters()}]

    assert len(param_groups) == len(optim_params)

    for param_group, group_opt in zip(param_groups, optim_params):
        param_group.update(group_opt)

    if split_optims:
        return [optim_func([pg]) for pg in param_groups]
    else:
        return optim_func(param_groups)


def strat_split_from_y(ds, split_ratio=0.5, y_idx=1):
    assert isinstance(ds, MyTensorDataset), ds
    n_classes = ds.tensors[y_idx].unique().numel()
    n_samples_per_class, mod = divmod(len(ds), n_classes)
    assert mod == 0
    n_samples_per_class *= split_ratio

    class_counts = defaultdict(int)
    train_data = []
    train_label = []
    test_data = []
    test_label = []

    # We can't iterate directly iterate on the dataset using
    # `for data, label in ds:` because we don't want the transforms to
    # be applied.
    for data, label in zip(*ds.tensors):
        c = label.item()
        if class_counts[c] < n_samples_per_class:
            train_data.append(data)
            train_label.append(label)
            class_counts[c] += 1
        else:
            test_data.append(data)
            test_label.append(label)
    train_data = torch.stack(train_data)
    train_label = torch.stack(train_label)
    train_name = torch.zeros(train_data.size(0)).int()
    test_data = torch.stack(test_data)
    test_label = torch.stack(test_label)
    test_name = torch.ones(test_data.size(0)).int()

    trans = ds.transforms
    return (MyTensorDataset(train_data, train_label, train_name,
                            transforms=trans),
            MyTensorDataset(test_data, test_label, test_name,
                            transforms=trans))


def _load_datasets_old(data_path, cur_loss_fn=None, past_tasks=None,
                   transforms=None, normalize=False):
    if transforms is None:
        transforms = [None] * len(data_path)
    datasets = []
    for split_path, trans in zip(data_path, transforms):
        x, y = torch.load(split_path)
        datasets.append(MyTensorDataset(x, y, transforms=trans))
    return datasets


def _load_datasets(task, splits=None, transforms=None, normalize=False):
    """
    Load the dataset associated with a task, all splits by default
    """
    if splits is None:
        splits = task['split_names']
    if isinstance(splits, str):
        splits = [splits]

    if transforms is None:
        transforms = [[] for _ in range(len(splits))]
    if normalize:
        assert 'statistics' in task
        t = torchvision.transforms.Normalize(**task['statistics'])
        transforms = [split_trans + [t] for split_trans in transforms]

    datasets = []
    for split, trans in zip(splits, transforms):
        split_idx = task['split_names'].index(split)
        split_path = task['data_path'][split_idx]
        x, y = torch.load(split_path)
        trans = torchvision.transforms.Compose(trans) if trans else None
        datasets.append(MyTensorDataset(x, y, transforms=trans))
    return datasets


def evaluate(model, dataset, batch_size, device, out_id=0):
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    n_classes = dataset.tensors[1][:, 0].unique().numel()
    out_transform = get_attr_transform(out_id)
    eval_metrics = {
        'accuracy': Accuracy(output_transform=out_transform),
        'confusion': ConfusionMatrix(num_classes=n_classes,
                                     output_transform=out_transform)
    }
    evaluator = create_supervised_evaluator(model, metrics=eval_metrics,
                                            device=device)
    evaluator.logger.setLevel(logging.WARNING)
    evaluator.run(val_loader)
    metrics = evaluator.state.metrics
    return metrics['accuracy'], metrics['confusion']


def evaluate_on_tasks(tasks, ll_model, batch_size, device, splits=None,
                      normalize=False, cur_task=None):
    if splits is None:
        splits = ['Test']
    assert isinstance(splits, (list, tuple))
    res = defaultdict(lambda: defaultdict(list))
    for t_id, task in enumerate(tqdm(tasks, desc='Evaluation on tasks',
                                     leave=False, disable=True)):
        t_id = t_id if cur_task is None else min(t_id, cur_task)
        eval_model = ll_model.get_model(task_id=t_id)
        for split in splits:
            split_dataset = _load_datasets(task, split, normalize=normalize)[0]
            acc, conf_mat = evaluate(eval_model, split_dataset, batch_size,
                                     device)
            res[split]['accuracy'].append(acc)
            res[split]['confusion'].append(conf_mat)
        eval_model.cpu()
        torch.cuda.empty_cache()
    return res


def get_attr_transform(attr_idx):
    def out_transform(out):
        y_pred, y = out

        if isinstance(y_pred, (tuple, list)):
            y_pred = y_pred[attr_idx]

        if y.dim() > 1:
            y = y[:, attr_idx]

        return y_pred, y

    return out_transform


def normalize_splits(splits):
    train_split = splits[0]
    if train_split[0].dim() == 3:
        # Images
        assert train_split.size(1) == 3
        means = [train_split[:, i, :, :].mean() for i in range(3)]
        stds = [train_split[:, i, :, :].std() for i in range(3)]
        splits = (norm_batch(s, means, stds) for s in splits)
    else:
        # Vectors
        mean = train_split.mean()
        std = train_split.std()
        splits = (x.sub(mean).div(std) for x in splits)

    return splits


def norm_batch(split, mean, std):
    return torch.stack([tf.normalize(x, mean, std) for x in split.unbind(0)])
