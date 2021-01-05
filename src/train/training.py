# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
from collections import defaultdict

import torch
from ignite.engine import Events
from ignite.handlers import Timer
from ignite.metrics import Accuracy, Loss, RunningAverage, Average
from torch.utils.data import DataLoader

from src.train.ignite_utils import create_supervised_trainer, _prepare_batch
from src.train.utils import create_supervised_evaluator, \
    StopAfterIterations, get_attr_transform

logger = logging.getLogger(__name__)


def get_classic_dataloaders(datasets, batch_sizes, n_workers=0):
    train_loader = DataLoader(datasets[0],
                              batch_size=batch_sizes[0],
                              shuffle=True, num_workers=n_workers)
    eval_loaders = [DataLoader(ds, batch_size=batch_sizes[1],
                               num_workers=n_workers)
                    for ds in datasets]
    return train_loader, eval_loaders


def train(model, train_loader, eval_loaders, optimizer, loss_fn,
          n_it_max, patience, split_names, select_metric='Val accuracy_0',
          select_mode='max', viz=None, device='cpu', lr_scheduler=None, name=None, log_steps=None,
          log_epoch=False, _run=None, prepare_batch=_prepare_batch,
          single_pass=False, n_ep_max=None):

    # print(model)

    if not log_steps and not log_epoch:
        logger.warning('/!\\ No logging during training /!\\')

    if log_steps is None:
        log_steps = []

    epoch_steps = len(train_loader)
    if log_epoch:
        log_steps.append(epoch_steps)

    if single_pass:
        max_epoch = 1
    elif n_ep_max is None:
        assert n_it_max is not None
        max_epoch = int(n_it_max / epoch_steps) + 1
    else:
        assert n_it_max is None
        max_epoch = n_ep_max

    all_metrics = defaultdict(dict)
    trainer = create_supervised_trainer(model, optimizer, loss_fn,
                                        device=device,
                                        prepare_batch=prepare_batch)

    if hasattr(model, 'new_epoch_hook'):
        trainer.add_event_handler(Events.EPOCH_STARTED, model.new_epoch_hook)
    if hasattr(model, 'new_iter_hook'):
        trainer.add_event_handler(Events.ITERATION_STARTED,
                                     model.new_iter_hook)

    trainer._logger.setLevel(logging.WARNING)

    # trainer output is in the format (x, y, y_pred, loss, optionals)
    train_loss = RunningAverage(output_transform=lambda out: out[3].item(),
                                epoch_bound=True)
    train_loss.attach(trainer, 'Trainer loss')
    if hasattr(model, 's'):
        met = Average(output_transform=lambda _: float('nan') if model.s is None else model.s)
        met.attach(trainer, 'cur_s')
        trainer.add_event_handler(Events.ITERATION_COMPLETED, met.completed, 'cur_s')

    if hasattr(model, 'arch_sampler') and model.arch_sampler.distrib_dim > 0:
        met = Average(output_transform=lambda _: float('nan') if model.cur_split is None else model.cur_split)
        met.attach(trainer, 'Trainer split')
        trainer.add_event_handler(Events.ITERATION_COMPLETED, met.completed, 'Trainer split')
        # trainer.add_event_handler(Events.EPOCH_STARTED, met.started)
        all_ent = Average(
            output_transform=lambda out: out[-1]['arch_entropy_avg'].item())
        all_ent.attach(trainer, 'Trainer all entropy')
        trainer.add_event_handler(Events.ITERATION_COMPLETED, all_ent.completed, 'Trainer all entropy')
        train_ent = Average(
            output_transform=lambda out: out[-1]['arch_entropy_sample'].item())
        train_ent.attach(trainer, 'Trainer sampling entropy')
        trainer.add_event_handler(Events.ITERATION_COMPLETED, train_ent.completed, 'Trainer sampling entropy')
        trainer.add_event_handler(Events.EPOCH_COMPLETED,
                                  lambda engine: model.check_arch_freezing(
                                      ent=train_ent.compute(),
                                      epoch=engine.state.iteration/(epoch_steps*max_epoch))
                                  )
        def log_always(engine, name):
            val = engine.state.output[-1][name]
            all_metrics[name][engine.state.iteration/epoch_steps] = val.mean().item()

        def log_always_dict(engine, name):
            for node, val in engine.state.output[-1][name].items():
                all_metrics['node {} {}'.format(node, name)][engine.state.iteration/epoch_steps] = val.mean().item()
        trainer.add_event_handler(Events.ITERATION_COMPLETED, log_always_dict, name='arch_grads')
        trainer.add_event_handler(Events.ITERATION_COMPLETED, log_always_dict, name='arch_probas')
        trainer.add_event_handler(Events.ITERATION_COMPLETED, log_always_dict, name='node_grads')
        trainer.add_event_handler(Events.ITERATION_COMPLETED, log_always, name='task all_loss')
        trainer.add_event_handler(Events.ITERATION_COMPLETED, log_always, name='arch all_loss')
        trainer.add_event_handler(Events.ITERATION_COMPLETED, log_always, name='entropy all_loss')

    if n_it_max is not None:
        StopAfterIterations([n_it_max]).attach(trainer)
    # epoch_pbar = ProgressBar(bar_format='{l_bar}{bar}{r_bar}', desc=name,
    #                          persist=True, disable=not (_run or viz))
    # epoch_pbar.attach(trainer, metric_names=['Train loss'])
    #
    # training_pbar = ProgressBar(bar_format='{l_bar}{bar}{r_bar}', desc=name,
    #                             persist=True, disable=not (_run or viz))
    # training_pbar.attach(trainer, event_name=Events.EPOCH_COMPLETED,
    #                      closing_event_name=Events.COMPLETED)
    total_time = Timer(average=False)
    eval_time = Timer(average=False)
    eval_time.pause()
    data_time = Timer(average=False)
    forward_time = Timer(average=False)
    forward_time.attach(trainer, start=Events.EPOCH_STARTED,
                        pause=Events.ITERATION_COMPLETED,
                        resume=Events.ITERATION_STARTED,
                        step=Events.ITERATION_COMPLETED)
    epoch_time = Timer(average=False)
    epoch_time.attach(trainer, start=Events.EPOCH_STARTED,
                      pause=Events.EPOCH_COMPLETED,
                      resume=Events.EPOCH_STARTED,
                      step=Events.EPOCH_COMPLETED)

    def get_loss(y_pred, y):
        l = loss_fn(y_pred, y)
        if not torch.is_tensor(l):
            l, *l_details = l
        return l.mean()

    def get_member(x, n=0):
        if isinstance(x, (list, tuple)):
            return x[n]
        return x

    eval_metrics = {'loss': Loss(get_loss)}

    for i in range(model.n_out):
        out_trans = get_attr_transform(i)
        def extract_ys(out):
            x, y, y_pred, loss, _ = out
            return out_trans((y_pred, y))

        train_acc = Accuracy(extract_ys)
        train_acc.attach(trainer, 'Trainer accuracy_{}'.format(i))
        trainer.add_event_handler(Events.ITERATION_COMPLETED,
                                  train_acc.completed, 'Trainer accuracy_{}'.format(i))
        eval_metrics['accuracy_{}'.format(i)] = \
            Accuracy(output_transform=out_trans)
        # if isinstance(model, SSNWrapper):
        #     model.arch_sampler.entropy().mean()

    evaluator = create_supervised_evaluator(model, metrics=eval_metrics,
                                            device=device,
                                            prepare_batch=prepare_batch)
    last_iteration = 0
    patience_counter = 0

    best = {'value': float('inf') * 1 if select_mode == 'min' else -1,
            'iter': -1,
            'state_dict': None
            }

    def is_better(new, old):
        if select_mode == 'min':
            return new < old
        else:
            return new > old

    def log_results(evaluator, data_loader, iteration, split_name):
        evaluator.run(data_loader)
        metrics = evaluator.state.metrics

        log_metrics = {}

        for metric_name, metric_val in metrics.items():
            log_name = '{} {}'.format(split_name, metric_name)
            if viz:
                first = iteration == 0 and split_name == split_names[0]
                viz.line([metric_val], X=[iteration], win=metric_name,
                         name=log_name,
                         update=None if first else 'append',
                         opts={'title': metric_name,
                               'showlegend': True,
                               'width': 500, 'xlabel': 'iterations'})
                viz.line([metric_val], X=[iteration/epoch_steps],
                         win='{}epoch'.format(metric_name),
                         name=log_name,
                         update=None if first else 'append',
                         opts={'title': metric_name,
                               'showlegend': True,
                               'width': 500, 'xlabel': 'epoch'})
            if _run:
                _run.log_scalar(log_name, metric_val, iteration)
            log_metrics[log_name] = metric_val
            all_metrics[log_name][iteration] = metric_val

        return log_metrics

    if lr_scheduler is not None:
        @trainer.on(Events.EPOCH_COMPLETED)
        def step(_):
            lr_scheduler.step()
            # logger.warning('current lr {:.5e}'.format(
            #     optimizer.param_groups[0]['lr']))

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_event(trainer):
        iteration = trainer.state.iteration if trainer.state else 0
        nonlocal last_iteration, patience_counter, best

        if not log_steps or not \
                (iteration in log_steps or iteration % log_steps[-1] == 0):
            return
        epoch_time.pause()
        eval_time.resume()
        all_metrics['training_epoch'][iteration] = iteration / epoch_steps
        all_metrics['training_iteration'][iteration] = iteration
        if hasattr(model, 'arch_sampler'):
            all_metrics['training_archs'][iteration] = \
                model.arch_sampler().squeeze().detach()
        # if hasattr(model, 'distrib_gen'):
        #     entropy = model.distrib_gen.entropy()
        #     all_metrics['entropy'][iteration] = entropy.mean().item()
        # if trainer.state and len(trainer.state.metrics) > 1:
        #     raise ValueError(trainer.state.metrics)
        all_metrics['data time'][iteration] = data_time.value()
        all_metrics['data time_ps'][iteration] = data_time.value() / max(data_time.step_count, 1.)
        all_metrics['forward time'][iteration] = forward_time.value()
        all_metrics['forward time_ps'][iteration] = forward_time.value() / max(forward_time.step_count, 1.)
        all_metrics['epoch time'][iteration] = epoch_time.value()
        all_metrics['epoch time_ps'][iteration] = epoch_time.value() / max(epoch_time.step_count, 1.)

        if trainer.state:
            # logger.warning(trainer.state.metrics)
            for metric, value in trainer.state.metrics.items():
                all_metrics[metric][iteration] = value
                if viz:
                    viz.line([value], X=[iteration], win=metric.split()[-1],
                         name=metric,
                         update=None if iteration==0 else 'append',
                         opts={'title': metric,
                               'showlegend': True,
                               'width': 500, 'xlabel': 'iterations'})

        iter_this_step = iteration - last_iteration
        for d_loader, name in zip(eval_loaders, split_names):
            if name == 'Train':
                if iteration == 0:
                    all_metrics['Trainer loss'][iteration] = float('nan')
                    all_metrics['Trainer accuracy_0'][iteration] = float('nan')
                    if hasattr(model, 'arch_sampler'):
                        all_metrics['Trainer all entropy'][iteration] = float('nan')
                        all_metrics['Trainer sampling entropy'][iteration] = float('nan')
                    # if hasattr(model, 'cur_split'):
                        all_metrics['Trainer split'][iteration] = float('nan')
                continue
            split_metrics = log_results(evaluator, d_loader, iteration, name)
            if select_metric not in split_metrics:
                continue
            if is_better(split_metrics[select_metric], best['value']):
                best['value'] = split_metrics[select_metric]
                best['iter'] = iteration
                best['state_dict'] = copy.deepcopy(model.state_dict())
                if patience > 0:
                    patience_counter = 0
            elif patience > 0:
                patience_counter += iter_this_step
                if patience_counter >= patience:
                    logger.info('#####')
                    logger.info('# Early stopping Run')
                    logger.info('#####')
                    trainer.terminate()
        last_iteration = iteration
        eval_time.pause()
        eval_time.step()
        all_metrics['eval time'][iteration] = eval_time.value()
        all_metrics['eval time_ps'][iteration] = eval_time.value() / eval_time.step_count
        all_metrics['total time'][iteration] = total_time.value()
        epoch_time.resume()

    log_event(trainer)

    #
    # @trainer.on(Events.EPOCH_COMPLETED)
    # def log_epoch(trainer):
    #     iteration = trainer.state.iteration if trainer.state else 0
    #     epoch = iteration/epoch_steps
    #     fw_t = forward_time.value()
    #     fw_t_ps = fw_t / forward_time.step_count
    #     d_t = data_time.value()
    #     d_t_ps = d_t / data_time.step_count
    #     e_t = epoch_time.value()
    #     e_t_ps = e_t / epoch_time.step_count
    #     ev_t = eval_time.value()
    #     ev_t_ps = ev_t / eval_time.step_count
    #     logger.warning('<{}> Epoch {}/{} finished (Forward: {:.3f}s({:.3f}), '
    #                    'data: {:.3f}s({:.3f}), epoch: {:.3f}s({:.3f}),'
    #                    ' Eval: {:.3f}s({:.3f}), Total: '
    #                    '{:.3f}s)'.format(type(model).__name__, epoch,
    #                                      max_epoch, fw_t, fw_t_ps, d_t, d_t_ps,
    #                                      e_t, e_t_ps, ev_t, ev_t_ps,
    #                                      total_time.value()))

    data_time.attach(trainer, start=Events.STARTED,
                     pause=Events.ITERATION_STARTED,
                     resume=Events.ITERATION_COMPLETED,
                     step=Events.ITERATION_STARTED)

    if hasattr(model, 'iter_per_epoch'):
        model.iter_per_epoch = len(train_loader)
    trainer.run(train_loader, max_epochs=max_epoch)
    return trainer.state.iteration, all_metrics, best
