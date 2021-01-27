# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
import logging
import os
import time
from functools import partial

import numpy as np
import ray
import torch
from ray import tune
from ray.tune import Experiment
from ray.tune.logger import JsonLogger, CSVLogger
from ray.tune.suggest.variant_generator import generate_variants
from torch import nn
from torchvision import transforms

from src.models.base import get_block_model
from src.models.utils import get_conv_out_size
from src.train.ignite_utils import _prepare_batch
from src.train.ray_training import TrainLLModel, OSTrainLLModel, \
    convert_to_tune_search_space
from src.train.training import train, get_classic_dataloaders
from src.train.utils import set_optim_params, _load_datasets
from src.utils.misc import get_env_url, rename_class
from src.utils.plotting import plot_res_dataframe, plot_trajectory, \
    list_top_archs, list_arch_scores, update_summary

logger = logging.getLogger(__name__)


class LifelongLearningModel(nn.Module, abc.ABC):
    def __init__(self, grid_params, ray_resources, *args, **kwargs):
        super(LifelongLearningModel, self).__init__(*args, **kwargs)
        self.models = nn.ModuleList([])
        self.grid_params = grid_params

        self.ray_resources = ray_resources

    def get_model(self, task_id, **task_infos):
        if task_id >= len(self.models):
            # this is a new task

            # New tasks should always give the x_dim and n_classes infos.
            assert 'x_dim' in task_infos and 'n_classes' in task_infos
            assert task_id == len(self.models)

            model = self._new_model(task_id=task_id, **task_infos)
            self.models.append(model)
        return self.models[task_id]

    @abc.abstractmethod
    def _new_model(self, **kwargs):
        raise NotImplementedError

    def n_params(self, t_id):
        """
        Return the total number of parameters used by the lifelong learner for
        models up to the task with id `t_id` (included).
        :param t_id:
        :return:
        """
        all_params = set()
        for i in range(t_id+1):
            model = self.get_model(i)
            all_params.update(model.parameters())
            all_params.update(model.buffers())

        return sum(map(torch.numel, all_params))

    def new_params(self, t_id):
        return self.n_params(t_id) - self.n_params(t_id-1)

    def get_search_space(self):
        # params = {k: tune.grid_search(v) for k, v in self.grid_params.items()}
        # return params
        return convert_to_tune_search_space(self.grid_params)

    def finish_task(self, dataset, task_id, viz=None, path=None):
        """
        Use the datastet to perform reauired post taks operations and compute
        statistics to track.

        :param dataset:
        :param task_id:
        :param viz:
        :return:
        """
        return {}

    def forward(self, *input):
        raise NotImplementedError

    def train_model_on_task(self, task, task_viz, exp_dir, use_ray,
                            use_ray_logging, grace_period,
                            num_hp_samplings, local_mode,
                            redis_address, lca_n, **training_params):
        logger.info("Training dashboard: {}".format(get_env_url(task_viz)))
        t_id = task['id']

        trainable = self.get_trainable(use_ray_logging=use_ray_logging)
        past_tasks = training_params.pop('past_tasks')
        normalize = training_params.pop('normalize')
        augment_data = training_params.pop('augment_data')

        transformations = []
        if augment_data:
            transformations.extend([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor()
            ])
        t_trans = [[] for _ in range(len(task['split_names']))]
        t_trans[0] = transformations
        datasets = trainable._load_datasets(task,
                                            task['loss_fn'],
                                            past_tasks, t_trans, normalize)
        train_loader, eval_loaders = get_classic_dataloaders(datasets,
                                                             training_params.pop(
                                                                 'batch_sizes'))
        model = self.get_model(task_id=t_id, x_dim=task['x_dim'],
                               n_classes=task['n_classes'],
                               descriptor=task['descriptor'],
                               dataset=eval_loaders[:2])

        if use_ray:
            if not ray.is_initialized():
                ray.init(address=redis_address)

            scheduler = None

            training_params['loss_fn'] = tune.function(
                training_params['loss_fn'])
            training_params['optim_func'] = tune.function(self.optim_func)

            init_model_path = os.path.join(exp_dir, 'model_initializations')
            model_file_name = '{}_init.pth'.format(training_params['name'])
            model_path = os.path.join(init_model_path, model_file_name)
            torch.save(model, model_path)

            training_params['model_path'] = model_path
            config = {**self.get_search_space(),
                      'training-params': training_params}
            if use_ray_logging:
                stop_condition = {'training_iteration':
                                      training_params['n_it_max']}
                checkpoint_at_end = False
                keep_checkpoints_num = 1
                checkpoint_score_attr = 'min-Val nll'
            else:
                stop_condition = None
                # loggers = [JsonLogger, MyCSVLogger]
                checkpoint_at_end = False
                keep_checkpoints_num = None
                checkpoint_score_attr = None

            trainable = rename_class(trainable, training_params['name'])
            experiment = Experiment(
                name=training_params['name'],
                run=trainable,
                stop=stop_condition,
                config=config,
                resources_per_trial=self.ray_resources,
                num_samples=num_hp_samplings,
                local_dir=exp_dir,
                loggers=(JsonLogger, CSVLogger),
                checkpoint_at_end=checkpoint_at_end,
                keep_checkpoints_num=keep_checkpoints_num,
                checkpoint_score_attr=checkpoint_score_attr)

            analysis = tune.run(experiment,
                                scheduler=scheduler,
                                verbose=1,
                                raise_on_failed_trial=True,
                                # max_failures=-1,
                                # with_server=True,
                                # server_port=4321
                                )
            os.remove(model_path)
            logger.info("Training dashboard: {}".format(get_env_url(task_viz)))

            all_trials = {t.logdir: t for t in analysis.trials}
            best_logdir = analysis.get_best_logdir('Val nll', 'min')
            best_trial = all_trials[best_logdir]

            # picked_metric = 'accuracy_0'
            # metric_names = {s: '{} {}'.format(s, picked_metric) for s in
            #                 ['Train', 'Val', 'Test']}

            logger.info('Best trial: {}'.format(best_trial))
            best_res = best_trial.checkpoint.result
            best_point = (best_res['training_iteration'], best_res['Val nll'])

            # y_keys = ['mean_loss' if use_ray_logging else 'Val nll', 'train_loss']
            y_keys = ['Val nll', 'Train nll']

            epoch_key = 'training_epoch'
            it_key = 'training_iteration'
            plot_res_dataframe(analysis, training_params['name'], best_point,
                               task_viz, epoch_key, it_key, y_keys)
            if 'entropy' in next(iter(analysis.trial_dataframes.values())):
                plot_res_dataframe(analysis, training_params['name'], None,
                                    task_viz, epoch_key, it_key, ['entropy'])
            best_model = self.get_model(task_id=t_id)
            best_model.load_state_dict(torch.load(best_trial.checkpoint.value))

            train_accs = analysis.trial_dataframes[best_logdir]['Train accuracy_0']
            best_t = best_res['training_iteration']
            t = best_trial.last_result['training_iteration']
        else:
            search_space = self.get_search_space()
            rand_config = list(generate_variants(search_space))[0][1]
            learner_params = rand_config.pop('learner-params', {})
            optim_params = rand_config.pop('optim')


            split_optims = training_params.pop('split_optims')
            if hasattr(model, 'set_h_params'):
                model.set_h_params(**learner_params)
            if hasattr(model, 'train_loader_wrapper'):
                train_loader = model.train_loader_wrapper(train_loader)

            loss_fn = task['loss_fn']
            if hasattr(model, 'loss_wrapper'):
                loss_fn = model.loss_wrapper(task['loss_fn'])

            prepare_batch = _prepare_batch
            if hasattr(model, 'prepare_batch_wrapper'):
                prepare_batch = model.prepare_batch_wrapper(prepare_batch, t_id)

            optim_fact = partial(set_optim_params,
                                 optim_func=self.optim_func,
                                 optim_params=optim_params,
                                 split_optims=split_optims)
            if hasattr(model, 'train_func'):
                f = model.train_func
                t, metrics, b_state_dict = f(train_loader=train_loader,
                                                eval_loaders=eval_loaders,
                                                optim_fact=optim_fact,
                                                loss_fn=loss_fn,
                                                split_names=task['split_names'],
                                                viz=task_viz,
                                                prepare_batch=prepare_batch,
                                                **training_params)
            else:
                optim = optim_fact(model=model)
                t, metrics, b_state_dict = train(model=model,
                                                 train_loader=train_loader,
                                                 eval_loaders=eval_loaders,
                                                 optimizer=optim,
                                                 loss_fn=loss_fn,
                                                 split_names=task['split_names'],
                                                 viz=task_viz,
                                                 prepare_batch=prepare_batch,
                                                 **training_params)
            train_accs = metrics['Train accuracy_0']
            best_t = b_state_dict['iter']
            if 'training_archs' in metrics:
                plot_trajectory(model.ssn.graph, metrics['training_archs'],
                                model.ssn.stochastic_node_ids, task_viz)
                weights = model.arch_sampler().squeeze()
                archs = model.ssn.get_top_archs(weights, 5)
                list_top_archs(archs, task_viz)
                list_arch_scores(self.arch_scores[t_id], task_viz)
                update_summary(self.arch_scores[t_id], task_viz, 'scores')

        if len(train_accs) > lca_n:
            lca_accs = []
            for i in range(lca_n + 1):
                if i in train_accs:
                    lca_accs.append(train_accs[i])
                else:
                    logger.warning('Missing step for {}/{} for lca computation'
                                   .format(i, lca_n))
            lca = np.mean(lca_accs)
        else:
            lca = np.float('nan')
        stats = {}
        start = time.time()
        # train_idx = task['split_names'].index('Train')
        # train_path = task['data_path'][train_idx]
        # train_dataset = _load_datasets([train_path])[0]
        train_dataset = _load_datasets(task, 'Train')[0]
        stats.update(self.finish_task(train_dataset, t_id, task_viz,
                                      path='drawings'))
        stats['duration'] = {'iterations': t,
                             'finish': time.time() - start,
                             'best_iterations': best_t}
        stats['params'] = {'total': self.n_params(t_id),
                           'new': self.new_params(t_id)}
        stats['lca'] = lca
        return stats

    def get_trainable(self, use_ray_logging):
        if use_ray_logging:
            return TrainLLModel
        else:
            return OSTrainLLModel
