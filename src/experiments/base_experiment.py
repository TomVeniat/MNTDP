import logging
import os
import shutil
import tempfile
import threading
from collections import defaultdict

import numpy as np
import torch
import visdom
from ctrl.strategies.mixed_strategy import MixedStrategy

from src.models import HATLLModel
from src.models.utils import normalize_params_names
from src.utils import load_conf, VISDOM_CONF_PATH
from src.utils.plotting import get_env_name, update_summary, \
    plot_tasks_env_urls, write_text, update_avg_acc, update_pareto, \
    plot_accs, plot_times, plot_finish_times, plot_new_params, plot_lca, \
    plot_speeds, plot_best_speeds, plot_total_params, update_speed_plots, \
    update_avg_lca

logger = logging.getLogger(__name__)


class BaseExperiment(object):
    def __init__(self, task_gen, ll_models, cuda, n_it_max, n_ep_max,
                 augment_data, normalize, single_pass, n_tasks, patience,
                 grace_period, num_hp_samplings, visdom_traces_folder,
                 plot_all, batch_sizes, plot_tasks, lca, log_steps, log_epoch,
                 name, task_save_folder, load_tasks_from, use_ray,
                 use_ray_logging, redis_address, use_processes, local_mode,
                 smoke_test, stream_setting, sacred_run, log_dir, norm_models,
                 val_per_task, schedule_mode, split_optims, ref_params_id,
                 seed):
        self.task_gen = task_gen
        self.sims = None
        self.sims_comp = None
        self.name = name

        assert isinstance(ll_models, dict)

        self.ll_models = ll_models
        self.learner_names = list(self.ll_models.keys())
        self.norm_models = norm_models

        keys = list(self.ll_models.keys())
        self.norm_models_idx = [keys.index(nm) for nm in self.norm_models]

        if cuda and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.use_ray = use_ray
        self.redis_address = redis_address
        self.use_processes = use_processes
        self.local_mode = local_mode
        self.use_ray_logging = use_ray_logging

        self.single_pass = single_pass
        self.n_it_max = n_it_max
        self.n_ep_max = n_ep_max
        self.augment_data = augment_data
        self.normalize = normalize
        self.schedule_mode = schedule_mode

        self.n_tasks = n_tasks
        self.patience = patience
        self.grace_period = grace_period
        self.num_hp_samplings = num_hp_samplings
        self.stream_setting = stream_setting
        self.val_per_task = val_per_task
        self.split_optims = split_optims

        self.plot_tasks = plot_tasks
        self.batch_sizes = batch_sizes
        if os.path.isfile(VISDOM_CONF_PATH):
            self.visdom_conf = load_conf(VISDOM_CONF_PATH)
        else:
            self.visdom_conf = None

        self.lca = lca
        self.log_steps = log_steps
        self.log_epoch = log_epoch

        self.sacred_run = sacred_run
        self.seed = seed

        self.exp_name = get_env_name(sacred_run.config, sacred_run._id)
        self.exp_dir = os.path.join(log_dir, self.exp_name)
        self.init_model_path = os.path.join(self.exp_dir,
                                            'model_initializations')
        if not os.path.isdir(self.init_model_path):
            os.makedirs(self.init_model_path)
        if ref_params_id is None:
            self.ref_params_path = None
        else:
            assert isinstance(ref_params_id, int)
            self.ref_params_path = os.path.join(log_dir, str(ref_params_id),
                                      'model_initializations', 'ref.pth')

        self.visdom_traces_folder = os.path.join(visdom_traces_folder,
                                                 self.exp_name)

        self.load_tasks = load_tasks_from is not None
        if self.load_tasks:
            self.data_path = os.path.join(task_save_folder, str(load_tasks_from))
            assert os.path.isdir(self.data_path), \
                '{} doesn\'t exists'.format(self.data_path)
        else:
            self.data_path = os.path.join(task_save_folder, self.exp_name)
        main_env = get_env_name(sacred_run.config, sacred_run._id, main=True)
        trace_file = os.path.join(self.visdom_traces_folder, main_env)
        self.main_viz_params = {'env': main_env,
                                'log_to_filename': trace_file,
                                **self.visdom_conf}
        self.main_viz = visdom.Visdom(**self.main_viz_params)
        task_env = '{}_tasks'.format(self.exp_name)
        trace_file = '{}/{}'.format(self.visdom_traces_folder,
                                    task_env)
        self.task_env = visdom.Visdom(env=task_env,
                                      log_to_filename=trace_file,
                                      **self.visdom_conf)
        self.plot_all = plot_all

        self.summary = {'model': list(self.ll_models.keys()),
                        'speed': [float('nan')] * len(self.ll_models),
                        'accuracy_t': [float('nan')] * len(self.ll_models),
                        'accuracy_now': [float('nan')] * len(self.ll_models)
                        }
        update_summary(self.summary, self.main_viz)

        self.param_summary = defaultdict(list)
        self.param_summary['Task id'] = list(range(self.n_tasks))

        self.sacred_run.info['transfers'] = defaultdict(dict)
        self.task_envs_str = defaultdict(list)
        self.best_task_envs_str = defaultdict(list)

        # List of dicts. Each dict contains the parameters of a Visdom env for
        # the corresponding task per learner. In the current version this envs
        # are never used directly but modified for each training to contain
        # the actual parameters used.
        self.training_envs = []
        self.task_envs = []

        self.plot_labels = defaultdict()
        self.plot_labels.default_factory = self.plot_labels.__len__

        self.tune_register_lock = threading.Lock()
        self.eval_lock = threading.Lock()

        # Init metrics
        self.metrics = defaultdict(lambda: [[] for _ in self.ll_models])
        self.training_times_it = [[] for _ in self.ll_models]
        self.training_times_s = [[] for _ in self.ll_models]
        self.all_perfs = [[] for _ in self.ll_models]
        self.all_perfs_normalized = [[] for _ in self.ll_models]
        self.ideal_potentials = [[] for _ in self.ll_models]
        self.current_potentials = [[] for _ in self.ll_models]
        self.n_params = [[] for _ in self.ll_models]

    def run(self):
       raise NotImplementedError()

    def init_tasks(self):
        for i in range(self.n_tasks):
            task_name = '{}-T{}'.format(self.task_gen.concept_pool.name, i)
            if self.load_tasks:
                logger.info('Loading from {}'.format(self.data_path))
                t = self.task_gen.load_task(task_name, self.data_path)
            else:
                t = self.task_gen.add_task(task_name, self.data_path)

            self.training_envs.append({})
            for model in self.ll_models:
                training_name = '{}_{}_{}'.format(self.exp_name, model,
                                                  task_name)
                log_folder = '{}/{}'.format(self.visdom_traces_folder,
                                            training_name)
                train_viz_params = dict(env=training_name,
                                        log_to_filename=log_folder,
                                        **self.visdom_conf)
                self.training_envs[i][model] = train_viz_params
                task_env_name = '{}_{}'.format(self.exp_name, task_name)
                log_folder = '{}/{}'.format(self.visdom_traces_folder,
                                            task_env_name)
                task_viz_params = dict(env=task_env_name,
                                       log_to_filename=log_folder,
                                       **self.visdom_conf)
                self.task_envs.append(task_viz_params)
                plot_tasks_env_urls(self.task_envs_str, self.main_viz, 'best')

            if self.plot_tasks:
                self.task_gen.concept_pool.draw_tree(
                    highlighted_concepts=t.src_concepts,
                    viz=self.task_env,
                    title=task_name)
                task_env = '{}_T{}'.format(self.exp_name, i)
                trace_file = '{}/{}'.format(self.visdom_traces_folder,
                                            task_env)
                task_viz = visdom.Visdom(env=task_env,
                                         log_to_filename=trace_file,
                                         **self.visdom_conf)
                task_viz.text('<pre>{}</pre>'.format(t), win='task_descr',
                              opts={'width': 800, 'height': 350})
                t.plot_task(task_viz, 'T{}'.format(i))

            logger.info('###')
            logger.info('Task {}:'.format(i))
            logger.info(t)

        all_tasks = ''
        for i, t in enumerate(self.task_gen.task_pool):
            all_tasks += f'{i}\n{t}\n'
        # all_tasks = '\n'.join(str(i) for i in enumerate(str(t) for t in self.task_gen.task_pool))
        write_text(all_tasks, self.task_env)
        write_text(all_tasks, self.main_viz)

    def clean_tasks(self):
        for t in self.task_gen.task_pool:
            # Todo clean that
            meta = t._meta()
            t._infos['src_concepts'] = meta['source_concepts']
            t._infos['transformation'] = meta['transformation']

    def init_models(self, all_tasks):
        if not all_tasks:
            ref_learner = next(iter(self.ll_models.values()))
            ref_learner_name = next(iter(self.ll_models.keys()))
            t = self.task_gen.task_pool[0]
            if self.ref_params_path is None:
                first_mod = ref_learner.get_model(task_id=t.id, x_dim=t.x_dim,
                                               n_classes=t.n_classes.tolist(),
                                               descriptor=t.name)
                prune = not isinstance(ref_learner, HATLLModel)
                ref_params = normalize_params_names(first_mod.state_dict(),
                                                    prune_names=prune)
                logger.warning(f'Using {ref_learner_name} as ref params.')
            else:
                assert isinstance(self.ref_params_path, str)
                logger.warning(f'Loading ref params '
                               f'from {self.ref_params_path}.')
                ref_params = torch.load(self.ref_params_path)
            torch.save(ref_params, os.path.join(self.init_model_path,
                                                'ref.pth'))

            return self._sync_first_models(ref_params)

        params_count = defaultdict(list)
        for t in self.task_gen.task_pool:
            for name, ll_model in self.ll_models.items():
                mod = ll_model.get_model(task_id=t.id, x_dim=t.x_dim,
                                         n_classes=t.n_classes.tolist(),
                                         descriptor=t.name)
                n_params = sum(p.numel() for p in mod.parameters()
                               if p.requires_grad)
                params_count[name].append(n_params)
        return params_count

    def _sync_first_models(self, ref_params):
        t = self.task_gen.task_pool[0]
        descriptions = {}
        for name, ll_model in self.ll_models.items():
            first_mod = ll_model.get_model(task_id=t.id, x_dim=t.x_dim,
                                           n_classes=t.n_classes.tolist(),
                                           descriptor=t.name)
            prune = not isinstance(ll_model, HATLLModel)
            params = normalize_params_names(first_mod.state_dict(),
                                            prune_names=prune)

            for (ref_k, ref_t), (trg_k, trg_t) in zip(ref_params, params):
                assert ref_t.size() == trg_t.size(), f'{ref_k}, {trg_k}'
                trg_t.copy_(ref_t)
            descriptions[name] = first_mod.arch_repr()
            # descriptions[name] = count_params(first_mod)['trainable']
        return descriptions

    def init_sims(self):
        if isinstance(self.task_gen.strat, MixedStrategy) or \
                self.task_gen.contains_loaded_tasks:
            components = ''
        else:
            components = 'xyz'
        task_similarities = self.task_gen.get_similarities(components)

        for comp, sim in task_similarities.items():
            self.sacred_run.info['{}_similarities'.format(comp)] = sim.tolist()
            matrix_names = 'P({}) Similarity matrix'.format(comp)
            self.main_viz.heatmap(sim, opts={'title': matrix_names})

            vary_across = sim.numel() != sim.sum()
            if vary_across:
                if self.sims is not None:
                    logger.warning('Tasks are varying over 2 components '
                                   '({} and {}), which isn\'t supposed '
                                   'to happen.'.format(self.sims_comp, comp))
                self.sims_comp = comp
                self.sims = sim

        if self.sims is None:
            logging.warning('/!\\ All the tasks are identical /!\\')
            # Just a placeholder since all tasks are the same or we don't have
            # similarity information.
            self.sims = torch.ones(self.n_tasks, self.n_tasks)

    def init_plots(self):
        # pass
        # win_names = ['task_speeds']
        dummy_params = (float('nan'), float('nan'), self.learner_names[0],
                        self.main_viz, True)
        update_avg_acc(*dummy_params, 'Average Accuracies when seen')
        update_pareto([float('nan')], [float('nan')], *dummy_params[2:],
                      ['nothing'])
        update_pareto([float('nan')], [float('nan')], *dummy_params[2:],
                      ['nothing'], 'All')
        update_pareto([float('nan')], [float('nan')], *dummy_params[2:],
                      ['nothing'], 'Steps')
        update_pareto([float('nan')], [float('nan')], *dummy_params[2:],
                      ['nothing'], 'Steps_clean')

        update_avg_acc(*dummy_params, 'Average Accuracies now')

        plot_accs(*dummy_params)
        plot_times(*dummy_params)
        plot_speeds(*dummy_params)
        plot_best_speeds(*dummy_params)
        plot_finish_times(*dummy_params)
        plot_total_params(*dummy_params)
        plot_new_params(*dummy_params)
        update_speed_plots(*dummy_params)
        plot_lca(*dummy_params)
        update_avg_lca(*dummy_params)

        # for env_p in self.task_envs:
        #     vis = visdom.Visdom(**env_p)
        #     update_rescaled([float('nan')], [float('nan')], '___', vis, True)

    def update_stats(self, learner_id, task_id, stats):
        stats = stats.copy()
        learner_name = self.learner_names[learner_id]
        n_tasks_seen = task_id + 1

        train_time = stats.pop('duration')
        self.training_times_it[learner_id].append(train_time['iterations'])
        self.training_times_s[learner_id].append(train_time['seconds'])
        self.param_summary[learner_name].append(stats.pop('params'))
        for key, val in train_time.items():
            self.metrics['Train time {}'.format(key)][learner_id].append(val)

        test_accuracies = stats.pop('tasks_test')
        self.all_perfs[learner_id].append(test_accuracies)

        self.sacred_run.info['transfers'][learner_name] = \
            self.all_perfs[learner_id]
        name = '{} Accuracies'.format(learner_name)
        self.sacred_run.log_scalar(name, test_accuracies)

        new_speed_val = np.mean(self.training_times_it[learner_id])
        new_acc_val_t = np.mean([self.all_perfs[learner_id][t][t]
                                 for t in range(n_tasks_seen)])
        new_acc_val_now = np.mean([self.all_perfs[learner_id][-1][t]
                                   for t in range(n_tasks_seen)])

        self.summary['speed'][learner_id] = new_speed_val
        self.summary['accuracy_t'][learner_id] = new_acc_val_t
        self.summary['accuracy_now'][learner_id] = new_acc_val_now

    def save_traces(self):
        logger.warning('Archiving traces folder ...')
        with tempfile.TemporaryDirectory() as dir:
            archive_name = os.path.join(dir, '{}_traces'.format(self.exp_name))
            shutil.make_archive(archive_name, 'zip', self.visdom_traces_folder)
            self.sacred_run.add_artifact('{}.zip'.format(archive_name))
        shutil.rmtree(self.visdom_traces_folder)
