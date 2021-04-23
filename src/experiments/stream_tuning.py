import logging
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from copy import deepcopy
from functools import partial
from os import path

import numpy as np
import pandas
import ray
import torch
import visdom
from ray import tune
from ray.tune import CLIReporter
from ray.tune.logger import JsonLogger, CSVLogger
from torchvision.transforms import transforms

from src.experiments.base_experiment import BaseExperiment
from src.models.ExhaustiveSearch import ExhaustiveSearch
from src.models.utils import execute_step
from src.train.ignite_utils import _prepare_batch
from src.train.training import train, get_classic_dataloaders
from src.train.utils import set_dropout, set_optim_params, \
    _load_datasets, evaluate_on_tasks
from src.utils.misc import get_env_url, fill_matrix, \
    get_training_vis_conf
from src.utils.plotting import update_summary, plot_tasks_env_urls, \
    plot_heatmaps, \
    plot_trajectory, list_top_archs, process_final_results

visdom.logger.setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)


class StreamTuningExperiment(BaseExperiment):
    def run(self):
        if self.task_gen.concept_pool.attribute_similarities is not None:
            attr_sim = self.task_gen.concept_pool.attribute_similarities
            self.main_viz.heatmap(attr_sim,
                                  opts={'title': 'Attribute similarities'})
        if self.plot_tasks:
            self.task_gen.concept_pool.draw_tree(viz=self.main_viz,
                                                 title='Full tree')
            self.task_gen.concept_pool.draw_attrs(viz=self.main_viz)
            self.task_gen.concept_pool.plot_concepts(self.main_viz)

        self.init_tasks()
        self.init_sims()
        self.clean_tasks()
        if not self.stream_setting:
            self.init_models(True)
        else:
            details = self.init_models(False)
            logger.info('Architecture details for the first models:')
            for learner, det in details.items():
                logger.info(f'{learner}: {det} ({sum(det.values())}, '
                            f'{4*sum(det.values())/1e6})')
        self.init_plots()

        logger.info("General dashboard: {}".format(get_env_url(self.main_viz)))
        logger.info('Tasks: {}'.format(get_env_url(self.task_env)))
        # if self.use_ray and not self.use_processes:
        #     if self.redis_address and not self.local_mode:
        #         ray.init(redis_address=self.redis_address)
        #     else:
        #         logger.warning('Launching a new ray cluster')
        #         ray.init(object_store_memory=int(1e7), include_webui=True,
        #                  local_mode=self.local_mode, num_gpus=0)

        train_calls = []
        for model_name, ll_model in self.ll_models.items():
            vis_params = [vis_params[model_name]
                          for vis_params in self.training_envs]
            params = dict(learner=ll_model,
                          stream=self.task_gen.stream_infos(True),
                          task_level_tuning=self.val_per_task,
                          learner_name=model_name,
                          # exp_name=self.exp_name,
                          vis_params=vis_params,
                          plot_all=self.plot_all,
                          batch_sizes=self.batch_sizes,
                          n_it_max=self.n_it_max,
                          n_ep_max=self.n_ep_max,
                          augment_data=self.augment_data,
                          normalize=self.normalize,
                          schedule_mode=self.schedule_mode,
                          patience=self.patience,
                          # grace_period=self.grace_period,
                          num_hp_samplings=self.num_hp_samplings,
                          device=self.device,
                          log_steps=self.log_steps,
                          log_epoch=self.log_epoch,
                          exp_dir=self.exp_dir,
                          lca=self.lca,
                          single_pass=self.single_pass,
                          stream_setting=self.stream_setting,
                          split_optims=self.split_optims,
                          # use_ray=self.use_ray,
                          # use_ray_logging=self.use_ray_logging,
                          local_mode=self.local_mode,
                          redis_address=self.redis_address,
                          seed=self.seed
                          )
            train_calls.append(partial(tune_learner_on_stream, **params))

        ctx = torch.multiprocessing.get_context('spawn')
        # ctx = None
        results_array = execute_step(train_calls, self.use_processes, ctx=ctx)
        res = dict(zip(self.ll_models.keys(), results_array))

        summ = process_final_results(self.main_viz, res, self.exp_name,
                                     self.visdom_conf, self.task_envs_str,
                                     len(self.task_gen.task_pool),
                                     self.best_task_envs_str, self.val_per_task,
                                     self.visdom_traces_folder)

        plot_tasks_env_urls(self.task_envs_str, self.main_viz, 'all')
        plot_tasks_env_urls(self.best_task_envs_str, self.main_viz, 'best')
        self.save_traces()

        res_py = {k: [itm.to_dict('list') for itm in v] for k, v in res.items()}
        # res_2 = {k: [pandas.DataFrame(itm) for itm in v] for k, v in res_py.items()}

        # for (k1, v1), (k2, v2) in zip(res.items(), res_2.items()):
            # assert k1 == k2
            # print([i1.equals(i2) for i1, i2 in zip(v1, v2)])
        logger.info(f'Args {" ".join(sys.argv[2:])}')
        print(pandas.DataFrame(summ).set_index('model'))
        return [res_py, self.task_gen.stream_infos(full=False)]




def tune_learner_on_stream(learner, learner_name, task_level_tuning,
                           stream, redis_address, local_mode, num_hp_samplings,
                           vis_params, exp_dir, seed, **training_params):
    """
    Returns 2 dataframes:
     - The first one contains information about the best trajectory and
     contains as many rows as there are tasks. Each row corresponding to the
     model trained on the corresponding task in the best trajectory.
      - The second contains one row per hyper-parameters combination. Each
      Row corresponds contains information about the results on all tasks for
      this specific hp combination. Note that, *in the task-level hp optim
      settting*, this DF is useful to investigate the behaviors of specific
      trainings, but rows *DOES NOT* correspond to actual trajectories.
    """

    exp_name = os.path.basename(exp_dir)
    init_path = path.join(exp_dir, 'model_initializations', learner_name)
    torch.save(learner, init_path)
    config = {**learner.get_search_space(),
              'training-params': training_params,
              'tasks': stream,
              'vis_params': vis_params,
              # 'learner': learner,
              'learner_path': init_path,
              'task_level_tuning': task_level_tuning,
              # 'env': learner_name
              'seed': seed
              }


    def trial_name_creator(trial):
        return learner_name
        # return '{}_{}'.format(learner_name, trial.trial_id)

    reporter = CLIReporter(max_progress_rows=10)
    # reporter.add_metric_column('avg_acc_val')
    reporter.add_metric_column('avg_acc_val_so_far', 'avg_val')
    reporter.add_metric_column('avg_acc_test_so_far', 'avg_test')
    reporter.add_metric_column('total_params')
    # reporter.add_metric_column('fw_t')
    # reporter.add_metric_column('data_t')
    # reporter.add_metric_column('eval_t')
    # reporter.add_metric_column('epoch_t')
    reporter.add_metric_column('duration_model_creation', 'creat_t')
    reporter.add_metric_column('duration_training', 'train_t')
    reporter.add_metric_column('duration_postproc', 'pp_t')
    reporter.add_metric_column('duration_finish', 'fin_t')
    reporter.add_metric_column('duration_eval', 'ev_t')
    reporter.add_metric_column('duration_sum', 'sum_t')
    reporter.add_metric_column('duration_seconds', 'step_t')
    reporter.add_metric_column('total_t')
    reporter.add_metric_column('t')

    ray_params = dict(
        loggers=[JsonLogger, CSVLogger],
        name=learner_name,
        resources_per_trial=learner.ray_resources,
        num_samples=num_hp_samplings,
        local_dir=exp_dir,
        verbose=1,
        progress_reporter=reporter,
        trial_name_creator=trial_name_creator,
        max_failures=3,
    )
    envs = []
    all_val_accs = defaultdict(list)
    all_test_accs = defaultdict(list)
    if task_level_tuning:
        best_trials_df = []
        config['ray_params'] = ray_params
        config['local_mode'] = local_mode
        config['redis_address'] = redis_address
        analysis, selected = train_on_tasks(config)
        for t_id, (task, task_an) in enumerate(zip(stream, analysis)):
            # envs.append([])
            for trial_n, t in enumerate(task_an.trials):
                if len(envs) <= trial_n:
                    envs.append([])
                env = '{}_Trial_{}_{}_{}'.format(exp_name, t, t.experiment_tag,
                                                 task['descriptor'])
                envs[trial_n].append(env)
                if selected[t_id] == t.experiment_tag:
                    all_val_accs[t.experiment_tag].append(
                        '<span style="font-weight:bold">{}</span>'.format(
                        t.last_result[f'Val_T{t_id}']))
                else:
                    all_val_accs[t.experiment_tag].append(
                        t.last_result[f'Val_T{t_id}'])
                all_test_accs[t.experiment_tag].append(
                    t.last_result[f'Test_T{t_id}']
                )

            best_trial = max(task_an.trials,
                         key=lambda trial: trial.last_result['avg_acc_val_so_far'])

            df = task_an.trial_dataframes[best_trial.logdir]
            best_trials_df.append(df)

        return_df = pandas.concat(best_trials_df, ignore_index=True)
        analysis = analysis[-1]
        results = sorted(analysis.trials, reverse=True,
                         key=lambda trial: trial.last_result['avg_acc_val_so_far'])
    else:
        if not ray.is_initialized():
            if local_mode:
                ray.init(local_mode=local_mode)
            else:
                ray.init(redis_address)
                # logging_level=logging.DEBUG)
        ray_params['config'] = config
        analysis = tune.run(train_on_tasks, **ray_params)

        results = sorted(analysis.trials, reverse=True,
                         key=lambda trial: trial.last_result['avg_acc_val_so_far'])
        for t in results:
            envs.append([])
            for task in stream:
                env = '{}_Trial_{}_{}_{}'.format(exp_name, t, t.experiment_tag,
                                                 task['descriptor'])
                envs[-1].append(env)
        return_df = analysis.trial_dataframes[results[0].logdir]
    summary = {
        'model': [t.experiment_tag for t in results],
        'Avg acc Val': [t.last_result['avg_acc_val'] for t in results],
        'Acc Val': [all_val_accs[t.experiment_tag] for t in results],
        'Avg acc Test': [t.last_result['avg_acc_test'] for t in results],
        'Acc Test': [all_test_accs[t.experiment_tag] for t in results],
        'Params': [t.last_result['total_params'] for t in results],
        'Steps': [t.last_result['total_steps'] for t in results],
        'paths': [t.logdir for t in results],
        'evaluated_params': [t.evaluated_params for t in results],
        'envs': envs
    }
    summary = pandas.DataFrame(summary)

    return return_df, summary


def train_on_tasks(config):
    """Config can either be the sampled configuration given by ray during a run
    or all the parameters including thos to pass to ray under the 'ray_config'
    key"""
    seed = config['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    tasks = config.pop('tasks')

    task_vis_params = config.pop('vis_params')

    # all_stats = []
    transfer_matrix = defaultdict(list)
    total_steps = 0

    if 'learner' in config:
        learner = config.pop('learner')
    else:
        learner_path = config.pop('learner_path')
        learner = torch.load(learner_path)
    task_level_tuning = config.pop('task_level_tuning')
    if task_level_tuning:
        ray_params = config.pop('ray_params')
        local_mode = config.pop('local_mode')
        redis_address = config.pop('redis_address')
        all_analysis = []
        selected_tags = []
    for t_id, (task, vis_p) in enumerate(zip(tasks, task_vis_params)):
        #todo sync transfer matrix
        static_params = dict(
            t_id=t_id, task=task, tasks=tasks, vis_p=vis_p,
            transfer_matrix=transfer_matrix, total_steps=total_steps
        )

        if task_level_tuning:
            if not ray.is_initialized():
                if local_mode:
                    ray.init(local_mode=local_mode)
                else:
                    ray.init(redis_address,
                             log_to_driver=False,
                             logging_level=logging.ERROR)

            config['static_params'] = static_params
            config['learner_path'] = learner_path
            config['seed'] += t_id

            # reporter = CLIReporter(max_progress_rows=10)
            # print(reporter._metric_columns)
            # print(reporter.DEFAULT_COLUMNS)
            # reporter.add_metric_column('avg_acc_val')
            # reporter.add_metric_column('total_params')
            # reporter.add_metric_column('fw_t')
            # reporter.add_metric_column('data_t')
            # reporter.add_metric_column('eval_t')
            # reporter.add_metric_column('epoch_t')
            # reporter.add_metric_column('total_t')
            # ray_params['progress_reporter'] = reporter
            analysis = tune.run(train_t, config=config, **ray_params)

            all_analysis.append(analysis)

            def get_key(trial):
                # return trial.last_result['avg_acc_val_so_far']
                return trial.last_result['best_val']
            best_trial = max(analysis.trials, key=get_key)
            for trial in analysis.trials:
                if trial != best_trial:
                    trial_path = trial.logdir
                    shutil.rmtree(trial_path)
            # am = np.argmax(list(map(get_key, analysis.trials)))
            # print("BEST IS {}: {}".format(am, best_trial.last_result['avg_acc_val']))

            # t = best_trial.last_result['duration_iterations']
            total_steps = best_trial.last_result['total_steps']
            selected_tags.append(best_trial.experiment_tag)
            best_learner_path = os.path.join(best_trial.logdir, 'learner.pth')
            learner = torch.load(best_learner_path, map_location='cpu')
            shutil.rmtree(best_trial.logdir)

            #todo UPDATE LEARNER AND SAVE
            torch.save(learner, learner_path)
        else:
            rescaled, t, metrics, b_state_dict, \
            stats = train_single_task(config=deepcopy(config), learner=learner,
                                                                          **static_params)

        # all_stats.append(stats)
        # update_rescaled(list(rescaled.values()), list(rescaled.keys()), tag,
        #                  g_task_vis, False)

    if task_level_tuning:
        return all_analysis, selected_tags
    else:
        save_path = path.join(tune.get_trial_dir(), 'learner.pth')
        logger.info('Saving {} to {}'.format(learner, save_path))
        torch.save(learner, save_path)


def train_t(config):
    seed = config.pop('seed')
    static_params = config.pop('static_params')

    torch.backends.cudnn.enabled = True
    if static_params['t_id'] == 0:
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
    else:
        torch.backends.cudnn.deterministic = False

    if 'PSSN' in tune.get_trial_name() or static_params['t_id'] == 0:
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    if 'learner' in config:
        learner = config.pop('learner')
    else:
        learner_path = config.pop('learner_path')
        learner = torch.load(learner_path)

    rescaled, t, metrics, b_state_dict, stats = train_single_task(config=config, learner=learner, **static_params)

    learner_save_path = os.path.join(tune.get_trial_dir(), 'learner.pth')
    # raise ValueError(learner_save_path)
    torch.save(learner, learner_save_path)


def train_single_task(t_id, task, tasks, vis_p, learner, config, transfer_matrix,
                      total_steps):

    training_params = config.pop('training-params')
    learner_params = config.pop('learner-params', {})
    assert 'model-params' not in config, "Can't have model-specific " \
                                         "parameters while tuning at the " \
                                         "stream level."

    if learner_params:
        learner.set_h_params(**learner_params)

    batch_sizes = training_params.pop('batch_sizes')
    # optim_func = training_params.pop('optim_func')
    optim_func = learner.optim_func
    optim_params = config.pop('optim')
    schedule_mode = training_params.pop('schedule_mode')
    split_optims = training_params.pop('split_optims')

    dropout = config.pop('dropout') if 'dropout' in config else None

    stream_setting = training_params.pop('stream_setting')
    plot_all = training_params.pop('plot_all')
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
    lca_n = training_params.pop('lca')

    if plot_all:
        vis_p = get_training_vis_conf(vis_p, tune.get_trial_dir())
        # print('NEW vis: ', vis_p)
        task_vis = visdom.Visdom(**vis_p)
        # env = [env[0], env[-1]]
        # vis_p['env'] = '_'.join(env)
        # vis_p['log_to_filename'] = os.path.join(vis_logdir, vis_p['env'])
        # g_task_vis = visdom.Visdom(**vis_p)

        logger.info(get_env_url(task_vis))
    else:
        task_vis = None

    t_trans = [[] for _ in range(len(task['split_names']))]
    t_trans[0] = transformations.copy()

    datasets_p = dict(task=task,
                      transforms=t_trans,
                      normalize=normalize)
    datasets = _load_datasets(**datasets_p)
    train_loader, eval_loaders = get_classic_dataloaders(datasets,
                                                         batch_sizes)

    assert t_id == task['id']

    start1 = time.time()
    model = learner.get_model(task['id'], x_dim=task['x_dim'],
                              n_classes=task['n_classes'],
                              descriptor=task['descriptor'],
                              dataset=eval_loaders[:2])
    model_creation_time = time.time() - start1

    loss_fn = task['loss_fn']
    training_params['loss_fn'] = loss_fn

    prepare_batch = _prepare_batch
    if hasattr(model, 'prepare_batch_wrapper'):
        prepare_batch = model.prepare_batch_wrapper(prepare_batch, t_id)

    if hasattr(model, 'loss_wrapper'):
        training_params['loss_fn'] = \
            model.loss_wrapper(training_params['loss_fn'])

    # if hasattr(model, 'backward_hook'):
    #     training_params[]

    # optim = set_optim_params(optim_func, optim_params, model, split_optims)
    optim_fact = partial(set_optim_params,
                         optim_func=optim_func,
                         optim_params=optim_params,
                         split_optims=split_optims)
    # if schedule_mode == 'steps':
    #     lr_scheduler = torch.optim.lr_scheduler.\
    #         MultiStepLR(optim[0], milestones=[25, 40])
    # elif schedule_mode == 'cos':
    #     lr_scheduler = torch.optim.lr_scheduler.\
    #         CosineAnnealingLR(optim[0], T_max=200, eta_min=0.001)
    # elif schedule_mode is None:
    #     lr_scheduler = None
    # else:
    #     raise NotImplementedError()
    if dropout is not None:
        set_dropout(model, dropout)

    assert not config, config
    start2 = time.time()
    rescaled, t, metrics, b_state_dict = train_model(model, datasets_p,
                                                     batch_sizes, optim_fact,
                                                     prepare_batch, task,
                                                     train_loader, eval_loaders,
                                                     training_params, config)

    training_time = time.time() - start2
    start3 = time.time()
    if not isinstance(model, ExhaustiveSearch):
        #todo Handle the state dict loading uniformly for all learners RN only
        # the exhaustive search models load the best state dict after training
        model.load_state_dict(b_state_dict['state_dict'])

    iterations = list(metrics.pop('training_iteration').values())
    epochs = list(metrics.pop('training_epoch').values())

    assert len(iterations) == len(epochs)
    index = dict(epochs=epochs, iterations=iterations)
    update_summary(index, task_vis, 'index', 0.5)

    grouped_xs = dict()
    grouped_metrics = defaultdict(list)
    grouped_legends = defaultdict(list)
    for metric_n, metric_v in metrics.items():
        split_n = metric_n.split()
        if len(split_n) < 2:
            continue
        name = ' '.join(split_n[:-1])
        grouped_metrics[split_n[-1]].append(list(metric_v.values()))
        grouped_legends[split_n[-1]].append(name)
        if split_n[-1] in grouped_xs:
            if len(metric_v) > len(grouped_xs[split_n[-1]]):
                longer_xs = list(metric_v.keys())
                assert all(a == b for a, b in zip(longer_xs,
                                                  grouped_xs[split_n[-1]]))
                grouped_xs[split_n[-1]] = longer_xs
        else:
            grouped_xs[split_n[-1]] = list(metric_v.keys())

    for (plot_name, val), (_, legends) in sorted(zip(grouped_metrics.items(),
                                                     grouped_legends.items())):
        assert plot_name == _
        val = fill_matrix(val)
        if len(val) == 1:
            val = np.array(val[0])
        else:
            val = np.array(val).transpose()
        x = grouped_xs[plot_name]
        task_vis.line(val, X=x, win=plot_name,
                      opts={'title': plot_name, 'showlegend': True,
                            'width': 500, 'legend': legends,
                            'xlabel': 'iterations', 'ylabel': plot_name})

    avg_data_time = list(metrics['data time_ps'].values())[-1]
    avg_forward_time = list(metrics['forward time_ps'].values())[-1]
    avg_epoch_time = list(metrics['epoch time_ps'].values())[-1]
    avg_eval_time = list(metrics['eval time_ps'].values())[-1]
    total_time = list(metrics['total time'].values())[-1]

    entropies, ent_legend = [], []
    for metric_n, metric_v in metrics.items():
        if metric_n.startswith('Trainer entropy'):
            entropies.append(list(metric_v.values()))
            ent_legend.append(metric_n)

    if entropies:
        task_vis.line(np.array(entropies).transpose(), X=iterations,
                      win='ENT',
                      opts={'title': 'Arch entropy', 'showlegend': True,
                            'width': 500, 'legend': ent_legend,
                            'xlabel': 'Iterations', 'ylabel': 'Loss'})

    if hasattr(learner, 'arch_scores') and hasattr(learner, 'get_top_archs'):
        update_summary(learner.arch_scores[t_id], task_vis, 'scores')
        archs = model.get_top_archs(5)
        list_top_archs(archs, task_vis)

    if 'training_archs' in metrics:
        plot_trajectory(model.ssn.graph, metrics['training_archs'],
                        model.ssn.stochastic_node_ids, task_vis)

    postproc_time = time.time() - start3
    start4 = time.time()
    save_path = tune.get_trial_dir()
    finish_res = learner.finish_task(datasets[0], t_id,
                                     task_vis, save_path)
    finish_time = time.time() - start4

    start5 = time.time()
    eval_tasks = tasks
    # eval_tasks = tasks[:t_id + 1] if stream_setting else tasks
    evaluation = evaluate_on_tasks(eval_tasks, learner, batch_sizes[1],
                                   training_params['device'],
                                   ['Val', 'Test'], normalize,
                                   cur_task=t_id)
    assert evaluation['Val']['accuracy'][t_id] == b_state_dict['value']

    stats = {}
    eval_time = time.time() - start5

    stats.update(finish_res)

    test_accs = metrics['Test accuracy_0']
    if not test_accs:
        lca = np.float('nan')
    else:
        if len(test_accs) <= lca_n:
            last_key = max(test_accs.keys())
            assert len(test_accs) == last_key + 1,\
                f"Can't compute LCA@{lca_n} if steps were skipped " \
                f"(got {list(test_accs.keys())})"
            test_accs = test_accs.copy()
            last_acc = test_accs[last_key]
            for i in range(last_key + 1, lca_n+1):
                test_accs[i] = last_acc
        lca = np.mean([test_accs[i] for i in range(lca_n + 1)])

    accs = {}
    key = 'accuracy'
    # logger.warning(evaluation)
    for split in evaluation.keys():
        transfer_matrix[split].append(evaluation[split][key])
        for i in range(len(tasks)):
            split_acc = evaluation[split][key]
            if i < len(split_acc):
                accs['{}_T{}'.format(split, i)] = split_acc[i]
            else:
                accs['{}_T{}'.format(split, i)] = float('nan')
    plot_heatmaps(list(transfer_matrix.keys()),
                  list(map(fill_matrix, transfer_matrix.values())),
                  task_vis)


    # logger.warning(t_id)
    # logger.warning(transfer_matrix)

    avg_val = np.mean(evaluation['Val']['accuracy'])
    avg_val_so_far = np.mean(evaluation['Val']['accuracy'][:t_id+1])
    avg_test = np.mean(evaluation['Test']['accuracy'])
    avg_test_so_far = np.mean(evaluation['Test']['accuracy'][:t_id+1])

    step_time_s = time.time() - start1
    step_sum = model_creation_time + training_time + postproc_time + \
               finish_time + eval_time
    best_it = b_state_dict.get('cum_best_iter', b_state_dict['iter'])
    tune.report(t=t_id,
                best_val=b_state_dict['value'],
                avg_acc_val=avg_val,
                avg_acc_val_so_far=avg_val_so_far,
                avg_acc_test_so_far=avg_test_so_far,
                lca=lca,
                avg_acc_test=avg_test,
                test_acc=evaluation['Test']['accuracy'][t_id],
                duration_seconds=step_time_s,
                duration_iterations=t,
                duration_best_it=best_it,
                duration_finish=finish_time,
                duration_model_creation=model_creation_time,
                duration_training=training_time,
                duration_postproc=postproc_time,
                duration_eval=eval_time,
                duration_sum=step_sum,
                # entropy=stats.pop('entropy'),
                new_params=learner.new_params(t_id),
                total_params=learner.n_params(t_id),
                total_steps=total_steps + t,
                fw_t=round(avg_forward_time * 1000) / 1000,
                data_t=round(avg_data_time * 1000) / 1000,
                epoch_t=round(avg_epoch_time * 1000) / 1000,
                eval_t=round(avg_eval_time * 1000) / 1000,
                total_t=round(total_time * 1000) / 1000,
                env_url=get_env_url(vis_p),
                **accs, **stats)
    return rescaled, t, metrics, b_state_dict, stats


def train_model(model, datasets_p, batch_sizes, optim_fact, prepare_batch,
                task, train_loader, eval_loaders, training_params, config):
    if hasattr(model, 'train_func'):
        assert not config, config
        f = model.train_func
        t, metrics, b_state_dict = f(datasets_p=datasets_p,
                                     b_sizes=batch_sizes,
                                     optim_fact=optim_fact,
                                     # lr_scheduler=lr_scheduler,
                                     # viz=task_vis,
                                     prepare_batch=prepare_batch,
                                     split_names=task['split_names'],
                                     # viz=task_vis,
                                     **training_params)
        rescaled = list(
            filter(lambda itm: 'rescaled' in itm[0], metrics.items()))
        rescaled = rescaled[0][1]
    else:
        optim = optim_fact(model=model)
        if hasattr(model, 'train_loader_wrapper'):
            train_loader = model.train_loader_wrapper(train_loader)
        t, metrics, b_state_dict = train(model, train_loader, eval_loaders,
                                         optimizer=optim,
                                         # lr_scheduler=lr_scheduler,
                                         # viz=task_vis,
                                         prepare_batch=prepare_batch,
                                         split_names=task['split_names'],
                                         # viz=task_vis,
                                         **training_params)
        rescaled = metrics['Val accuracy_0']

    return rescaled, t, metrics, b_state_dict
