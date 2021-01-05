# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
from collections import defaultdict
from functools import partial
from math import pi
from numbers import Number
from operator import itemgetter

import imageio
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import torch
import visdom
from ctrl.commons.utils import hex_to_rgb
from json2html import json2html
from matplotlib import patches
from pandas import DataFrame, MultiIndex
from plotly.express import parallel_coordinates
from sklearn.metrics import auc

from constants import MODEL_ALIASES, MODELS, FILE_NAMES, BASE_RESNET_FLOPS, \
    BATCH_SIZE, BASE_RESNET_PARAMS, M_AVG_ACC, M_FORGETTING, M_COMPUTE, \
    M_MEMORY, M_LCA, M_T_OUT, OUTER_CROSS_VAL, M_T_IN, INCH, BASE_ALEX_FLOPS, \
    RADAR_NAMES, RADAR_ALIASES, BARPLOT_IDS, CNN, LONG_TRAJ_KEEPERS, M_BW
from src.utils.misc import get_env_url, paretize_exp, count_params

logger = logging.getLogger(__name__)

PARA_COORD_MOCK_VALUES = {'dropout': -1}
PARA_COORD_LOG_SCALE = ['optim/lr', 'optim/0/lr', 'optim/1/lr']

REF_WIDTH = 600
REF_HEIGHT = 350
DEFAULT_OPTS = dict(showlegend=True,
                    markersize=9,
                    width=REF_WIDTH,
                    height=REF_HEIGHT,
                    colormap='Alphabet')

COLORS = hex_to_rgb(px.colors.qualitative.Alphabet)


def dd_counter():
    d = defaultdict()
    d.default_factory = d.__len__
    return d


color_gen = defaultdict(dd_counter)


def get_color(win, name):
    return np.array([COLORS[color_gen[win][name] % len(COLORS)]])


def get_env_name(config, _id, main=False):
    assert _id is not None
    env = str(_id)
    if main:
        return env + '_main'
    else:
        return env


    var_domain = config['datasets']['task_gen']['strat']['domain']
    train_samples = config['datasets']['task_gen']['samples_per_class'][0]
    name = f'{var_domain}-{train_samples}'
    # name = '{}'.format(config['experiment']['name'])
    if _id is not None:
        name = f'{name}-{_id}'
    if config['experiment']['single_pass']:
        name = f'{name}-single'
    if main:
        name += '_main'

    return name


def get_aucs(values, normalize, ks=None):
    """
    Get the AUCs for one task
    :param values: dict of format {'split_1' :{x1:y1, x2:y2, ...},
                                   'split_2' :{x1:y1, x2:y2, ...}}
    :param normalize: Will divide auc@k by k if True
    :return: {'split_1' :([x1, x2, ...], [auc@x1, auc@x2, ...]),
              'split_2' :([x1, x2, ...], [auc@x1, auc@x2, ...])}
    """
    res_at = {}
    for split_name, split_metrics in values.items():
        x, y = zip(*split_metrics.items())
        auc_at = ([], [])
        for i in range(1, len(x)):
            if ks and x[i] not in ks:
                continue
            auc_at[0].append(x[i])
            auc_at_i = auc(x[:i + 1], y[:i + 1])
            if normalize:
                auc_at_i /= x[i]
            auc_at[1].append(auc_at_i)
        res_at[split_name] = auc_at
    return res_at


def plot_transfers(all_perfs, similarities, task_names, task_viz,
                   agg_func=np.mean):
    """
    Plot the Final accuracy vs similarity scatter plot using two settings:
        - 'Ideal' where the accuracy used are the ones obtained after training
        on each task (diag of transfer matrix)
        - 'Current' where the accuracy used are the one on all task at current
        timestep.
    :param all_perfs: Nxt transfer matrix where N is the total number of tasks
    and t the number of tasks experienced so far
    :param similarities: array of size N containing the similarity between each
    task and the current one.
    :param task_names:
    :param task_viz:
    :return:
    """
    if not all_perfs:
        return 0, 0
    n_seen_tasks = len(all_perfs)

    prev_perf_after_training = [all_perfs[i][i] for i in range(n_seen_tasks)]
    prev_perf_now = all_perfs[-1][:n_seen_tasks]

    ideal = list(zip(similarities, prev_perf_after_training))
    dists_ideal = torch.tensor(ideal).norm(dim=1).tolist()

    current = list(zip(similarities, prev_perf_now))
    dists_current = torch.tensor(current).norm(dim=1).tolist()
    labels = torch.arange(n_seen_tasks) + 1

    opts = {'markersize': 5, 'legend': task_names, 'xtickmin': 0,
            'xtickmax': 1, 'ytickmin': 0, 'ytickmax': 1,
            'xlabel': 'similarity', 'width': 600}
    task_viz.scatter(ideal, labels,
                     opts={'title': 'Ideal Transfer',
                           'textlabels': dists_ideal,
                           'ylabel': 'acc after training',
                           **opts})
    task_viz.scatter(current, labels, opts={'title': 'Current Transfer',
                                            'textlabels': dists_current,
                                            'ylabel': 'current acc',
                                            **opts})

    return agg_func(dists_ideal), agg_func(dists_current)


def plot_aucs(all_aucs, task_viz, main_vis):
    """
    Update the auc curves:
        - Draw k vs auc@k in the environment of this training for the current
        task
        - Draw k vs average auc@k on all tasks seen by this model so far (in
        the training and global envs).
    :param all_aucs: List containing the auc of a given model for all tasks.
    Each element of this list is in the format
    returned by `get_aucs`: {'split_1' :([x1, x2, ...], [auc@x1, auc@x2, ...]),
                             'split_2' :([x1, x2, ...], [auc@x1, auc@x2, ...]),
                             ...}
    :param task_viz: The Visdom environment of the concerned training (model
    and task).
    :param main_vis: The Visdom environment of the global experiment.
    :return:
    """
    ### Update AUC plots
    opts = {'legend': list(all_aucs[-1].keys()), 'markersize': 3,
            'xlabel': 'n iterations', 'ylabel': 'AuC'}

    # Current task
    all_points, labels = [], []
    last_auc = all_aucs[-1]
    for i, (split, (x, y)) in enumerate(last_auc.items()):
        all_points.extend(zip(x, y))
        labels.extend([i + 1] * len(x))
    task_viz.scatter(all_points, labels, win='task_auc',
                     opts={'title': 'Task AUCs', **opts})

    # All tasks so far
    split_aucs = defaultdict(list)
    for t_aucs in all_aucs:
        for split, (x, y) in t_aucs.items():
            assert x == all_aucs[0][split][
                0], 'All AUC of a split should be computed at the same points'
            split_aucs[split].append(y)

    ys = np.mean(list(split_aucs.values()), 1).reshape(-1)
    xs = all_aucs[0]['Train'][0]
    labels = np.repeat(range(len(split_aucs)), len(xs)) + 1
    xs = np.tile(xs, len(split_aucs))

    task_viz.scatter(np.array([xs, ys]).transpose(), labels, opts={
        'title': 'Average task AUCs {}'.format(len(all_aucs)),
        **opts})
    main_vis.scatter(np.array([xs, ys]).transpose(), labels, win='task_aucs',
                     opts={'title': 'Task AUCs {}'.format(len(all_aucs)),
                           **opts})


def plot_potential_speed(models_aucs, potentials, potential_type, main_vis,
                         model_names, plot_labels, splits):
    """
    Update the AUC vs transfer potential plots.
    :param models_aucs:
    :param potentials:
    :param potential_type:
    :param main_vis:
    :param model_names:
    :param plot_labels:
    :param splits:
    :return:
    """
    aucs_at_x = defaultdict(lambda: defaultdict(list))
    for model_auc, pot, model_name in zip(models_aucs,
                                          potentials,
                                          model_names):
        # model auc is [{split1:([x1...], [y_1...]), split2:([],[])},
        #               {split1:...}, ...]
        for split, (xs, ys) in model_auc.items():
            if splits and split not in splits:
                continue
            for i, (x, y) in enumerate(zip(xs, ys)):
                aucs_at_x[x]['values'].append((pot, y))
                trace_name = '{}_{}'.format(model_name, split)
                aucs_at_x[x]['labels'].append(plot_labels[trace_name])
                aucs_at_x[x]['legend'].append(trace_name)

    for x, d in aucs_at_x.items():
        opts = {'legend': d['legend'], 'markersize': 5,
                'xlabel': 'Transfer Potential', 'ylabel': 'AuC', 'width': 600,
                'height': 400}

        main_vis.scatter(d['values'], np.array(d['labels']) + 1,
                         win='{}_transaccbis{}'.format(potential_type, x),
                         update='append', opts={
                'title': '{} speed transfer@{}'.format(potential_type, x),
                **opts})

    return aucs_at_x


def plot_heatmaps(models, metrics, viz, **kwargs):
    """
    Plot the acc
    :param metrics: list of M matrix of size (N*t), where M is the number of
    models, N the total number of tasks and t
    the number of tasks seen so far.
    :param models: List of the names of the M models
    :param viz: The Visdom env in which the heatmaps will be drawn.
    :return:
    """
    kwargs['xlabel'] = kwargs.get('xlabel', 'Task seen')
    kwargs['ylabel'] = kwargs.get('ylabel', 'Task perf')

    for name, model_perfs in zip(models, metrics):
        opts = kwargs.copy()
        opts['title'] = opts.get('title', '{} transfer matrix'.format(name))
        if not torch.is_tensor(model_perfs):
            model_perfs = torch.tensor(model_perfs)
        viz.heatmap(model_perfs.t(), win='{}_heatmap'.format(name), opts=opts)


def plot_speeds(val, task, name, viz, first):
    new_point = [(task, val)]
    update_scatter(new_point, name, viz, 'tasks_speeds', first, 'N task seen',
                   'n iterations to train', 'Training Durations')


def plot_best_speeds(val, task, name, viz, first):
    new_point = [(task, val)]
    update_scatter(new_point, name, viz, 'tasks_best_speeds', first,
                   'N task seen', 'n iterations to find the best model',
                   'Convergence Durations')


def plot_times(val, task, name, viz, first):
    new_point = [(task, val)]
    update_scatter(new_point, name, viz, 'tasks_times', first, 'N task seen',
                   'Time (s) to find the model',
                   'Time taken by the learner to give the model')

def plot_total_params(val, task, name, viz, first):
    new_point = [(task, val)]
    update_scatter(new_point, name, viz, 'total_params', first, 'N task seen',
                   '# params', 'Total # of params used by the learner')

def plot_new_params(val, task, name, viz, first):
    new_point = [(task, val)]
    update_scatter(new_point, name, viz, 'new_params', first, 'N task seen',
                   '# params', '# of new params introduced')

def plot_lca(val, task, name, viz, first, n=5):
    new_point = [(task, val)]
    update_scatter(new_point, name, viz, 'lca_{}'.format(n), first, 'Task id',
                   'lca@{}'.format(n),
                   'Area under the learning curve after {} batches'.format(n))


def plot_creation_times(val, task, name, viz, first):
    new_point = [(task, val)]
    update_scatter(new_point, name, viz, 'tasks_creation_times', first,
                   'N task seen', 'Time (s) to create the model',
                   'Time taken by the learner to instantiate the new model')

def plot_finish_times(val, task, name, viz, first):
    new_point = [(task, val)]
    update_scatter(new_point, name, viz, 'tasks_finish_times', first,
                   'N task seen', 'Time (s) to finish after training',
                   'Time taken by the learner to finalize the training')


def plot_accs(val, task, name, viz, first):
    new_point = [(task, val)]
    update_scatter(new_point, name, viz, 'tasks_acc', first, 'Task id',
                   'Test acc', 'Learning_accuracies')


def update_scatter(new_points, name, viz, win, first, x, y, title, lines=True):
    symbol = 'diamond' if 'search' in name.lower() else \
        None if 'pssn' in name.lower() else 'cross'
    viz.scatter(new_points, win=win, name=name,
                update=None if first else 'append',
                opts={**DEFAULT_OPTS, 'xlabel': x,
                      'ylabel': y, 'title': title,
                      'markersymbol': symbol,
                      'linecolor': get_color(win, name),
                      'mode': 'lines+markers' if lines else 'markers'})


def plot_accs_data(all_accs, model_names, n_samples, viz):
    task_id = len(all_accs[0]) - 1
    last_task_accs = torch.tensor(all_accs)[:, -1, task_id]

    new_points = list(
        (n_samples[task_id], acc.item()) for acc in last_task_accs.unbind())
    labels = list(range(1, len(model_names) + 1))
    viz.scatter(new_points, labels, win='accdatatasks',
                update='append' if len(all_accs[0]) > 1 else None,
                opts={'legend': list(model_names), 'markersize': 5,
                      'xlabel': 'n samples per class',
                      'ylabel': 'Task test accuracy', 'width': 600,
                      'height': 400, 'title': 'Learning accuracies wrt data '
                                              'quantity'})


def plot_speed_vs_tp(training_times, potentials, potential_type, model_names,
                     viz):
    new_point = [[potential[-1], times[-1]] for times, potential in
                 zip(training_times, potentials)]
    labels = list(range(1, len(model_names) + 1))
    viz.scatter(new_point, labels,
                update='append' if len(training_times[0]) > 1 else None,
                win='{}_speeds_tp'.format(potential_type),
                opts={'legend': list(model_names), 'markersize': 5,
                      'xlabel': 'Transfer Potential',
                      'ylabel': 'Time to converge', 'width': 600,
                      'height': 400,
                      'title': '{} speed-TP'.format(potential_type)})


def plot_corr_coeffs(all_aucs, potentials, model_names, viz):
    all_potential_auc_at_k = defaultdict(lambda: defaultdict(list))
    for model_aucs, model_potentials, model_name in zip(all_aucs, potentials,
                                                        model_names):
        for task_aucs, potential in zip(model_aucs, model_potentials):
            for split, (xs, ys) in task_aucs.items():
                name = '{}-{}'.format(model_name, split)
                for k, auc_at_k in zip(xs, ys):
                    all_potential_auc_at_k[k][name].append(
                        (potential, auc_at_k))

    corr_coeffs = defaultdict(dict)

    for k, trace in all_potential_auc_at_k.items():
        for trace_name, trace_values in trace.items():
            corr = np.corrcoef(trace_values, rowvar=False)
            corr_coeffs[k][trace_name] = corr if isinstance(corr, Number) \
                                              else corr[0, 1]

    names = list(list(corr_coeffs.values())[0].keys())
    ks = ['@{}'.format(k) for k in corr_coeffs.keys()]
    corr_coeffs = [list(vals.values()) for vals in corr_coeffs.values()]
    viz.heatmap(corr_coeffs, opts={'columnnames': names, 'rownames': ks,
                                   'title': 'TP-AUC correlation after {}'
                                            'tasks'.format(len(all_aucs[0]))})

    return all_potential_auc_at_k


def plot_res_dataframe(analysis, plot_name, best_point, viz, epoch_key, it_key,
                       y_keys, width=1500, height=500):
    logdir_to_trial = {t.logdir: t for t in analysis.trials}
    res_df = pd.concat([t[y_keys] for t in
                        analysis.trial_dataframes.values()], axis=1)

    cols = []
    for logdir in analysis.trial_dataframes.keys():
        for key in y_keys:
            tag = logdir_to_trial[logdir].experiment_tag
            cols.append('{}-{}'.format(key, tag))

    res_df.columns = cols

    longest_df = max(analysis.trial_dataframes.values(), key=len)
    x_epochs = longest_df[epoch_key]
    x_iterations = longest_df[it_key]

    y_label = str(y_keys)
    _plot_df(x_epochs, res_df, viz, plot_name, width, height, y_label)
    _plot_df(x_iterations, res_df, viz, plot_name, width, height, y_label,
             best_point)


def _plot_df(x_series, df, viz, plot_name, width, height, y_label,
             best_point=None):
    assert len(x_series) == len(df)

    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Scatter(x=x_series.values, y=df[col], name=col,
                                 line_width=1))
    fig.layout.title.text = plot_name
    fig.layout.xaxis.title.text = x_series.name
    fig.layout.yaxis.title.text = y_label
    if best_point:
        fig.add_trace(go.Scatter(x=[best_point[0]],
                                 y=[best_point[1]],
                                 mode='markers',
                                 name='best checkpoint',
                                 marker_color='red',
                                 marker_size=5))
    win = viz.plotlyplot(fig)
    viz.update_window_opts(win=win, opts=dict(width=width, height=height,
                                              showlegend=True))


def plot_tasks_env_urls(urls, viz, name='all'):
    all_models_urls = []
    for model, env_urls in urls.items():
        links = []
        for url in env_urls:
            links.append('<a href="{}">{}</a>'.format(url, url))
        model_urls = '<br/>'.join(links)
        all_models_urls.append('{}:<br/>{}'.format(model, model_urls))

    all_urls = '<br/><br/>'.join(all_models_urls)
    viz.text(all_urls, win='urls{}'.format(name),
             opts={'width': 3*REF_WIDTH, 'height': REF_HEIGHT})


def update_acc_plots(all_perfs, name, viz):
    n_tasks = len(all_perfs)
    current_accuracies = all_perfs[-1]
    accuracies_when_seen = [all_perfs[i][i] for i in range(n_tasks)]

    mean_perf_on_all_tasks = np.mean(current_accuracies)
    mean_perf_on_seen_tasks = np.mean(current_accuracies[:n_tasks])
    mean_perf_when_seen = np.mean(accuracies_when_seen)
    viz.line(Y=[[mean_perf_on_all_tasks],
                [mean_perf_on_seen_tasks],
                [mean_perf_when_seen]],
             X=[n_tasks],
             win='{}aggrac'.format(name),
             update='append' if n_tasks > 1 else None,
             opts={'title': '{} Average Test Accuracies'.format(name),
                   'xlabel': 'N tasks seen',
                   'ylabel': 'Accuracy',
                   'width': 600,
                   'height': 400,
                   'legend': ['Current acc on all tasks',
                              'Current acc on seen tasks',
                              'Acc on tasks when seen']})


def update_avg_acc_old(all_accs, model_names, viz, title):
    n_tasks = len(all_accs[0])
    acc_when_seen = [[mod_acc[i][i] for i in range(n_tasks)] for mod_acc in
                     all_accs]
    mean_accs = np.mean(acc_when_seen, axis=1, keepdims=True)
    viz.line(Y=mean_accs,
             X=[n_tasks], win='mean_accs_{}'.format(title),
             update='append' if n_tasks > 1 else None,
             opts={'title': title,
                   'xlabel': 'N tasks seen',
                   'ylabel': 'Avg Accuracy',
                   'width': 600,
                   'height': 400,
                   'legend': model_names})
    return mean_accs.squeeze(-1)


def update_avg_acc(acc, x, name, viz, first, title):
    viz.line(Y=[acc],
             X=[x], win='mean_accs{}'.format(title),
             update=None if first else 'append',
             name=name,
             opts={'title': title,
                   'showlegend': True,
                   'xlabel': 'N tasks seen',
                   'ylabel': 'Avg Accuracy',
                   'width': REF_WIDTH,
                   'height': REF_HEIGHT,
                   })


def update_speed_plots(new_speed, new_x, name, viz, first):
    viz.line(Y=[new_speed],
             X=[new_x],
             win='meanspeed',
             update=None if first else 'append',
             name=name,
             opts={'title': 'Average Training Duration',
                   'showlegend': True,
                   'xlabel': 'N tasks seen',
                   'ylabel': 'Average N iterations to converge',
                   'width': REF_WIDTH,
                   'height': REF_HEIGHT,
                   })


def update_avg_lca(new_lca, new_x, name, viz, first, n=5):
    viz.line(Y=[new_lca],
             X=[new_x],
             win='mean_lca_{}'.format(n),
             update=None if first else 'append',
             name=name,
             opts={'title': 'Average Area under the Learning Curve after {} '
                            'batches'.format(n),
                   'showlegend': True,
                   'xlabel': 'N tasks seen',
                   'ylabel': 'lca@{}'.format(n),
                   'width': REF_WIDTH,
                   'height': REF_HEIGHT,
                   })

def update_rescaled(y, x, name, viz, first):
    viz.line(y, X=x, win='rescaled',
             update=None if first else 'append', name=name,
             opts={'title': 'Rescaled accuracy',
                   'showlegend': True,
                   'width': 3*REF_WIDTH,
                   'height': 3*REF_HEIGHT,
                   'xlabel': 'iterations',
                   'ylabel': 'acc'})


def update_summary(summary, viz, win='global', w_scale=1, h_scale=1):
    if not isinstance(summary, pandas.DataFrame):
        summary = pandas.DataFrame(summary)
    data = summary.round(3).to_dict('records')
    # else:
    #     data = []
    #     for items in zip(*summary.values()):
    #         row = {}
    #         for k, val in zip(summary.keys(), items):
    #             row[k] = "{:4.3f}".format(val) if isinstance(val, float) else val
    #         data.append(row)

    html_summary = json2html.convert(data, escape=False,
                                     table_attributes="id=\"info-table\" "
                                                      "class=\"table "
                                                      "table-bordered"
                                                      " table-hover\"")
    viz.text(html_summary, win=win,
             opts={'title': '{} summary'.format(win),
                   'width': REF_WIDTH * w_scale,
                   'height': REF_HEIGHT * h_scale})


def update_ents(model, value, task_id, viz):
    viz.line(Y=[value],
             X=[task_id],
             win='entropies',
             # title='Entropies',
             update='append',
             name=model,
             opts={'title': 'Average entropy',
                   'xlabel': 'N tasks seen',
                   'ylabel': 'Average entropy',
                   'width': 600,
                   'height': 400,
                   'showlegend': True
                   })


def write_text(txt, viz, name=None):
    viz.text('<pre>{}</pre>'.format(txt), win='txt{}'.format(name),
                       opts={'width': REF_WIDTH, 'height': REF_HEIGHT})


def update_plots(learner_name, task_id, main_vis, task_viz, first, all_perfs,
                 avg_speed, avg_acc_t, avg_acc_now, summaries, durations,
                 entropy, confusion, task_test, params, lca, avg_lca):
    first = False
    ###
    # Plot
    ###
    if task_viz:
        plot_heatmaps([learner_name], [all_perfs], task_viz)
        # categories = list(
        #     map(str, self.task_gen.task_pool[task_id].src_concepts))
        current_task_confusion = confusion[task_id]
        plot_heatmaps([learner_name], [current_task_confusion], task_viz,
                      title='Confusion matrix', width=600, height=600,
                      xlabel='Predicted category',
                      ylabel='Real category',
                    # rownames=categories,
                    # columnnames=categories
                    )
    if all_perfs is not None:
        update_acc_plots(all_perfs, learner_name, main_vis)
        plot_heatmaps([learner_name], [all_perfs], main_vis)

    if entropy is not None:
        update_ents(learner_name, entropy, task_id, main_vis)

    # for k in stats.keys():
    #     logger.warning('/!\ Statistics {} of model {} not logged'
    #                    .format(k, learner_name))

    n_tasks_seen = task_id + 1
    update_speed_plots(avg_speed, n_tasks_seen, learner_name, main_vis, first)
    update_avg_acc(avg_acc_t, n_tasks_seen, learner_name, main_vis, first,
                   'Average Accuracies when seen')
    update_avg_acc(avg_acc_now, n_tasks_seen, learner_name, main_vis, first,
                   'Average Accuracies now')

    for name, summary in summaries.items():
        update_summary(summary, main_vis, name)

    # update_summary(self.summary, main_vis)
    # update_summary(self.param_summary, main_vis, 'parameters')

    plot_accs(task_test, n_tasks_seen, learner_name, main_vis, first)
    if durations:
        new_speed = durations['iterations']
        plot_speeds(new_speed, n_tasks_seen, learner_name, main_vis, first)
        new_speed = durations['best_iterations']
        plot_best_speeds(new_speed, n_tasks_seen, learner_name, main_vis,
                         first)
        new_time = durations['seconds']
        plot_times(new_time, n_tasks_seen, learner_name, main_vis, first)
        new_finish_time = durations['finish']
        plot_finish_times(new_finish_time, n_tasks_seen, learner_name,
                          main_vis, first)
        if 'model_creation' in durations:
            new_creation_time = durations['model_creation']
            plot_creation_times(new_creation_time, n_tasks_seen, learner_name,
                             main_vis, first)

    if params:
        plot_total_params(params['total'], n_tasks_seen, learner_name,
                          main_vis, first)
        plot_new_params(params['new'], n_tasks_seen, learner_name, main_vis,
                          first)
    if lca:
        plot_lca(lca, n_tasks_seen, learner_name, main_vis, first)
        update_avg_lca(avg_lca, n_tasks_seen, learner_name, main_vis, first)


def update_pareto(xs, ys, label, viz, first, orig, plot_name='clean',
                  x_name='params'):
    points = list(zip(xs, ys))
    # symbol = None if 'pssn' in label.lower() else 'cross'
    symbol = 'diamond' if 'search' in label.lower() else \
        None if 'pssn' in label.lower() else 'cross'
    win = 'pareto_{}'.format(plot_name)
    viz.scatter(points, name=str(label), win=win,
                update=None if first else 'append',
                opts={**DEFAULT_OPTS,
                      'markersymbol': symbol,
                      'xlabel': 'Total # {}'.format(x_name),
                      'ylabel': 'Test acc',
                      'linecolor': get_color(win, str(label)),
                      'title': '{} Cost/Accuracy trade-off'.format(plot_name),
                      'textlabels': orig,
                      'mode': 'markers'})


def plot_grid_result(df, color, title, viz):
    fig = parallel_coordinates(df, color=color, title=title,
                               width=2*REF_WIDTH, height=2*REF_HEIGHT)
    viz.plotlyplot(fig, opts={'width': 2*REF_WIDTH,
                                    'height': 2*REF_HEIGHT})


def plot_svg(svg, name, viz):
    viz.svg(svg, opts=dict(title=name, width=REF_WIDTH, height=2*REF_HEIGHT))


def graph_to_svg(g, save_path=None, png=False, scat_fact=1, edge_labels=None,
                 show_labels=True, penwidth=True):
    if True:
        return b'<svg SKIP graph plotting></svg>'
    edge_labels = edge_labels or {}
    graph = g.copy()
    max_x = 0
    max_y = 0
    out_nodes = []
    for n in g.nodes:
        if isinstance(n, tuple) and len(n) == 2:
            if isinstance(n[1], int):
                graph.node[n]['pos'] = '"%d,%d!"' % (n[0] * scat_fact,
                                                     n[1] * scat_fact)
                max_x = max(max_x, n[0])
                max_y = max(max_y, n[1])
            elif isinstance(n[1], str) and n[1].startswith('IN'):
                graph.node[n]['pos'] = '"%d,%d!"' % (n[0] * scat_fact,
                                                     -scat_fact)
            elif isinstance(n[1], str) and n[1].startswith('OUT'):
                out_nodes.append(n)
        elif isinstance(n, tuple) and len(n) > 2:
            node = graph.nodes[n]
            p = graph.pred[n]
            s = graph.succ[n]
            assert len(p) == 1 and len(s) == 1
            p = list(p)[0]
            s = list(s)[0]
            if count_params(node['module'])['total'] > 0:
                style = 'solid'
            else:
                style = 'dashed'
            graph.add_edge(p, s, style=style, color=node.get('color', 'black'))
            if edge_labels.get(n) is not None:
                if show_labels:
                    graph.edges[p, s]['label'] = edge_labels.get(n)
                if penwidth:
                    graph.edges[p, s]['penwidth'] = float(edge_labels.get(n))

            graph.remove_node(n)
        else:
            raise ValueError('Nodes should be tupes of len at least 2, '
                             'got {}'.format(n))
    for n in out_nodes:
        graph.node[n]['pos'] = '"%d,%d!"' % (n[0] * scat_fact,
                                             (max_y + 1) * scat_fact)
    p = nx.drawing.nx_pydot.to_pydot(graph)
    if save_path:
        p.write_png(save_path)
    # p.write_png('b.png')
    svg = p.create_svg(prog=['fdp', '-n'])
    if png == True:
        png = p.create_png(prog=['fdp', '-n'])
        return svg, png
    else:
        return svg


def plot_trajectory(graph, weights_traj, node_ids, viz):
    if True:
        return
    first = True
    for it, weights in weights_traj.items():
        weights = weights.tolist()
        weights_dict = {n: '{:.2f}'.format(weights[idx])
                        for n, idx in node_ids.items()}
        _, img_bytes = graph_to_svg(graph, png=True, edge_labels=weights_dict)
        img_arr = imageio.imread(img_bytes).transpose(2, 0, 1)[:3, ...]
        viz.image(img_arr, win='traj_all', update=not first,
                  opts=dict(store_history=True, jpgquality=1))
        _, img_bytes = graph_to_svg(graph, png=True, edge_labels=weights_dict,
                                    show_labels=False)
        img_arr = imageio.imread(img_bytes).transpose(2, 0, 1)[:3, ...]
        viz.image(img_arr, win='traj_width', update=not first,
                  opts=dict(store_history=True, jpgquality=1))
        first = False


def list_top_archs(archs, viz):
    res = defaultdict(list)
    for i, (k, v) in enumerate(archs.items()):
        res['top {}'.format(i)].append('{:.5f}'.format(v))
        for node in k:
            res['top {}'.format(i)].append(str(node))
    update_summary(res, viz, 'archs')
    #
    #
    # archs = OrderedDict((str(k), v) for k, v in archs.items())
    # repr = pformat(archs)
    # print("teteetetxt")
    # print(repr)
    # viz.text('<pre>{}</pre> LOOOOOOOOOOOOOOOOL'.format(repr), win='viz_archs',
    #          opts={'width': REF_WIDTH, 'height': REF_HEIGHT})


def list_arch_scores(scores, viz):
    viz.text('<pre>{}</pre>'.format(scores), win='viz_archs',
             opts={'width': REF_WIDTH, 'height': REF_HEIGHT})


def plot_para_coord(params, criterion, title, viz):
    para_coord_dict = defaultdict(list)
    for params, perf in zip(params, criterion):
        para_coord_dict['Val acc'].append(perf)
        for k, v in params.items():
            if v is None:
                if k not in PARA_COORD_MOCK_VALUES:
                    raise ValueError('Don\'t know how to replace {} for'
                                     'key {}'.format(v, k))
                else:
                    v = PARA_COORD_MOCK_VALUES[k]
            elif isinstance(v, bool):
                v = int(v)
            elif k in PARA_COORD_LOG_SCALE:
                k = 'log({})'.format(k)
                v = np.log10(v)
            para_coord_dict[k].append(v)

    df = DataFrame(para_coord_dict)

    plot_grid_result(df, color='Val acc', title=title, viz=viz)


def process_final_results(main_vis, res_dict, exp_name, visdom_conf,
                          task_envs_str, n_task,
                          best_task_envs_str, simplify_pareto=True,
                          traces_folder=None, plot=True):

    global_summary = defaultdict(list)
    first_plot = True
    for ll_name, (best_traj, exp_summary) in res_dict.items():
        if plot:
            exp_env = '{}_{}'.format(exp_name, ll_name)
            if traces_folder is not None:
                log_file = '{}/{}'.format(traces_folder, exp_env)
            else:
                log_file = None
            exp_viz = visdom.Visdom(env=exp_env, log_to_filename=log_file,
                                    **visdom_conf)
            env_url = get_env_url(exp_viz)
            task_envs_str[ll_name].append(env_url)

            update_summary(exp_summary, main_vis, ll_name)

            val_accs_detailed_summary = defaultdict(list)
            val_accs_detailed_summary['Tag'] = exp_summary['model']
            for trial_accs in exp_summary['Acc Val']:
                for i, acc in enumerate(trial_accs):
                    val_accs_detailed_summary[f'T{i}'].append(acc)

            update_summary(val_accs_detailed_summary, exp_viz, ll_name + 'vaccs')

            parto_mem = paretize_exp(exp_summary, 'Params', 'Avg acc Val',
                                     ['Avg acc Test', 'model', 'paths'], )
            if simplify_pareto:
                parto_mem = {k: v[-1:] for k, v in parto_mem.items()}

            update_pareto(exp_summary['Params'].tolist(),
                          exp_summary['Avg acc Test'].tolist(),
                          ll_name, main_vis, first_plot,
                          exp_summary['model'].tolist(), 'All')
            update_pareto(parto_mem['Params'], parto_mem['Avg acc Test'],
                          ll_name, main_vis, first_plot,
                          parto_mem['model'])

            update_pareto(exp_summary['Steps'].tolist(),
                          exp_summary['Avg acc Test'].tolist(),
                          ll_name, main_vis, first_plot,
                          exp_summary['model'].tolist(), 'Steps', 'steps')
            pareto_steps = paretize_exp(exp_summary, 'Steps', 'Avg acc Val',
                                        ['Avg acc Test', 'model', 'paths'])
            if simplify_pareto:
                pareto_steps = {k: v[-1:] for k, v in pareto_steps.items()}
            update_pareto(pareto_steps['Steps'], pareto_steps['Avg acc Test'],
                          ll_name, main_vis, first_plot,
                          pareto_steps['model'], 'Steps_clean', 'steps')

        all_test_accuracies = []
        sum_acc_t = 0
        sum_durations = 0
        sum_lca = 0
        if not len(best_traj) == n_task and n_task is not None:
            logger.warning('There was an issue with the results, revieved '
                           '{} results while the stream contains {} tasks.'
                           .format(len(best_traj), n_task))
            raise RuntimeError
        for t_id, result in best_traj.iterrows():
            # for eval_t in range(t_id + 1):
            arr = [result['Test_T{}'.format(eval_t)]
                   for eval_t in range(len(best_traj))]
            all_test_accuracies.append(arr)

            durations = {'iterations': result['duration_iterations'],
                         'finish': result['duration_finish'],
                         'seconds': result['duration_seconds'],
                         'best_iterations': result['duration_best_it']
                         }
            if 'duration_model_creation' in result:
                durations['model_creation'] = result['duration_model_creation'],
            params = {'total': result['total_params'],
                      'new': result['new_params']}
            sum_acc_t += result['test_acc']
            sum_durations += result['duration_best_it']
            sum_lca += result['lca']
            avg_duration = sum_durations / (t_id + 1)
            avg_acc_t = sum_acc_t / (t_id + 1)
            avg_lca = sum_lca / (t_id + 1)
            if plot:
                update_plots(ll_name, t_id, main_vis, None, False,
                             all_test_accuracies, avg_duration,
                             avg_acc_t, result['avg_acc_test'], {},
                             durations, result.get('entropy'), None,
                             result['test_acc'], params, result['lca'],
                             avg_lca)
                # if isinstance(best_ll_model, ProgressiveSSN):
                #     for i, trial_tag in enumerate(parto_mem['model']):
                #         tag = trial_tag.split('_')[0]
                #         env = '{}_Pareto_{}-{}_T{}'.format(self.exp_name, ll_name,
                #                                     tag, t_id)
                #         log_file = '{}/{}'.format(self.visdom_traces_folder,
                #                                   env)
                #         viz = visdom.Visdom(env=env, log_to_filename=log_file,
                #                             **self.visdom_conf)
                #         trial_path = parto_mem['paths'][i]
                #         viz.text('<pre>{}</pre>'.format(trial_tag))
                #         self.task_envs_str[ll_name].append(
                #             get_env_url(viz))
                #         files = ['trained', 'pruned', 'cleaned',
                #                  'full', 'newget']
                #         for f in files:
                #             file = path.join(trial_path, 'model_T{}_{}.svg'
                #                              .format(t_id, f))
                #             if path.isfile(file):
                #                 plot_svg(str(open(file).readlines()), f, viz)
                task_envs = exp_summary['envs']
                for trial_envs in task_envs:
                    params = {**visdom_conf, 'env': trial_envs[t_id]}
                    task_envs_str[ll_name].append(get_env_url(params))

                best_task_envs_str[ll_name].append(result['env_url'])
            ### Update task plots

        global_summary['model'].append(ll_name)
        global_summary['speed'].append(avg_duration)
        global_summary['LCA'].append(avg_lca)
        global_summary['Acc now'].append(result['avg_acc_test'])
        global_summary['Acc t'].append(avg_acc_t)
        global_summary['Params'].append(result['total_params'])
        global_summary['Steps'].append(result['total_steps'])
        update_summary(global_summary, main_vis)

        # best_ll_model = torch.load(path.join(exp_summary['paths'][0],
        #                                      'learner.pth'))
        #
        # # if isinstance(best_ll_model, ProgressiveSSN):
        # if 'ProgressiveSSN' in type(best_ll_model).__name__:
        #     for t_id, _ in best_traj.iterrows():
        #         viz_params = training_envs[t_id][ll_name]
        #         viz = visdom.Visdom(**viz_params)
        #         best_model = best_ll_model.get_model(t_id)
        #         if 'ZeroModel' in type(best_model).__name__:
        #             continue
        #         svg = graph_to_svg(best_model.get_graph())
        #         viz.svg(svgstr=str(svg),
        #                 win='best_{}'.format(t_id),
        #                 opts=dict(title='best_{}'.format(t_id)))

        if plot:
            plot_para_coord(exp_summary['evaluated_params'],
                            exp_summary['Avg acc Val'],
                            ll_name,
                            exp_viz)

            first_plot = False
    return global_summary


# MODEL_ALIASES = defaultdict(lambda x: x)

# MODELS = sorted(MODELS)
# raise ValueError(MODELS)


def _get_steps(best_traj, ):
    steps = best_traj['duration_best_it'].sum()
    if '/env/2527_Trial' in best_traj['env_url'][0] \
            and 'PSSN-search-6-fw' in best_traj['env_url'][0]:
        assert steps == 2570, steps
        steps = 7 * steps
    if '/env/2492_Trial' in best_traj['env_url'][0] \
            and 'PSSN-search-6-fw' in best_traj['env_url'][0]:
        assert steps == 30530, steps
        steps = 4 * steps
    return steps


def _get_macs(best_traj):
    env_string = best_traj['env_url'][0]
    print(env_string)
    if 'ewc-online' in env_string:
        print('EWC-O')
        steps_first = best_traj['duration_best_it'][0]
        macs = steps_first * BASE_RESNET_FLOPS * BATCH_SIZE

        remaining_steps = sum(best_traj['duration_best_it'][1:])
        assert remaining_steps + steps_first == _get_steps(best_traj)
        step_macs = BATCH_SIZE * BASE_RESNET_FLOPS + 2 * BASE_RESNET_PARAMS
        macs += remaining_steps * step_macs
    elif 'ewc-full' in env_string:
        print('EWC-F')
        macs = 0
        for i, iters in enumerate(best_traj['duration_best_it']):
            step_macs = BATCH_SIZE * BASE_RESNET_FLOPS \
                        + i * 2 * BASE_RESNET_PARAMS
            macs += iters * step_macs
    elif 'PNN' in env_string:
        print('PNN')
        macs = 0
        for i, iters in enumerate(best_traj['duration_best_it']):
            macs += iters * (2 * i + 1) * BASE_RESNET_FLOPS * BATCH_SIZE
    elif 'er-ring' in env_string or 'er-reservoir' in env_string:
        print('ER')
        steps = _get_steps(best_traj)
        macs = steps * BASE_RESNET_FLOPS * 2 * BATCH_SIZE
    elif 'wider-hat' in env_string:
        steps = _get_steps(best_traj)
        macs = steps * BASE_ALEX_FLOPS[6.5] * BATCH_SIZE
    elif 'wide-hat' in env_string:
        steps = _get_steps(best_traj)
        macs = steps * BASE_ALEX_FLOPS[3.2 if CNN else 4] * BATCH_SIZE
    elif 'hat' in env_string or 'Alex' in env_string:
        steps = _get_steps(best_traj)
        macs = steps * BASE_ALEX_FLOPS[1] * BATCH_SIZE
    else:
        print('Other')
        steps = _get_steps(best_traj)
        macs = steps * BASE_RESNET_FLOPS * BATCH_SIZE
    return macs


def _get_results(best_traj, tasks, mod_summary):
    tasks, metrics = tasks
    res = {}
    if tasks:
        t1 = tasks[0]
        first_t = t1 + 1
        acc_t1 = best_traj[f'Test_T{t1}'][t1]
        res[f'Acc $T_{first_t}$'] = np.round(acc_t1, 2)
        for i, t2 in enumerate(tasks[1:], 1):
            t_alias = 'T_{}{}'.format(first_t, "'" * i)
            acc_t2 = best_traj[f'Test_T{t2}'][t2]
            res[f'Acc ${t_alias}$'] = np.round(acc_t2, 2)
            diff = np.round(acc_t2 - acc_t1, 2)
            res[f'$\Delta_{{T_{first_t}, {t_alias}}}$'] = diff
    if M_BW in metrics:
        l = len(best_traj['Test_T0'])
        res[M_BW] = best_traj['Test_T0'][l-1] - best_traj['Test_T0'][0]
        # res[M_BW+'2'] = best_traj['Test_T0'][-1] - best_traj['Test_T0'][0]

    for met in metrics:
        if met == M_AVG_ACC:
            res[M_AVG_ACC] = np.round(best_traj['avg_acc_test'].iloc[-1],
                                          2)
        elif met == M_FORGETTING:
            acc_t = mod_summary['Acc t']
            res[met] = np.round(res[M_AVG_ACC] - acc_t, 2)

        # elif met == 'Steps':
        #     steps = _get_steps(best_traj)
        #     res[met] = np.round(steps / 1.e3, 1)
        elif met == M_COMPUTE:
            res[met] = int(np.round(2*_get_macs(best_traj)/1.e12))
        elif met == M_MEMORY:
            params = best_traj['total_params'].iloc[-1]
            mem = np.round(4*params/1.e6, 1)
            res[met] = mem
        elif met == M_LCA:
            res[met] = np.round(best_traj['lca'].mean(), 2)
        else:
            logger.warning(f'Unknown metric {met}')
    return res


def fill_result(best_traj, tasks, table, mod_summary):
    res = _get_results(best_traj, tasks, mod_summary)
    for k, v in res.items():
        table[k].append(v)


def fill_ph(tasks, table, ph=float('nan')):
    tasks, metrics = tasks
    if tasks:
        first_t = tasks[0] + 1
        table[f'Acc $T_{first_t}$'].append(ph)
        for i, t2 in enumerate(tasks[1:], 1):
            t_alias = 'T_{}{}'.format(first_t, "'" * i)
            table[f'Acc ${t_alias}$'].append(ph)
            table[f'$\Delta_{{T_{first_t}, {t_alias}}}$'].append(ph)
    for met in metrics:
        table[met].append(ph)


def normalize_table(table, ref, columns):
    if ref not in table['Model']:
        logger.warning(f'Trying to normalize by {ref}, but not present in table !')
        return
    idx = table['Model'].index(ref)
    for prop in columns:
        column = table[prop]
        ref = column[idx]
        if ref is float('nan'):
            logger.warning(f'No value for column {prop}')
            continue
        table[prop] = [f'{n} ({(n/ref)}x)' for n in column]


def plot_delta_table(tasks, res, global_summary, vis):
    all_models = _remove_ducplicates(MODELS + list(res.keys()))

    table = defaultdict(list)
    for ll_name in all_models:
        table['Model'].append(MODEL_ALIASES[ll_name] if ll_name in MODEL_ALIASES else ll_name)
        if ll_name in res:
            best_traj, _ = res.get(ll_name)
            mod_summary = global_summary.loc[ll_name]
            fill_result(best_traj, tasks, table, mod_summary)
        else:
            fill_ph(tasks, table)

    df = pandas.DataFrame(table).set_index('Model')
    # df = df.sort_values('Model')
    html = df.to_html(classes=['table', 'table-bordered', 'table-hover'],
                      na_rep='-')
    latex = df.to_latex(index=True, bold_rows=True, escape=False, na_rep='-')
    vis.text(html)
    vis.text(f'<pre>{latex}</pre>')
    print(latex)
    return df


def get_agg_results(results_df, std=True, string=True):
    strats = set(map(lambda x: f'{x["strat"]} {x["n_tasks"]}',
                     results_df.values()))
    if len(strats) == 0:
        logger.warning('No results to show')
        return
    elif len(strats) > 1:
        logger.warning("Got results from differrent strats, can't " \
                             "aggregate that {}".format(strats))
        return

    cat = pd.concat(map(itemgetter('df'), results_df.values()),
                    keys=results_df.keys())
    exp_ids = defaultdict(list)
    for k, v in results_df.items():
        exp_ids[v['seed']].append(k)
    # exp_ids = set(cat.index.get_level_values(0))

    groups = cat.groupby(by='Model')
    # std = False
    if std and string:
        res = groups.agg(aggreg_row_std_str)
    elif std:
        res = groups.agg([aggreg_row_mean, aggreg_row_er])
    else:
        res = groups.agg(aggreg_row_simple)
        # res = groups.agg(['mean', 'std'])

    return exp_ids, res, strats.pop()
    # print(res)
    # res = res.groupby(level=[0, 1]).apply(
    #     lambda x: x.round(3).astype(str).apply('±'.join, 0))
    # print(res)
    # for k, v in results_df.items():
    #     print(k, v)


def aggreg_row_simple(row, ref='independent', columns=['Steps (K)']):
    if row.dropna().empty:
        return float('nan')
    else:
        return f'{np.mean(row.dropna().values).round(3)}'

def aggreg_row_mean(row, ref='independent', columns=['Steps (K)']):
    if row.dropna().empty:
        return float('nan')
    else:
        return np.mean(row.dropna().values)


def aggreg_row_er(row):
    if row.dropna().empty:
        return float('nan')
    else:
        return np.std(row.dropna().values) / len(row.dropna())

def aggreg_row_std_str(row):
    if row.dropna().empty:
        return float('nan')
    else:
        return f'{np.mean(row.dropna().values).round(2)} ' \
               f'$\pm$ {(np.std(row.dropna().values) / len(row.dropna())).round(2)} '\
               # f'({len(row.dropna())})'


def _remove_ducplicates(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]


def normalize_df(df, offset=.2, inv=None):
    if inv is None:
        inv = []
    for col in inv:
        if col in df:
            df[col] = -df[col]
    df -= df.min()
    df /= df.max()
    df = df * (1-offset) + offset
    return df


def radar_chart(full_res, multi=True, rows=None, columns=None):
    if rows is None:
        rows = full_res.index
    plot_df = full_res[columns].loc[rows]
    # plot_df.columns = plot_df.columns.droplevel(0)
    plot_df = normalize_df(plot_df, inv=[M_COMPUTE, M_MEMORY])
    # plot_df = plot_df.transpose().reset_index()

    plot_df = plot_df.fillna(value=1)
    if multi:
        res = radar_chart_matplotlib_multi(plot_df)
    else:
        res = radar_chart_matplotlib_simple(plot_df)
    res.show()
    return res

def radar_char_column_names(columns):
    categories = []
    for col in columns:
        # if col == M_MEMORY:
        #     col = f'         $-${M_MEMORY}'
        # elif col == M_COMPUTE:
        #     col = f'            $-${M_COMPUTE}'
        # elif col == M_FORGETTING:
        #     col = f'        $-${M_FORGETTING}'
        # elif col == M_T_IN:
        #     col = f'{M_T_IN}        '
        # elif col == M_T_OUT:
        #     col = f'{M_T_OUT}       '
        col = RADAR_NAMES[col]
        categories.append(col)
    categories = [cat.replace('\mbox{', '\\;') for cat in categories]
    return categories

def radar_chart_matplotlib_simple(df, size=23):
    # ------- PART 1: Create background
    # plt.style.use('seaborn')
    # number of variable
    # categories = [col.replace('\mbox{', '\\;') for col in df.columns]
    categories = radar_char_column_names(df.columns)
    # categories = list(df.columns)
    N = len(categories)


    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories, size=size)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([.2, .4, .6, .8, 1], [".2", ".4", ".6", ".8", "1"],
               color="grey", size=size*0.8, visible=False)
    plt.ylim(0, 1)

    # ------- PART 2: Add plots

    # Plot each individual = each line of the data
    # I don't do a loop, because plotting more than 3 groups makes the chart unreadable

    # Ind1
    my_palette = plt.cm.get_cmap("Set2", 6)#len(df.index)+1)
    line_styles = ['-', '--', '-.', ':', (-2.5, (7, 2)), (0, (1, 4))]
    for i, (name, trace) in enumerate(df.iterrows()):
        # ax = plt.subplot(i,2,2, polar=True)
        values = trace.values.flatten().tolist()
        values += values[:1]
        w = 2
        if name.startswith('MNTDP'):
            w = 2.5
        c = my_palette(i)
        # print(c)
        # input()
        # if name=='HAT (Alexnet)':
        #     c = 'blue'
        #     i=3
        # if name == 'HAT (Wide Alexnet)':
        #     i=4
        #     c = my_palette(i)
        if name in RADAR_ALIASES:
            name = RADAR_ALIASES[name]
        ax.plot(angles, values, color=c, linewidth=w,
                linestyle=line_styles[i], label=name)
        ax.fill(angles, values, color=c, alpha=0.1)

    # # Ind2
    # values = df.loc[1].drop('group').values.flatten().tolist()
    # values += values[:1]
    # ax.plot(angles, values, linewidth=1, linestyle='solid', label="group B")
    # ax.fill(angles, values, 'r', alpha=0.1)

    # Add legend
    plt.legend(loc='lower left',
               # bbox_to_anchor=(1.5, 0.1),
               # bbox_to_anchor=(1.15, 0.65),
               bbox_to_anchor=(-0.7, 0.8),
               prop={'size': size})
    # plt.gcf().set_size_inches(9, 5)
    plt.gcf().set_size_inches(12, 7)
    plt.tight_layout(1)
    # plt.legend()
    plt.show()
    return plt


def radar_chart_matplotlib_multi(df):
    categories = radar_char_column_names(df.columns)
    # categories = list(df.columns)
    N = len(categories)
    plt.cla()
    plt.clf()
    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    num_traces = len(df.index)
    n_cols = 3
    n_rows = np.ceil(num_traces/n_cols)

    def make_spider(idx, title, trace, color):
        # Initialise the spider plot
        ax = plt.subplot(n_rows, n_cols, idx+1, polar=True, )

        # If you want the first axis to be on top:
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)

        # Draw one axe per variable + add labels labels yet
        if idx == 0:
            plt.xticks(angles[:-1], categories, color='grey', size=12)
        else:
            plt.xticks(angles[:-1], ['']*len(categories), color='grey', size=1)

        # Draw ylabels
        ax.set_rlabel_position(0)
        plt.yticks([.2, .4, .6, .8, 1], [".2", ".4", ".6", ".8", "1"],
                   color="grey", size=7, visible=False)
        plt.ylim(0, 1)

        # Ind1
        values = trace.values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
        ax.fill(angles, values, color=color, alpha=0.4)

        # Add a title
        plt.title(title, size=11, color=color, y=1.15)

    # ------- PART 2: Apply to all individuals
    # initialize the figure
    # my_dpi = 96
    # plt.figure(figsize=(1000 / my_dpi, 1000 / my_dpi), dpi=my_dpi)
    plt.gcf().set_size_inches(9, 12)

    # Create a color palette:
    my_palette = plt.cm.get_cmap("Set2", len(df.index))

    # Loop to plot
    for idx, (name, trace) in enumerate(df.iterrows()):
        make_spider(idx=idx, title=name, trace=trace,
                    color=my_palette(idx))

    # plt.gca().get_legend()
    plt.tight_layout()
    return plt


def do_the_barplots(df, strat, path, keepers=None, legend=False, cols=4):
    return
    # MNIST
    if keepers is None:
        keepers = df.index
    n_cols = 3
    if isinstance(df.columns, MultiIndex):
        with_error = True
        df_err = df.xs('aggreg_row_er', level=1, axis=1)
        plot_df_err = df_err[df_err.columns[-n_cols:]]
        plot_df_err = plot_df_err.loc[keepers].transpose()
        df = df.xs('aggreg_row_mean', level=1, axis=1)
        print(f'Got errors: {plot_df_err}')
    else:
        with_error = False
    plot_df = df[df.columns[-n_cols:]]
    plot_df = plot_df.loc[keepers].transpose()
    # idx = list(plot_df.index)
    # idx[2], idx[3] = idx[3], idx[2]
    # plot_df = plot_df.reindex(idx)
    # colors = plt.cm.get_cmap("tab20", len(plot_df.columns))
    # n_styles = len(plot_df.columns)
    n_styles = 7
    colors = plt.cm.get_cmap("tab20", n_styles)
    colors = [colors(i) for i in range(n_styles)]
    patterns = ["||", "\\\\", "//", "+", "---", ".", "*", "x", "o", "O",
                '...', '.-', '+++', '/.', '.|']
    fig, axes = plt.subplots(nrows=1, ncols=n_cols)
    # fig, axes = plt.subplots(nrows=2, ncols=2)

    print(plot_df.index)

    if with_error:
        items = zip(plot_df.iterrows(), plot_df_err.iterrows())
    else:
        items = plot_df.iterrows()

    handles = None
    for i, data in enumerate(items):
        if with_error:
            ((metric_v, row_v), (metric_err, row_err)) = data
        else:
            metric_v, row_v = data
            metric_err = metric_v
            row_err = [None] * len(row_v)
        assert metric_v == metric_err
        metric = metric_v
        # ax = axes[int(i/2), i % 2]
        ax = axes[i]
        new_handles = []
        for j, ((model, itm), err) in enumerate(zip(row_v.iteritems(), row_err)):
            if model in BARPLOT_IDS:
                style_id = BARPLOT_IDS[model]
            else:
                style_id = j
            ax.bar(
                x=j,
                height=itm,
                color=colors[style_id],
                width=0.9,
                yerr=err,
                # hatch=patterns[j]
            )

            ax.bar(x=j,
                   height=itm,
                   color='None',
                   alpha=0.5,
                   edgecolor='black',
                   hatch=patterns[style_id%len(patterns)])

            rect = patches.Rectangle((0, 0), 1, 1, color=colors[style_id])
            rect.set_hatch(patterns[style_id % len(patterns)])
            rect.set_edgecolor('k')
            new_handles.append(rect)

        if handles is None:
            handles = new_handles

        # row.plot.bar(
        #     title=metric,
        #     colors=colors,
        #     ax=ax,
        #     width=0.9,
        #     hatch=patterns[i])
        ax.set_xlabel('')
        ax.set_ylabel('')
        # ax.set_ylabel(metric)
        if metric == M_MEMORY:
            ax.set_yscale('log')
            ax.set_title(f'Log {metric}$\\downarrow$')
        elif metric == M_FORGETTING:
            ax.invert_yaxis()
            ax.set_title(metric + '$\\downarrow$')
        elif metric == M_AVG_ACC:
            margin = 0.05 if CNN else 0.01
            ax.set_ylim([row_v.min() - margin, min(row_v.max() + margin, 1)])
            # ax.set_ylim([0.5, 1])
            ax.set_title(metric + '$\\uparrow$')
        else:
            ax.set_title(metric + '$\\downarrow$')


        # ax.set_x_tick(None)
        ax.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom edge are off
        ax.tick_params(
            axis='y',  # changes apply to the x-axis
            # nbins=4,
            # style='sci',
            which='both',  # both major and minor ticks are affected
            rotation=90)  # labels along the bottom edge are off
        ax.set_axisbelow(True)
        for label in ax.get_yticklabels():
            label.set_verticalalignment('center')
        if metric != M_MEMORY:
            ax.locator_params(nbins=3, axis='y')
            ax.ticklabel_format(axis='y', style='sci')
        ax.yaxis.grid(color='gray', linestyle='dashed')
    # bar = plot_df.plot.bar(color=colors, subplots=True)
    # fig = bar.get_figure()
    # handles = [patches.Rectangle((0, 0), 1, 1, color=colors[i],)
    #            for i in range(n_styles)]
    for i, h in enumerate(handles):
        h.set_hatch(patterns[i%len(patterns)])
        h.set_edgecolor('k')
    # edgecolor = 'black',


    # hatch = patterns[i])

    if legend:
        # 1x4 legend
        if len(plot_df.columns) <= 11:
            plt.legend(handles, list(plot_df.columns),
                       loc='lower left',
                       ncol=1,
                       bbox_to_anchor=(1, 0),
                       prop={'size': 13})
        else:
            plt.legend(handles, list(plot_df.columns),
                       loc='right',
                       ncol=1,
                       bbox_to_anchor=(2.4, 0.55),
                       prop={'size': 8})

    for ax in axes:
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(15)
            # ax.set_yticklabels(va='bottom')
            # ax.tick_params(axis="y", va='bottom')
    # 4x4 legend
    # plt.legend(handles, list(plot_df.columns),
    #     loc='upper left', bbox_to_anchor=(1.3, 3), prop={'size': 7})

    mult = 1 if legend else 0.8
    fig.set_size_inches(8.5*mult, 2.5)
    fig.tight_layout()
    f_name = FILE_NAMES[strat][:-4]

    fig.savefig(path/(f_name + '_barplot.svg'))
    fig.savefig(path/(f_name + '_barplot.png'))
    fig.savefig(path/(f_name + '_barplot.pdf'))

    fig.show()
    # input()
    # raise ValueError

    print(plot_df)

def plot_traces(x, y, y_err, y_name, traces_names, ax, labels=True):
    # fig, ax = plt.subfigures(21, figsize=(8, 5))
    my_palette = plt.cm.get_cmap("Set2", len(traces_names))
    line_styles = ['-', '--', '-.', (1, (5, 2, 18, 2)), (-2.5, (8, 1)), ':']
    print(len(traces_names))
    print(len(traces_names))
    print(len(traces_names))
    print(traces_names)
    for i, (name, y, y_err) in enumerate(zip(traces_names, y, y_err)):
        ax.plot(x, y, color=my_palette(i), linestyle=line_styles[i%len(line_styles)],
                label=name if labels else None, linewidth=2)
        ax.fill_between(x, y - y_err, y + y_err, alpha=0.3,
                        facecolor=my_palette(i))

    ax.set(ylabel=y_name)
    # plt.plot(x, y1,
    #          color='red',
    #          linewidth=1.0,
    #          linestyle='--'
    #          )


def plot_traj_curves(data_dict, out_path):
    res_dict = {}
    for idx, data in data_dict.items():
        for mod, res in data['res'].items():
            assert (idx, mod) not in res_dict
            res_dict[(idx, mod)] = res[0]

    print(res_dict.keys())
    out_cross_val = defaultdict(list)
    for k in list(res_dict.keys()):
        for pref in OUTER_CROSS_VAL:
            if k[1].startswith(pref):
                out_cross_val[(k[0], pref)].append((k, res_dict.pop(k)))
                break

    print(res_dict.keys())
    print(f'OUTER CROSSVAL : {out_cross_val.keys()}')
    for k, candidates in out_cross_val.items():
        best = None
        best_val = None
        for name, res in candidates:
            avg_val = res['avg_acc_val_so_far'][-1]
            if best is None or avg_val > best_val:
                best = name
                best_val = avg_val
                best_res = res
        print(f'BEST: {best}')
        best_k = '-'.join(best[1].split('-')[:-1])
        print(f'BEST k: {best_k}')
        res_dict[(best[0], f'best-{best_k}')] = best_res

    print(res_dict.keys())

    df = pd.DataFrame(res_dict, )
    keep = ['avg_acc_test_so_far', 'lca', 'total_params', 't']
    df = df.loc[keep]
    df2 = df
    fig, axs = plt.subplots(3,1, sharex=True)
    x = df.loc['t'].tolist()[0]


    # df2 = df.groupby(level=1, axis=1).agg(aggregate_trajectories)
    df = df.transpose()
    df = df.groupby(level=1).agg([ #aggregate_trajectories)
        mean_s,
        se_s,
    ])

    df = df.loc[LONG_TRAJ_KEEPERS]
    # input(df)
    trace_names = [MODEL_ALIASES[n] for n in df.index]

    avg_acc = [np.array(itm) for itm in df['avg_acc_test_so_far', 'mean_s'].values]
    avg_acc_err = [np.array(itm) for itm in df['avg_acc_test_so_far', 'se_s'].values]
    # lca = [np.array(itm) for itm in df.loc['lca'].values]
    n_params = [np.array(itm) * 4 for itm in df['total_params', 'mean_s'].values]
    n_params_err = [np.array(itm) * 4 for itm in df['total_params', 'se_s'].values]
    plot_traces(x, avg_acc, avg_acc_err, M_AVG_ACC, trace_names, axs[2])
    plot_traces(x, avg_acc, avg_acc_err, M_AVG_ACC, trace_names, axs[1])
    plot_traces(x, n_params, n_params_err, M_MEMORY, trace_names, axs[0], labels=False)
    axs[2].set(xlabel='Task id')
    lgd = fig.legend(
        loc='bottom left',
        # bbox_to_anchor=(.532, .923),
        bbox_to_anchor=(0.63, 1.06),
        fontsize=12,
        ncol=int(np.ceil(len(trace_names) / 3)),
    )
    for ax in axs:
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(11)
    save_p = {
        'bbox_extra_artists': (lgd,),
        'bbox_inches': 'tight'
    }

    fig.subplots_adjust(hspace=0.0)
    # fig.set_size_inches(3*6.5/INCH, 3*3.2/INCH)
    fig.set_size_inches(3*7/INCH, 3*4/INCH)
    fig.savefig(out_path/'long-stream-traj.svg', **save_p)
    fig.savefig(out_path/'long-stream-traj.pdf', **save_p)
    fig.savefig(out_path/'long-stream-traj.png', **save_p)

    fig.show()
    return fig


def aggregate_trajectories(traj, method='mean'):
    return traj.apply(partial(mean_std_series, method=method), axis=1)


def se_s(s):
    return mean_std_series(s, 'se')

def std_s(s):
    return mean_std_series(s, 'std')


def mean_s(s):
    return mean_std_series(s, 'mean')


def mean_std_series(series, method):
    vals = np.array(series.values.tolist())
    if method == 'mean':
        return vals.mean(axis=0).tolist()
    elif method == 'std':
        return vals.std(axis=0).tolist()
    elif method == 'se':
        se = vals.std(axis=0) / vals.shape[0]
        return se.tolist()
    else:
        raise ValueError(f'Unknown method {method}')
