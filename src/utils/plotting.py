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
    print('y')
    print([[mean_perf_on_all_tasks],
           [mean_perf_on_seen_tasks],
           [mean_perf_when_seen]])
    print('x')
    print([n_tasks])
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

