# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os
import re
import shutil
import tempfile
import time
from collections import defaultdict
from operator import itemgetter

import pandas
import visdom
from tqdm import tqdm

import src.utils.external_resources as external

logger = logging.getLogger(__name__)

def get_env_url(visdom_client, replace=('devfair054', 'localhost')):
    if isinstance(visdom_client, dict):
        res = '{}:{}/env/{}'.format(visdom_client['server'],
                                    visdom_client['port'],
                                    visdom_client['env'])
    else:
        res = '{}:{}/env/{}'.format(visdom_client.server,
                                    visdom_client.port,
                                    visdom_client.env)
    if not res.startswith('http://'):
        res = 'http://' + res
    if replace:
        res = res.replace(*replace)
    return res


def rename_class(cls, new_name):
    cls.__name__ = new_name
    qualname = cls.__qualname__.split('.')
    qualname[-1] = new_name
    cls.__qualname__ = '.'.join(qualname)
    return cls


def fill_matrix(matrix, val=float('nan')):
    assert isinstance(matrix, list)
    max_len = max(map(len, matrix))
    matrix = [pad_seq(seq, max_len, val) for seq in matrix]
    return matrix


def pad_seq(seq, length, val):
    new_elems = [val] * (length - len(seq))
    return seq + new_elems


def paretize_exp(data, x_name, crit_name, keep_keys):
    # Data needs to be sorted following x axis

    # for x1, x2 in zip(data[x_name], data[x_name][1:]):
    #     assert x1 <= x2
    if isinstance(data, pandas.DataFrame):
        data = data.to_dict('list')

    keep_keys = set(keep_keys + [x_name, crit_name])

    res = defaultdict(list)
    cur_best_crit = None
    cur_best_x = None
    keys = list(data.keys())
    data = [dict(zip(keys, vals)) for vals in zip(*data.values())]
    for d in sorted(data, key=itemgetter(x_name)):
        cur_crit = d[crit_name]
        if len(res) == 0 or cur_crit > cur_best_crit:
            if d[x_name] == cur_best_x:
                for k in keep_keys:
                    res[k][-1] = d[k]
            else:
                for k in keep_keys:
                    res[k].append(d[k])
            cur_best_crit = cur_crit
            cur_best_x = d[x_name]
    return res

def get_runs(sacred_ids=None, slurm_ids=None, mongo_conf_path=None):
    if sacred_ids:
        assert not slurm_ids, 'Can\'t specify both sacred and slurm ids'
        ids = sacred_ids
        req = {'_id': {"$in": ids}}
    elif slurm_ids:
        ids = [str(id) for id in slurm_ids]
        req = {'host.ENV.SLURM_JOB_ID': {"$in": ids}}
    else:
        raise ValueError('Should specify one of --sacred-ids or --slurm-ids.')

    mongo_collection = external.get_mongo_collection(mongo_path=mongo_conf_path)
    runs = mongo_collection.find(req, no_cursor_timeout=True)
    n_res = mongo_collection.count_documents(req)
    if n_res < len(ids):
        retrieved = set(r['_id'] if sacred_ids else
                        r['host']['ENV']['SLURM_JOB_ID'] for r in runs)
        missing = set(ids) - retrieved
        raise ValueError('Missing runs, expected {} but got {} (missing {}).'
                         .format(len(ids), runs.count(), missing))
    if n_res > len(ids):
        raise ValueError('Big problem: More results that ids (runs coming from '
                         'different Slurm clusters ?)')
    return runs


def replay_run(run, viz, args, mongo_path, logger):
    if not run['artifacts']:
        logger.warning(
            'Run {} doesn\'t have any stored file'.format(run['_id']))
        return None, 0, 0

    selected_artifact = None
    for artifact in reversed(run['artifacts']):
        if artifact['name'].endswith(args.name_end):
            selected_artifact = artifact
            break

    if selected_artifact is None:
        available = [a['name'] for a in run['artifacts']]
        raise ValueError('No artifact ending with \'{}\' in run {}. Available'
                         ' artifacts are {}'.format(args.name_end, args.id,
                                                    available))
    start_time = time.time()
    gridfs = external.get_gridfs(mongo_path=mongo_path)
    object = gridfs.get(selected_artifact['file_id'])

    # with tempfile.TemporaryDirectory() as dir:
    with tempfile.TemporaryDirectory(dir='/local/veniat') as dir:
        file_path = os.path.join(dir, selected_artifact['name'])
        with open(file_path, 'wb') as file:
            file.write(object.read())

        n_replayed, main_env = replay_from_path(file_path, viz, run['_id'],
                                            args.all_envs)
    tot_time = time.time() - start_time
    logger.info('Replayed {} envs in {:.3f} seconds.'.format(n_replayed,
                                                             tot_time))
    if main_env is None:
        logger.warning('No main env file found.')
    else:
        logger.info('Main env is {}.'.format(main_env))
        viz.env = main_env
    logger.info(get_env_url(viz))
    return get_env_url(viz), n_replayed, tot_time

def replay_from_path(archive_path, viz, run_id, all_envs=False):
    target = archive_path[:-4]
    # print(archive_path)
    # print(os.path.getsize(archive_path))
    # shutil.copy(file_path, selected_artifact['name'])
    shutil.unpack_archive(archive_path, target)
    env_files = os.listdir(target)
    logger.info('Replaying envs for {}'.format(run_id))
    n_replayed = 0
    main_env = None
    for file in tqdm(env_files):
        if 'main' in file:
            main_env = file
        if all_envs or 'Trial' not in file or 'PSSN-search-6-bw' in file:
            # or 'main' in file or 'tasks' in file:
            # print(file)
            viz.delete_env(file)
            env_path = os.path.join(target, file)
            try:
                viz.replay_log(env_path)
            except ValueError as r:
                logger.warning('Can\' replay {}: {}'.format(env_path, r))
            except IsADirectoryError as r:
                logger.warning('Can\' replay dir {}: {}'.format(env_path, r))
            n_replayed += 1

    return n_replayed, main_env


def get_training_vis_conf(vis_p, trial_dir):
    vis_p = vis_p.copy()
    # if trial_dir.endswith('/'):
    #     trial_dir = trial_dir[:-1]
    #
    # # 19  chars for the date + 8 chars for mktmp directory + 1 sep
    # tag = os.path.basename(trial_dir)[:-28]

    split = re.split('_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}', trial_dir)
    assert len(split) == 2
    tag = os.path.basename(split[0])

    env = vis_p['env'].split('_')
    assert len(env) == 3, '{} is too long'.format(env)  # exp_model_task
    env[1] = tag
    env.insert(1, 'Trial')
    vis_p['env'] = '_'.join(env)

    vis_logdir = os.path.dirname(vis_p['log_to_filename'])
    vis_p['log_to_filename'] = os.path.join(vis_logdir, vis_p['env'])

    return vis_p


def count_params(model):
    n_params = 0
    n_trainable_params = 0
    for p in model.parameters():
        n_params += p.numel()
        if p.requires_grad:
            n_trainable_params += p.numel()

    return {'total': n_params, 'trainable': n_trainable_params}


if __name__ == '__main__':
    vis = visdom.Visdom()
    path = '/home/tom/Downloads/new_archive.zip'

    print(replay_from_path(path, vis, 1888, False))