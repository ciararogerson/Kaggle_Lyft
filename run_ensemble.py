######################################
# IMPORTS
######################################

import numpy as np
import pandas as pd
import os
import random
import pickle
import inspect
from timeit import default_timer as timer
from tqdm import tqdm
import itertools
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import math
from datetime import datetime
from copy import deepcopy

from l5kit.evaluation.metrics import neg_multi_log_likelihood
from l5kit.evaluation import write_pred_csv, read_gt_csv

from settings import BASE_DIR, DATA_DIR, SUBMISSIONS_DIR, SINGLE_MODE_SUBMISSION, MULTI_MODE_SUBMISSION

from configs import *
from utils import *

###########################
# SET SEEDS
###########################

SEED = 9999
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)


######################################
# SET UP / GLOBALS
######################################

NUM_WORKERS = 16

######################################
# ENSEMBLING FUNCTIONALITY
######################################

def numpy_neg_multi_log_likelihood(gt, pred, confidences, avails):
    """
    numpy version of l5kit's neg_multi_log_likelihood
    """

    # add modes and cords
    gt = np.expand_dims(gt, 1)
    avails = np.expand_dims(np.expand_dims(avails, 1), -1)

    # error (batch_size, num_modes, future_len), reduce coords and use availability
    error = np.sum(((gt - pred) * avails) ** 2, axis=-1)

    with np.errstate(divide="ignore"):  # when confidence is 0 log goes to -inf, but we're fine with it
        # error (batch_size, num_modes)
        error = np.log(confidences) - 0.5 * np.sum(error, axis=-1)  # reduce time

    # use max aggregator on modes for numerical stability
    # error (batch_size, num_modes)
    max_value = np.max(error, axis=1).reshape(-1, 1)  # error are negative at this point, so max() gives the minimum one
    error = -np.log(np.sum(np.exp(error - max_value), axis=-1)).reshape(-1, 1) - max_value  # reduce modes
    # print("error", error)
    return np.mean(error)


def combine_predictions(predictions, weights):
    
    predictions['preds'] = np.multiply(predictions['preds'], np.array(weights).reshape(-1, 1, 1, 1, 1))
    predictions['preds'] = np.divide(np.sum(predictions['preds'], axis=0), np.sum(weights))

    predictions['confs'] = np.multiply(predictions['confs'], np.array(weights).reshape(-1, 1, 1))
    predictions['confs'] = np.divide(np.sum(predictions['confs'], axis=0), np.sum(weights))
    
    return predictions


def weighted_nll(args_in): 
    predictions, y, weights = args_in  
    predictions = combine_predictions(predictions, weights)
    return numpy_neg_multi_log_likelihood(y['truth'], predictions['preds'], predictions['confs'], y['avails'])


def opt_gridsearch_mp(predictions, y, weights, opt_fn):

    n_weights = len(weights)
    arg_list = [(predictions, y, weights[i]) for i in range(n_weights)]
    results = execute_mp_map(opt_fn, arg_list, boo_async = True, display_iter = 10)
    results = np.array(results)

    return results


def opt(predictions, y, weights_min, weights_max, steps=[0.2, 0.1, 0.05]):

    weights_working_min, weights_working_max = weights_min, weights_max

    for step in steps:

        _weights = [np.minimum(weights_max, np.arange(weights_min, weights_max + step, step)) for weights_min, weights_max in zip(weights_working_min, weights_working_max)]

        weights_combs = np.meshgrid(*_weights)
        weights_combs = np.concatenate([x.ravel()[:, np.newaxis] for x in weights_combs], axis = 1)
        weights_combs = weights_combs[np.sum(weights_combs, axis=1) > 0]
        weights_combs = list(np.unique(weights_combs, axis=0))
        
        # Carry out your grid search
        results = opt_gridsearch_mp(predictions, y, weights_combs, weighted_nll)
        
        # Find your optimal combs and reset working_min, working_max
        min_index = np.argwhere(results == np.min(results))[0][0]

        print(' '.join(('Step', str(step), 'Min NLL:', str(results[min_index]), str(weights_combs[min_index]))))

        weights_working_min = [x - step for x in weights_combs[min_index]]
        weights_working_max = [x + step for x in weights_combs[min_index]]


    # You now have your opt results
    opt_weights = weights_combs[min_index]

    p = combine_predictions(predictions.copy(), opt_weights)
    opt_nll = numpy_neg_multi_log_likelihood(y['truth'], p['preds'], p['confs'], y['avails'])

    base_p = combine_predictions(predictions.copy(), np.ones_like(opt_weights))

    # Sanity check:
    print(''.join(('predictions nll: ', str(numpy_neg_multi_log_likelihood(y['truth'], base_p['preds'], base_p['confs'], y['avails'])))))
    print(''.join(('opt adj predictions nll: ', str(opt_nll))))

    return opt_weights, opt_nll


def get_column_names():

    sample_submission = pd.read_csv(MULTI_MODE_SUBMISSION)

    cols = list(sample_submission.columns)

    confs = cols[2:5]
    conf0 = cols[5:105]
    conf1 = cols[105:205]
    conf2 = cols[205:305]

    return confs, conf0, conf1, conf2

confs, *conf_names = get_column_names()
all_names = confs + [_c for c in conf_names for _c in c]


def check_validity(subs):
    return all([np.array_equal(s.timestamp.values, subs[0].timestamp.values) and np.array_equal(s.track_id.values, subs[0].track_id.values) for s in subs[1:]])  
    

def sort_sub_by_distances(sub):

    # Sort by distance from first point (0, 0) to last point
    distances = np.concatenate([np.sqrt((sub[conf_names[i]].values[:, 0] - sub[conf_names[i]].values[:, -2])**2 + 
                                        (sub[conf_names[i]].values[:, 1] - sub[conf_names[i]].values[:, -1])**2).reshape(-1, 1) for i in range(3)], axis=1)
    dist_order = np.argsort(distances, axis=1)

    # Create a holder for new distance-ordered values
    sub_dist_order = sub[all_names].values.copy()

    # Reorder values according to sorted distance
    u_dist_order = np.unique(dist_order, axis=0)

    for order in u_dist_order:

        # Only reorder if the rows are not already in order
        if not np.array_equal(order, np.array([0, 1, 2])):

            # Find rows that correspond to this order
            idx = np.argwhere(np.all(np.concatenate([(dist_order[:, i] == order[i]).reshape(-1, 1) for i in range(len(order))], axis=1), axis=1)).reshape(-1,)

            # Reorder these rows so that they are distance-sorted
            _sub = sub.iloc[idx]
            sub_dist_order[idx, :] = np.concatenate([_sub[confs[order[0]]].values.reshape(-1, 1), 
                                                    _sub[confs[order[1]]].values.reshape(-1, 1),
                                                    _sub[confs[order[2]]].values.reshape(-1, 1),
                                                    _sub[conf_names[order[0]]].values, 
                                                    _sub[conf_names[order[1]]].values,
                                                    _sub[conf_names[order[2]]].values], axis=1)
    # Set sub to new distance-order
    sub[all_names] = sub_dist_order

    return sub


def generate_ground_truth(timestamps_trackid, gt_path=os.path.join(BASE_DIR, 'scenes/validate_chopped_100','gt.csv')):

    gt = {}
    for row in tqdm(read_gt_csv(gt_path)):
        gt[str(row['timestamp']) + str(row['track_id'])] = {'coords': row["coord"], 'avails': row['avail']}

    truth, avails = [], []
    for i in tqdm(range(timestamps_trackid.shape[0])):
        timestamp, track_id = int(timestamps_trackid[i, 0]), int(timestamps_trackid[i, 1])
        truth.append(gt[str(timestamp) + str(track_id)]['coords'])
        avails.append(gt[str(timestamp) + str(track_id)]['avails'])

    gt_out = {'truth': np.stack(truth, axis=0), 'avails': np.stack(avails, axis=0)}

    return gt_out
        

def generate_subs(sub_paths):

    subs = [pd.read_csv(sub) for sub in sub_paths]
    subs = [sub.sort_values(['timestamp', 'track_id']) for sub in subs]
    subs = [sort_sub_by_distances(sub) for sub in subs]

    assert check_validity(subs)

    return subs


def estimate_ensemble(sub_paths):

    subs = generate_subs(sub_paths)

    y = generate_ground_truth(subs[0][['timestamp', 'track_id']].values)

    predictions = {'preds': np.stack([np.concatenate([sub[conf_names[j]].values.reshape(-1, 1, 50, 2) for j in range(3)], axis=1) for sub in subs], axis=0),
                    'confs': np.stack([sub[confs].values for sub in subs], axis=0)}

    n = predictions['preds'].shape[0]

    base_p = combine_predictions(deepcopy(predictions), np.ones((n,)))
    orig_nll = numpy_neg_multi_log_likelihood(y['truth'], base_p['preds'], base_p['confs'], y['avails'])
    print(''.join(('Original nll: ', str(orig_nll))))

    weights, opt_nll = opt(predictions, y, weights_min=[0]*n, weights_max=[1.0]*n)

    print('*********************************************************************************')
    print(' : '.join((str(weights), str(opt_nll), str(orig_nll))))
    print('*********************************************************************************')

    return weights, opt_nll


def generate_subs_from_val(val_path):

    if 'pkl' in os.path.splitext(val_path)[-1]:

        # val_path is a dictionary
        val_sub_path = val_path.replace('pkl', 'csv')

        if not os.path.exists(val_sub_path):
            # Create sub from dict
            val_dict = load_from_pickle(val_path)
            write_pred_csv(val_sub_path, val_dict['timestamps'], val_dict['track_ids'], val_dict['preds'], val_dict['conf'])
    else:

        val_sub_path = val_path

    return val_sub_path


def estimate_validation_weights(val_paths):

    val_sub_paths = [generate_subs_from_val(val_path) for val_path in val_paths]

    return estimate_ensemble(val_sub_paths)


def generate_ensemble_submission(submission_paths, weights):

    subs = generate_subs(submission_paths)

    predictions = {'preds': np.stack([np.concatenate([sub[conf_names[j]].values.reshape(-1, 1, 50, 2) for j in range(3)], axis=1) for sub in subs], axis=0),
                    'confs': np.stack([sub[confs].values for sub in subs], axis=0)}

    ensembled_predictions = combine_predictions(deepcopy(predictions), weights)

    write_pred_csv(os.path.join(SUBMISSIONS_DIR, 'submission.csv'), 
                    subs[0]['timestamps'].values, 
                    subs[0]['track_ids'].values, 
                    ensembled_predictions['preds'], 
                    ensembled_predictions['confs'])


def generate_ensemble_prediction(submission_paths, weights=None):

    if weights is None:
        weights = [1.] * len(submission_paths)

    generate_ensemble_submission(submission_paths, weights)



if __name__ == '__main__':

    val_paths = [os.path.join(DATA_DIR, 'val_test_transform_LyftResnet18Transform_double_channel_agents_ego_map_dayhour_create_config_train_chopped_neg_log_likelihood_transform_320_0.5_0.5_0.25_0.5_600_256_17000_10_10_50_3_40_resnet18_fit_fastai_transform_heavycoarsedropoutblur__.pkl'),
                os.path.join(DATA_DIR, 'val_test_transform_LyftResnet18Transform_double_channel_agents_ego_map_transform_create_config_train_chopped_randego_neg_log_likelihood_transform_320_0.5_0.5_0.25_0.5_600_256_17000_5_5_50_3_40_resnet18_fit_fastai_transform_none__.pkl')]
    
    estimate_validation_weights(val_paths)