######################################
# IMPORTS
######################################

import numpy as np
import pandas as pd
import os
import random
from timeit import default_timer as timer
from tqdm import tqdm
import scipy as sp
import scipy.optimize
import matplotlib.pyplot as plt
import math
from datetime import datetime
from copy import deepcopy
from functools import partial

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
    
    preds, confs = calc_weighted_ensemble(predictions['preds'], predictions['confs'], weights)

    predictions['preds'] = preds
    predictions['confs'] = confs
    
    return predictions


def conf_weighted_nll(coefs, gt, avails, preds, confs):

    p, c = calc_weighted_ensemble(preds, confs, coefs)

    return numpy_neg_multi_log_likelihood(gt, p, c, avails)


def generate_distance_indicator(preds, confs, threshold=20):

    max_pred_distance = np.mean(np.sqrt(np.square(preds[:, :, -1, -1, :])), axis=(0, 2))
    long_dist = np.logical_and(max_pred_distance > threshold, np.mean(confs[:, :, -1], axis=0) > 0.1)

    return long_dist


def dist_weighted_nll(coefs, gt, avails, preds, confs):

    p, c = calc_weighted_ensemble(preds, confs, coefs)

    return numpy_neg_multi_log_likelihood(gt, p, c, avails)


def calc_weighted_ensemble(preds, confs, alpha_weights):

    n = len(alpha_weights) // 2

    if len(alpha_weights) - 1 > preds.shape[0]:
        long_dist = generate_distance_indicator(preds, confs)
        indicators = [long_dist, np.logical_not(long_dist)]
        w = [alpha_weights[:n], alpha_weights[n:]]
    else:
        indicators = [np.ones((preds.shape[1],), dtype=np.bool)]
        w = [alpha_weights]

    p = np.zeros_like(preds[0])
    c = np.zeros_like(confs[0])

    for _w, ind in zip(w, indicators):

        p[ind], c[ind] = calc_weighted_ensemble_internal(preds[:, ind], confs[:, ind], _w)

    return p, c


def calc_weighted_ensemble_internal(preds, confs, alpha_weights):

    alpha = alpha_weights[0]
    weights = alpha_weights[1:]

    n, n_samples, n_modes = preds.shape[:3]

    c = np.power(confs, alpha)
    c = np.divide(c, np.sum(c, axis=-1)[:, :, None])

    weights = np.array(weights) / np.sum(weights)
    w = np.multiply(np.ones((n, n_samples, n_modes)), weights.reshape(-1, 1, 1))

    w = np.multiply(w, confs)

    pw = np.multiply(preds, w[:, :, :, None, None])
    pw = np.divide(np.sum(pw, axis=0), np.sum(w, axis=0)[:, :, None, None])

    cw = np.multiply(confs, w)
    cw = np.divide(np.sum(cw, axis=0), np.sum(w, axis=0))
    cw = np.divide(cw, np.sum(cw, axis=-1)[:, None])

    return pw, cw
    

def opt_partial(opt_fn, gt, avails, preds, confs, init_coefs=None):

    if init_coefs is None:
        # No conf impact and equal weights
        init_coefs = np.array([0] + [1] * preds.shape[0])
        if opt_fn == dist_weighted_nll:
            init_coefs = np.concatenate([init_coefs, init_coefs])

    loss_partial = partial(opt_fn, gt=gt, avails=avails, preds=preds, confs=confs)
    coef = sp.optimize.minimize(loss_partial, init_coefs, bounds=[(0, 1) for i in range(len(init_coefs))], tol=1e-10)

    return coef.x, coef.fun


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


def generate_ground_truth(timestamps_trackid, gt_path):

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


def estimate_ensemble(sub_paths, gt_path, opt_fn):

    subs = generate_subs(sub_paths)

    assert check_validity(subs)

    y = generate_ground_truth(subs[0][['timestamp', 'track_id']].values, gt_path)

    predictions = {'preds': np.stack([np.concatenate([sub[conf_names[j]].values.reshape(-1, 1, 50, 2) for j in range(3)], axis=1) for sub in subs], axis=0),
                    'confs': np.stack([sub[confs].values for sub in subs], axis=0)}

    n = predictions['preds'].shape[0]

    base_p = combine_predictions(deepcopy(predictions), np.array([0] + [1.]*n))
    orig_nlls = [numpy_neg_multi_log_likelihood(y['truth'], predictions['preds'][i], predictions['confs'][i], y['avails']) for i in range(n)]
    avg_nll = numpy_neg_multi_log_likelihood(y['truth'], base_p['preds'], base_p['confs'], y['avails'])

    weights, opt_nll = opt_partial(opt_fn, y['truth'], y['avails'], predictions['preds'], predictions['confs'])

    print('*********************************************************************************')
    print(' : '.join(('weights', 'opt nll', 'avg nll', 'orig_nlls')))
    print(' : '.join((str(weights), str(opt_nll), str(avg_nll), str(orig_nlls))))
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


def estimate_validation_weights(val_paths, gt_path=os.path.join(BASE_DIR, 'scenes/validate_chopped_100','gt.csv'), opt_fn=conf_weighted_nll):

    val_sub_paths = [generate_subs_from_val(val_path) for val_path in val_paths]

    return estimate_ensemble(val_sub_paths, gt_path, opt_fn)


def generate_ensemble_submission(submission_paths, weights):

    subs = generate_subs(submission_paths)

    assert check_validity(subs)

    predictions = {'preds': np.stack([np.concatenate([sub[conf_names[j]].values.reshape(-1, 1, 50, 2) for j in range(3)], axis=1) for sub in subs], axis=0),
                    'confs': np.stack([sub[confs].values for sub in subs], axis=0)}

    ensembled_predictions = combine_predictions(deepcopy(predictions), weights)

    # Save weights and submission
    sub_name = '_'.join(('submission', datetime.now().strftime('%Y%m%d%H%M%S')))

    save_as_pickle(os.path.join(SUBMISSIONS_DIR, '_'.join(('weights', sub_name, '.pkl'))), weights)

    write_pred_csv(os.path.join(SUBMISSIONS_DIR, '_'.join((sub_name, '.csv'))), 
                    subs[0]['timestamp'].values, 
                    subs[0]['track_id'].values, 
                    ensembled_predictions['preds'], 
                    ensembled_predictions['confs'])


def generate_ensemble_prediction(submission_paths, weights=None):

    if weights is None:
        weights = [0] + [1.] * len(submission_paths)

    generate_ensemble_submission(submission_paths, weights)



if __name__ == '__main__':

    gt_path = os.path.join(BASE_DIR, 'scenes/validate_chopped_100', 'gt.csv')

    val_paths = [os.path.join(DATA_DIR,'val_test_transform_LyftResnest50_double_channel_agents_ego_map_transform_create_config_multi_chopped_lite_val10_neg_log_likelihood_transform_128_2800_256_1062_1_5_50_3_False_7_resnet18_fit_fastai_trainloss_none__.pkl'),
                 os.path.join(DATA_DIR, 'params_v17.chopped.1600_20chops_last_valid100.csv'),
                 os.path.join(DATA_DIR, 'params_v17.chopped.2240_20chops_last_valid100.csv'),
                 os.path.join(DATA_DIR, 'params_v17.chopped.1960_20chops_296_last_valid100.csv')]

    
    weights, nll = estimate_validation_weights(val_paths, gt_path, opt_fn=conf_weighted_nll)

    sub_paths = [os.path.join(SUBMISSIONS_DIR,'test_test_transform_LyftResnest50_double_channel_agents_ego_map_transform_create_config_multi_chopped_lite_val10_neg_log_likelihood_transform_128_2800_256_1062_1_5_50_3_False_7_resnet18_fit_fastai_trainloss_none__.csv'),
                 os.path.join(SUBMISSIONS_DIR, 'submission_rp_1261.csv'),
                 os.path.join(SUBMISSIONS_DIR, 'submission_rp_1230.csv'),
                 os.path.join(SUBMISSIONS_DIR, 'submission_rp_12496.csv')]

    generate_ensemble_submission(sub_paths, weights)


