

import os
import numpy as np
import pandas as pd
from copy import deepcopy

RUN_LOCAL = False

if RUN_LOCAL:

    BASE_DIR                = '/media/user/3a0db8e5-2fba-4de7-be88-04c227b14704/lyft-motion-prediction-autonomous-vehicles/'
    SUBMISSIONS_DIR         = os.path.join(BASE_DIR, 'submissions')
    DATA_DIR                = SUBMISSIONS_DIR
    SAMPLE_SUBMISSION_PATH  = os.path.join(BASE_DIR, 'multi_mode_sample_submission.csv')

    SAMPLE_SUBMISSION       = pd.read_csv(SAMPLE_SUBMISSION_PATH)

else:

    READ_DIR                = '/kaggle/input'
    WRITE_DIR               = '/kaggle/working'
    BASE_DIR                = READ_DIR
    DATA_DIR                = os.path.join(BASE_DIR, 'lyft-motion-prediction-subs')
    SUBMISSIONS_DIR         = WRITE_DIR

    SAMPLE_SUBMISSION_PATH  = os.path.join(BASE_DIR, 'lyft-motion-prediction-autonomous-vehicles', 'multi_mode_sample_submission.csv')

    SAMPLE_SUBMISSION       = pd.read_csv(SAMPLE_SUBMISSION_PATH)

        
def get_column_names():

    cols = list(SAMPLE_SUBMISSION.columns)

    confs = cols[2:5]
    conf0 = cols[5:105]
    conf1 = cols[105:205]
    conf2 = cols[205:305]

    return confs, conf0, conf1, conf2

confs, *conf_names = get_column_names()
all_names = confs + [_c for c in conf_names for _c in c]


def check_validity(subs):
    return all([np.array_equal(s.timestamp.values, subs[0].timestamp.values) and np.array_equal(s.track_id.values, subs[0].track_id.values) for s in subs[1:]])  
    


def clip_sub_confs(sub, clip_min, clip_max):

    sub_confs = np.clip(sub[confs].values, clip_min, clip_max)
    sub_confs = np.divide(sub_confs, np.sum(sub_confs, axis=1).reshape(-1, 1))

    sub[confs] = sub_confs

    return sub


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



def combine_multi_subs_dist_order(subs, weights, logscale=False):

    assert len(weights) == len(subs), 'You must have one weight for each sub in sub_list'

    subs = [sort_sub_by_distances(s) for s in subs]

    assert check_validity(subs)

    comb_sub = deepcopy(subs[0])

    for c in all_names:
        if logscale and c in confs:
            comb_sub[c] = np.exp(np.sum(np.stack([np.log(s[c].values) * w for s, w in zip(subs, weights)], axis=-1), axis=-1) / np.sum(weights))
        else:
            comb_sub[c] = np.sum(np.stack([s[c].values * w for s, w in zip(subs, weights)], axis=-1), axis=-1) / np.sum(weights)

    # Renormalise confs
    comb_sub = clip_sub_confs(comb_sub, 0.001, 0.999)
        
    # Having problems with formats... Overwrite SAMPLE_SUBMISSION with correct values to avoid this
    submission = SAMPLE_SUBMISSION[["timestamp", "track_id"]].merge(comb_sub, on=["timestamp", "track_id"])
    
    return submission


def create_sub_from_multis_dist_order(subs, weights, logscale=False):

    submission = combine_multi_subs_dist_order(subs, weights, logscale)

    submission.to_csv(os.path.join(SUBMISSIONS_DIR, 'submission.csv'), index=False, float_format='%.6g')
    
    

def save_as_sub(sub_path, clip_values=None):

    submission = pd.read_csv(sub_path)

    if clip_values is not None:
        submission = clip_sub_confs(submission, *clip_values)

    submission = SAMPLE_SUBMISSION[["timestamp", "track_id"]].merge(submission, on=["timestamp", "track_id"])
    
    submission.to_csv(os.path.join(SUBMISSIONS_DIR, 'submission.csv'), index=False, float_format='%.6g')


if __name__ == '__main__':

    sub_paths = [
                os.path.join(DATA_DIR, 'submission_20201116115433_.csv'), # weights based on val chopped 100
                os.path.join(DATA_DIR, 'submission_20201117152334_.csv'), # weights based on val chopped 150
                os.path.join(DATA_DIR, 'submission_20201117152818_.csv'), # weights based on val chopped 50
                ]
    
    sub_list = [pd.read_csv(sub) for sub in sub_paths]
    
    create_sub_from_multis_dist_order(sub_list, weights=[1/3]*3, logscale=True)
 

 
 

 

