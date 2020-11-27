######################################
# IMPORTS
######################################

import numpy as np
import os
import torch
import random


######################################
# SET SEEDS
######################################

SEED = 9999
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


######################################
# DIRECTORIES
######################################

BASE_DIR = '/path/to/lyft-motion-prediction-autonomous-vehicles'

DATA_DIR = os.path.join(BASE_DIR, 'data')
CACHE_DIR = os.path.join(BASE_DIR, 'cache')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
SUBMISSIONS_DIR = os.path.join(BASE_DIR, 'submissions')

if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
if not os.path.exists(CACHE_DIR): os.makedirs(CACHE_DIR)
if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
if not os.path.exists(SUBMISSIONS_DIR): os.makedirs(SUBMISSIONS_DIR)

SINGLE_MODE_SUBMISSION = os.path.join(BASE_DIR, 'single_mode_sample_submission.csv')
MULTI_MODE_SUBMISSION = os.path.join(BASE_DIR, 'multi_mode_sample_submission.csv')


######################################
# ENVIRONMENT / MISC
######################################

# set env variable for data
os.environ["L5KIT_DATA_FOLDER"] = BASE_DIR

# Set number of threads
NUM_WORKERS = 10


if __name__ == '__main__':
    pass
