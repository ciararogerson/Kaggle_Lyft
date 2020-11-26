######################################
# IMPORTS
######################################

import numpy as np
import os
import torch
import random
import math
import pickle
import inspect
import bisect
import json
import cv2
from typing import Tuple, cast, Optional, Dict
from timeit import default_timer as timer
from tqdm import tqdm
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim.optimizer import Optimizer
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torchvision import utils
from torch.utils import model_zoo

from adamp import AdamP

import fastai
from fastai.vision import *
from fastai.callbacks import SaveModelCallback, ReduceLROnPlateauCallback
from fastai.callback import *
from fastai.torch_core import add_metrics

from l5kit.rasterization.rasterizer_builder import build_rasterizer
from l5kit.evaluation import write_pred_csv, compute_metrics_csv, read_gt_csv, write_gt_csv
from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace
from l5kit_adaptations import AgentDatasetCF, AgentDatasetTL, build_rasterizer_tl

from typing import List, Optional, Tuple

import albumentations
from albumentations import OneOf
from albumentations.augmentations.transforms import GaussNoise, MotionBlur, MedianBlur, Blur, CoarseDropout

from datetime import datetime

from settings import BASE_DIR, DATA_DIR, CACHE_DIR, MODEL_DIR, SUBMISSIONS_DIR, SINGLE_MODE_SUBMISSION, \
    MULTI_MODE_SUBMISSION, NUM_WORKERS

from configs import *

from utils import *


###########################
# SET SEEDS
###########################

SEED = 9999
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

######################################
# SET UP / GLOBALS
######################################

DEBUG = False

# GPU
USE_MULTI_GPU = True
USE_CUDA = torch.cuda.is_available()
print(' '.join(('USE_CUDA set to', str(USE_CUDA))))
DEVICE = torch.device("cuda:0" if USE_CUDA else "cpu")

# MULTIPROCESSING
CREATE_CACHE = False


##############################################
# AUGMENTATION OPS
##############################################


def make_transform(transform_list):
    """
    transform list = list of transform names:
    'blur', 'coarsedropout'
    """
    out_list = []

    if transform_list is not None:
        for transform in transform_list:
            out_list.extend(map_transform(transform))

    return albumentations.Compose(out_list)


def map_transform(transform, p=0.3):
    out = []

    if transform.lower() == 'coarsedropout':
        if np.random.uniform() <= p:
            out = [CoarseDropout(max_holes=6, min_holes=1, p=1.0)]
    if transform.lower() == 'heavycoarsedropout':
        if np.random.uniform() <= 2 * p:
            out = [CoarseDropout(max_holes=10, min_holes=1, p=1.0)]
    elif transform.lower() == 'blur':
        if np.random.uniform() <= p:
            out = [OneOf([MotionBlur(p=1.0), MedianBlur(p=1.0, blur_limit=3), Blur(p=1.0, blur_limit=3)])]

    return out


def augment_img(img, transforms):
    if transforms is not None:
        return transforms(image=img)['image']
    else:
        return img


##############################################
# DATASETS
##############################################

def get_dataset(cfg):
    if 'tl_persistence' in cfg['raster_params'] and cfg['raster_params']['tl_persistence']:
        ds = AgentDatasetTL
    else:
        ds = AgentDatasetCF
    return ds


class MotionPredictDataset(Dataset):
    """
    l5kit Motion prediction dataset wrapper.
    """

    def __init__(self,
                 cfg,
                 args_dict={},
                 str_loader='train_data_loader',
                 fn_rasterizer=build_rasterizer,
                 fn_create=None):

        self.cfg = cfg
        self.args_dict = args_dict
        self.str_loader = str_loader
        self.fn_rasterizer = fn_rasterizer
        self.fn_create = fn_create  # Function that takes a filename input and creates a model input

        self.group_scenes = self.args_dict['group_scenes'] if 'group_scenes' in self.args_dict else False
        self.weight_by_agent_count = self.args_dict[
            'weight_by_agent_count'] if 'weight_by_agent_count' in self.args_dict else 0

        self.setup()

    def setup(self):

        self.dm = LocalDataManager(None)
        self.rasterizer = self.fn_rasterizer(self.cfg, self.dm)
        self.data_zarr = ChunkedDataset(self.dm.require(self.cfg[self.str_loader]["key"])).open(cached=False)

        raw_data_file = os.path.splitext(self.cfg[self.str_loader]["key"])[0].replace('scenes/', '')

        if 'mask_path' in self.cfg[self.str_loader]:
            mask = np.load(self.cfg[self.str_loader]['mask_path'])["arr_0"]
            self.ds = get_dataset(self.cfg)(raw_data_file, self.cfg, self.str_loader, self.data_zarr, self.rasterizer,
                                            agents_mask=mask)
        else:
            self.ds = get_dataset(self.cfg)(raw_data_file, self.cfg, self.str_loader, self.data_zarr, self.rasterizer)

        self.sample_size = min(self.cfg[self.str_loader]['samples_per_epoch'], len(self.ds))

        self.shuffle = True if 'train_data_loader' in self.str_loader else False

        self.add_output = True if self.str_loader == 'test_data_loader' else False  # Add timestamp and track_id to output

        self.set_all_idx()

    def __getitem__(self, index):

        idx = self.map_index(index)

        out = self.fn_create(self.ds, idx, self.args_dict, self.cfg, self.str_loader)

        # Add timestamps and track_ids in the case of test/val
        if self.add_output:
            return out
        else:
            return out[:-2]

    def __len__(self):
        return self.sample_size

    def set_all_idx(self):
        """
        Reset sample indexes for the whole dataset
        """
        self.current_idx = 0

        if self.str_loader in ['test_data_loader', 'val_data_loader']:

            self.set_all_idx_default()

        elif self.group_scenes:

            self.set_all_idx_group_scenes()

        elif self.weight_by_agent_count > 0:

            self.set_all_idx_weight_by_agent_count()

        else:

            self.set_all_idx_default()

    def set_all_idx_default(self):

        self.all_idx = list(range(len(self.ds)))
        if self.shuffle: random.shuffle(self.all_idx)

    def set_all_idx_group_scenes(self):

        def grp_range(a):
            idx = a.cumsum()
            id_arr = np.ones(idx[-1], dtype=int)
            id_arr[0] = 0
            id_arr[idx[:-1]] = -a[:-1] + 1
            return id_arr.cumsum()

        # First shuffle the indices to ensure that there is no order in which the scene samples occur
        idx = list(range(len(self.ds)))
        random.shuffle(idx)

        if DEBUG: print('Shuffling scene indices...')
        self.ds.scene_indices = [self.ds.scene_indices[i] for i in idx]
        self.ds.frame_indices = [self.ds.frame_indices[i] for i in idx]
        self.ds.timestamps = [self.ds.timestamps[i] for i in idx]
        self.ds.track_ids = [self.ds.track_ids[i] for i in idx]

        # Within each scene, number the agents/frames 0 -> n
        count = np.unique(self.ds.scene_indices, return_counts=1)[1]
        scene_cumcount = grp_range(count)[np.argsort(self.ds.scene_indices, kind='mergesort').argsort(
            kind='mergesort')]  # Use mergesort to guarantee same results each time

        # Create all_idx by selecting one agent/frame from each scene consecutively
        cumcounts = range(int(scene_cumcount.max())) if self.weight_by_agent_count <= 0 else range(
            min(self.weight_by_agent_count, int(scene_cumcount.max())))
        self.all_idx = list(np.concatenate([np.argwhere(scene_cumcount == i).reshape(-1, ) for i in tqdm(cumcounts)]))

    def set_all_idx_weight_by_agent_count(self):

        # Calculate the count for each frame index in the datasetd = number of agents in that frame
        frame_bincounts = np.bincount(self.ds.frame_indices)
        agent_counts = frame_bincounts[self.ds.frame_indices]

        # Select a sample from each agent_count group
        # For e.g. if there are 3000 samples where there are two agents in the frame and self.weight_by_agent_count = 1, we sample 1500 of these 3000 agents
        # Use weight_by_agent_count to govern how agressively we sub-sample: The lower the number the more we downweight frames with many agents.
        all_idx = []
        for agent_count in sorted(np.unique(agent_counts)):
            _idx = np.argwhere(agent_counts == agent_count).reshape(-1, )

            select_count = int(len(_idx) * min(1, (self.weight_by_agent_count / agent_count)))

            all_idx.append(np.random.choice(_idx, size=select_count))

        all_idx = np.concatenate(all_idx)

        random.shuffle(all_idx)

        self.all_idx = all_idx

        if DEBUG: print(
            ' : '.join(('Reset all_idx, total sample length', str(len(self.all_idx)), 'from', str(len(self.ds)))))

    def map_index(self, index):

        if self.current_idx + index >= len(self.all_idx):
            self.set_all_idx()
        return self.all_idx[self.current_idx + index]

    def sample_ds(self):
        """
        Select self.sample_size indices from the dataset.
        If you have already sampled the dataset in its entirety then reshuffle
        and start sampling again.
        """
        if DEBUG: print('Sampling dataset...')

        if self.current_idx >= len(self.all_idx):
            self.set_all_idx()
        else:
            self.current_idx += self.sample_size

    def plot_index(self, index):

        data = self.ds[index]

        im = data["image"].transpose(1, 2, 0)
        im = self.ds.rasterizer.to_rgb(im)
        target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2],
                                                   data["raster_from_world"])
        draw_trajectory(im, target_positions_pixels, TARGET_POINTS_COLOR, yaws=data["target_yaws"], radius=1)

        plt.imshow(im[::-1])
        plt.show()


class MultiMotionPredictDataset(Dataset):
    """
    Holder for multiple MotionPredictDatasets
    """

    def __init__(self,
                 cfg,
                 args_dict={},
                 str_loader=['train_data_loader'],
                 fn_rasterizer=build_rasterizer,
                 fn_create=None):

        self.cfg = cfg
        self.args_dict = args_dict
        self.str_loader = str_loader
        self.fn_rasterizer = fn_rasterizer
        self.fn_create = fn_create  # Function that takes a filename input and creates a model input

        self.setup()

    def setup(self):

        assert isinstance(self.str_loader,
                          (list, tuple)), 'str_loader must be a list/tuple for use in MultiMotionPredictDataset '

        self.dataset_list = [
            MotionPredictDataset(self.cfg, self.args_dict, str_loader, self.fn_rasterizer, self.fn_create) for
            str_loader in self.str_loader]

        self.dataset_sizes = [len(dataset) for dataset in self.dataset_list]

        self.sample_size = int(np.sum(self.dataset_sizes))

        self.cumulative_dataset_sizes = np.cumsum(self.dataset_sizes)

        self.shuffle = True if all(['train_data_loader' in s for s in self.str_loader]) else False

        self.all_idx = list(range(self.sample_size))
        if self.shuffle: random.shuffle(self.all_idx)

    def sample_ds(self):

        if DEBUG: print('Sampling all sub datasets...')

        for ds in self.dataset_list:
            ds.sample_ds()

    def __getitem__(self, index):

        dataset_loc = np.argmax(np.less(index, self.cumulative_dataset_sizes))
        idx = index if dataset_loc == 0 else index - self.cumulative_dataset_sizes[dataset_loc - 1]

        try:
            out = self.dataset_list[dataset_loc][idx]
        except:
            print('********************************')
            print(self.str_loader[dataset_loc])
            print(idx)
            print('********************************')

        return out

    def __len__(self):
        return self.sample_size


def return_idx(dataset, idx):
    return dataset[idx]


def get_cache_filename(idx, args_dict, cfg, str_fn_create, str_data_loader):
    str_input_size = '_'.join(([str(i) for i in cfg['raster_params']['raster_size']]))
    str_pixel_size = '_'.join(([str(i) for i in cfg['raster_params']['pixel_size']]))
    str_ego_center = '_'.join(([str(i) for i in cfg['raster_params']['ego_center']]))
    str_history_num_frames = str(args_dict['history_num_frames'])
    str_future_num_frames = str(args_dict['future_num_frames'])

    idx_filename = str(idx) + '.pkl'
    subfolder = os.path.join(CACHE_DIR, '_'.join((str_fn_create, str_data_loader, cfg['raster_params']['map_type'],
                                                  str_input_size, str_pixel_size, str_ego_center,
                                                  str_history_num_frames, str_future_num_frames)))

    return os.path.join(subfolder, idx_filename)


def double_channel_agents_ego_map_transform(dataset, idx, args_dict, cfg, str_data_loader, info=False, info_dict=None):
    """
    double_channel_agents_ego_map tailored to multi mode output model
    including centroid and raster_from_world matrix
    """
    if info:
        n_input_channels = 5  # Each ego/agent is condensed into two channels, each map is condensed into 1
        n_output = info_dict['n_modes'] + (info_dict['future_num_frames'] * 3 * info_dict[
            'n_modes'])  # n_confs + (future_num_frames * (x, y, yaws) * modes)
        return n_input_channels, n_output

    data = check_load(
        get_cache_filename(idx, args_dict, cfg, 'double_channel_agents_ego_map_transform', str_data_loader),
        return_idx, idx, CREATE_CACHE, (dataset, idx))

    im = data["image"].transpose(1, 2, 0)

    n, im_map, im_agents_history, im_agents_current, im_ego_history, im_ego_current = split_im(im)

    history_idx = generate_history_idx(n - 1, args_dict['sample_history_num_frames'], args_dict['SHUFFLE'])

    im_map = np.sum(im_map, axis=-1)

    im_agents_history = np.sum(im_agents_history[:, :, history_idx], axis=-1)

    im_ego_history = np.sum(im_ego_history[:, :, history_idx], axis=-1)

    im_reduced = np.stack([im_agents_history, im_agents_current, im_ego_history, im_ego_current, im_map], axis=-1)

    transforms = make_transform(args_dict['TRANSFORMS']) if 'TRANSFORMS' in args_dict else None
    im_reduced = augment_img(im_reduced, transforms)

    x = numpy_to_torch(im_reduced)

    y, transform_matrix, centroid, ego_center = create_y_transform_tensor(data, cfg)

    return [x, transform_matrix, centroid, ego_center], y, int(data['timestamp']), int(data['track_id'])


def double_channel_agents_ego_map_dayhour_tl(dataset, idx, args_dict, cfg, str_data_loader, info=False, info_dict=None):
    """
    double_channel_agents_ego_map tailored to multi mode output model
    including centroid and raster_from_world matrix.
    Includes channel for traffic light persistence, and another for day/hour
    """
    if info:
        n_input_channels = 7  # Each ego/agent is condensed into two channels, each map is condensed into 1, traffic light persistence + day hour
        n_output = info_dict['n_modes'] + (info_dict['future_num_frames'] * 3 * info_dict[
            'n_modes'])  # n_confs + (future_num_frames * (x, y, yaws) * modes)
        return n_input_channels, n_output

    data = check_load(
        get_cache_filename(idx, args_dict, cfg, 'double_channel_agents_ego_map_dayhour_tl', str_data_loader),
        return_idx, idx, CREATE_CACHE, (dataset, idx))

    im = data["image"].transpose(1, 2, 0)

    n, im_map, im_agents_history, im_agents_current, im_ego_history, im_ego_current, im_tl = split_im_tl(im)

    history_idx = generate_history_idx(n - 1, args_dict['sample_history_num_frames'], args_dict['SHUFFLE'])

    im_map = np.sum(im_map, axis=-1)

    im_tl = im_tl * 2

    im_agents_history = np.sum(im_agents_history[:, :, history_idx], axis=-1)

    im_ego_history = np.sum(im_ego_history[:, :, history_idx], axis=-1)

    im_reduced = np.stack([im_agents_history, im_agents_current, im_ego_history, im_ego_current, im_map, im_tl],
                          axis=-1)

    transforms = make_transform(args_dict['TRANSFORMS']) if 'TRANSFORMS' in args_dict else []
    im_reduced = augment_img(im_reduced, transforms)

    _date = datetime.fromtimestamp(data['timestamp'] // 10 ** 9)
    weekday = _date.weekday()
    hour = _date.hour
    dayhour = 10 * (weekday + (hour / 25))

    im_reduced = np.concatenate([im_reduced, np.ones(im_reduced.shape[:2] + (1,)) * dayhour], axis=-1)

    x = numpy_to_torch(im_reduced)

    y, transform_matrix, centroid, ego_center = create_y_transform_tensor(data, cfg)

    return [x, transform_matrix, centroid, ego_center], y, int(data['timestamp']), int(data['track_id'])


def double_channel_agents_ego_map_relativecoords(dataset, idx, args_dict, cfg, str_data_loader, info=False,
                                                 info_dict=None):
    """
    double_channel_agents_ego_map tailored to multi mode output model
    including centroid and raster_from_world matrix.
    Includes x/y relative coordinates as 2 additional channels
    """
    if info:
        n_input_channels = 7  # Each ego/agent is condensed into two channels, each map is condensed into 1
        n_output = info_dict['n_modes'] + (info_dict['future_num_frames'] * 3 * info_dict[
            'n_modes'])  # n_confs + (future_num_frames * (x, y, yaws) * modes)
        return n_input_channels, n_output

    data = check_load(
        get_cache_filename(idx, args_dict, cfg, 'double_channel_agents_ego_map_transform', str_data_loader),
        return_idx, idx, CREATE_CACHE, (dataset, idx))

    im = data["image"].transpose(1, 2, 0)

    n, im_map, im_agents_history, im_agents_current, im_ego_history, im_ego_current = split_im(im)

    history_idx = generate_history_idx(n - 1, args_dict['sample_history_num_frames'], args_dict['SHUFFLE'])

    im_map = np.sum(im_map, axis=-1)

    im_agents_history = np.sum(im_agents_history[:, :, history_idx], axis=-1)

    im_ego_history = np.sum(im_ego_history[:, :, history_idx], axis=-1)

    im_coords = generate_relative_coordinate_channels(cfg)

    im_reduced = np.stack([im_agents_history, im_agents_current, im_ego_history, im_ego_current, im_map], axis=-1)

    transforms = make_transform(args_dict['TRANSFORMS']) if 'TRANSFORMS' in args_dict else None
    im_reduced = augment_img(im_reduced, transforms)

    im_reduced = np.concatenate([im_reduced, im_coords], axis=-1)

    x = numpy_to_torch(im_reduced)

    y, transform_matrix, centroid, ego_center = create_y_transform_tensor(data, cfg)

    return [x, transform_matrix, centroid, ego_center], y, int(data['timestamp']), int(data['track_id'])


def split_im(im):
    """
    Split image into

    - agents current
    - agents history
    - ego current
    - ego history
    - semantic map
    """
    n = (im.shape[-1] - 3) // 2

    im_agents_current = im[:, :, 0]
    im_agents_history = im[:, :, 1:n]

    im_ego_current = im[:, :, n]
    im_ego_history = im[:, :, n + 1:-3]

    im_map = im[:, :, -3:]

    return n, im_map, im_agents_history, im_agents_current, im_ego_history, im_ego_current


def split_im_tl(im):
    """
    Split image into

    - agents current
    - agents history
    - ego current
    - ego history
    - semantic map
    - traffic light channel
    """
    im_tl = im[:, :, -1]
    n, im_map, im_agents_history, im_agents_current, im_ego_history, im_ego_current = split_im(im[:, :, :-1])

    return n, im_map, im_agents_history, im_agents_current, im_ego_history, im_ego_current, im_tl


def generate_history_idx(n, history_num_frames, shuffle=False):
    """
    Select frames from history
    If shuffle, select random frames.
    If not shuffle, select evenly spaced frames.
    """
    if shuffle:
        history_idx = random.sample(list(range(n)), history_num_frames)
    else:
        n_history = n // history_num_frames
        history_idx = list(range(0, n, n_history))
    return sorted(history_idx[:n])


def generate_coordinate_channels(cfg):
    h = cfg['raster_params']['raster_size'][0]

    ch_h, ch_v = np.meshgrid(np.arange(h), np.arange(h))
    ch_c = np.sqrt((ch_h - cfg['raster_params']['ego_center'][0] * h) ** 2 + (
                ch_v - cfg['raster_params']['ego_center'][1] * h) ** 2)

    coords = np.stack([ch_v, ch_h, ch_c], axis=-1)

    return coords


def generate_relative_coordinate_channels(cfg):
    h = cfg['raster_params']['raster_size'][0]

    ch_h, ch_v = np.meshgrid(np.arange(h), np.arange(h))

    h_offset = cfg['raster_params']['ego_center'][0] * h
    v_offset = cfg['raster_params']['ego_center'][1] * h

    coords = np.stack([ch_v - v_offset, ch_h - h_offset], axis=-1)

    return coords


def create_y_transform_tensor(data, cfg):
    # Target_positions are in raster coordinates in metres
    y_pos_transform = torch.Tensor(data['target_positions']).float()
    yaws = torch.Tensor(data['target_yaws']).float()
    y_avail = torch.Tensor(data['target_availabilities'].reshape(-1, 1)).float()

    world_from_agent = torch.Tensor(data['world_from_agent']).float()
    centroid = torch.Tensor(data['centroid'][None, :]).float()

    if 'ego_center' in data:
        ego_center = torch.Tensor(data['ego_center'][None, :]).float()
    else:
        ego_center = torch.Tensor(np.array(cfg['raster_params']['ego_center'])[None, :]).float()

    # y_pos = world coordinates
    y_pos = transform_points(y_pos_transform, world_from_agent) - centroid[:2]

    # CHECKS:
    if False:
        # If we go from y_pos_transform -> y_pos do we get back to the original correctly?
        # (Checking for use of reverse_transform_y later...)
        world_pos = transform_points(data['target_positions'], data['world_from_agent']) - centroid.numpy()[:2]
        y_pos_test = reverse_transform_y(y_pos_transform.clone().unsqueeze(0), centroid, world_from_agent, 1).squeeze(0)
        np.allclose(y_pos_test.numpy(), world_pos, atol=0.001)

    y = torch.cat([y_pos, yaws, y_pos_transform, yaws, y_avail], dim=1)

    return y, world_from_agent, centroid, ego_center


def reverse_transform_y(y, centroid, world_from_agent, n_modes, run_check=False):
    """
    Transform agent -> world coordinates
    """
    if run_check: y_orig = y.clone()

    device = y.device

    is_4d = y.ndim == 4

    if not is_4d:
        y = y[None, :, :, :]
        centroid = centroid[None, :, :]
        world_from_agent = world_from_agent[None, :, :]

    batch_size = y.shape[0]
    n_future_frames = y.shape[2]

    y = torch.cat([y, torch.ones((batch_size, n_modes, n_future_frames, 1)).to(device)], dim=3)
    y = torch.stack([torch.matmul(world_from_agent.to(device), y[:, i].transpose(1, 2)) for i in range(y.shape[1])],
                    dim=1)
    y = y.transpose(2, 3)[:, :, :, :2]

    if run_check:
        # Check that matrix transform works the same as the l5kit version
        test = transform_points(y_orig[0, 0], world_from_agent[0])
        np.allclose(test.detach().cpu().numpy(), y[0, 0].detach().cpu().numpy())

    y = y - centroid[:, None, :, :].to(device)

    if not is_4d:
        y = y.squeeze(0)

    return y


def numpy_to_torch(img):
    img = img.astype(np.float32)
    img = img.transpose(2, 0, 1)
    img = torch.tensor(img)
    with torch.no_grad():
        img = img / 255
    return img


def batch_numpy_to_torch(img):
    img = img.astype(np.float32)
    img = np.moveaxis(img, -1, 1)
    img = torch.tensor(img)
    with torch.no_grad():
        img = img / 255
    return img


def batch_torch_to_numpy(img):
    img = img * 255
    img = img.cpu().numpy()
    img = np.moveaxis(img, 1, -1)
    return img


##############################################
# DATA LOADERS
##############################################

def create_data_loaders(fn_rasterizer, fn_create, fn_cfg, cfg_model_params, input_size, pixel_size, ego_center,
                        batch_size, num_workers, args_dicts, str_loaders, drop_last=False):
    samples_per_epoch = args_dicts[0]['samples_per_epoch']
    sample_history_num_frames = args_dicts[0]['sample_history_num_frames']
    history_num_frames = args_dicts[0]['history_num_frames']
    history_step_size = args_dicts[0]['history_step_size']
    future_num_frames = args_dicts[0]['future_num_frames']
    max_agents = args_dicts[0]['max_agents']
    n_modes = args_dicts[0]['n_modes']
    str_network = args_dicts[0]['str_network']

    n_input_channels, n_output_channels = fn_create(None, None, None, None, None, info=True, info_dict=args_dicts[0])

    cfg = fn_cfg(str_network, cfg_model_params, input_size, pixel_size, ego_center, batch_size, num_workers,
                 samples_per_epoch, sample_history_num_frames, history_num_frames, future_num_frames, n_modes,
                 max_agents, n_input_channels, n_output_channels, history_step_size)

    data_loaders = [
        create_loader(args_dict['clsDataset'], fn_rasterizer, fn_create, cfg, args_dict, str_loader, drop_last) for
        args_dict, str_loader in zip(args_dicts, str_loaders)]

    return tuple(data_loaders) + (cfg,)


def create_loader(clsDataset, fn_rasterizer, fn_create, cfg, args_dict, str_loader, drop_last=False):
    _dataset = clsDataset(cfg, args_dict, str_loader, fn_rasterizer, fn_create)

    # Checks:
    # _dataset.plot_index(0)
    for i in range(1): _dataset[i]

    if isinstance(str_loader, (list, tuple)):
        loader = DataLoader(_dataset,
                            shuffle=cfg[str_loader[0]]["shuffle"],
                            batch_size=cfg[str_loader[0]]["batch_size"],
                            num_workers=cfg[str_loader[0]]["num_workers"],
                            pin_memory=False if USE_CUDA else True,
                            drop_last=drop_last)
    else:
        loader = DataLoader(_dataset,
                            shuffle=cfg[str_loader]["shuffle"],
                            batch_size=cfg[str_loader]["batch_size"],
                            num_workers=cfg[str_loader]["num_workers"],
                            pin_memory=False if USE_CUDA else True,
                            drop_last=drop_last)

    return loader


##############################################
# VISUALISATION
##############################################

def plot_trajectories(trajectory_list, batch_size, future_num_frames):
    fig = plt.figure(figsize=(12, 12))
    n_trajectories = len(trajectory_list)

    trajectories = [trajectory[:, 0].reshape(batch_size, future_num_frames, -1)[:, :, :2] for trajectory in
                    trajectory_list]

    for i in range(batch_size):
        for j, trajectory in enumerate(trajectories):
            plt.subplot(1, n_trajectories, j + 1)
            plt.plot(trajectory[i, :, 0].numpy(), trajectory[i, :, 1].numpy())

    plt.show()


##############################################
# LOSSES
##############################################


def data_transform_to_modes(pred, truth):
    """
    Convert model output format to format compatible with l5kit's neg_multi_log_likelihood
    pred = centroid (x + y), n_modes (confidence for each mode) + (future_num_frames * 3 (x, y, yaw) * 2 (original, img coordinates))
    """

    batch_size = pred.shape[0]
    centroid = pred[:, :2]

    pred = pred[:, 2:]

    n_modes = max(1, pred.shape[1] // (truth.shape[1] * (truth.shape[2] - 1) + 1))  # truth includes mask
    future_num_frames = truth.shape[1]
    n_points = 3  # (x, y, yaws)

    # Extract mask/truth
    mask = truth[:, :, -1]
    mask = torch.stack([mask] * (truth.shape[-1] - 1), dim=-1)
    mask = mask.reshape(batch_size, -1)

    if n_modes > 1:
        # Extract confidence
        conf = pred[:, :n_modes].softmax(dim=1)
        pred = pred[:, n_modes:]
    else:
        conf = torch.ones((batch_size, n_modes))

    # Truth is created by stacking y_orig, yaws, y_transform, yaws, mask
    # We now need to extract orig/transform
    truth = truth[:, :, :-1].reshape(batch_size, future_num_frames, -1)
    truth_orig = truth[:, :, :n_points].reshape(batch_size, -1)
    truth_transform = truth[:, :, n_points:].reshape(batch_size, -1)

    # Split pred into original vs image coordinates
    # pred is created via torch.cat...x_orig.reshape(batch_size, -1), x_transform.reshape(batch_size, -1)
    n = pred.shape[-1] // 2
    pred_orig = pred[:, :n].reshape(batch_size, n_modes, -1)
    pred_transform = pred[:, n:].reshape(batch_size, n_modes, -1)

    # Sanity check:
    if False:
        plot_trajectories([truth_orig.unsqueeze(1), truth_transform.unsqueeze(1)], batch_size, future_num_frames)
        plot_trajectories([pred_orig.detach().cpu(), pred_transform.detach().cpu()], batch_size, future_num_frames)

    return pred_orig, pred_transform, truth_orig, truth_transform, conf, mask, batch_size, n_modes, future_num_frames, centroid


def torch_neg_multi_log_likelihood(gt, pred, confidences, avails, use_weights=False):
    """
    pytorch version of l5kit's neg_multi_log_likelihood
    """

    # add modes and cords
    gt = gt.unsqueeze(1)
    avails = avails.unsqueeze(1).unsqueeze(-1)

    if use_weights:
        weights = torch.ones_like(avails)
        weights[:, :, :-10,
        :] = 0.5  # sum(weights for last 10 points**2) = 10 => 40 * w**2 = 10 => w = sqrt(10/40) = 0.5
        avails = avails * weights

    # error (batch_size, num_modes, future_len), reduce coords and use availability
    error = torch.sum(((gt - pred) * avails) ** 2, dim=-1)

    with np.errstate(divide="ignore"):  # when confidence is 0 log goes to -inf, but we're fine with it
        # error (batch_size, num_modes)
        error = torch.log(confidences) - 0.5 * torch.sum(error, dim=-1)  # reduce time

    # use max aggregator on modes for numerical stability
    # error (batch_size, num_modes)
    max_value, _ = error.max(dim=1, keepdim=True)  # error are negative at this point, so max() gives the minimum one
    error = -torch.log(torch.sum(torch.exp(error - max_value), dim=-1, keepdim=True)) - max_value  # reduce modes
    # print("error", error)
    return error


def neg_log_likelihood_transform(pred, truth, calctype='transform', reduction='mean'):
    """
    pred = n_modes (confidence for each mode) + (future_num_frames * 3 (x, y, yaw) * 2)
    """
    pred_orig, pred_transform, truth_orig, truth_transform, conf, mask, batch_size, n_modes, future_num_frames, centroid = data_transform_to_modes(
        pred.cpu(), truth.cpu())

    if calctype.lower() == 'orig':
        nll = torch_neg_multi_log_likelihood(truth_orig.reshape(batch_size, future_num_frames, -1)[:, :, :2],
                                             pred_orig.reshape(batch_size, n_modes, future_num_frames, -1)[:, :, :, :2],
                                             # ignore yaws
                                             conf.reshape(batch_size, n_modes),
                                             mask.reshape(batch_size, future_num_frames, -1)[:, :, 0])
    elif calctype.lower() == 'transform':
        nll = torch_neg_multi_log_likelihood(truth_transform.reshape(batch_size, future_num_frames, -1)[:, :, :2],
                                             pred_transform.reshape(batch_size, n_modes, future_num_frames, -1)[:, :, :,
                                             :2],  # ignore yaws
                                             conf.reshape(batch_size, n_modes),
                                             mask.reshape(batch_size, future_num_frames, -1)[:, :, 0])
    else:
        nll = torch.Tensor([0]).float()

    if reduction.lower() == 'mean':
        return nll.mean()
    elif reduction.lower() == 'sum':
        return nll.sum()
    else:
        return nll


def neg_log_likelihood_transform_orig(pred, truth, reduction='mean'):
    # For assessing original coordinate system under transform data format
    return neg_log_likelihood_transform(pred, truth, calctype='orig', reduction=reduction)


def neg_log_likelihood_weighted(pred, truth, calctype='transform', reduction='mean'):
    """
    pred = n_modes (confidence for each mode) + (future_num_frames * 3 (x, y, yaw) * 2)
    """
    pred_orig, pred_transform, truth_orig, truth_transform, conf, mask, batch_size, n_modes, future_num_frames, centroid = data_transform_to_modes(
        pred.cpu(), truth.cpu())

    if calctype.lower() == 'orig':
        nll = torch_neg_multi_log_likelihood(truth_orig.reshape(batch_size, future_num_frames, -1)[:, :, :2],
                                             pred_orig.reshape(batch_size, n_modes, future_num_frames, -1)[:, :, :, :2],
                                             # ignore yaws
                                             conf.reshape(batch_size, n_modes),
                                             mask.reshape(batch_size, future_num_frames, -1)[:, :, 0],
                                             use_weights=True)
    elif calctype.lower() == 'transform':
        nll = torch_neg_multi_log_likelihood(truth_transform.reshape(batch_size, future_num_frames, -1)[:, :, :2],
                                             pred_transform.reshape(batch_size, n_modes, future_num_frames, -1)[:, :, :,
                                             :2],  # ignore yaws
                                             conf.reshape(batch_size, n_modes),
                                             mask.reshape(batch_size, future_num_frames, -1)[:, :, 0],
                                             use_weights=True)
    else:
        nll = torch.Tensor([0]).float()

    if reduction.lower() == 'mean':
        return nll.mean()
    elif reduction.lower() == 'sum':
        return nll.sum()
    else:
        return nll


@dataclass
class MultiModeNegLogLossTransform(Callback):

    def on_epoch_begin(self, **kwargs):
        self.nll = torch.tensor([])
        self.nll_transform = torch.tensor([])

    def on_batch_end(self, last_output: Tensor, last_target: Tensor, **kwargs):
        nll_val = neg_log_likelihood_transform(last_output, last_target, calctype='orig', reduction='none')
        nll_val_transform = neg_log_likelihood_transform(last_output, last_target, calctype='transform',
                                                         reduction='none')
        self.nll = torch.cat((self.nll, nll_val))
        self.nll_transform = torch.cat((self.nll_transform, nll_val_transform))

    def on_epoch_end(self, last_metrics, **kwargs):
        print('nll orig / transform ' + str(self.nll.mean()) + ' / ' + str(self.nll_transform.mean()))
        return add_metrics(last_metrics, self.nll.mean())


##############################################
# MODELS
##############################################

class Network(object):

    def __init__(self,
                 net,
                 train_loader,
                 val_loader,
                 lr=0.001,
                 weight_decay=0,
                 step_lr=False,
                 init_model_weights_path=None,
                 model_checkpoint_path=os.path.join(MODEL_DIR, 'model-checkpoint.pth'),
                 optimizer_checkpoint_path=os.path.join(MODEL_DIR, 'opt-checkpoint.pth'),
                 save_net=False):

        super(Network, self).__init__()

        self.net = net

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_net = save_net

        self.lr = lr
        self.weight_decay = weight_decay
        # self.optimizer = torch.optim.Adam(self.get_opt_params(), lr=self.lr, weight_decay=self.weight_decay)
        self.optimizer = AdamP(self.get_opt_params(), lr=self.lr, weight_decay=self.weight_decay)

        if step_lr:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=2, gamma=0.5, last_epoch=-1)
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.3, patience=50,
                                                                        verbose=True)

        self.history = {'train_loss': [], 'val_loss': [], 'model_loss': []}
        self.epochs_done = 0
        self.iteration = 0
        self.device = DEVICE

        self.init_model_weights_path = init_model_weights_path
        self.model_checkpoint_path = model_checkpoint_path
        self.optimizer_checkpoint_path = optimizer_checkpoint_path

        self.confusion_matrix = None
        self.n_classes = 0

    def get_opt_params(self):
        return self.net.parameters()

    def test_transform(self, data_loader, silent=False, df_only=True):

        self.set_state('eval')

        assert hasattr(data_loader.dataset, 'add_output'), 'data_loader must have addtional timestamp/track_id output'
        add_output = data_loader.dataset.add_output
        data_loader.dataset.add_output = True

        # Added to ensure when running for val dataset it uses the whole thing
        orig_sample_size = data_loader.dataset.sample_size
        if data_loader.dataset.sample_size < len(data_loader.dataset.ds):
            print(' '.join(('Setting data_loader sample size from', str(orig_sample_size), 'to',
                            str(len(data_loader.dataset.ds)))))
            data_loader.dataset.sample_size = len(data_loader.dataset.ds)

        y_pred = []
        y_conf = []
        timestamps = []
        track_ids = []
        centroids = []

        with tqdm(total=len(data_loader), desc="Test transform prediction", leave=False, disable=silent) as pbar:

            for data in data_loader:
                with torch.no_grad():
                    x = data[0]
                    pseudo_y = data[1]
                    timestamp = data[2]
                    track_id = data[3]

                    out = self.predict_net(x)

                    pred_orig, pred_transform, truth_orig, truth_transform, conf, mask, batch_size, n_modes, future_num_frames, centroid = data_transform_to_modes(
                        out, pseudo_y)

                    # Shape predictions correctly and take just the first two (target_x, target_y)
                    pred = pred_orig.reshape(batch_size, n_modes, future_num_frames, -1)[:, :, :50, :2]

                y_pred.append(pred.detach().cpu().numpy())
                y_conf.append(conf.detach().cpu().numpy())
                timestamps.append(timestamp.numpy())
                track_ids.append(track_id.numpy())
                centroids.append(centroid.detach().cpu().numpy())

                pbar.update()

        test_dict = {'preds': np.concatenate(y_pred), 'conf': np.concatenate(y_conf),
                     'centroids': np.concatenate(centroids), 'timestamps': np.concatenate(timestamps),
                     'track_ids': np.concatenate(track_ids)}

        data_loader.dataset.add_output = add_output
        data_loader.dataset.sample_size = orig_sample_size

        return test_dict

    def test_tta_10(self, data_loader, silent=False, df_only=True, n_tta=10):

        self.set_state('eval')

        assert hasattr(data_loader.dataset, 'add_output'), 'data_loader must have addtional timestamp/track_id output'
        add_output = data_loader.dataset.add_output
        data_loader.dataset.add_output = True

        # Added to ensure when running for val dataset it uses the whole thing
        orig_sample_size = data_loader.dataset.sample_size
        if data_loader.dataset.sample_size < len(data_loader.dataset.ds):
            print(' '.join(('Setting data_loader sample size from', str(orig_sample_size), 'to',
                            str(len(data_loader.dataset.ds)))))
            data_loader.dataset.sample_size = len(data_loader.dataset.ds)

        transforms = albumentations.Compose([OneOf([MotionBlur(p=0.5),
                                                    Blur(p=0.5, blur_limit=3)]),
                                             OneOf([CoarseDropout(max_holes=6, min_holes=1, max_height=8, max_width=8,
                                                                  p=0.5),
                                                    CoarseDropout(max_holes=12, min_holes=1, max_height=4, max_width=4,
                                                                  p=0.5)])])

        y_pred = []
        y_conf = []
        timestamps = []
        track_ids = []
        centroids = []

        for data in tqdm(data_loader, total=len(data_loader), desc="Test tta prediction"):

            with torch.no_grad():

                x = data[0]
                pseudo_y = data[1]
                timestamp = data[2]
                track_id = data[3]

                out = self.predict_net(x)

                pred_orig, pred_transform, truth_orig, truth_transform, conf, mask, batch_size, n_modes, future_num_frames, centroid = data_transform_to_modes(
                    out, pseudo_y)

                # Shape predictions correctly and take just the first two (target_x, target_y)
                pred = pred_orig.reshape(batch_size, n_modes, future_num_frames, -1)[:, :, :50, :2]

                for tta in range(n_tta):
                    imgs = batch_torch_to_numpy(x[0].clone())

                    aug_img = np.stack([transforms(image=imgs[i])['image'] for i in range(imgs.shape[0])], axis=0)

                    _x = [batch_numpy_to_torch(aug_img).to(DEVICE)] + x[1:]

                    out_tta = self.predict_net(_x)

                    pred_tta, _, _, _, conf_tta, _, _, _, _, _ = data_transform_to_modes(out_tta, pseudo_y)

                    pred_tta = pred_tta.reshape(batch_size, n_modes, future_num_frames, -1)[:, :, :50, :2]

                    pred = pred + pred_tta
                    conf = conf + conf_tta

                pred = pred / (n_tta + 1)
                conf = conf / (n_tta + 1)

            y_pred.append(pred.detach().cpu().numpy())
            y_conf.append(conf.detach().cpu().numpy())
            timestamps.append(timestamp.numpy())
            track_ids.append(track_id.numpy())
            centroids.append(centroid.detach().cpu().numpy())

        test_dict = {'preds': np.concatenate(y_pred), 'conf': np.concatenate(y_conf),
                     'centroids': np.concatenate(centroids), 'timestamps': np.concatenate(timestamps),
                     'track_ids': np.concatenate(track_ids)}

        data_loader.dataset.add_output = add_output
        data_loader.dataset.sample_size = orig_sample_size

        return test_dict

    def evaluate(self, data_loader, loss_fn, silent=False, df_only=False):

        self.set_state('eval')

        model_loss = 0
        total_examples = 0
        y_pred = []
        y_actual = []

        with tqdm(total=len(data_loader), desc="Evaluation", leave=False, disable=silent) as pbar:

            for batch_idx, data in enumerate(data_loader):

                with torch.no_grad():

                    batch_size = data[-1].shape[0]

                    if isinstance(data[0], list):
                        x = [d.to(self.device) for d in data[0]]
                    else:
                        x = data[0].to(self.device)

                    y_label = data[-1].to(self.device)

                    out = self.predict_net(x)

                    loss = loss_fn(out, y_label)

                    model_loss += loss.item() * batch_size

                    y_pred.append(out.detach().cpu().numpy())
                    y_actual.append(y_label.detach().cpu().numpy())

                total_examples += batch_size
                pbar.update()

        model_loss /= total_examples

        val_dict = {'actual': np.concatenate(y_actual), 'preds': np.concatenate(y_pred)}

        if df_only:
            return val_dict
        else:
            return model_loss, val_dict

    def fit_trainloss(self, epochs, start_epoch=0, resample=True, loss_fn=neg_log_likelihood_transform):

        # Load weights
        if self.init_model_weights_path is not None:
            print(' '.join(('Loading weights from', self.init_model_weights_path)))
            assert os.path.exists(self.init_model_weights_path)
            self.net = load_weights(self.net, self.init_model_weights_path)

        # Training
        with tqdm(total=len(self.train_loader), leave=False) as pbar:

            for epoch in range(epochs):

                pbar.reset()
                pbar.set_description("Epoch %d" % (self.epochs_done + 1))

                # Resample dataset
                if resample:
                    self.train_loader.dataset.sample_ds()

                model_loss = 0
                total_examples = 0

                self.set_state('train')

                for batch_idx, data in enumerate(self.train_loader):
                    batch_size = data[-1].shape[0]

                    x = data[0]
                    y = data[1].to(self.device)

                    self.optimizer.zero_grad()

                    out = self.predict_net(x)

                    loss = loss_fn(out, y)

                    # loss.backward()
                    loss.mean().backward()  # Need the mean() for multi-gpus??
                    self.optimizer.step()

                    batch_loss = loss.item()
                    model_loss += batch_loss * batch_size
                    # self.history["train_loss"].append(batch_loss)

                    total_examples += batch_size
                    self.iteration += 1

                    pbar.set_postfix(loss=batch_loss)
                    pbar.update()

                model_loss /= total_examples
                self.epochs_done += 1

                print("Epoch: %3d, train loss: %.4f" % (self.epochs_done, model_loss))

                # val_model_loss, _ = self.evaluate(self.val_loader, loss_fn, silent=True)

                # print("              val loss: %.4f" % (val_model_loss,))

                self.scheduler.step(model_loss)

                if self.save_net and (epoch == 0 or model_loss < np.min(self.history['model_loss'])):
                    print('Saving model state...')
                    self.save_state()

                self.history["model_loss"].append(model_loss)

    def fit_fastai_transform(self, epochs, start_epoch=0, resample=True, loss_fn=neg_log_likelihood_transform):

        data = DataBunch(train_dl=self.train_loader, valid_dl=self.val_loader)

        checkpoint_path = os.path.splitext(self.model_checkpoint_path)[0]

        learn = Learner(data,
                        self.net,
                        path=checkpoint_path,
                        loss_func=loss_fn,
                        metrics=[MultiModeNegLogLossTransform()]).to_fp16()
        learn.clip_grad = 1.0
        learn.split([self.net.head])
        learn.unfreeze()

        learn_callbacks = [SaveModelCallback(learn, name=f'model', monitor='valid_loss')]

        if resample:
            class DataSamplingCallback(Callback):
                def on_epoch_begin(self, **kwargs):
                    learn.data.train_dl.dl.dataset.sample_ds()

            learn_callbacks = learn_callbacks + [DataSamplingCallback()]

        if hasattr(loss_fn, 'update'):
            class LossUpdateCallback(Callback):
                def on_epoch_begin(self, epoch: int, **kwargs):
                    loss_fn.update(epoch)

            learn_callbacks = learn_callbacks + [LossUpdateCallback()]

        if self.init_model_weights_path is not None:
            print(' '.join(('Loading weights from', self.init_model_weights_path)))
            assert os.path.exists(self.init_model_weights_path)
            learn.model = load_weights(learn.model, self.init_model_weights_path)

        if start_epoch > 0:
            assert os.path.exists(os.path.join(checkpoint_path, 'models', 'model.pth'))
            learn.load(os.path.join(checkpoint_path, 'models', 'model'))

        # learn.fit(epochs = epochs, lr=self.lr, callbacks=[SaveModelCallback(learn,name=f'model',monitor='auroc'),
        #                                        ReduceLROnPlateauCallback(learn,monitor='valid_loss', factor=10, patience=3, min_lr = 1e-10)])
        learn.fit_one_cycle(epochs, max_lr=self.lr, div_factor=100, pct_start=0.0, callbacks=learn_callbacks)

        # Load best weights
        learn.load(os.path.join(checkpoint_path, 'models', 'model'))

        self.net = learn.model.float()

        self.save_state()

        val_model_loss, val_dict = self.evaluate(self.val_loader, loss_fn, silent=True)
        print('Val loss: ' + str(val_model_loss))

        return val_dict

    def fit_fastai_ralamb(self, epochs, start_epoch=0, resample=True, loss_fn=neg_log_likelihood_transform):

        data = DataBunch(train_dl=self.train_loader, valid_dl=self.val_loader)

        checkpoint_path = os.path.splitext(self.model_checkpoint_path)[0]

        learn = Learner(data,
                        self.net,
                        path=checkpoint_path,
                        loss_func=loss_fn,
                        opt_func=Over9000,
                        metrics=[MultiModeNegLogLossTransform()]).to_fp16()
        learn.clip_grad = 1.0
        learn.split([self.net.head])
        learn.unfreeze()

        learn_callbacks = [SaveModelCallback(learn, name=f'model', monitor='valid_loss')]

        if resample:
            class DataSamplingCallback(Callback):
                def on_epoch_begin(self, **kwargs):
                    learn.data.train_dl.dl.dataset.sample_ds()

            learn_callbacks = learn_callbacks + [DataSamplingCallback()]

        if hasattr(loss_fn, 'update'):
            class LossUpdateCallback(Callback):
                def on_epoch_begin(self, epoch: int, **kwargs):
                    loss_fn.update(epoch)

            learn_callbacks = learn_callbacks + [LossUpdateCallback()]

        if self.init_model_weights_path is not None:
            print(' '.join(('Loading weights from', self.init_model_weights_path)))
            assert os.path.exists(self.init_model_weights_path)
            learn.model = load_weights(learn.model, self.init_model_weights_path)

        if start_epoch > 0:
            assert os.path.exists(os.path.join(checkpoint_path, 'models', 'model.pth'))
            learn.load(os.path.join(checkpoint_path, 'models', 'model'))

        # learn.fit(epochs = epochs, lr=self.lr, callbacks=[SaveModelCallback(learn,name=f'model',monitor='auroc'),
        #                                        ReduceLROnPlateauCallback(learn,monitor='valid_loss', factor=10, patience=3, min_lr = 1e-10)])
        learn.fit_one_cycle(epochs, max_lr=self.lr, div_factor=100, pct_start=0.0, callbacks=learn_callbacks)

        # Load best weights
        learn.load(os.path.join(checkpoint_path, 'models', 'model'))

        self.net = learn.model.float()

        self.save_state()

        val_model_loss, val_dict = self.evaluate(self.val_loader, loss_fn, silent=True)
        print('Val loss: ' + str(val_model_loss))

        return val_dict

    def fit_fastai_trainloss(self, epochs, start_epoch=0, resample=True, loss_fn=neg_log_likelihood_transform):

        data = DataBunch(train_dl=self.train_loader, valid_dl=self.val_loader)

        checkpoint_path = os.path.splitext(self.model_checkpoint_path)[0]

        learn = Learner(data,
                        self.net,
                        path=checkpoint_path,
                        loss_func=loss_fn,
                        opt_func=Over9000,
                        metrics=[MultiModeNegLogLossTransform()]).to_fp16()
        learn.clip_grad = 1.0
        learn.split([self.net.head])
        learn.unfreeze()

        learn_callbacks = [SaveModelCallback(learn, name=f'model', monitor='train_loss')]

        if resample:
            class DataSamplingCallback(Callback):
                def on_epoch_begin(self, **kwargs):
                    learn.data.train_dl.dl.dataset.sample_ds()

            learn_callbacks = learn_callbacks + [DataSamplingCallback()]

        if hasattr(loss_fn, 'update'):
            class LossUpdateCallback(Callback):
                def on_epoch_begin(self, epoch: int, **kwargs):
                    loss_fn.update(epoch)

            learn_callbacks = learn_callbacks + [LossUpdateCallback()]

        if self.init_model_weights_path is not None:
            print(' '.join(('Loading weights from', self.init_model_weights_path)))
            assert os.path.exists(self.init_model_weights_path)
            learn.model = load_weights(learn.model, self.init_model_weights_path)

        if start_epoch > 0:
            assert os.path.exists(os.path.join(checkpoint_path, 'models', 'model.pth'))
            learn.load(os.path.join(checkpoint_path, 'models', 'model_' + str(start_epoch)))

        # learn.fit(epochs = epochs, lr=self.lr, callbacks=[SaveModelCallback(learn,name=f'model',monitor='auroc'),
        #                                        ReduceLROnPlateauCallback(learn,monitor='valid_loss', factor=10, patience=3, min_lr = 1e-10)])
        learn.fit_one_cycle(epochs, max_lr=self.lr, div_factor=100, pct_start=0.0, callbacks=learn_callbacks)

        # Load best weights
        learn.load(os.path.join(checkpoint_path, 'models', 'model'))

        self.net = learn.model.float()

        self.save_state()

        val_model_loss, val_dict = self.evaluate(self.val_loader, loss_fn, silent=True)
        print('Val loss: ' + str(val_model_loss))

        return val_dict

    def set_state(self, state):

        boo_train = True if state == 'train' else False

        self.net.train(boo_train)

    def predict_net(self, x):
        return self.net(*x) if isinstance(x, list) else self.net(x)

    def save_state(self):
        torch.save(self.net.state_dict(), self.model_checkpoint_path)
        # torch.save(self.optimizer.state_dict(), self.optimizer_checkpoint_path)


class LyftResnet18Small(nn.Module):

    def __init__(self, cfg: Dict):
        super().__init__()
        self.cfg = cfg
        self.generate_network()

    def generate_network(self):
        num_in_channels = self.cfg['model_params']['n_input_channels']

        m = models.resnet18(pretrained=True)

        m.conv1 = nn.Conv2d(num_in_channels,
                            m.conv1.out_channels,
                            kernel_size=m.conv1.kernel_size,
                            stride=m.conv1.stride,
                            padding=m.conv1.padding,
                            bias=False,
                            )

        self.backbone = nn.Sequential(*list(m.children())[:-1])

        # This is 512 for resnet18 and resnet34;
        # And it is 2048 for the other resnets
        backbone_out_features = 512

        # X, Y coords for the future positions (output shape: Bx50x2)
        num_targets = self.cfg['model_params']['n_output']

        self.head = nn.Sequential(
            nn.Linear(in_features=backbone_out_features, out_features=2048),
            nn.Linear(2048, out_features=num_targets)
        )

        if USE_MULTI_GPU:
            self.backbone = nn.DataParallel(self.backbone)
            self.head = nn.DataParallel(self.head)

        self.backbone = self.backbone.to(DEVICE)
        self.head = self.head.to(DEVICE)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.head(x)

        return x


class LyftResnet18Transform(LyftResnet18Small):

    def forward(self, *_x):

        x = _x[0]
        world_from_agent = _x[1]
        centroid = _x[2]
        ego_center = _x[3]

        batch_size = x.shape[0]
        n_modes = self.cfg['model_params']['n_modes']
        future_num_frames = self.cfg['model_params']['future_num_frames']

        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.head(x)

        # We predict img coordinates transforms directly from the model,
        # but we also append original coordinates by calling reverse_transform_y() on the predictions.
        x_confs = x[:, :n_modes]
        x_transform = x[:, n_modes:].reshape(batch_size, n_modes, future_num_frames, -1)[:, :, :, :2].float()
        x_orig = reverse_transform_y(x_transform.clone(), centroid.float(), world_from_agent.float(), n_modes)

        yaws = torch.ones((batch_size, n_modes, future_num_frames, 1)).to(DEVICE)

        centroid = centroid.squeeze(1).float().to(DEVICE)

        if x.dtype == torch.float16:
            x_transform = x_transform.half()
            x_orig = x_orig.half()
            yaws = yaws.half()
            centroid = centroid.half()

        # For consistency with other models we include yaws as well as x, y coordinates
        include_yaws = True
        if include_yaws:
            x_orig = torch.cat([x_orig, yaws], dim=-1)
            x_transform = torch.cat([x_transform, yaws], dim=-1)

        out = torch.cat([centroid, x_confs, x_orig.reshape(batch_size, -1), x_transform.reshape(batch_size, -1)],
                        dim=-1)

        return out


class LyftResnest50(nn.Module):

    def __init__(self, cfg: Dict):
        super().__init__()
        self.cfg = cfg
        self.generate_network()

    def generate_network(self):
        num_in_channels = self.cfg['model_params']['n_input_channels']

        m = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50_fast_1s1x64d', pretrained=True)

        m.conv1[0] = nn.Conv2d(num_in_channels,
                               m.conv1[0].out_channels,
                               kernel_size=m.conv1[0].kernel_size,
                               stride=m.conv1[0].stride,
                               padding=m.conv1[0].padding,
                               bias=False,
                               )

        self.backbone = nn.Sequential(*list(m.children())[:-1])

        # This is 512 for resnet18 and resnet34;
        # And it is 2048 for the other resnets
        backbone_out_features = 2048

        # X, Y coords for the future positions (output shape: Bx50x2)
        num_targets = self.cfg['model_params']['n_output']

        self.head = nn.Sequential(
            nn.Linear(in_features=backbone_out_features, out_features=4096),
            nn.Linear(4096, out_features=num_targets)
        )

        if USE_MULTI_GPU:
            self.backbone = nn.DataParallel(self.backbone)
            self.head = nn.DataParallel(self.head)

        self.backbone = self.backbone.to(DEVICE)
        self.head = self.head.to(DEVICE)

    def forward(self, *_x):

        x = _x[0]
        world_from_agent = _x[1]
        centroid = _x[2]
        ego_center = _x[3]

        batch_size = x.shape[0]
        n_modes = self.cfg['model_params']['n_modes']
        future_num_frames = self.cfg['model_params']['future_num_frames']

        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.head(x)

        # We predict img coordinates transforms directly from the model,
        # but we also append original coordinates by calling reverse_transform_y() on the predictions.
        x_confs = x[:, :n_modes]
        x_transform = x[:, n_modes:].reshape(batch_size, n_modes, future_num_frames, -1)[:, :, :, :2].float()
        x_orig = reverse_transform_y(x_transform.clone(), centroid.float(), world_from_agent.float(), n_modes)

        yaws = torch.ones((batch_size, n_modes, future_num_frames, 1)).to(DEVICE)

        centroid = centroid.squeeze(1).float().to(DEVICE)

        if x.dtype == torch.float16:
            x_transform = x_transform.half()
            x_orig = x_orig.half()
            yaws = yaws.half()
            centroid = centroid.half()

        # For consistency with other models we include yaws as well as x, y coordinates
        include_yaws = True
        if include_yaws:
            x_orig = torch.cat([x_orig, yaws], dim=-1)
            x_transform = torch.cat([x_transform, yaws], dim=-1)

        out = torch.cat([centroid, x_confs, x_orig.reshape(batch_size, -1), x_transform.reshape(batch_size, -1)],
                        dim=-1)

        return out


# RAdam + LARS
class Lookahead(Optimizer):
    def __init__(self, base_optimizer, alpha=0.5, k=6):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')
        defaults = dict(lookahead_alpha=alpha, lookahead_k=k, lookahead_step=0)
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.defaults = base_optimizer.defaults
        self.defaults.update(defaults)
        self.state = defaultdict(dict)
        # manually add our defaults to the param groups
        for name, default in defaults.items():
            for group in self.param_groups:
                group.setdefault(name, default)

    def update_slow(self, group):
        for fast_p in group["params"]:
            if fast_p.grad is None:
                continue
            param_state = self.state[fast_p]
            if 'slow_buffer' not in param_state:
                param_state['slow_buffer'] = torch.empty_like(fast_p.data)
                param_state['slow_buffer'].copy_(fast_p.data)
            slow = param_state['slow_buffer']
            slow.add_(group['lookahead_alpha'], fast_p.data - slow)
            fast_p.data.copy_(slow)

    def sync_lookahead(self):
        for group in self.param_groups:
            self.update_slow(group)

    def step(self, closure=None):
        # print(self.k)
        # assert id(self.param_groups) == id(self.base_optimizer.param_groups)
        loss = self.base_optimizer.step(closure)
        for group in self.param_groups:
            group['lookahead_step'] += 1
            if group['lookahead_step'] % group['lookahead_k'] == 0:
                self.update_slow(group)
        return loss

    def state_dict(self):
        fast_state_dict = self.base_optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict['state']
        param_groups = fast_state_dict['param_groups']
        return {
            'state': fast_state,
            'slow_state': slow_state,
            'param_groups': param_groups,
        }

    def load_state_dict(self, state_dict):
        fast_state_dict = {
            'state': state_dict['state'],
            'param_groups': state_dict['param_groups'],
        }
        self.base_optimizer.load_state_dict(fast_state_dict)

        # We want to restore the slow state, but share param_groups reference
        # with base_optimizer. This is a bit redundant but least code
        slow_state_new = False
        if 'slow_state' not in state_dict:
            print('Loading state_dict from optimizer without Lookahead applied.')
            state_dict['slow_state'] = defaultdict(dict)
            slow_state_new = True
        slow_state_dict = {
            'state': state_dict['slow_state'],
            'param_groups': state_dict['param_groups'],  # this is pointless but saves code
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.param_groups = self.base_optimizer.param_groups  # make both ref same container
        if slow_state_new:
            # reapply defaults to catch missing lookahead specific ones
            for name, default in self.defaults.items():
                for group in self.param_groups:
                    group.setdefault(name, default)


class Ralamb(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(Ralamb, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Ralamb, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('Ralamb does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                # v_t
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]

                if state['step'] == buffered[0]:
                    N_sma, radam_step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        radam_step_size = math.sqrt(
                            (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                    N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        radam_step_size = 1.0 / (1 - beta1 ** state['step'])
                    buffered[2] = radam_step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # more conservative since it's an approximated value
                radam_step = p_data_fp32.clone()
                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    radam_step.addcdiv_(-radam_step_size * group['lr'], exp_avg, denom)
                else:
                    radam_step.add_(-radam_step_size * group['lr'], exp_avg)

                radam_norm = radam_step.pow(2).sum().sqrt()
                weight_norm = p.data.pow(2).sum().sqrt().clamp(0, 10)
                if weight_norm == 0 or radam_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / radam_norm

                state['weight_norm'] = weight_norm
                state['adam_norm'] = radam_norm
                state['trust_ratio'] = trust_ratio

                if N_sma >= 5:
                    p_data_fp32.addcdiv_(-radam_step_size * group['lr'] * trust_ratio, exp_avg, denom)
                else:
                    p_data_fp32.add_(-radam_step_size * group['lr'] * trust_ratio, exp_avg)

                p.data.copy_(p_data_fp32)

        return loss


def Over9000(params, alpha=0.5, k=6, *args, **kwargs):
    ralamb = Ralamb(params, *args, **kwargs)
    return Lookahead(ralamb, alpha, k)


def clean_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module' in k:
            k = k.replace(".module", '')  # removing ‘.moldule’ from key
            k = k.replace("module.", '')  # removing ‘.moldule’ from key
        new_state_dict[k] = v
    return new_state_dict


def load_weights(model, model_checkpoint_path):
    assert os.path.exists(model_checkpoint_path), 'File does not exist: ' + model_checkpoint_path
    state_dict = torch.load(model_checkpoint_path, map_location=lambda storage, loc: storage)
    try:
        model.load_state_dict(state_dict)
    except:
        model.load_state_dict(clean_state_dict(state_dict))

    return model


##############################################
# FITTING
##############################################

def fit_multitrain_motion_predict(n_epochs, train_args_dict, val_args_dict,
                                  init_model_weights_path=None,
                                  model_checkpoint_path=None, lr=3e-4,
                                  start_epoch=0,
                                  fit_fn='fit_transform', val_fn='test_transform',
                                  clsModel=LyftResnet18Transform,
                                  loss_fn=neg_log_likelihood_transform,
                                  loader_fn=double_channel_agents_ego_map_transform,
                                  str_train_loaders=['train_data_loader_100', 'train_data_loader_30'],
                                  str_val_loader='val_data_loader',
                                  cfg_fn=create_config,
                                  rasterizer_fn=build_rasterizer,
                                  cfg_model_params=None,
                                  action='fit'):
    # Init variables
    input_size = train_args_dict['INPUT_SIZE']
    pixel_size = train_args_dict['PIXEL_SIZE']
    ego_center = train_args_dict['EGO_CENTER']
    batch_size = train_args_dict['BATCH_SIZE']

    # Create data loaders
    if action == 'test':
        train_loader, val_loader = None, None
        val_fn_loader, cfg = create_data_loaders(rasterizer_fn, loader_fn, cfg_fn, cfg_model_params, input_size,
                                                 pixel_size, ego_center, batch_size, NUM_WORKERS, [val_args_dict],
                                                 ['test_data_loader'])
    else:
        train_loader, val_loader, cfg = create_data_loaders(rasterizer_fn, loader_fn, cfg_fn, cfg_model_params,
                                                            input_size, pixel_size, ego_center, batch_size, NUM_WORKERS,
                                                            [train_args_dict, val_args_dict],
                                                            [str_train_loaders, str_val_loader])
        val_fn_loader = val_loader

    # Init model
    model = clsModel(cfg=cfg)

    # Create network
    net = Network(model, train_loader, val_loader, lr=lr, init_model_weights_path=init_model_weights_path,
                  model_checkpoint_path=model_checkpoint_path, save_net=True)

    # Fit
    if action == 'fit':
        val_dict = getattr(net, fit_fn)(n_epochs, loss_fn=loss_fn)
        # Clear train_data_loader as it can be memory hogging
        train_loader, net.train_loader = None, None

    # Load best weights and predict
    load_weights(net.net, model_checkpoint_path)
    val_dict = getattr(net, val_fn)(val_fn_loader, loss_fn, df_only=True)

    # Print val results
    if action != 'test':
        print('Val results:')

    return val_dict, net


##############################################
# RUN HELPERS
##############################################

def setup_args_dicts(clsTrainDataset, clsValDataset, aug, str_network, in_size, pixel_size, ego_center, batch_size,
                     samples_per_epoch, n_epochs, sample_history_num_frames, history_num_frames, history_step_size,
                     future_num_frames, n_modes, max_agents, group_scenes, weight_by_agent_count):
    transforms = []

    if aug == 'coarsedropout':
        transforms = transforms + ['coarsedropout']
    elif aug == 'heavycoarsedropout':
        transforms = transforms + ['heavycoarsedropout']
    elif aug == 'coarsedropout_blur':
        transforms = transforms + ['coarsedropout', 'blur']
    elif aug == 'heavycoarsedropoutblur':
        transforms = transforms + ['heavycoarsedropout', 'blur']

    print(transforms)

    train_args_dict = {'INPUT_SIZE': in_size,
                       'PIXEL_SIZE': pixel_size,
                       'EGO_CENTER': ego_center,
                       'BATCH_SIZE': batch_size,
                       'TRANSFORMS': transforms,
                       'str_network': str_network,
                       'samples_per_epoch': samples_per_epoch,
                       'n_epochs': n_epochs,
                       'sample_history_num_frames': sample_history_num_frames,
                       'history_num_frames': history_num_frames,
                       'history_step_size': history_step_size,
                       'future_num_frames': future_num_frames,
                       'n_modes': n_modes,
                       'max_agents': max_agents,
                       'group_scenes': group_scenes,
                       'weight_by_agent_count': weight_by_agent_count,
                       'clsDataset': clsTrainDataset,
                       'SHUFFLE': True}

    val_args_dict = {'INPUT_SIZE': in_size,
                     'PIXEL_SIZE': pixel_size,
                     'EGO_CENTER': ego_center,
                     'BATCH_SIZE': batch_size,
                     'TRANSFORMS': None,
                     'str_network': str_network,
                     'samples_per_epoch': samples_per_epoch,
                     'n_epochs': n_epochs,
                     'sample_history_num_frames': sample_history_num_frames,
                     'history_num_frames': history_num_frames,
                     'history_step_size': history_step_size,
                     'future_num_frames': future_num_frames,
                     'n_modes': n_modes,
                     'max_agents': max_agents,
                     'group_scenes': group_scenes,
                     'clsDataset': clsValDataset,
                     'SHUFFLE': False}

    return train_args_dict, val_args_dict


def setup_test_args_dict(clsDataset, aug, str_network, in_size, pixel_size, ego_center, batch_size, samples_per_epoch,
                         n_epochs, sample_history_num_frames, history_num_frames, history_step_size, future_num_frames,
                         n_modes, max_agents, group_scenes, weight_by_agent_count):
    test_args_dict = {'INPUT_SIZE': in_size,
                      'PIXEL_SIZE': pixel_size,
                      'EGO_CENTER': ego_center,
                      'BATCH_SIZE': batch_size,
                      'TRANSFORMS': None,
                      'str_network': str_network,
                      'samples_per_epoch': samples_per_epoch,
                      'n_epochs': n_epochs,
                      'sample_history_num_frames': sample_history_num_frames,
                      'history_num_frames': history_num_frames,
                      'history_step_size': history_step_size,
                      'future_num_frames': future_num_frames,
                      'n_modes': n_modes,
                      'max_agents': max_agents,
                      'group_scenes': group_scenes,
                      'clsDataset': clsDataset,
                      'SHUFFLE': False}

    return test_args_dict


def create_base_filename(clsTrainDataset, clsValDataset, clsModel, val_fn, rasterizer_fn, loader_fn, cfg_fn, fit_fn,
                         loss_fn, in_size, pixel_size, ego_center, n_epochs, batch_size, samples_per_epoch,
                         sample_history_num_frames, history_num_frames, history_step_size, future_num_frames, n_modes,
                         max_agents, group_scenes, weight_by_agent_count, str_network, aug, model_str):
    str_train_dataset = str(clsTrainDataset).split('.')[-1].split("'")[0]
    str_val_dataset = str(clsValDataset).split('.')[-1].split("'")[0]
    str_model = str(clsModel).split('.')[-1].split("'")[0]
    str_loader_fn = str(loader_fn).split(' ')[1]
    str_cfg_fn = str(cfg_fn).split(' ')[1]
    str_loss_fn = str(loss_fn).split(' ')[1]
    str_rasterizer_fn = str(rasterizer_fn).split(' ')[1]

    str_pixel_size = '_'.join(([str(i) for i in pixel_size]))
    str_ego_center = '_'.join(([str(i) for i in ego_center]))

    # base_filename = '_'.join((val_fn, str_train_dataset, str_val_dataset, str_model, str_rasterizer_fn, str_loader_fn, str_cfg_fn, str_loss_fn, str(in_size), str_pixel_size, str_ego_center, str(n_epochs),str(batch_size), str(samples_per_epoch), str(sample_history_num_frames), str(history_num_frames), str(future_num_frames), str(n_modes), str_network, fit_fn, aug, model_str, '.pkl'))
    # base_filename = '_'.join((val_fn, str_model, str_rasterizer_fn, str_loader_fn, str_cfg_fn, str_loss_fn, str(in_size), str_pixel_size, str_ego_center, str(n_epochs),str(batch_size), str(samples_per_epoch), str(sample_history_num_frames), str(history_num_frames), str(future_num_frames), str(n_modes), str(group_scenes), str_network, fit_fn, aug, model_str, '.pkl'))
    # base_filename = '_'.join((val_fn, str_model, str_loader_fn, str_cfg_fn, str_loss_fn, str(in_size), str_pixel_size, str_ego_center, str(n_epochs),str(batch_size), str(samples_per_epoch), str(sample_history_num_frames), str(history_num_frames), str(future_num_frames), str(n_modes), str(group_scenes), str_network, fit_fn, aug, model_str, '.pkl'))
    # base_filename = '_'.join((val_fn, str_model, str_loader_fn, str_cfg_fn, str_loss_fn, str(in_size), str_pixel_size, str_ego_center, str(n_epochs),str(batch_size), str(samples_per_epoch), str(sample_history_num_frames), str(history_num_frames), str(future_num_frames), str(n_modes), str(max_agents), str_network, fit_fn, aug, model_str, '.pkl'))
    # base_filename = '_'.join((val_fn, str_model, str_loader_fn, str_cfg_fn, str_loss_fn, str(in_size), str(n_epochs),str(batch_size), str(samples_per_epoch), str(sample_history_num_frames), str(history_num_frames), str(future_num_frames), str(n_modes), str(max_agents), str(weight_by_agent_count), str_network, fit_fn, aug, model_str, '.pkl'))
    # base_filename = '_'.join((val_fn, str_model, str_loader_fn, str_cfg_fn, str_loss_fn, str(in_size), str(n_epochs),str(batch_size), str(samples_per_epoch), str(history_step_size), str(history_num_frames), str(future_num_frames), str(n_modes), str(max_agents), str(weight_by_agent_count), str_network, fit_fn, aug, model_str, '.pkl'))
    base_filename = '_'.join((val_fn, str_model, str_loader_fn, str_cfg_fn, str_loss_fn, str(in_size), str(n_epochs),
                              str(batch_size), str(samples_per_epoch), str(history_step_size), str(history_num_frames),
                              str(future_num_frames), str(n_modes), str(group_scenes), str(weight_by_agent_count),
                              str_network, fit_fn, aug, model_str, '.pkl'))

    return base_filename


def create_val_dict_filepath(clsTrainDataset, clsValDataset, clsModel, val_fn, rasterizer_fn, loader_fn, cfg_fn, fit_fn,
                             loss_fn, in_size, pixel_size, ego_center, n_epochs, batch_size, samples_per_epoch,
                             sample_history_num_frames, history_num_frames, history_step_size, future_num_frames,
                             n_modes, max_agents, group_scenes, weight_by_agent_count, str_network, aug, model_str):
    base_filename = create_base_filename(clsTrainDataset, clsValDataset, clsModel, val_fn, rasterizer_fn, loader_fn,
                                         cfg_fn, fit_fn, loss_fn, in_size, pixel_size, ego_center, n_epochs, batch_size,
                                         samples_per_epoch, sample_history_num_frames, history_num_frames,
                                         history_step_size, future_num_frames, n_modes, max_agents, group_scenes,
                                         weight_by_agent_count, str_network, aug, model_str)

    return os.path.join(DATA_DIR, 'val_' + base_filename)


def create_test_dict_filepath(clsTrainDataset, clsValDataset, clsModel, val_fn, rasterizer_fn, loader_fn, cfg_fn,
                              fit_fn, loss_fn, in_size, pixel_size, ego_center, n_epochs, batch_size, samples_per_epoch,
                              sample_history_num_frames, history_num_frames, history_step_size, future_num_frames,
                              n_modes, max_agents, group_scenes, weight_by_agent_count, str_network, aug, model_str):
    base_filename = create_base_filename(clsTrainDataset, clsValDataset, clsModel, val_fn, rasterizer_fn, loader_fn,
                                         cfg_fn, fit_fn, loss_fn, in_size, pixel_size, ego_center, n_epochs, batch_size,
                                         samples_per_epoch, sample_history_num_frames, history_num_frames,
                                         history_step_size, future_num_frames, n_modes, max_agents, group_scenes,
                                         weight_by_agent_count, str_network, aug, model_str)

    return os.path.join(DATA_DIR, 'test_' + base_filename)


def create_model_checkpoint_path(clsTrainDataset, clsValDataset, clsModel, val_fn, rasterizer_fn, loader_fn, cfg_fn,
                                 fit_fn, loss_fn, in_size, pixel_size, ego_center, n_epochs, batch_size,
                                 samples_per_epoch, sample_history_num_frames, history_num_frames, history_step_size,
                                 future_num_frames, n_modes, max_agents, group_scenes, weight_by_agent_count,
                                 str_network, aug, model_str):
    str_train_dataset = str(clsTrainDataset).split('.')[-1].split("'")[0]
    str_val_dataset = str(clsValDataset).split('.')[-1].split("'")[0]
    str_model = str(clsModel).split('.')[-1].split("'")[0]
    str_loader_fn = str(loader_fn).split(' ')[1]
    str_cfg_fn = str(cfg_fn).split(' ')[1]
    str_loss_fn = str(loss_fn).split(' ')[1]
    str_rasterizer_fn = str(rasterizer_fn).split(' ')[1]

    str_pixel_size = '_'.join(([str(i) for i in pixel_size]))
    str_ego_center = '_'.join(([str(i) for i in ego_center]))

    # model_filename = '_'.join(('model_checkpoint', str_train_dataset, str_val_dataset, str_model, str_rasterizer_fn, str_loader_fn, str_cfg_fn, str_loss_fn, str(in_size), str_pixel_size, str_ego_center, str(n_epochs),str(batch_size), str(samples_per_epoch), str(sample_history_num_frames), str(history_num_frames), str(future_num_frames), str(n_modes), str_network, fit_fn, aug, model_str, '.pth'))
    # model_filename = '_'.join(('chkpt', str_model, str_rasterizer_fn, str_loader_fn, str_cfg_fn, str_loss_fn, str(in_size), str_pixel_size, str_ego_center, str(n_epochs),str(batch_size), str(samples_per_epoch), str(sample_history_num_frames), str(history_num_frames), str(future_num_frames), str(n_modes), str(max_agents), str_network, fit_fn, aug, model_str, '.pth'))
    # model_filename = '_'.join(('chkpt', str_model, str_rasterizer_fn, str_loader_fn, str_cfg_fn, str_loss_fn, str(in_size), str(n_epochs),str(batch_size), str(samples_per_epoch), str(sample_history_num_frames), str(history_num_frames), str(future_num_frames), str(n_modes), str(max_agents), str(weight_by_agent_count), str_network, fit_fn, aug, model_str, '.pth'))
    # model_filename = '_'.join(('chkpt', str_model, str_rasterizer_fn, str_loader_fn, str_cfg_fn, str_loss_fn, str(in_size), str(n_epochs),str(batch_size), str(samples_per_epoch), str(history_step_size), str(history_num_frames), str(future_num_frames), str(n_modes), str(max_agents), str(weight_by_agent_count), str_network, fit_fn, aug, model_str, '.pth'))
    model_filename = '_'.join(('chkpt', str_model, str_rasterizer_fn, str_loader_fn, str_cfg_fn, str_loss_fn,
                               str(in_size), str(n_epochs), str(batch_size), str(samples_per_epoch),
                               str(history_step_size), str(history_num_frames), str(future_num_frames), str(n_modes),
                               str(group_scenes), str(weight_by_agent_count), str_network, fit_fn, aug, model_str,
                               '.pth'))
    model_filename = model_filename.replace('-', '_')

    return os.path.join(MODEL_DIR, model_filename)


def create_submission(submission_dict, submission_dict_filepath):
    submission_filepath = submission_dict_filepath.replace(DATA_DIR, SUBMISSIONS_DIR).replace('pkl', 'csv')

    if submission_dict['preds'].shape[1] == 1:
        # Single mode prediction
        write_pred_csv(submission_filepath,
                       timestamps=submission_dict['timestamps'],
                       track_ids=submission_dict['track_ids'],
                       coords=np.squeeze(submission_dict['preds']))
    else:
        # Multi mode prediction
        write_pred_csv(submission_filepath,
                       timestamps=submission_dict['timestamps'],
                       track_ids=submission_dict['track_ids'],
                       coords=submission_dict['preds'],
                       confs=submission_dict['conf'])


##############################################
# RUN FUNCTIONS
##############################################

def run_tests_multi_motion_predict(model_str='', str_network='resnet18',
                                   n_epochs=20, in_size=224, pixel_size=[0.5, 0.5], ego_center=[0.25, 0.5],
                                   batch_size=24, samples_per_epoch=17000, lr=3e-4,
                                   sample_history_num_frames=10, history_num_frames=10, history_step_size=1,
                                   future_num_frames=50, n_modes=3, max_agents=40,
                                   group_scenes=False, weight_by_agent_count=False,
                                   fit_fn='fit_transform', val_fn='test_transform', aug='none',
                                   clsTrainDataset=MotionPredictDataset, clsValDataset=MotionPredictDataset,
                                   clsModel=LyftResnet18Transform, init_model_weights_path=None, cfg_model_params=None,
                                   rasterizer_fn=build_rasterizer,
                                   loss_fn=neg_log_likelihood_transform,
                                   str_train_loaders=['train_data_loader_100', 'train_data_loader_30'],
                                   str_val_loader='val_data_loader',
                                   loader_fn=double_channel_agents_ego_map_transform,
                                   cfg_fn=create_config_multi_train_chopped_lite):

    val_dict_filepath = create_val_dict_filepath(clsTrainDataset, clsValDataset, clsModel, val_fn, rasterizer_fn,
                                                 loader_fn, cfg_fn, fit_fn, loss_fn, in_size, pixel_size, ego_center,
                                                 n_epochs, batch_size, samples_per_epoch, sample_history_num_frames,
                                                 history_num_frames, history_step_size, future_num_frames, n_modes,
                                                 max_agents,
                                                 group_scenes, weight_by_agent_count, str_network, aug, model_str)

    if not os.path.exists(val_dict_filepath):

        print(' : '.join(('Training model', str_network, 'for input size', str(in_size), 'batch_size', str(batch_size),
                          'augmentation', aug, 'val_file', os.path.split(val_dict_filepath)[-1])))

        # Set up args_dict inputs
        train_args_dict, val_args_dict = setup_args_dicts(clsTrainDataset, clsValDataset, aug, str_network, in_size,
                                                          pixel_size, ego_center, batch_size, samples_per_epoch,
                                                          n_epochs, sample_history_num_frames, history_num_frames,
                                                          history_step_size, future_num_frames, n_modes, max_agents,
                                                          group_scenes, weight_by_agent_count)

        # Fit / evaluate model
        model_checkpoint_path = create_model_checkpoint_path(clsTrainDataset, clsValDataset, clsModel, val_fn,
                                                             rasterizer_fn, loader_fn, cfg_fn, fit_fn, loss_fn, in_size,
                                                             pixel_size, ego_center, n_epochs, batch_size,
                                                             samples_per_epoch, sample_history_num_frames,
                                                             history_num_frames, history_step_size, future_num_frames,
                                                             n_modes, max_agents, group_scenes, weight_by_agent_count,
                                                             str_network, aug, model_str)

        val_dicts, net = fit_multitrain_motion_predict(n_epochs, train_args_dict, val_args_dict,
                                                       init_model_weights_path=init_model_weights_path,
                                                       model_checkpoint_path=model_checkpoint_path,
                                                       fit_fn=fit_fn, loss_fn=loss_fn, val_fn=val_fn,
                                                       loader_fn=loader_fn,
                                                       cfg_fn=cfg_fn,
                                                       str_train_loaders=str_train_loaders,
                                                       str_val_loader=str_val_loader,
                                                       rasterizer_fn=rasterizer_fn,
                                                       cfg_model_params=cfg_model_params,
                                                       clsModel=clsModel, lr=lr,
                                                       action='fit' if not os.path.exists(
                                                           model_checkpoint_path) else 'evaluate')

        # Save out of sample data
        val_dict = concatenate_list_of_dicts(val_dicts) if isinstance(val_dicts, list) else val_dicts
        save_as_pickle(val_dict_filepath, val_dict)

    else:
        # Already evaluated, load from disk
        print('Completed ' + val_dict_filepath)
        val_dict = load_from_pickle(val_dict_filepath)
        model_checkpoint_path = None
        net = None

    return model_checkpoint_path, net, val_dict


def run_valset_multi_motion_predict(model_str='', str_network='resnet18',
                                    n_epochs=20, in_size=224, pixel_size=[0.5, 0.5], ego_center=[0.25, 0.5],
                                    batch_size=24, samples_per_epoch=17000, lr=3e-4,
                                    sample_history_num_frames=10, history_num_frames=10, history_step_size=1,
                                    future_num_frames=50, n_modes=3, max_agents=40,
                                    group_scenes=False, weight_by_agent_count=False,
                                    fit_fn='fit_transform', val_fn='test_transform', aug='none',
                                    clsTrainDataset=MotionPredictDataset, clsValDataset=MotionPredictDataset,
                                    clsModel=LyftResnet18Transform, init_model_weights_path=None, cfg_model_params=None,
                                    rasterizer_fn=build_rasterizer,
                                    loss_fn=neg_log_likelihood_transform,
                                    str_train_loaders=['train_data_loader_100', 'train_data_loader_30'],
                                    str_val_loader='val_data_loader',
                                    loader_fn=double_channel_agents_ego_map_transform,
                                    cfg_fn=create_config_multi_train_chopped_lite):

    val_dict_filepath = create_val_dict_filepath(clsTrainDataset, clsValDataset, clsModel, val_fn, rasterizer_fn,
                                                 loader_fn, cfg_fn, fit_fn, loss_fn, in_size, pixel_size, ego_center,
                                                 n_epochs, batch_size, samples_per_epoch, sample_history_num_frames,
                                                 history_num_frames, history_step_size, future_num_frames, n_modes,
                                                 max_agents,
                                                 group_scenes, weight_by_agent_count, str_network, aug, model_str)

    val_num = [int(s) for s in str_val_loader.split("_") if s.isdigit()]
    assert len(val_num) == 1, 'You can only process one val_data_loader at a time'

    val_dict_filepath = val_dict_filepath.replace('.pkl', str(val_num[0]) + '.pkl')

    if not os.path.exists(val_dict_filepath):

        print(' : '.join(('Training model', str_network, 'for input size', str(in_size), 'batch_size', str(batch_size),
                          'augmentation', aug, 'val_file', os.path.split(val_dict_filepath)[-1])))

        # Set up args_dict inputs
        train_args_dict, val_args_dict = setup_args_dicts(clsTrainDataset, clsValDataset, aug, str_network, in_size,
                                                          pixel_size, ego_center, batch_size, samples_per_epoch,
                                                          n_epochs, sample_history_num_frames, history_num_frames,
                                                          history_step_size, future_num_frames, n_modes, max_agents,
                                                          group_scenes, weight_by_agent_count)

        # Fit / evaluate model
        model_checkpoint_path = create_model_checkpoint_path(clsTrainDataset, clsValDataset, clsModel, val_fn,
                                                             rasterizer_fn, loader_fn, cfg_fn, fit_fn, loss_fn, in_size,
                                                             pixel_size, ego_center, n_epochs, batch_size,
                                                             samples_per_epoch, sample_history_num_frames,
                                                             history_num_frames, history_step_size, future_num_frames,
                                                             n_modes, max_agents, group_scenes, weight_by_agent_count,
                                                             str_network, aug, model_str)

        val_dicts, net = fit_multitrain_motion_predict(n_epochs, train_args_dict, val_args_dict,
                                                       init_model_weights_path=init_model_weights_path,
                                                       model_checkpoint_path=model_checkpoint_path,
                                                       fit_fn=fit_fn, loss_fn=loss_fn, val_fn=val_fn,
                                                       loader_fn=loader_fn,
                                                       cfg_fn=cfg_fn,
                                                       str_train_loaders=str_train_loaders,
                                                       str_val_loader=str_val_loader,
                                                       rasterizer_fn=rasterizer_fn,
                                                       cfg_model_params=cfg_model_params,
                                                       clsModel=clsModel, lr=lr,
                                                       action='fit' if not os.path.exists(
                                                           model_checkpoint_path) else 'evaluate')

        # Save out of sample data
        val_dict = concatenate_list_of_dicts(val_dicts) if isinstance(val_dicts, list) else val_dicts
        save_as_pickle(val_dict_filepath, val_dict)

    else:
        # Already evaluated, load from disk
        print('Completed ' + val_dict_filepath)
        val_dict = load_from_pickle(val_dict_filepath)
        model_checkpoint_path = None
        net = None

    return model_checkpoint_path, net, val_dict


def run_forecast_multi_motion_predict(model_str='', str_network='resnet18',
                                      n_epochs=20, in_size=224, pixel_size=[0.5, 0.5], ego_center=[0.25, 0.5],
                                      batch_size=24, samples_per_epoch=17000, lr=3e-4,
                                      sample_history_num_frames=10, history_num_frames=10, history_step_size=1,
                                      future_num_frames=50, n_modes=3, max_agents=40,
                                      group_scenes=False, weight_by_agent_count=False,
                                      fit_fn='fit_transform', val_fn='test_transform', aug='none',
                                      clsTrainDataset=MotionPredictDataset,
                                      clsValDataset=MotionPredictDataset,
                                      clsTestDataset=MotionPredictDataset,
                                      clsModel=LyftResnet18Transform,
                                      init_model_weights_path=None,
                                      rasterizer_fn=build_rasterizer,
                                      loss_fn=neg_log_likelihood_transform,
                                      cfg_model_params=None,
                                      str_train_loaders=['train_data_loader_100', 'train_data_loader_30'],
                                      str_val_loader='val_data_loader',
                                      loader_fn=double_channel_agents_ego_map_transform,
                                      cfg_fn=create_config_multi_train_chopped_lite):

    test_dict_filepath = create_test_dict_filepath(clsTrainDataset, clsValDataset, clsModel, val_fn, rasterizer_fn,
                                                   loader_fn, cfg_fn, fit_fn, loss_fn, in_size, pixel_size, ego_center,
                                                   n_epochs, batch_size, samples_per_epoch, sample_history_num_frames,
                                                   history_num_frames, history_step_size, future_num_frames, n_modes,
                                                   max_agents, group_scenes, weight_by_agent_count, str_network, aug,
                                                   model_str)

    if not os.path.exists(test_dict_filepath):

        print(' : '.join(('Forecasting for model', str_network, 'for input size', str(in_size), 'batch_size',
                          str(batch_size), 'augmentation', aug, 'test_file', os.path.split(test_dict_filepath)[-1])))

        # Set up args_dict inputs
        test_args_dict = setup_test_args_dict(clsTestDataset, aug, str_network, in_size, pixel_size, ego_center,
                                              batch_size, samples_per_epoch, n_epochs, sample_history_num_frames,
                                              history_num_frames, history_step_size, future_num_frames, n_modes,
                                              max_agents, group_scenes, weight_by_agent_count)

        # Fit / evaluate model
        model_checkpoint_path = create_model_checkpoint_path(clsTrainDataset, clsValDataset, clsModel, val_fn,
                                                             rasterizer_fn, loader_fn, cfg_fn, fit_fn, loss_fn, in_size,
                                                             pixel_size, ego_center, n_epochs, batch_size,
                                                             samples_per_epoch, sample_history_num_frames,
                                                             history_num_frames, history_step_size, future_num_frames,
                                                             n_modes, max_agents, group_scenes, weight_by_agent_count,
                                                             str_network, aug, model_str)

        test_dicts, net = fit_multitrain_motion_predict(n_epochs, test_args_dict, test_args_dict,
                                                        init_model_weights_path=init_model_weights_path,
                                                        model_checkpoint_path=model_checkpoint_path,
                                                        fit_fn=fit_fn, loss_fn=loss_fn, val_fn=val_fn,
                                                        loader_fn=loader_fn,
                                                        rasterizer_fn=rasterizer_fn, cfg_fn=cfg_fn,
                                                        str_train_loaders=str_train_loaders,
                                                        cfg_model_params=cfg_model_params,
                                                        clsModel=clsModel, lr=lr,
                                                        action='test')

        # Save out of sample data
        test_dict = concatenate_list_of_dicts(test_dicts) if isinstance(test_dicts, list) else test_dicts
        save_as_pickle(test_dict_filepath, test_dict)

    else:
        # Already evaluated, load from disk
        print('Completed ' + test_dict_filepath)
        test_dict = load_from_pickle(test_dict_filepath)

    # create submission
    create_submission(test_dict, test_dict_filepath)

    return test_dict


if __name__ == '__main__':


    chop_indices = list(range(10, 201, 10))

    run_tests_multi_motion_predict(n_epochs=1000, in_size=128, batch_size=256,
                                   samples_per_epoch=17000 // len(chop_indices),
                                   sample_history_num_frames=5, history_num_frames=5, history_step_size=1,
                                   future_num_frames=50,
                                   group_scenes=False, weight_by_agent_count=7,
                                   clsTrainDataset=MultiMotionPredictDataset,
                                   clsValDataset=MotionPredictDataset,
                                   clsModel=LyftResnest50,
                                   fit_fn='fit_fastai_transform', val_fn='test_transform',
                                   loss_fn=neg_log_likelihood_transform,
                                   aug='none',
                                   loader_fn=double_channel_agents_ego_map_transform,
                                   cfg_fn=create_config_multi_train_chopped_lite,
                                   str_train_loaders=['train_data_loader_' + str(i) for i in chop_indices],
                                   rasterizer_fn=build_rasterizer)

    run_forecast_multi_motion_predict(n_epochs=1000, in_size=128, batch_size=256,
                                      samples_per_epoch=17000 // len(chop_indices),
                                      sample_history_num_frames=5, history_num_frames=5, history_step_size=1,
                                      future_num_frames=50,
                                      group_scenes=False, weight_by_agent_count=7,
                                      clsTrainDataset=MultiMotionPredictDataset,
                                      clsValDataset=MotionPredictDataset,
                                      clsModel=LyftResnest50,
                                      fit_fn='fit_fastai_transform', val_fn='test_transform',
                                      loss_fn=neg_log_likelihood_transform,
                                      aug='none',
                                      loader_fn=double_channel_agents_ego_map_transform,
                                      cfg_fn=create_config_multi_train_chopped_lite,
                                      str_train_loaders=['train_data_loader_' + str(i) for i in chop_indices],
                                      rasterizer_fn=build_rasterizer)



