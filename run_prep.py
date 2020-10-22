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

import argparse
from pathlib import Path

from zarr import convenience

from l5kit.data import DataManager, LocalDataManager, ChunkedDataset, get_agents_slice_from_frames
from l5kit.data.filter import filter_tl_faces_by_status, get_tl_faces_slice_from_frames
from l5kit.data.map_api import MapAPI
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.dataset.select_agents import TH_DISTANCE_AV, TH_EXTENT_RATIO, TH_YAW_DEGREE, select_agents
from l5kit.kinematic import Perturbation
from l5kit.rasterization.rasterizer_builder import build_rasterizer, _load_metadata
from l5kit.rasterization.sem_box_rasterizer import SemBoxRasterizer
from l5kit.rasterization.box_rasterizer import BoxRasterizer
from l5kit.rasterization.rasterizer import Rasterizer
from l5kit.rasterization.semantic_rasterizer import SemanticRasterizer, elements_within_bounds, cv2_subpixel, CV2_SHIFT, CV2_SHIFT_VALUE
from l5kit.rasterization import StubRasterizer

from l5kit.evaluation import write_pred_csv, compute_metrics_csv, read_gt_csv, write_gt_csv, create_chopped_dataset, export_zarr_to_csv
from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace
from l5kit.geometry import transform_points
from l5kit.visualization import PREDICTED_POINTS_COLOR, TARGET_POINTS_COLOR, draw_trajectory

from concurrent.futures import ThreadPoolExecutor

from utils import *
from settings import BASE_DIR, DATA_DIR, CACHE_DIR, MODEL_DIR
from configs import create_prep_config



######################################
# SET UP / GLOBALS
######################################

NUM_WORKERS = 16
MIN_FUTURE_STEPS = 10


######################################
# L5KIT ADAPTATIONS
######################################

def create_chopped_dataset_lite(
    zarr_path: str, th_agent_prob: float, num_frames_to_copy: int, num_frames_gt: int, min_frame_future: int, history_num_frames: int
) -> str:
    """
    Create a chopped version of the zarr that can be used as a test set.
    This function was used to generate the test set for the competition so that the future GT is not in the data.

    Store:
     - a dataset where each scene has been chopped at `num_frames_to_copy` frames;
     - a mask for agents for those final frames based on the original mask and a threshold on the future_frames;
     - the GT csv for those agents

     For the competition, only the first two (dataset and mask) will be available in the notebooks

    Args:
        zarr_path (str): input zarr path to be chopped
        th_agent_prob (float): threshold over agents probabilities used in select_agents function
        num_frames_to_copy (int):  number of frames to copy from the beginning of each scene, others will be discarded
        min_frame_future (int): minimum number of frames that must be available in the future for an agent
        num_frames_gt (int): number of future predictions to store in the GT file

    Returns:
        str: the parent folder of the new datam
    """
    zarr_path = Path(zarr_path)
    dest_path = zarr_path.parent / f"{zarr_path.stem}_chopped_{num_frames_to_copy}_lite"
    chopped_path = dest_path / zarr_path.name
    gt_path = dest_path / "gt.csv"

    if not os.path.exists(gt_path):
        # Create standard mask for the dataset so we can use it to filter out unreliable agents
        zarr_dt = ChunkedDataset(str(zarr_path))
        zarr_dt.open()

        agents_mask_path = Path(zarr_path) / f"agents_mask/{th_agent_prob}"
        if not agents_mask_path.exists():  # don't check in root but check for the path
            select_agents(
                zarr_dt,
                th_agent_prob=th_agent_prob,
                th_yaw_degree=TH_YAW_DEGREE,
                th_extent_ratio=TH_EXTENT_RATIO,
                th_distance_av=TH_DISTANCE_AV,
            )
        agents_mask_origin = np.asarray(convenience.load(str(agents_mask_path)))

        # create chopped dataset
        chopped_info_filename = os.path.join(os.path.split(chopped_path)[0], 'chopped_info.pkl')
        chopped_info = check_load(chopped_info_filename,  zarr_scenes_chop_lite, str(chopped_path), save_to_file=True, args_in=(str(zarr_path), str(chopped_path), num_frames_to_copy, history_num_frames), verbose=True)
        
        zarr_chopped = ChunkedDataset(str(chopped_path))
        zarr_chopped.open()

        # Compute the agent mask for the chopped dataset
        chopped_agents_mask_path = Path(chopped_path) / f"agents_mask/{th_agent_prob}"
        if not chopped_agents_mask_path.exists():  
            select_agents(
                    zarr_chopped,
                    th_agent_prob=th_agent_prob,
                    th_yaw_degree=TH_YAW_DEGREE,
                    th_extent_ratio=TH_EXTENT_RATIO,
                    th_distance_av=TH_DISTANCE_AV,
                )

        # compute original boolean mask limited to frames of interest for GT csv
        agents_mask_orig_bool = np.zeros(len(zarr_dt.agents), dtype=np.bool)

        for idx in tqdm(range(len(zarr_dt.scenes)), desc='Extracting masks'):

            scene = zarr_dt.scenes[idx]

            frame_original = zarr_dt.frames[scene["frame_index_interval"][0] + num_frames_to_copy - 1]
            slice_agents_original = get_agents_slice_from_frames(frame_original)

            mask = agents_mask_origin[slice_agents_original][:, 1] >= min_frame_future
            agents_mask_orig_bool[slice_agents_original] = mask.copy()

        export_zarr_to_csv(zarr_dt, str(gt_path), num_frames_gt, th_agent_prob, agents_mask=agents_mask_orig_bool)

    else:

        print(' : '.join((str(gt_path), 'COMPLETED')))

    return str(dest_path)


def zarr_scenes_chop_lite(input_zarr: str, output_zarr: str, num_frames_to_copy: int, history_num_frames: int) -> None:
    """
    Copy `num_frames_to_keep` from each scene in input_zarr and paste them into output_zarr

    Args:
        input_zarr (str): path to the input zarr
        output_zarr (str): path to the output zarr
        num_frames_to_copy (int): how many frames to copy from the start of each scene

    Returns:
        chopped_indices (list[int])
    """

    input_dataset = ChunkedDataset(input_zarr)
    input_dataset.open()

    # check we can actually copy the frames we want from each scene
    #assert np.all(np.diff(input_dataset.scenes["frame_index_interval"], 1) > num_frames_to_copy), "not enough frames"

    output_dataset = ChunkedDataset(output_zarr)
    output_dataset.initialize()

    # current indices where to copy in the output_dataset
    cur_scene_idx, cur_frame_idx, cur_agent_idx, cur_tl_face_idx = 0, 0, 0, 0
    chopped_indices = []
    chopped_agents_slice = []

    for idx in tqdm(range(len(input_dataset.scenes)), desc="copying"):

        # get data and immediately chop frames, agents and traffic lights
        scene = input_dataset.scenes[idx]
        
        first_frame_idx = scene["frame_index_interval"][0]
        last_frame_idx = scene["frame_index_interval"][-1]

        if (last_frame_idx - first_frame_idx - num_frames_to_copy) >= 0 and num_frames_to_copy >= history_num_frames:

            frames = input_dataset.frames[first_frame_idx + num_frames_to_copy - history_num_frames: first_frame_idx + num_frames_to_copy]
            agents = input_dataset.agents[get_agents_slice_from_frames(*frames[[0, -1]])]
            tl_faces = input_dataset.tl_faces[get_tl_faces_slice_from_frames(*frames[[0, -1]])]

            chopped_frame_agents_slice = get_agents_slice_from_frames(frames[-1])

            # reset interval relative to our output (subtract current history and add output history)
            scene["frame_index_interval"][0] = cur_frame_idx
            scene["frame_index_interval"][1] = cur_frame_idx + num_frames_to_copy  # address for less frames

            frames["agent_index_interval"] += cur_agent_idx - frames[0]["agent_index_interval"][0]
            frames["traffic_light_faces_index_interval"] += (
                cur_tl_face_idx - frames[0]["traffic_light_faces_index_interval"][0]
            )

            # write in dest using append (slow)
            output_dataset.scenes.append(scene[None, ...])  # need 2D array to concatenate
            output_dataset.frames.append(frames)
            output_dataset.agents.append(agents)
            output_dataset.tl_faces.append(tl_faces)

            # increase indices in output
            cur_scene_idx += len(scene)
            cur_frame_idx += len(frames)
            cur_agent_idx += len(agents)
            cur_tl_face_idx += len(tl_faces)

            # Add to chopped info
            chopped_indices.append(idx)
            chopped_agents_slice.append(chopped_frame_agents_slice)

        else:

            print(' : '.join(('Excluded', str(idx), str(last_frame_idx - first_frame_idx))))

    return chopped_indices, chopped_agents_slice


def create_chopped_dataset_CF(
    zarr_path: str, th_agent_prob: float, num_frames_to_copy: int, num_frames_gt: int, min_frame_future: int
) -> str:
    """
    Create a chopped version of the zarr that can be used as a test set.
    This function was used to generate the test set for the competition so that the future GT is not in the data.

    Store:
     - a dataset where each scene has been chopped at `num_frames_to_copy` frames;
     - a mask for agents for those final frames based on the original mask and a threshold on the future_frames;
     - the GT csv for those agents

     For the competition, only the first two (dataset and mask) will be available in the notebooks

    Args:
        zarr_path (str): input zarr path to be chopped
        th_agent_prob (float): threshold over agents probabilities used in select_agents function
        num_frames_to_copy (int):  number of frames to copy from the beginning of each scene, others will be discarded
        min_frame_future (int): minimum number of frames that must be available in the future for an agent
        num_frames_gt (int): number of future predictions to store in the GT file

    Returns:
        str: the parent folder of the new datam
    """
    zarr_path = Path(zarr_path)
    dest_path = zarr_path.parent / f"{zarr_path.stem}_chopped_{num_frames_to_copy}_CF"
    chopped_path = dest_path / zarr_path.name
    gt_path = dest_path / "gt.csv"
    mask_chopped_path = dest_path / "mask"

    if not os.path.exists(gt_path):
        # Create standard mask for the dataset so we can use it to filter out unreliable agents
        zarr_dt = ChunkedDataset(str(zarr_path))
        zarr_dt.open()

        agents_mask_path = Path(zarr_path) / f"agents_mask/{th_agent_prob}"
        if not agents_mask_path.exists():  # don't check in root but check for the path
            select_agents(
                zarr_dt,
                th_agent_prob=th_agent_prob,
                th_yaw_degree=TH_YAW_DEGREE,
                th_extent_ratio=TH_EXTENT_RATIO,
                th_distance_av=TH_DISTANCE_AV,
            )
        agents_mask_origin = np.asarray(convenience.load(str(agents_mask_path)))

        # create chopped dataset
        chopped_indices_filename = os.path.join(os.path.split(chopped_path)[0], 'chopped_indices.pkl')
        chopped_indices = check_load(chopped_indices_filename,  zarr_scenes_chop_CF, str(chopped_path), save_to_file=True, args_in=(str(zarr_path), str(chopped_path), num_frames_to_copy), verbose=True)

        zarr_chopped = ChunkedDataset(str(chopped_path))
        zarr_chopped.open()

        # compute the chopped boolean mask, but also the original one limited to frames of interest for GT csv
        agents_mask_chop_bool = np.zeros(len(zarr_chopped.agents), dtype=np.bool)
        agents_mask_orig_bool = np.zeros(len(zarr_dt.agents), dtype=np.bool)

        for idx in tqdm(range(len(zarr_dt.scenes)), desc='Extracting masks'):

            scene = zarr_dt.scenes[idx]

            frame_original = zarr_dt.frames[scene["frame_index_interval"][0] + num_frames_to_copy - 1]
            slice_agents_original = get_agents_slice_from_frames(frame_original)

            mask = agents_mask_origin[slice_agents_original][:, 1] >= min_frame_future
            agents_mask_orig_bool[slice_agents_original] = mask.copy()

            if idx in chopped_indices:

                frame_chopped = zarr_chopped.frames[zarr_chopped.scenes[chopped_indices.index(idx)]["frame_index_interval"][-1] - 1]
                slice_agents_chopped = get_agents_slice_from_frames(frame_chopped)

                agents_mask_chop_bool[slice_agents_chopped] = mask.copy()

        # store the mask and the GT csv of frames on interest
        np.savez(str(mask_chopped_path), agents_mask_chop_bool)
        export_zarr_to_csv(zarr_dt, str(gt_path), num_frames_gt, th_agent_prob, agents_mask=agents_mask_orig_bool)

    else:

        print(' : '.join((str(gt_path), 'COMPLETED')))

    return str(dest_path)


def zarr_scenes_chop_CF(input_zarr: str, output_zarr: str, num_frames_to_copy: int) -> None:
    """
    Copy `num_frames_to_keep` from each scene in input_zarr and paste them into output_zarr

    Args:
        input_zarr (str): path to the input zarr
        output_zarr (str): path to the output zarr
        num_frames_to_copy (int): how many frames to copy from the start of each scene

    Returns:
        chopped_indices (list[int])
    """

    input_dataset = ChunkedDataset(input_zarr)
    input_dataset.open()

    # check we can actually copy the frames we want from each scene
    #assert np.all(np.diff(input_dataset.scenes["frame_index_interval"], 1) > num_frames_to_copy), "not enough frames"

    output_dataset = ChunkedDataset(output_zarr)
    output_dataset.initialize()

    # current indices where to copy in the output_dataset
    cur_scene_idx, cur_frame_idx, cur_agent_idx, cur_tl_face_idx = 0, 0, 0, 0
    chopped_indices = []

    for idx in tqdm(range(len(input_dataset.scenes)), desc="copying"):

        # get data and immediately chop frames, agents and traffic lights
        scene = input_dataset.scenes[idx]
        
        first_frame_idx = scene["frame_index_interval"][0]
        last_frame_idx = scene["frame_index_interval"][-1]

        if (last_frame_idx - first_frame_idx - num_frames_to_copy) >= 0:

            chopped_indices.append(idx)

            frames = input_dataset.frames[first_frame_idx : first_frame_idx + num_frames_to_copy]
            agents = input_dataset.agents[get_agents_slice_from_frames(*frames[[0, -1]])]
            tl_faces = input_dataset.tl_faces[get_tl_faces_slice_from_frames(*frames[[0, -1]])]

            # reset interval relative to our output (subtract current history and add output history)
            scene["frame_index_interval"][0] = cur_frame_idx
            scene["frame_index_interval"][1] = cur_frame_idx + num_frames_to_copy  # address for less frames

            frames["agent_index_interval"] += cur_agent_idx - frames[0]["agent_index_interval"][0]
            frames["traffic_light_faces_index_interval"] += (
                cur_tl_face_idx - frames[0]["traffic_light_faces_index_interval"][0]
            )

            # write in dest using append (slow)
            output_dataset.scenes.append(scene[None, ...])  # need 2D array to concatenate
            output_dataset.frames.append(frames)
            output_dataset.agents.append(agents)
            output_dataset.tl_faces.append(tl_faces)

            # increase indices in output
            cur_scene_idx += len(scene)
            cur_frame_idx += len(frames)
            cur_agent_idx += len(agents)
            cur_tl_face_idx += len(tl_faces)

        else:

            print(' : '.join(('Excluded', str(idx), str(last_frame_idx - first_frame_idx))))

    return chopped_indices



######################################
# DATA PREP
######################################

def save_chopped_ds_lite(cfg, str_data_loader='train_data_loader', num_frames_to_chop=100, min_future_steps=10, history_num_frames=10):

    dm = LocalDataManager(None)
    chopped_ds_path = create_chopped_dataset_lite(dm.require(cfg[str_data_loader]["key"]), cfg["raster_params"]["filter_agents_threshold"], 
                                            num_frames_to_chop, cfg["model_params"]["future_num_frames"], min_future_steps, history_num_frames)

    return chopped_ds_path


def save_chopped_ds_CF(cfg, str_data_loader='train_data_loader', num_frames_to_chop=100, min_future_steps=10):

    dm = LocalDataManager(None)
    chopped_ds_path = create_chopped_dataset_CF(dm.require(cfg[str_data_loader]["key"]), cfg["raster_params"]["filter_agents_threshold"], 
                                            num_frames_to_chop, cfg["model_params"]["future_num_frames"], min_future_steps)

    return chopped_ds_path


def save_multi_datasets_CF(config = create_prep_config(), str_data_loader='train_data_loader', num_frames_to_chop=[30, 100, 180]):

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as e:
        for n in num_frames_to_chop:
            e.submit(save_chopped_ds_CF, config, str_data_loader, n)


def save_multi_datasets_lite(config = create_prep_config(), str_data_loader='train_data_loader', num_frames_to_chop=[30, 100, 180]):
    """
    The creation of an agents mask while saving a lite dataset requires Multiprocessing, so we can't put it on a thread
    """
    for n in num_frames_to_chop:
        save_chopped_ds_lite(config, str_data_loader, n)


if __name__ == '__main__':

    """
    Depending on your hardware specs you may be able to run 
    multiple chops concurrently. 
    Mine maxes out memory at 3 or 4 chops at a time.
    Defaulting to single runs here
    """

    chop_indices = [10, 30, 50, 70, 90, 110, 130, 150, 180, 200]
    chop_indices = [30]
    for str_loader in ['train_data_loader']:
        for n in chop_indices:
            print(' : '.join((str_loader, str(n))))
            save_multi_datasets_lite(str_data_loader=str_loader, num_frames_to_chop=[n] if not isinstance(n, list) else n)


