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

from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torchvision import utils
from torch.utils import model_zoo

from adamp import AdamP

from efficientnet_pytorch import EfficientNet

import fastai
from fastai.vision import *
from fastai.callbacks import SaveModelCallback, ReduceLROnPlateauCallback
from fastai.torch_core import add_metrics

from l5kit.data import DataManager, LocalDataManager, ChunkedDataset, TL_FACE_DTYPE, filter_agents_by_labels, filter_tl_faces_by_frames, get_agents_slice_from_frames, get_tl_faces_slice_from_frames
from l5kit.data.filter import filter_tl_faces_by_status, get_frames_slice_from_scenes, filter_agents_by_frames, filter_agents_by_track_id
from l5kit.data.map_api import MapAPI
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.kinematic import Perturbation
from l5kit.rasterization.rasterizer_builder import build_rasterizer, _load_metadata, get_hardcoded_world_to_ecef
from l5kit.rasterization.sem_box_rasterizer import SemBoxRasterizer
from l5kit.rasterization.box_rasterizer import BoxRasterizer
from l5kit.rasterization.rasterizer import Rasterizer, EGO_EXTENT_HEIGHT, EGO_EXTENT_LENGTH, EGO_EXTENT_WIDTH
from l5kit.rasterization.semantic_rasterizer import SemanticRasterizer, elements_within_bounds, cv2_subpixel, CV2_SHIFT, CV2_SHIFT_VALUE
from l5kit.rasterization.render_context import RenderContext
from l5kit.evaluation import write_pred_csv, compute_metrics_csv, read_gt_csv, write_gt_csv
from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace
from l5kit.geometry import transform_point, transform_points, rotation33_as_yaw, compute_agent_pose
from l5kit.visualization import PREDICTED_POINTS_COLOR, TARGET_POINTS_COLOR, draw_trajectory
from l5kit.sampling.slicing import get_future_slice, get_history_slice
from l5kit.sampling.agent_sampling import _create_targets_for_deep_prediction

from typing import List, Optional, Tuple

import albumentations
from albumentations import OneOf
from albumentations.augmentations.transforms import GaussNoise, MotionBlur, MedianBlur, Blur, CoarseDropout

from datetime import datetime


from settings import BASE_DIR, DATA_DIR, CACHE_DIR, MODEL_DIR, SUBMISSIONS_DIR, SINGLE_MODE_SUBMISSION, MULTI_MODE_SUBMISSION

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
NUM_WORKERS = 10
CREATE_CACHE = False
GROUP_SCENES = False  # Whether we should order training/val samples according to single samples from each scene. Legacy.

# set env variable for data
os.environ["L5KIT_DATA_FOLDER"] = BASE_DIR


##############################################
# L5KIT ADAPTATIONS
##############################################

class AgentDatasetCF(AgentDataset):
    """
    Exposes scene_id, frame_id, track_id and timestamp for each agent
    so that these can be used in sampling strategy
    """
    def __init__(
        self,
        raw_data_file: str,
        cfg: dict,
        str_loader: str,
        zarr_dataset: ChunkedDataset,
        rasterizer: Rasterizer,
        perturbation: Optional[Perturbation] = None,
        agents_mask: Optional[np.ndarray] = None,
        min_frame_history: int = 10,
        min_frame_future: int = 10,  # Changed from 1 to 10 2020-09-20
    ):
        assert perturbation is None, "AgentDataset does not support perturbation (yet)"

        super(AgentDatasetCF, self).__init__(cfg, zarr_dataset, rasterizer, perturbation, agents_mask, min_frame_history, min_frame_future)

        self.raw_data_file = raw_data_file
        self.str_loader = str_loader

        self.gt_path = self.cfg[self.str_loader]['gt_path'] if 'gt_path' in self.cfg[self.str_loader] else None
        
        self.load_indices()

        self.load_gt()


    def load_gt(self):
        # Cache ground truth
        if self.gt_path is not None:
            gt = {}
            for row in read_gt_csv(self.gt_path):
                gt[row["track_id"] + row["timestamp"]] = [row["coord"], row["avail"]]
            self.gt = gt
        else:
            self.gt = None


    def load_indices(self):

        filename = os.path.join(DATA_DIR, self.raw_data_file + '_ids.pkl')
        frame_indices, scene_indices, track_ids, timestamps = check_load(filename, self.get_track_frame_scene_timestamps, None, save_to_file=True, args_in=None, verbose=True)
        
        self.frame_indices = frame_indices
        self.scene_indices = scene_indices
        self.track_ids = track_ids
        self.timestamps = timestamps


    def get_track_frame_scene_timestamps(self):

        frame_indices = [bisect.bisect_right(self.cumulative_sizes_agents, index) for index in tqdm(self.agents_indices)]
        scene_indices = [bisect.bisect_right(self.cumulative_sizes, frame_index) for frame_index in tqdm(frame_indices)]

        # Load track_ids in chunks as otherwise we can get memory errors
        track_ids = self.load_agent_item_in_chunks('track_id')

        # We only need track_id and timestamp for reference purposes here (and for csv output)
        # In the case of train_full.zarr we can't load all of the timestamps as we run out of memory.
        # Add a failsafe in this situation
        if 'train_full.zarr' in self.cfg[self.str_loader]:

            all_timestamps = self.dataset.frames[:]['timestamp']
            timestamps = [all_timestamps[frame_index] for frame_index in tqdm(frame_indices)]

        else:

            print('Failed to load timestamps for ' + self.str_loader)
            timestamps = [0] * len(scene_indices)

        return frame_indices, scene_indices, track_ids, timestamps
        

    def load_agent_item_in_chunks(self, str_item='track_id', n_chunks=10):

        n = len(self.dataset.agents)
        index_chunks = np.arange(0, n, n//n_chunks).astype(np.int)
        index_chunks[-1] = n 

        item_ids = np.empty((len(self.agents_indices),), dtype=self.dataset.agents[0][str_item].dtype)

        for i in tqdm(range(1, len(index_chunks)), desc='Loading agent items'):

            agents_valid = np.argwhere(np.logical_and(self.agents_indices >= index_chunks[i-1], self.agents_indices < index_chunks[i])).reshape(-1,)

            _item_ids = self.dataset.agents[slice(index_chunks[i-1], index_chunks[i])][str_item]

            item_ids[agents_valid] = _item_ids[self.agents_indices[agents_valid] - index_chunks[i - 1]]

            _item_ids = None

        # Check that we've populated everything
        assert np.all(item_ids >= 0)

        return item_ids


    def __getitem__(self, index: int) -> dict:
        """
        Differs from parent by accessing indices directly
        """
        track_id = self.track_ids[index]
        frame_index = self.frame_indices[index]
        scene_index = self.scene_indices[index]

        if scene_index == 0:
            state_index = frame_index
        else:
            state_index = frame_index - self.cumulative_sizes[scene_index - 1]
        return self.get_frame(scene_index, state_index, track_id=track_id)


    def get_frame(self, scene_index: int, state_index: int, track_id: Optional[int] = None) -> dict:
        """
        A utility function to get the rasterisation and trajectory target for a given agent in a given frame

        Args:
            scene_index (int): the index of the scene in the zarr
            state_index (int): a relative frame index in the scene
            track_id (Optional[int]): the agent to rasterize or None for the AV
        Returns:
            dict: the rasterised image, the target trajectory (position and yaw) along with their availability,
            the 2D matrix to center that agent, the agent track (-1 if ego) and the timestamp

        """
        frames = self.dataset.frames[get_frames_slice_from_scenes(self.dataset.scenes[scene_index])]
        data = self.sample_function(state_index, frames, self.dataset.agents, self.dataset.tl_faces, track_id)
        
        # 0,1,C -> C,0,1
        image = data["image"].transpose(2, 0, 1)

        history_positions = np.array(data["history_positions"], dtype=np.float32)
        history_yaws = np.array(data["history_yaws"], dtype=np.float32)

        timestamp = frames[state_index]["timestamp"]
        track_id = np.int64(-1 if track_id is None else track_id)  # always a number to avoid crashing torch

        target_positions = np.array(data["target_positions"], dtype=np.float32)
        target_yaws = np.array(data["target_yaws"], dtype=np.float32)

        target_availabilities = data["target_availabilities"]

        # Get target_positions from ground truth if self.gt is available
        if self.gt is not None:
            assert str(track_id) + str(timestamp) in self.gt, 'self.gt (ground truth) does not contain requested track_id/timestamp combination. We have got a problem somewhere!'
            target_positions = np.array(self.gt[str(track_id) + str(timestamp)][0], dtype=np.float32)
            target_positions = transform_points(target_positions + data['centroid'][:2], data['agent_from_world'] )
            target_availabilities = np.array(self.gt[str(track_id) + str(timestamp)][1], dtype=np.float32)

        ego_center = data['ego_center'] if 'ego_center' in data else np.array(self.cfg['raster_params']['ego_center'])

        return {
            "image": image,
            "target_positions": target_positions,
            "target_yaws": target_yaws,
            "target_availabilities": target_availabilities,
            "history_positions": history_positions,
            "history_yaws": history_yaws,
            "history_availabilities": data["history_availabilities"],
            "world_to_image": data["world_to_image"],
            "raster_from_world": data["raster_from_world"],
            "raster_from_agent": data["raster_from_agent"],
            "world_from_agent": data["world_from_agent"],
            "agent_from_world": data["agent_from_world"],
            "track_id": track_id,
            "timestamp": timestamp,
            "centroid": data["centroid"],
            "ego_center": ego_center,
            "yaw": data["yaw"],
            "extent": data["extent"],
        }


class AgentDatasetRandomEgoCentre(AgentDatasetCF):
    """
    Exposes scene_id, frame_id, track_id and timestamp for each agent
    so that these can be used in sampling strategy
    """
    def __init__(
        self,
        raw_data_file: str,
        cfg: dict,
        str_loader: str,
        zarr_dataset: ChunkedDataset,
        rasterizer: Rasterizer,
        perturbation: Optional[Perturbation] = None,
        agents_mask: Optional[np.ndarray] = None,
        min_frame_history: int = 10,
        min_frame_future: int = 10,  # Changed from 1 to 10 2020-09-20
    ):
        assert perturbation is None, "AgentDataset does not support perturbation (yet)"

        super(AgentDatasetRandomEgoCentre, self).__init__(raw_data_file, 
                                                            cfg, 
                                                            str_loader, 
                                                            zarr_dataset, 
                                                            rasterizer, 
                                                            perturbation, 
                                                            agents_mask, 
                                                            min_frame_history, 
                                                            min_frame_future)

        render_context = RenderContext(
            raster_size_px=np.array(cfg["raster_params"]["raster_size"]),
            pixel_size_m=np.array(cfg["raster_params"]["pixel_size"]),
            center_in_raster_ratio=np.array(cfg["raster_params"]["ego_center"]),
        )
        # Overwrite sample function:
        # build a partial so we don't have to access cfg each time
        self.sample_function = partial(generate_agent_sample_random_ego_center,
                                        render_context=render_context,
                                        history_num_frames=cfg["model_params"]["history_num_frames"],
                                        history_step_size=cfg["model_params"]["history_step_size"],
                                        future_num_frames=cfg["model_params"]["future_num_frames"],
                                        future_step_size=cfg["model_params"]["future_step_size"],
                                        filter_agents_threshold=cfg["raster_params"]["filter_agents_threshold"],
                                        rasterizer=rasterizer,
                                        perturbation=perturbation,
                                        )


class SemanticTLRasterizer(SemanticRasterizer):
    """
    Rasteriser for the vectorised semantic map with historic traffic lights as a separate channel(generally loaded from json files).
    """

    def rasterize(
        self,
        history_frames: np.ndarray,
        history_agents: List[np.ndarray],
        history_tl_faces: List[np.ndarray],
        agent: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if agent is None:
            ego_translation_m = history_frames[0]["ego_translation"]
            ego_yaw_rad = rotation33_as_yaw(history_frames[0]["ego_rotation"])
        else:
            ego_translation_m = np.append(agent["centroid"], history_frames[0]["ego_translation"][-1])
            ego_yaw_rad = agent["yaw"]

        raster_from_world = self.render_context.raster_from_world(ego_translation_m, ego_yaw_rad)
        world_from_raster = np.linalg.inv(raster_from_world)

        # get XY of center pixel in world coordinates
        center_in_raster_px = np.asarray(self.raster_size) * (0.5, 0.5)
        center_in_world_m = transform_point(center_in_raster_px, world_from_raster)

        sem_im = self.render_semantic_map(center_in_world_m, raster_from_world, history_tl_faces)
        return sem_im.astype(np.float32) / 255

    def render_semantic_map(
        self, center_world: np.ndarray, raster_from_world: np.ndarray, history_tl_faces: List[np.ndarray]
    ) -> np.ndarray:
        """Renders the semantic map at given x,y coordinates.

        Args:
            center_world (np.ndarray): XY of the image center in world ref system
            raster_from_world (np.ndarray):
        Returns:
            np.ndarray: RGB raster

        """

        img = 255 * np.ones(shape=(self.raster_size[1], self.raster_size[0], 3), dtype=np.uint8)
        tl_imgs = [255 * np.ones(shape=(self.raster_size[1], self.raster_size[0], 3), dtype=np.uint8) for i in range(len(history_tl_faces) - 1)]

        # filter using half a radius from the center
        raster_radius = float(np.linalg.norm(self.raster_size * self.pixel_size)) / 2

        # get active traffic light faces
        history_active_tl_ids = [set(filter_tl_faces_by_status(tl_faces, "ACTIVE")["face_id"].tolist()) for tl_faces in history_tl_faces]
        
        # plot lanes
        lanes_lines = [defaultdict(list) for tl_faces in history_tl_faces]

        for idx in elements_within_bounds(center_world, self.bounds_info["lanes"]["bounds"], raster_radius):
            lane = self.proto_API[self.bounds_info["lanes"]["ids"][idx]].element.lane

            # get image coords
            lane_coords = self.proto_API.get_lane_coords(self.bounds_info["lanes"]["ids"][idx])
            xy_left = cv2_subpixel(transform_points(lane_coords["xyz_left"][:, :2], raster_from_world))
            xy_right = cv2_subpixel(transform_points(lane_coords["xyz_right"][:, :2], raster_from_world))
            lanes_area = np.vstack((xy_left, np.flip(xy_right, 0)))  # start->end left then end->start right

            # Note(lberg): this called on all polygons skips some of them, don't know why
            cv2.fillPoly(img, [lanes_area], (17, 17, 31), lineType=cv2.LINE_AA, shift=CV2_SHIFT)

            for tl_idx, active_tl_ids in enumerate(history_active_tl_ids):
                lane_type = "default"  # no traffic light face is controlling this lane
                lane_tl_ids = set([MapAPI.id_as_str(la_tc) for la_tc in lane.traffic_controls])
                for tl_id in lane_tl_ids.intersection(active_tl_ids):
                    if self.proto_API.is_traffic_face_colour(tl_id, "red"):
                        lane_type = "red"
                    elif self.proto_API.is_traffic_face_colour(tl_id, "green"):
                        lane_type = "green"
                    elif self.proto_API.is_traffic_face_colour(tl_id, "yellow"):
                        lane_type = "yellow"

                lanes_lines[tl_idx][lane_type].extend([xy_left, xy_right])

        cv2.polylines(img, lanes_lines[0]["default"], False, (255, 217, 82), lineType=cv2.LINE_AA, shift=CV2_SHIFT)
        cv2.polylines(img, lanes_lines[0]["green"], False, (0, 255, 0), lineType=cv2.LINE_AA, shift=CV2_SHIFT)
        cv2.polylines(img, lanes_lines[0]["yellow"], False, (255, 255, 0), lineType=cv2.LINE_AA, shift=CV2_SHIFT)
        cv2.polylines(img, lanes_lines[0]["red"], False, (255, 0, 0), lineType=cv2.LINE_AA, shift=CV2_SHIFT)

        # Fill in tl history
        for tl_idx in range(1, len(lanes_lines)):
            cv2.polylines(tl_imgs[tl_idx-1], lanes_lines[tl_idx]["default"], False, (255, 217, 82), lineType=cv2.LINE_AA, shift=CV2_SHIFT)
            cv2.polylines(tl_imgs[tl_idx-1], lanes_lines[tl_idx]["green"], False, (0, 255, 0), lineType=cv2.LINE_AA, shift=CV2_SHIFT)
            cv2.polylines(tl_imgs[tl_idx-1], lanes_lines[tl_idx]["yellow"], False, (255, 255, 0), lineType=cv2.LINE_AA, shift=CV2_SHIFT)
            cv2.polylines(tl_imgs[tl_idx-1], lanes_lines[tl_idx]["red"], False, (255, 0, 0), lineType=cv2.LINE_AA, shift=CV2_SHIFT)

        # plot crosswalks
        crosswalks = []
        for idx in elements_within_bounds(center_world, self.bounds_info["crosswalks"]["bounds"], raster_radius):
            crosswalk = self.proto_API.get_crosswalk_coords(self.bounds_info["crosswalks"]["ids"][idx])

            xy_cross = cv2_subpixel(transform_points(crosswalk["xyz"][:, :2], raster_from_world))
            crosswalks.append(xy_cross)

        cv2.polylines(img, crosswalks, True, (255, 117, 69), lineType=cv2.LINE_AA, shift=CV2_SHIFT)

        tl_imgs = [np.sum(tl_img, axis=-1) for tl_img in tl_imgs]
        tl_img = np.sum(np.stack(tl_imgs, axis=-1), axis=-1)

        return np.concatenate([img, tl_img[:, :, np.newaxis]], axis=-1)

    def to_rgb(self, in_im: np.ndarray, **kwargs: dict) -> np.ndarray:
        # Exclude individual traffic light channel from rgb
        return (in_im[:, :, :-1] * 255).astype(np.uint8)


class SemBoxTLRasterizer(SemBoxRasterizer):
    """Combine a Semantic Map and a Box Rasterizers into a single class
    """

    def __init__(
        self,
        render_context: RenderContext,
        filter_agents_threshold: float,
        history_num_frames: int,
        semantic_map_path: str,
        world_to_ecef: np.ndarray,
    ):
        super(SemBoxTLRasterizer, self).__init__(render_context, filter_agents_threshold, history_num_frames, semantic_map_path, world_to_ecef)

        # Change self.sat_rast reference to include traffic lights as separate channel
        self.sat_rast = SemanticTLRasterizer(render_context, semantic_map_path, world_to_ecef)


def generate_agent_sample_random_ego_center(
    state_index: int,
    frames: np.ndarray,
    agents: np.ndarray,
    tl_faces: np.ndarray,
    selected_track_id: Optional[int],
    render_context: RenderContext,
    history_num_frames: int,
    history_step_size: int,
    future_num_frames: int,
    future_step_size: int,
    filter_agents_threshold: float,
    rasterizer: Optional[Rasterizer] = None,
    perturbation: Optional[Perturbation] = None,
) -> dict:
    """Generates the inputs and targets to train a deep prediction model. A deep prediction model takes as input
    the state of the world (here: an image we will call the "raster"), and outputs where that agent will be some
    seconds into the future.

    This function has a lot of arguments and is intended for internal use, you should try to use higher level classes
    and partials that use this function.

    Args:
        state_index (int): The anchor frame index, i.e. the "current" timestep in the scene
        frames (np.ndarray): The scene frames array, can be numpy array or a zarr array
        agents (np.ndarray): The full agents array, can be numpy array or a zarr array
        tl_faces (np.ndarray): The full traffic light faces array, can be numpy array or a zarr array
        selected_track_id (Optional[int]): Either None for AV, or the ID of an agent that you want to
        predict the future of. This agent is centered in the raster and the returned targets are derived from
        their future states.
        raster_size (Tuple[int, int]): Desired output raster dimensions
        pixel_size (np.ndarray): Size of one pixel in the real world
        ego_center (np.ndarray): Where in the raster to draw the ego, [0.5,0.5] would be the center
        history_num_frames (int): Amount of history frames to draw into the rasters
        history_step_size (int): Steps to take between frames, can be used to subsample history frames
        future_num_frames (int): Amount of history frames to draw into the rasters
        future_step_size (int): Steps to take between targets into the future
        filter_agents_threshold (float): Value between 0 and 1 to use as cutoff value for agent filtering
        based on their probability of being a relevant agent
        rasterizer (Optional[Rasterizer]): Rasterizer of some sort that draws a map image
        perturbation (Optional[Perturbation]): Object that perturbs the input and targets, used
        to train models that can recover from slight divergence from training set data

    Raises:
        ValueError: A ValueError is returned if the specified ``selected_track_id`` is not present in the scene
        or was filtered by applying the ``filter_agent_threshold`` probability filtering.

    Returns:
        dict: a dict object with the raster array, the future offset coordinates (meters),
        the future yaw angular offset, the future_availability as a binary mask
    """
    ego_center = render_context.center_in_raster_ratio

    # Augment ego_center
    rand_ego_center = np.array([ego_center[0] * np.random.uniform(0.6, 1.8), ego_center[1] * np.random.uniform(0.6, 1.8)])

    #  the history slice is ordered starting from the latest frame and goes backward in time., ex. slice(100, 91, -2)
    history_slice = get_history_slice(state_index, history_num_frames, history_step_size, include_current_state=True)
    future_slice = get_future_slice(state_index, future_num_frames, future_step_size)

    history_frames = frames[history_slice].copy()  # copy() required if the object is a np.ndarray
    future_frames = frames[future_slice].copy()

    sorted_frames = np.concatenate((history_frames[::-1], future_frames))  # from past to future

    # get agents (past and future)
    agent_slice = get_agents_slice_from_frames(sorted_frames[0], sorted_frames[-1])
    agents = agents[agent_slice].copy()  # this is the minimum slice of agents we need
    history_frames["agent_index_interval"] -= agent_slice.start  # sync interval with the agents array
    future_frames["agent_index_interval"] -= agent_slice.start  # sync interval with the agents array
    history_agents = filter_agents_by_frames(history_frames, agents)
    future_agents = filter_agents_by_frames(future_frames, agents)

    try:
        tl_slice = get_tl_faces_slice_from_frames(history_frames[-1], history_frames[0])  # -1 is the farthest
        # sync interval with the traffic light faces array
        history_frames["traffic_light_faces_index_interval"] -= tl_slice.start
        history_tl_faces = filter_tl_faces_by_frames(history_frames, tl_faces[tl_slice].copy())
    except ValueError:
        history_tl_faces = [np.empty(0, dtype=TL_FACE_DTYPE) for _ in history_frames]

    if perturbation is not None:
        history_frames, future_frames = perturbation.perturb(
            history_frames=history_frames, future_frames=future_frames
        )

    # State you want to predict the future of.
    cur_frame = history_frames[0]
    cur_agents = history_agents[0]

    if selected_track_id is None:
        agent_centroid = cur_frame["ego_translation"][:2]
        agent_yaw = rotation33_as_yaw(cur_frame["ego_rotation"])
        agent_extent = np.asarray((EGO_EXTENT_LENGTH, EGO_EXTENT_WIDTH, EGO_EXTENT_HEIGHT))
        agent_label_probabilities = None
        selected_agent = None
    else:
        # this will raise IndexError if the agent is not in the frame or under agent-threshold
        # this is a strict error, we cannot recover from this situation
        try:
            agent = filter_agents_by_track_id(
                filter_agents_by_labels(cur_agents, filter_agents_threshold), selected_track_id
            )[0]
        except IndexError:
            raise ValueError(f" track_id {selected_track_id} not in frame or below threshold")
        agent_centroid = agent["centroid"]
        agent_yaw = float(agent["yaw"])
        agent_extent = agent["extent"]
        agent_label_probabilities = agent["label_probabilities"]
        selected_agent = agent

    input_im = (
        None
        if not rasterizer
        else rasterizer.rasterize(history_frames, history_agents, history_tl_faces, selected_agent)
    )

    world_from_agent = compute_agent_pose(agent_centroid, agent_yaw)
    agent_from_world = np.linalg.inv(world_from_agent)
    raster_from_world = render_context.raster_from_world(agent_centroid, agent_yaw)


    future_coords_offset, future_yaws_offset, future_availability = _create_targets_for_deep_prediction(
        future_num_frames, future_frames, selected_track_id, future_agents, agent_centroid[:2], agent_yaw,
    )

    # history_num_frames + 1 because it also includes the current frame
    history_coords_offset, history_yaws_offset, history_availability = _create_targets_for_deep_prediction(
        history_num_frames + 1, history_frames, selected_track_id, history_agents, agent_centroid[:2], agent_yaw,
    )

    return {
        "image": input_im,
        "target_positions": future_coords_offset,
        "target_yaws": future_yaws_offset,
        "target_availabilities": future_availability,
        "history_positions": history_coords_offset,
        "history_yaws": history_yaws_offset,
        "history_availabilities": history_availability,
        "raster_from_world": raster_from_world,
        "world_from_agent": world_from_agent,
        "agent_from_world": agent_from_world,
        "centroid": agent_centroid,
        "ego_center": rand_ego_center,
        "yaw": agent_yaw,
        "extent": agent_extent,
        "label_probabilities": agent_label_probabilities,
    }


def build_rasterizer_tl(cfg: dict, data_manager: DataManager) -> Rasterizer:
    """Factory function for rasterizers, reads the config, loads required data and initializes the correct rasterizer.

    Args:
        cfg (dict): Config.
        data_manager (DataManager): Datamanager that is used to require files to be present.

    Raises:
        NotImplementedError: Thrown when the ``map_type`` read from the config doesn't have an associated rasterizer
        type in this factory function. If you have custom rasterizers, you can wrap this function in your own factory
        function and catch this error.

    Returns:
        Rasterizer: Rasterizer initialized given the supplied config.
    """
    raster_cfg = cfg["raster_params"]
    map_type = raster_cfg["map_type"]
    dataset_meta_key = raster_cfg["dataset_meta_key"]

    render_context = RenderContext(
            raster_size_px=np.array(raster_cfg["raster_size"]),
            pixel_size_m=np.array(raster_cfg["pixel_size"]),
            center_in_raster_ratio=np.array(raster_cfg["ego_center"]),
        )
    filter_agents_threshold = raster_cfg["filter_agents_threshold"]
    history_num_frames = cfg["model_params"]["history_num_frames"]

    semantic_map_filepath = data_manager.require(raster_cfg["semantic_map_key"])
    try:
        dataset_meta = _load_metadata(dataset_meta_key, data_manager)
        world_to_ecef = np.array(dataset_meta["world_to_ecef"], dtype=np.float64)
    except (KeyError, FileNotFoundError):  # TODO remove when new dataset version is available
        world_to_ecef = get_hardcoded_world_to_ecef()
    
    return SemBoxTLRasterizer(render_context, filter_agents_threshold, history_num_frames, semantic_map_filepath, world_to_ecef)


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
        if np.random.uniform() <= 2*p:
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
    ds = AgentDatasetRandomEgoCentre if 'random_ego_center' in cfg['raster_params'] and cfg['raster_params']['random_ego_center'] else AgentDatasetCF
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
        self.weight_by_agent_count = self.args_dict['weight_by_agent_count'] if 'weight_by_agent_count' in self.args_dict else 0

        self.setup()


    def setup(self):

        self.dm = LocalDataManager(None)
        self.rasterizer = self.fn_rasterizer(self.cfg, self.dm)
        self.data_zarr = ChunkedDataset(self.dm.require(self.cfg[self.str_loader]["key"])).open(cached=False)

        raw_data_file = os.path.splitext(self.cfg[self.str_loader]["key"])[0].replace('scenes/', '')

        if 'mask_path' in self.cfg[self.str_loader]:
            mask = np.load(self.cfg[self.str_loader]['mask_path'])["arr_0"]
            self.ds = get_dataset(self.cfg)(raw_data_file, self.cfg, self.str_loader, self.data_zarr, self.rasterizer, agents_mask=mask)
        else:
            self.ds = get_dataset(self.cfg)(raw_data_file, self.cfg, self.str_loader, self.data_zarr, self.rasterizer)

        self.sample_size = min(self.cfg[self.str_loader]['samples_per_epoch'], len(self.ds))

        self.shuffle = True if 'train_data_loader' in self.str_loader else False

        self.add_output = True if self.str_loader == 'test_data_loader' else False # Add timestamp and track_id to output

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
            id_arr = np.ones(idx[-1],dtype=int)
            id_arr[0] = 0
            id_arr[idx[:-1]] = -a[:-1]+1
            return id_arr.cumsum()

        # First shuffle the indices to ensure that there is no order in which the scene samples occur
        idx = list(range(len(self.ds)))
        random.shuffle(idx)

        print('Shuffling scene indices...')
        self.ds.scene_indices = [self.ds.scene_indices[i] for i in tqdm(idx)]
        self.ds.frame_indices = [self.ds.frame_indices[i] for i in tqdm(idx)]
        self.ds.timestamps = [self.ds.timestamps[i] for i in tqdm(idx)]
        self.ds.track_ids = [self.ds.track_ids[i] for i in tqdm(idx)]

        # Within each scene, number the agents/frames 0 -> n
        count = np.unique(self.ds.scene_indices, return_counts=1)[1]
        scene_cumcount = grp_range(count)[np.argsort(self.ds.scene_indices, kind='mergesort').argsort(kind='mergesort')]  # Use mergesort to guarantee same results each time

        # Create all_idx by selecting one agent/frame from each scene consecutively
        cumcounts = range(int(scene_cumcount.max()))
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

            _idx = np.argwhere(agent_counts==agent_count).reshape(-1,)

            select_count = int(len(_idx) * min(1, (self.weight_by_agent_count/agent_count)))

            all_idx.append(np.random.choice(_idx, size=select_count))

        all_idx = np.concatenate(all_idx)

        random.shuffle(all_idx)

        self.all_idx = all_idx

        if DEBUG: print(' : '.join(('Reset all_idx, total sample length', str(len(self.all_idx)), 'from', str(len(self.ds)))))


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
        target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["raster_from_world"])
        draw_trajectory(im, target_positions_pixels, TARGET_POINTS_COLOR, yaws=data["target_yaws"], radius=1)

        plt.imshow(im[::-1])
        plt.show()


class FullMotionPredictDataset(Dataset):
    """
    l5kit Motion prediction dataset wrapper for train_full.zarr
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
        
        self.weight_by_agent_count = self.args_dict['weight_by_agent_count'] if 'weight_by_agent_count' in self.args_dict else 0

        self.setup()


    def setup(self):

        self.dm = LocalDataManager(None)
        self.rasterizer = self.fn_rasterizer(self.cfg, self.dm)
        self.data_zarr = ChunkedDataset(self.dm.require(self.cfg[self.str_loader]["key"])).open(cached=False)

        if 'mask_path' in self.cfg[self.str_loader]:
            mask = np.load(self.cfg[self.str_loader]['mask_path'])["arr_0"]
            self.ds = AgentDataset(self.cfg,self.data_zarr, self.rasterizer, None, agents_mask=mask, min_frame_history=10, min_frame_future=10)
        else:
            self.ds = AgentDataset(self.cfg, self.data_zarr, self.rasterizer, None, None, min_frame_history=10, min_frame_future=10)

        self.sample_size = min(self.cfg[self.str_loader]['samples_per_epoch'], len(self.ds))

        self.shuffle = True if 'train_data_loader' in self.str_loader else False

        self.add_output = True if self.str_loader == 'test_data_loader' else False # Add timestamp and track_id to output

        self.n_epochs = self.args_dict['n_epochs'] if 'n_epochs' in self.args_dict else 1000

        self.set_all_idx()


    def __getitem__(self, index):
        
        out = self.fn_create(self.ds, index, self.args_dict, self.cfg, self.str_loader)

        # Include timestamps and track_ids in the case of test/val
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
        
        if self.weight_by_agent_count > 0:
            self.set_all_idx_weight_by_agent_count()
        else:
            self.set_all_idx_default()


    def set_all_idx_default(self):

        total_samples = self.cfg[self.str_loader]['samples_per_epoch'] * self.n_epochs
        N = len(self.ds)

        self.all_idx = []
        counter = 0
        while len(self.all_idx) < total_samples:
            self.all_idx.extend(list(range(counter, N, N//total_samples)))
            counter += 1
        self.all_idx = self.all_idx[:total_samples]
        
        if self.shuffle: random.shuffle(self.all_idx)


    def set_all_idx_weight_by_agent_count(self):
        # TODO
        #frame_index = bisect.bisect_right(self.cumulative_sizes_agents, index)
        pass


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

        assert isinstance(self.str_loader, (list, tuple)), 'str_loader must be a list/tuple for use in MultiMotionPredictDataset '

        self.dataset_list = [MotionPredictDataset(self.cfg, self.args_dict, str_loader, self.fn_rasterizer, self.fn_create) for str_loader in self.str_loader]

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
        idx = index if dataset_loc==0 else index - self.cumulative_dataset_sizes[dataset_loc - 1]

        out = self.dataset_list[dataset_loc][idx]

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
    subfolder = os.path.join(CACHE_DIR, '_'.join((str_fn_create,  str_data_loader, cfg['raster_params']['map_type'], str_input_size, str_pixel_size, str_ego_center, str_history_num_frames, str_future_num_frames)))

    return os.path.join(subfolder, idx_filename)


def double_channel_agents_ego_map_transform(dataset, idx, args_dict, cfg, str_data_loader, info=False, info_dict=None):
    """
    double_channel_agents_ego_map tailored to multi mode output model 
    including centroid and raster_from_world matrix
    """
    if info:
        n_input_channels = 5 # Each ego/agent is condensed into two channels, each map is condensed into 1
        n_output = info_dict['n_modes'] + (info_dict['future_num_frames'] * 3 * info_dict['n_modes']) # n_confs + (future_num_frames * (x, y, yaws) * modes)
        return n_input_channels, n_output

    data = check_load(get_cache_filename(idx, args_dict, cfg, 'double_channel_agents_ego_map_transform', str_data_loader),
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


def double_channel_agents_ego_map_avg_transform(dataset, idx, args_dict, cfg, str_data_loader, info=False, info_dict=None):
    """
    double_channel_agents_ego_map tailored to multi mode output model 
    including centroid and raster_from_world matrix
    """
    if info:
        n_input_channels = 5 # Each ego/agent is condensed into two channels, each map is condensed into 1
        n_output = info_dict['n_modes'] + (info_dict['future_num_frames'] * 3 * info_dict['n_modes']) # n_confs + (future_num_frames * (x, y, yaws) * modes)
        return n_input_channels, n_output

    data = check_load(get_cache_filename(idx, args_dict, cfg, 'double_channel_agents_ego_map_transform', str_data_loader),
                      return_idx, idx, CREATE_CACHE, (dataset, idx))

    im = data["image"].transpose(1, 2, 0)

    n, im_map, im_agents_history, im_agents_current, im_ego_history, im_ego_current = split_im(im)

    history_idx = generate_history_idx(n - 1, args_dict['sample_history_num_frames'], args_dict['SHUFFLE'])

    im_map = np.sum(im_map, axis=-1)

    im_agents_history = np.mean(im_agents_history[:, :, history_idx], axis=-1)

    im_ego_history = np.mean(im_ego_history[:, :, history_idx], axis=-1)

    im_reduced = np.stack([im_agents_history, im_agents_current, im_ego_history, im_ego_current, im_map], axis=-1)

    transforms = make_transform(args_dict['TRANSFORMS']) if 'TRANSFORMS' in args_dict else None
    im_reduced = augment_img(im_reduced, transforms)

    x = numpy_to_torch(im_reduced)

    y, transform_matrix, centroid, ego_center = create_y_transform_tensor(data, cfg)

    return [x, transform_matrix, centroid, ego_center], y, int(data['timestamp']), int(data['track_id'])


def double_channel_agents_ego_map_dayhour(dataset, idx, args_dict, cfg, str_data_loader, info=False, info_dict=None):
    """
    double_channel_agents_ego_map tailored to multi mode output model
    including centroid and raster_from_world matrix
    """
    if info:
        n_input_channels = 6 # Each ego/agent is condensed into two channels, each map is condensed into 1
        n_output = info_dict['n_modes'] + (info_dict['future_num_frames'] * 3 * info_dict['n_modes']) # n_confs + (future_num_frames * (x, y, yaws) * modes)
        return n_input_channels, n_output

    data = check_load(get_cache_filename(idx, args_dict, cfg, 'double_channel_agents_ego_map_date', str_data_loader),
                      return_idx, idx, CREATE_CACHE, (dataset, idx))

    im = data["image"].transpose(1, 2, 0)

    n, im_map, im_agents_history, im_agents_current, im_ego_history, im_ego_current = split_im(im)

    history_idx = generate_history_idx(n - 1, args_dict['sample_history_num_frames'], args_dict['SHUFFLE'])

    im_map = np.sum(im_map, axis=-1)

    im_agents_history = np.sum(im_agents_history[:, :, history_idx], axis=-1)

    im_ego_history = np.sum(im_ego_history[:, :, history_idx], axis=-1)

    im_reduced = np.stack([im_agents_history, im_agents_current, im_ego_history, im_ego_current, im_map], axis=-1)

    transforms = make_transform(args_dict['TRANSFORMS']) if 'TRANSFORMS' in args_dict else []
    im_reduced = augment_img(im_reduced, transforms)

    _date = datetime.fromtimestamp(data['timestamp'] // 10**9)
    weekday = _date.weekday()
    hour = _date.hour
    dayhour = weekday + (hour/25)

    im_reduced = np.concatenate([im_reduced, np.ones(im_reduced.shape[:2] + (1,)) * dayhour], axis=-1)

    x = numpy_to_torch(im_reduced)

    y, transform_matrix, centroid, ego_center = create_y_transform_tensor(data, cfg)

    return [x, transform_matrix, centroid, ego_center], y, int(data['timestamp']), int(data['track_id'])


def double_channel_agents_ego_map_coords(dataset, idx, args_dict, cfg, str_data_loader, info=False, info_dict=None):
    """
    double_channel_agents_ego_map tailored to multi mode output model 
    including centroid and raster_from_world matrix.
    Includes x/y/distance from ego centre as 3 additional channels
    """
    if info:
        n_input_channels = 8 # Each ego/agent is condensed into two channels, each map is condensed into 1
        n_output = info_dict['n_modes'] + (info_dict['future_num_frames'] * 3 * info_dict['n_modes']) # n_confs + (future_num_frames * (x, y, yaws) * modes)
        return n_input_channels, n_output

    data = check_load(get_cache_filename(idx, args_dict, cfg, 'double_channel_agents_ego_map_transform', str_data_loader),
                      return_idx, idx, CREATE_CACHE, (dataset, idx))

    im = data["image"].transpose(1, 2, 0)

    n, im_map, im_agents_history, im_agents_current, im_ego_history, im_ego_current = split_im(im)

    history_idx = generate_history_idx(n - 1, args_dict['sample_history_num_frames'], args_dict['SHUFFLE'])

    im_map = np.sum(im_map, axis=-1)

    im_agents_history = np.sum(im_agents_history[:, :, history_idx], axis=-1)

    im_ego_history = np.sum(im_ego_history[:, :, history_idx], axis=-1)

    im_coords = generate_coordinate_channels(cfg)

    im_reduced = np.stack([im_agents_history, im_agents_current, im_ego_history, im_ego_current, im_map], axis=-1)

    transforms = make_transform(args_dict['TRANSFORMS']) if 'TRANSFORMS' in args_dict else None
    im_reduced = augment_img(im_reduced, transforms)

    im_reduced = np.concatenate([im_reduced, im_coords], axis=-1)

    x = numpy_to_torch(im_reduced)

    y, transform_matrix, centroid, ego_center = create_y_transform_tensor(data, cfg)

    return [x, transform_matrix, centroid, ego_center], y, int(data['timestamp']), int(data['track_id'])


def double_channel_agents_ego_map_relativecoords(dataset, idx, args_dict, cfg, str_data_loader, info=False, info_dict=None):
    """
    double_channel_agents_ego_map tailored to multi mode output model 
    including centroid and raster_from_world matrix.
    Includes x/y relative coordinates as 2 additional channels
    """
    if info:
        n_input_channels = 7 # Each ego/agent is condensed into two channels, each map is condensed into 1
        n_output = info_dict['n_modes'] + (info_dict['future_num_frames'] * 3 * info_dict['n_modes']) # n_confs + (future_num_frames * (x, y, yaws) * modes)
        return n_input_channels, n_output

    data = check_load(get_cache_filename(idx, args_dict, cfg, 'double_channel_agents_ego_map_transform', str_data_loader),
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
    n = (im.shape[-1] - 3)//2

    im_agents_current = im[:, :, 0]
    im_agents_history = im[:, :, 1:n]

    im_ego_current = im[:, :, n]
    im_ego_history = im[:, :, n+1:-3]

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
    ch_c = np.sqrt((ch_h - cfg['raster_params']['ego_center'][0]*h)**2 + (ch_v - cfg['raster_params']['ego_center'][1]*h)**2)
    
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
    y = torch.stack([torch.matmul(world_from_agent.to(device), y[:, i].transpose(1, 2)) for i in range(y.shape[1])], dim=1)
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


##############################################
# DATA LOADERS
##############################################

def create_data_loaders(fn_rasterizer, fn_create, fn_cfg, cfg_model_params, input_size, pixel_size, ego_center, batch_size, num_workers, args_dicts, str_loaders, drop_last=False):

    samples_per_epoch = args_dicts[0]['samples_per_epoch']
    sample_history_num_frames = args_dicts[0]['sample_history_num_frames']
    history_num_frames = args_dicts[0]['history_num_frames']
    future_num_frames = args_dicts[0]['future_num_frames']
    max_agents = args_dicts[0]['max_agents']
    n_modes = args_dicts[0]['n_modes']
    str_network = args_dicts[0]['str_network']

    n_input_channels, n_output_channels = fn_create(None, None, None, None, None, info=True, info_dict=args_dicts[0])

    cfg = fn_cfg(str_network, cfg_model_params, input_size, pixel_size, ego_center, batch_size, num_workers, samples_per_epoch, sample_history_num_frames, history_num_frames, future_num_frames, n_modes, max_agents, n_input_channels, n_output_channels)

    data_loaders = [create_loader(args_dict['clsDataset'], fn_rasterizer, fn_create, cfg, args_dict, str_loader, drop_last) for args_dict, str_loader in zip(args_dicts, str_loaders)]

    return tuple(data_loaders) + (cfg,)


def create_loader(clsDataset, fn_rasterizer, fn_create, cfg, args_dict, str_loader, drop_last=False):

    _dataset = clsDataset(cfg, args_dict, str_loader, fn_rasterizer, fn_create)

    # Checks:
    #_dataset.plot_index(0)
    _dataset[0]

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

    fig = plt.figure(figsize = (12,12))
    n_trajectories = len(trajectory_list)
    
    trajectories = [trajectory[:, 0].reshape(batch_size, future_num_frames, -1)[:, :, :2] for trajectory in trajectory_list]

    for i in range(batch_size):
        for j, trajectory in enumerate(trajectories):
            plt.subplot(1, n_trajectories, j + 1)
            plt.plot(trajectory[i, : ,0].numpy(), trajectory[i, : ,1].numpy())

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
    n_points = 3 # (x, y, yaws)

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


def torch_neg_multi_log_likelihood(gt, pred, confidences, avails):
    """
    pytorch version of l5kit's neg_multi_log_likelihood
    """

    # add modes and cords
    gt = gt.unsqueeze(1)
    avails = avails.unsqueeze(1).unsqueeze(-1)

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
    pred_orig, pred_transform, truth_orig, truth_transform, conf, mask, batch_size, n_modes, future_num_frames, centroid = data_transform_to_modes(pred.cpu(), truth.cpu())

    if calctype.lower() == 'orig':
        nll = torch_neg_multi_log_likelihood(truth_orig.reshape(batch_size, future_num_frames, -1)[:, :, :2],
                                            pred_orig.reshape(batch_size, n_modes, future_num_frames, -1)[:, :, :, :2],  # ignore yaws
                                            conf.reshape(batch_size, n_modes), 
                                            mask.reshape(batch_size, future_num_frames, -1)[:, :, 0]) 
    elif calctype.lower() == 'transform':
        nll = torch_neg_multi_log_likelihood(truth_transform.reshape(batch_size, future_num_frames, -1)[:, :, :2],
                                            pred_transform.reshape(batch_size, n_modes, future_num_frames, -1)[:, :, :, :2],  # ignore yaws
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


@dataclass
class MultiModeNegLogLossTransform(Callback):

    def on_epoch_begin(self, **kwargs):
        self.nll = torch.tensor([])
        self.nll_transform = torch.tensor([])
        
    def on_batch_end(self, last_output:Tensor, last_target:Tensor, **kwargs):
        nll_val = neg_log_likelihood_transform(last_output, last_target, calctype='orig', reduction='none')
        nll_val_transform = neg_log_likelihood_transform(last_output, last_target, calctype='transform', reduction='none')
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
        #self.optimizer = torch.optim.Adam(self.get_opt_params(), lr=self.lr, weight_decay=self.weight_decay)
        self.optimizer = AdamP(self.get_opt_params(), lr=self.lr, weight_decay=self.weight_decay)

        if step_lr:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=2, gamma=0.5, last_epoch=-1)
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.1, patience=3,
                                                                        verbose=True)

        self.history = {'train_loss': [], 'train_auc': [], 'val_loss': [], 'val_auc': []}
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
            print(' '.join(('Setting data_loader sample size from', str(orig_sample_size), 'to', str(len(data_loader.dataset.ds)))))
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

                    pred_orig, pred_transform, truth_orig, truth_transform, conf, mask, batch_size, n_modes, future_num_frames, centroid = data_transform_to_modes(out, pseudo_y)

                    # Shape predictions correctly and take just the first two (target_x, target_y)
                    pred = pred_orig.reshape(batch_size, n_modes, future_num_frames, -1)[:, :, :, :2]

                y_pred.append(pred.cpu().numpy())
                y_conf.append(conf.cpu().numpy())
                timestamps.append(timestamp.numpy())
                track_ids.append(track_id.numpy())
                centroids.append(centroid.cpu().numpy())

                pbar.update()

        test_dict = {'preds': np.concatenate(y_pred), 'conf': np.concatenate(y_conf), 'centroids': np.concatenate(centroids), 'timestamps': np.concatenate(timestamps), 'track_ids': np.concatenate(track_ids)}

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
        
        out = torch.cat([centroid, x_confs, x_orig.reshape(batch_size, -1), x_transform.reshape(batch_size, -1)], dim=-1)

        return out


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
        val_fn_loader, cfg = create_data_loaders(rasterizer_fn, loader_fn, cfg_fn, cfg_model_params, input_size, pixel_size, ego_center, batch_size, NUM_WORKERS, [val_args_dict], ['test_data_loader'])
    else:
        train_loader, val_loader, cfg = create_data_loaders(rasterizer_fn, loader_fn, cfg_fn, cfg_model_params, input_size, pixel_size, ego_center, batch_size, NUM_WORKERS,
                                                       [train_args_dict, val_args_dict], 
                                                       [str_train_loaders, 'val_data_loader'])
        val_fn_loader = val_loader

    # Init model
    model = clsModel(cfg=cfg)

    # Create network
    net = Network(model, train_loader, val_loader, lr=lr, init_model_weights_path=init_model_weights_path, model_checkpoint_path=model_checkpoint_path, save_net=True)

    # Fit
    if action == 'fit':
        val_dict = getattr(net, fit_fn)(n_epochs, loss_fn=loss_fn)

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

def setup_args_dicts(clsTrainDataset, clsValDataset, aug, str_network, in_size, pixel_size, ego_center, batch_size, samples_per_epoch, n_epochs, sample_history_num_frames, history_num_frames, future_num_frames, n_modes, max_agents, group_scenes, weight_by_agent_count):

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
                        'future_num_frames': future_num_frames,
                        'n_modes': n_modes,
                        'max_agents': max_agents,
                        'group_scenes': group_scenes,
                        'clsDataset': clsValDataset,
                        'SHUFFLE': False}

    return train_args_dict, val_args_dict


def setup_test_args_dict(clsDataset, aug, str_network, in_size, pixel_size, ego_center, batch_size, samples_per_epoch, n_epochs, sample_history_num_frames, history_num_frames, future_num_frames, n_modes, max_agents, group_scenes, weight_by_agent_count):

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
                        'future_num_frames': future_num_frames,
                        'n_modes': n_modes,
                        'max_agents': max_agents,
                        'group_scenes': group_scenes,
                        'clsDataset': clsDataset,
                        'SHUFFLE': False}

    return test_args_dict


def create_base_filename(clsTrainDataset, clsValDataset, clsModel, val_fn, rasterizer_fn, loader_fn,  cfg_fn, fit_fn, loss_fn, in_size, pixel_size, ego_center, n_epochs, batch_size, samples_per_epoch, sample_history_num_frames, history_num_frames, future_num_frames, n_modes, max_agents, group_scenes, weight_by_agent_count, str_network, aug, model_str):

    str_train_dataset = str(clsTrainDataset).split('.')[-1].split("'")[0]
    str_val_dataset = str(clsValDataset).split('.')[-1].split("'")[0]
    str_model = str(clsModel).split('.')[-1].split("'")[0]
    str_loader_fn = str(loader_fn).split(' ')[1]
    str_cfg_fn = str(cfg_fn).split(' ')[1]
    str_loss_fn = str(loss_fn).split(' ')[1]
    str_rasterizer_fn = str(rasterizer_fn).split(' ')[1]

    str_pixel_size = '_'.join(([str(i) for i in pixel_size]))
    str_ego_center = '_'.join(([str(i) for i in ego_center]))

    #base_filename = '_'.join((val_fn, str_train_dataset, str_val_dataset, str_model, str_rasterizer_fn, str_loader_fn, str_cfg_fn, str_loss_fn, str(in_size), str_pixel_size, str_ego_center, str(n_epochs),str(batch_size), str(samples_per_epoch), str(sample_history_num_frames), str(history_num_frames), str(future_num_frames), str(n_modes), str_network, fit_fn, aug, model_str, '.pkl'))
    #base_filename = '_'.join((val_fn, str_model, str_rasterizer_fn, str_loader_fn, str_cfg_fn, str_loss_fn, str(in_size), str_pixel_size, str_ego_center, str(n_epochs),str(batch_size), str(samples_per_epoch), str(sample_history_num_frames), str(history_num_frames), str(future_num_frames), str(n_modes), str(group_scenes), str_network, fit_fn, aug, model_str, '.pkl'))
    #base_filename = '_'.join((val_fn, str_model, str_loader_fn, str_cfg_fn, str_loss_fn, str(in_size), str_pixel_size, str_ego_center, str(n_epochs),str(batch_size), str(samples_per_epoch), str(sample_history_num_frames), str(history_num_frames), str(future_num_frames), str(n_modes), str(group_scenes), str_network, fit_fn, aug, model_str, '.pkl'))
    #base_filename = '_'.join((val_fn, str_model, str_loader_fn, str_cfg_fn, str_loss_fn, str(in_size), str_pixel_size, str_ego_center, str(n_epochs),str(batch_size), str(samples_per_epoch), str(sample_history_num_frames), str(history_num_frames), str(future_num_frames), str(n_modes), str(max_agents), str_network, fit_fn, aug, model_str, '.pkl'))
    base_filename = '_'.join((val_fn, str_model, str_loader_fn, str_cfg_fn, str_loss_fn, str(in_size), str(n_epochs),str(batch_size), str(samples_per_epoch), str(sample_history_num_frames), str(history_num_frames), str(future_num_frames), str(n_modes), str(max_agents), str(weight_by_agent_count), str_network, fit_fn, aug, model_str, '.pkl'))

    return base_filename
    

def create_val_dict_filepath(clsTrainDataset, clsValDataset, clsModel, val_fn, rasterizer_fn, loader_fn,  cfg_fn, fit_fn, loss_fn, in_size, pixel_size, ego_center, n_epochs, batch_size, samples_per_epoch, sample_history_num_frames, history_num_frames, future_num_frames, n_modes, max_agents, group_scenes, weight_by_agent_count, str_network, aug, model_str):

    base_filename = create_base_filename(clsTrainDataset, clsValDataset, clsModel, val_fn, rasterizer_fn, loader_fn,  cfg_fn, fit_fn, loss_fn, in_size, pixel_size, ego_center, n_epochs, batch_size, samples_per_epoch, sample_history_num_frames, history_num_frames, future_num_frames, n_modes, max_agents, group_scenes, weight_by_agent_count, str_network, aug, model_str)

    return os.path.join(DATA_DIR, 'val_' + base_filename)

    
def create_test_dict_filepath(clsTrainDataset, clsValDataset, clsModel, val_fn, rasterizer_fn, loader_fn,  cfg_fn, fit_fn, loss_fn, in_size, pixel_size, ego_center, n_epochs, batch_size, samples_per_epoch, sample_history_num_frames, history_num_frames, future_num_frames, n_modes, max_agents, group_scenes, weight_by_agent_count, str_network, aug, model_str):

    base_filename = create_base_filename(clsTrainDataset, clsValDataset, clsModel, val_fn, rasterizer_fn, loader_fn,  cfg_fn, fit_fn, loss_fn, in_size, pixel_size, ego_center, n_epochs, batch_size, samples_per_epoch, sample_history_num_frames, history_num_frames, future_num_frames, n_modes, max_agents, group_scenes, weight_by_agent_count, str_network, aug, model_str)

    return os.path.join(DATA_DIR, 'test_' + base_filename)


def create_model_checkpoint_path(clsTrainDataset, clsValDataset, clsModel, val_fn, rasterizer_fn, loader_fn, cfg_fn, fit_fn, loss_fn, in_size, pixel_size, ego_center, n_epochs, batch_size, samples_per_epoch, sample_history_num_frames, history_num_frames, future_num_frames, n_modes, max_agents, group_scenes, weight_by_agent_count, str_network, aug, model_str):

    str_train_dataset = str(clsTrainDataset).split('.')[-1].split("'")[0]
    str_val_dataset = str(clsValDataset).split('.')[-1].split("'")[0]
    str_model = str(clsModel).split('.')[-1].split("'")[0]
    str_loader_fn = str(loader_fn).split(' ')[1]
    str_cfg_fn = str(cfg_fn).split(' ')[1]
    str_loss_fn = str(loss_fn).split(' ')[1]
    str_rasterizer_fn = str(rasterizer_fn).split(' ')[1]

    str_pixel_size = '_'.join(([str(i) for i in pixel_size]))
    str_ego_center = '_'.join(([str(i) for i in ego_center]))

    #model_filename = '_'.join(('model_checkpoint', str_train_dataset, str_val_dataset, str_model, str_rasterizer_fn, str_loader_fn, str_cfg_fn, str_loss_fn, str(in_size), str_pixel_size, str_ego_center, str(n_epochs),str(batch_size), str(samples_per_epoch), str(sample_history_num_frames), str(history_num_frames), str(future_num_frames), str(n_modes), str_network, fit_fn, aug, model_str, '.pth'))
    #model_filename = '_'.join(('chkpt', str_model, str_rasterizer_fn, str_loader_fn, str_cfg_fn, str_loss_fn, str(in_size), str_pixel_size, str_ego_center, str(n_epochs),str(batch_size), str(samples_per_epoch), str(sample_history_num_frames), str(history_num_frames), str(future_num_frames), str(n_modes), str(max_agents), str_network, fit_fn, aug, model_str, '.pth'))
    model_filename = '_'.join(('chkpt', str_model, str_rasterizer_fn, str_loader_fn, str_cfg_fn, str_loss_fn, str(in_size), str(n_epochs),str(batch_size), str(samples_per_epoch), str(sample_history_num_frames), str(history_num_frames), str(future_num_frames), str(n_modes), str(max_agents), str(weight_by_agent_count), str_network, fit_fn, aug, model_str, '.pth'))
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
                            n_epochs=20, in_size=224, pixel_size=[0.5, 0.5], ego_center=[0.25, 0.5], batch_size=24, samples_per_epoch=17000, lr=3e-4, 
                            sample_history_num_frames=10, history_num_frames=10, future_num_frames=50, n_modes=3, max_agents=40, 
                            group_scenes=False, weight_by_agent_count=False,
                            fit_fn='fit_transform', val_fn='test_transform', aug='none', 
                            clsTrainDataset=MotionPredictDataset, clsValDataset=MotionPredictDataset,
                            clsModel=LyftResnet18Transform, init_model_weights_path=None, cfg_model_params=None,
                            rasterizer_fn=build_rasterizer,
                            loss_fn=neg_log_likelihood_transform, 
                            str_train_loaders=['train_data_loader_100', 'train_data_loader_30'],
                            loader_fn=double_channel_agents_ego_map_transform, cfg_fn=create_config_multi_train_chopped):

    val_dict_filepath = create_val_dict_filepath(clsTrainDataset, clsValDataset, clsModel, val_fn, rasterizer_fn,
                                                 loader_fn, cfg_fn, fit_fn, loss_fn, in_size, pixel_size, ego_center,
                                                 n_epochs, batch_size, samples_per_epoch, sample_history_num_frames,
                                                 history_num_frames, future_num_frames, n_modes, max_agents,
                                                 group_scenes, weight_by_agent_count, str_network, aug, model_str)


    if not os.path.exists(val_dict_filepath):

        print(' : '.join(('Training model', str_network, 'for input size', str(in_size), 'batch_size', str(batch_size), 'augmentation', aug, 'val_file', os.path.split(val_dict_filepath)[-1])))

        # Set up args_dict inputs
        train_args_dict, val_args_dict = setup_args_dicts(clsTrainDataset, clsValDataset, aug, str_network, in_size, pixel_size, ego_center, batch_size, samples_per_epoch, n_epochs, sample_history_num_frames, history_num_frames, future_num_frames, n_modes, max_agents, group_scenes, weight_by_agent_count)

        # Fit / evaluate model
        model_checkpoint_path = create_model_checkpoint_path(clsTrainDataset, clsValDataset, clsModel, val_fn, rasterizer_fn, loader_fn, cfg_fn, fit_fn, loss_fn, in_size, pixel_size, ego_center, n_epochs, batch_size, samples_per_epoch, sample_history_num_frames, history_num_frames, future_num_frames, n_modes, max_agents, group_scenes, weight_by_agent_count, str_network, aug, model_str)

        val_dicts, net = fit_multitrain_motion_predict(n_epochs, train_args_dict, val_args_dict,
                                                    init_model_weights_path=init_model_weights_path,
                                                    model_checkpoint_path=model_checkpoint_path, 
                                                    fit_fn=fit_fn, loss_fn=loss_fn, val_fn=val_fn, loader_fn=loader_fn, 
                                                    cfg_fn=cfg_fn, str_train_loaders=str_train_loaders,
                                                    rasterizer_fn=rasterizer_fn,
                                                    cfg_model_params=cfg_model_params,
                                                    clsModel=clsModel, lr=lr,
                                                    action='fit' if not os.path.exists(model_checkpoint_path) else 'evaluate')


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
                                    n_epochs=20, in_size=224, pixel_size=[0.5, 0.5], ego_center=[0.25, 0.5], batch_size=24, samples_per_epoch=17000, lr=3e-4, 
                                    sample_history_num_frames=10, history_num_frames=10, future_num_frames=50, n_modes=3, max_agents=40, 
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
                                    loader_fn=double_channel_agents_ego_map_transform, cfg_fn=create_config_multi_train_chopped):
        
    test_dict_filepath = create_test_dict_filepath(clsTrainDataset, clsValDataset, clsModel, val_fn, rasterizer_fn, loader_fn, cfg_fn, fit_fn, loss_fn, in_size, pixel_size, ego_center, n_epochs, batch_size, samples_per_epoch, sample_history_num_frames, history_num_frames, future_num_frames, n_modes, max_agents, group_scenes, weight_by_agent_count, str_network, aug, model_str)
    
    if not os.path.exists(test_dict_filepath):

        print(' : '.join(('Forecasting for model', str_network, 'for input size', str(in_size), 'batch_size', str(batch_size), 'augmentation', aug, 'test_file', os.path.split(test_dict_filepath)[-1])))

        # Set up args_dict inputs
        test_args_dict = setup_test_args_dict(clsTestDataset, aug, str_network, in_size, pixel_size, ego_center, batch_size, samples_per_epoch, n_epochs, sample_history_num_frames, history_num_frames, future_num_frames, n_modes, max_agents, group_scenes, weight_by_agent_count)

        # Fit / evaluate model
        model_checkpoint_path = create_model_checkpoint_path(clsTrainDataset, clsValDataset, clsModel, val_fn, rasterizer_fn, loader_fn, cfg_fn, fit_fn, loss_fn, in_size, pixel_size, ego_center, n_epochs, batch_size, samples_per_epoch, sample_history_num_frames, history_num_frames, future_num_frames, n_modes, max_agents, group_scenes, weight_by_agent_count, str_network, aug, model_str)

        test_dicts, net = fit_multitrain_motion_predict(n_epochs, test_args_dict, test_args_dict,
                                                        init_model_weights_path=init_model_weights_path,
                                                        model_checkpoint_path=model_checkpoint_path, 
                                                        fit_fn=fit_fn, loss_fn=loss_fn, val_fn=val_fn, loader_fn=loader_fn,
                                                        rasterizer_fn=rasterizer_fn, cfg_fn=cfg_fn, str_train_loaders=str_train_loaders,
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


def test_agent_dataset(str_loader):
    cfg = create_config_multi_train_chopped('')
    
    dm = LocalDataManager(None)
    rasterizer = build_rasterizer(cfg, dm)
    data_zarr = ChunkedDataset(dm.require(cfg[str_loader]["key"])).open(cached=False)

    raw_data_file = os.path.splitext(cfg[str_loader]["key"])[0].replace('scenes/', '')

    mask = np.load(cfg[str_loader]['mask_path'])["arr_0"]
    ds = get_dataset(cfg)(raw_data_file, cfg, str_loader, data_zarr, rasterizer, agents_mask=mask) #AgentDataset(cfg, data_zarr, rasterizer, agents_mask=mask)#
        
    for i in range(300):
        data = ds[i]
        plt.plot(data['target_positions'][:, 0], data['target_positions'][:, 1])

    plt.show()



if __name__ == '__main__':
    """
    print('NEW L5KIT, NO AUG, FULL DATASET')

    run_tests_multi_motion_predict(n_epochs=200, in_size=128, batch_size=256, samples_per_epoch=17000,
                                   sample_history_num_frames=5, history_num_frames=5, future_num_frames=50,
                                   group_scenes=False, weight_by_agent_count=0,
                                   clsTrainDataset=FullMotionPredictDataset,
                                   clsValDataset=MotionPredictDataset,
                                   clsModel=LyftResnet18Transform,
                                   fit_fn='fit_fastai_transform', val_fn='test_transform',
                                   loss_fn=neg_log_likelihood_transform,
                                   aug='none',
                                   loader_fn=double_channel_agents_ego_map_transform,
                                   cfg_fn=create_config_train_full,
                                   str_train_loaders='train_data_loader',
                                   rasterizer_fn=build_rasterizer)
    """
    chop_indices = list(range(10, 201, 10))
    run_tests_multi_motion_predict(n_epochs=1000, in_size=320, batch_size=256, samples_per_epoch=17000//len(chop_indices),
                                   sample_history_num_frames=10, history_num_frames=10, future_num_frames=50,
                                   group_scenes=False, weight_by_agent_count=7,
                                   clsTrainDataset=MultiMotionPredictDataset,
                                   clsValDataset=MotionPredictDataset,
                                   clsModel=LyftResnet18Transform,
                                   fit_fn='fit_fastai_transform', val_fn='test_transform',
                                   loss_fn=neg_log_likelihood_transform,
                                   aug='none',
                                   loader_fn=double_channel_agents_ego_map_avg_transform,
                                   cfg_fn=create_config_multi_train_chopped,
                                   str_train_loaders=['train_data_loader_' + str(i) for i in chop_indices],
                                   rasterizer_fn=build_rasterizer)

    run_forecast_multi_motion_predict(n_epochs=1000, in_size=320, batch_size=256,
                                   samples_per_epoch=17000 // len(chop_indices),
                                   sample_history_num_frames=10, history_num_frames=10, future_num_frames=50,
                                   group_scenes=False, weight_by_agent_count=7,
                                   clsTrainDataset=MultiMotionPredictDataset,
                                   clsValDataset=MotionPredictDataset,
                                   clsModel=LyftResnet18Transform,
                                   fit_fn='fit_fastai_transform', val_fn='test_transform',
                                   loss_fn=neg_log_likelihood_transform,
                                   aug='none',
                                   loader_fn=double_channel_agents_ego_map_avg_transform,
                                   cfg_fn=create_config_multi_train_chopped,
                                   str_train_loaders=['train_data_loader_' + str(i) for i in chop_indices],
                                   rasterizer_fn=build_rasterizer)
    