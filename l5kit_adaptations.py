######################################
# IMPORTS
######################################
 
import numpy as np
import os
import random
import math
import inspect
import bisect
import json
from functools import partial
import cv2
from tqdm import tqdm
from collections import defaultdict
from typing import Tuple, cast, Optional, Dict, List

from l5kit.data import DataManager, LocalDataManager, ChunkedDataset, TL_FACE_DTYPE, filter_agents_by_labels, \
    filter_tl_faces_by_frames, get_agents_slice_from_frames, get_tl_faces_slice_from_frames
from l5kit.data.filter import filter_tl_faces_by_status, get_frames_slice_from_scenes, filter_agents_by_frames, \
    filter_agents_by_track_id
from l5kit.data.map_api import MapAPI
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.kinematic import Perturbation
from l5kit.rasterization.rasterizer_builder import build_rasterizer, _load_metadata, get_hardcoded_world_to_ecef
from l5kit.rasterization.sem_box_rasterizer import SemBoxRasterizer
from l5kit.rasterization.box_rasterizer import BoxRasterizer
from l5kit.rasterization.rasterizer import Rasterizer, EGO_EXTENT_HEIGHT, EGO_EXTENT_LENGTH, EGO_EXTENT_WIDTH
from l5kit.rasterization.semantic_rasterizer import SemanticRasterizer, elements_within_bounds, cv2_subpixel, CV2_SHIFT, \
    CV2_SHIFT_VALUE
from l5kit.rasterization.render_context import RenderContext
from l5kit.evaluation import write_pred_csv, compute_metrics_csv, read_gt_csv, write_gt_csv
from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace
from l5kit.geometry import transform_point, transform_points, rotation33_as_yaw, compute_agent_pose
from l5kit.visualization import PREDICTED_POINTS_COLOR, TARGET_POINTS_COLOR, draw_trajectory
from l5kit.sampling.slicing import get_future_slice, get_history_slice
from l5kit.sampling.agent_sampling import _create_targets_for_deep_prediction

from settings import BASE_DIR, DATA_DIR
from utils import *

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

        super(AgentDatasetCF, self).__init__(cfg, zarr_dataset, rasterizer, perturbation, agents_mask,
                                             min_frame_history, min_frame_future)

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
        frame_indices, scene_indices, track_ids, timestamps = check_load(filename,
                                                                         self.get_track_frame_scene_timestamps, None,
                                                                         save_to_file=True, args_in=None, verbose=True)

        self.frame_indices = frame_indices
        self.scene_indices = scene_indices
        self.track_ids = track_ids
        self.timestamps = timestamps

    def get_track_frame_scene_timestamps(self):

        frame_indices = [bisect.bisect_right(self.cumulative_sizes_agents, index) for index in
                         tqdm(self.agents_indices)]
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
        index_chunks = np.arange(0, n, n // n_chunks).astype(np.int)
        index_chunks[-1] = n

        item_ids = np.empty((len(self.agents_indices),), dtype=self.dataset.agents[0][str_item].dtype)

        for i in tqdm(range(1, len(index_chunks)), desc='Loading agent items'):
            agents_valid = np.argwhere(np.logical_and(self.agents_indices >= index_chunks[i - 1],
                                                      self.agents_indices < index_chunks[i])).reshape(-1, )

            _item_ids = self.dataset.agents[slice(index_chunks[i - 1], index_chunks[i])][str_item]

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
            assert str(track_id) + str(
                timestamp) in self.gt, 'self.gt (ground truth) does not contain requested track_id/timestamp combination. We have got a problem somewhere!'
            target_positions = np.array(self.gt[str(track_id) + str(timestamp)][0], dtype=np.float32)
            target_positions = transform_points(target_positions + data['centroid'][:2], data['agent_from_world'])
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


class AgentDatasetTL(AgentDatasetCF):
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

        super(AgentDatasetTL, self).__init__(raw_data_file,
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
        self.sample_function = partial(generate_agent_sample_tl_persistence,
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
        tl_img = 255 * np.ones(shape=(self.raster_size[1], self.raster_size[0], 3), dtype=np.uint8)

        # filter using half a radius from the center
        raster_radius = float(np.linalg.norm(self.raster_size * self.pixel_size)) / 2

        # get active traffic light faces
        all_active_tls = [filter_tl_faces_by_status(tl_faces, "ACTIVE") for tl_faces in history_tl_faces]
        curr_active_tl_ids = create_active_tl_dict(all_active_tls[0], all_active_tls[1:])

        # plot lanes
        lanes_lines = defaultdict(list)
        persistence_lines = defaultdict(list)

        for idx in elements_within_bounds(center_world, self.bounds_info["lanes"]["bounds"], raster_radius):

            lane = self.proto_API[self.bounds_info["lanes"]["ids"][idx]].element.lane

            # get image coords
            lane_coords = self.proto_API.get_lane_coords(self.bounds_info["lanes"]["ids"][idx])
            xy_left = cv2_subpixel(transform_points(lane_coords["xyz_left"][:, :2], raster_from_world))
            xy_right = cv2_subpixel(transform_points(lane_coords["xyz_right"][:, :2], raster_from_world))
            lanes_area = np.vstack((xy_left, np.flip(xy_right, 0)))  # start->end left then end->start right

            # Note(lberg): this called on all polygons skips some of them, don't know why
            cv2.fillPoly(img, [lanes_area], (17, 17, 31), lineType=cv2.LINE_AA, shift=CV2_SHIFT)

            # Create lane lines for the current index
            lane_type = "default"  # no traffic light face is controlling this lane
            lane_tl_ids = set([MapAPI.id_as_str(la_tc) for la_tc in lane.traffic_controls])
            for tl_id in lane_tl_ids.intersection(set(curr_active_tl_ids.keys())):
                if self.proto_API.is_traffic_face_colour(tl_id, "red"):
                    lane_type = "red"
                elif self.proto_API.is_traffic_face_colour(tl_id, "green"):
                    lane_type = "green"
                elif self.proto_API.is_traffic_face_colour(tl_id, "yellow"):
                    lane_type = "yellow"

                persistence_val = curr_active_tl_ids[tl_id]
                persistence_lines[persistence_val].extend([xy_left, xy_right])

            lanes_lines[lane_type].extend([xy_left, xy_right])

        cv2.polylines(img, lanes_lines["default"], False, (255, 217, 82), lineType=cv2.LINE_AA, shift=CV2_SHIFT)
        cv2.polylines(img, lanes_lines["green"], False, (0, 255, 0), lineType=cv2.LINE_AA, shift=CV2_SHIFT)
        cv2.polylines(img, lanes_lines["yellow"], False, (255, 255, 0), lineType=cv2.LINE_AA, shift=CV2_SHIFT)
        cv2.polylines(img, lanes_lines["red"], False, (255, 0, 0), lineType=cv2.LINE_AA, shift=CV2_SHIFT)

        # Fill in tl persistence
        for p in persistence_lines.keys():
            cv2.polylines(tl_img, persistence_lines[p], False, (0, 0, int(p)), lineType=cv2.LINE_AA, shift=CV2_SHIFT)
        # if True: plt.imshow(tl_img); plt.title(str(len(persistence_lines))); plt.show()

        # plot crosswalks
        crosswalks = []
        for idx in elements_within_bounds(center_world, self.bounds_info["crosswalks"]["bounds"], raster_radius):
            crosswalk = self.proto_API.get_crosswalk_coords(self.bounds_info["crosswalks"]["ids"][idx])

            xy_cross = cv2_subpixel(transform_points(crosswalk["xyz"][:, :2], raster_from_world))
            crosswalks.append(xy_cross)

        cv2.polylines(img, crosswalks, True, (255, 117, 69), lineType=cv2.LINE_AA, shift=CV2_SHIFT)

        tl_img = np.sum(tl_img, axis=-1)

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
        super(SemBoxTLRasterizer, self).__init__(render_context, filter_agents_threshold, history_num_frames,
                                                 semantic_map_path, world_to_ecef)

        # Change self.sat_rast reference to include traffic lights as separate channel
        self.sat_rast = SemanticTLRasterizer(render_context, semantic_map_path, world_to_ecef)


def create_active_tl_dict(curr_active_tl_ids, history_active_tl_ids):
    """
    Create a dictionary with keys active_tl_ids['face_id']
    and values the number of frames that this has persisted for
    """
    n = len(history_active_tl_ids)
    tl_dict = {face_id: n for face_id in [curr_tl['face_id'] for curr_tl in curr_active_tl_ids]}

    for curr_tl in curr_active_tl_ids:
        for i, hist_tl in enumerate(history_active_tl_ids):
            u_hist_tl = np.unique(hist_tl)
            tl_idx = np.argwhere(u_hist_tl['traffic_light_id'] == curr_tl['traffic_light_id']).reshape(-1, )
            if len(tl_idx) > 0:
                # It's possible that a traffic light has more than one face lit.
                # What do we do here? In this case we count it as a change (you could treat this differently)
                if np.any([u_hist_tl['face_id'][idx] != curr_tl['face_id'] for idx in tl_idx]):
                    tl_dict[curr_tl['face_id']] = i
                    break

    return tl_dict


def generate_agent_sample_tl_persistence(
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
    #  the history slice is ordered starting from the latest frame and goes backward in time., ex. slice(100, 91, -2)
    all_history_slice = get_history_slice(state_index, state_index, history_step_size, include_current_state=True)
    history_slice = get_history_slice(state_index, history_num_frames, history_step_size, include_current_state=True)
    future_slice = get_future_slice(state_index, future_num_frames, future_step_size)

    all_history_frames = frames[all_history_slice].copy()  # TL data will be based on all history
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

    # sync interval with the traffic light faces array
    tl_slice = get_tl_faces_slice_from_frames(all_history_frames[-1], all_history_frames[0])  # -1 is the farthest
    all_history_frames["traffic_light_faces_index_interval"] -= tl_slice.start
    history_tl_faces = filter_tl_faces_by_frames(all_history_frames, tl_faces[tl_slice].copy())

    # State you want to predict the future of.
    cur_frame = history_frames[0]
    cur_agents = history_agents[0]

    if selected_track_id is None:
        agent_centroid_m = cur_frame["ego_translation"][:2]
        agent_yaw_rad = rotation33_as_yaw(cur_frame["ego_rotation"])
        agent_extent_m = np.asarray((EGO_EXTENT_LENGTH, EGO_EXTENT_WIDTH, EGO_EXTENT_HEIGHT))
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
        agent_centroid_m = agent["centroid"]
        agent_yaw_rad = float(agent["yaw"])
        agent_extent_m = agent["extent"]
        selected_agent = agent

    input_im = (
        None
        if not rasterizer
        else rasterizer.rasterize(history_frames, history_agents, history_tl_faces, selected_agent)
    )

    world_from_agent = compute_agent_pose(agent_centroid_m, agent_yaw_rad)
    agent_from_world = np.linalg.inv(world_from_agent)
    raster_from_world = render_context.raster_from_world(agent_centroid_m, agent_yaw_rad)

    future_coords_offset, future_yaws_offset, future_availability = _create_targets_for_deep_prediction(
        future_num_frames, future_frames, selected_track_id, future_agents, agent_from_world, agent_yaw_rad
    )

    # history_num_frames + 1 because it also includes the current frame
    history_coords_offset, history_yaws_offset, history_availability = _create_targets_for_deep_prediction(
        history_num_frames + 1, history_frames, selected_track_id, history_agents, agent_from_world, agent_yaw_rad
    )

    return {
        "image": input_im,
        "target_positions": future_coords_offset,
        "target_yaws": future_yaws_offset,
        "target_availabilities": future_availability,
        "history_positions": history_coords_offset,
        "history_yaws": history_yaws_offset,
        "history_availabilities": history_availability,
        "world_to_image": raster_from_world,  # TODO deprecate
        "raster_from_agent": raster_from_world @ world_from_agent,
        "raster_from_world": raster_from_world,
        "agent_from_world": agent_from_world,
        "world_from_agent": world_from_agent,
        "centroid": agent_centroid_m,
        "yaw": agent_yaw_rad,
        "extent": agent_extent_m,
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

    return SemBoxTLRasterizer(render_context, filter_agents_threshold, history_num_frames, semantic_map_filepath,
                              world_to_ecef)




if __name__ == '__main__':
    pass