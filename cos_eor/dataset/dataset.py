from typing import Optional, List, Dict

from collections import OrderedDict, defaultdict
from copy import deepcopy
from pathlib import Path
import json
import os

import attr
import numpy as np
import pandas as pd

from habitat.core.dataset import Episode
from habitat.core.utils import DatasetFloatJSONEncoder, not_none_validator
from habitat.datasets.pointnav.pointnav_dataset import (
    CONTENT_SCENES_PATH_FIELD,
    DEFAULT_SCENE_PATH_PREFIX,
    PointNavDatasetV1,
)
from habitat.tasks.nav.nav import NavigationEpisode
from habitat.core.registry import registry
from habitat.config.default import Config
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower


@attr.s(auto_attribs=True, kw_only=True)
class CosRearrangementSpec:
    r"""Specifications that capture a particular position of final position
    or initial position of the object.
    """
    bbox: List[List[float]] = attr.ib(default=None)
    position: List[float] = attr.ib(default=None, validator=not_none_validator)
    rotation: List[float] = attr.ib(default=None, validator=not_none_validator)
    info: Optional[Dict[str, str]] = attr.ib(default=None)
    scale: List[float] = attr.ib(
        default=[1.0, 1.0, 1.0], validator=not_none_validator
    )
    file: str = attr.ib(default=None, validator=not_none_validator)
    key: str = attr.ib(default=None, validator=not_none_validator)
    type: str = attr.ib(default=None, validator=not_none_validator)
    or_type: str = attr.ib(default=None, validator=not_none_validator)
    receptacles: str = attr.ib(default=None)
    objects: List = attr.ib(default=None)
    sidenote: List = attr.ib(default=None)
    pickable: bool = attr.ib(default=False, validator=not_none_validator)
    object_id: str = attr.ib(default=None, validator=not_none_validator)

    receptacles_rank: List[int] = attr.ib(default=None)
    paired_recs_positions: List = attr.ib(default=None)
    paired_recs_ranks: List = attr.ib(default=None)
    paired_recs_keys: List = attr.ib(default=None)


@attr.s(auto_attribs=True, kw_only=True)
class CosRearrangementEpisode(Episode):
    r"""Specification of episode that includes initial position and rotation
    of agent, all goal specifications, all object specifications

    Args:
        episode_id: id of episode in the dataset
        scene_id: id of scene inside the simulator.
        start_position: numpy ndarray containing 3 entries for (x, y, z).
        start_rotation: numpy ndarray with 4 entries for (x, y, z, w)
            elements of unit quaternion (versor) representing agent 3D
            orientation.
        goal: object's goal position and rotation
        object: object's start specification defined with object type,
            position, and rotation.

    """
    default_matrix_shape: List[int] = attr.ib(default=None, validator=not_none_validator)
    start_matrix: List[int] = attr.ib(default=None, validator=not_none_validator)
    end_matrix: List[int] = attr.ib(default=None, validator=not_none_validator)
    recs_keys: List[str] = attr.ib(default=None, validator=not_none_validator)
    objs_keys: List[str] = attr.ib(default=None, validator=not_none_validator)
    recs_cats: List[str] = attr.ib(default=None, validator=not_none_validator)
    objs_cats: List[str] = attr.ib(default=None, validator=not_none_validator)
    oracle_steps_solve: List[int] = attr.ib(default=None, validator=not_none_validator)
    objs_files: List[str] = attr.ib(default=None, validator=not_none_validator)
    objs_pos: List = attr.ib(default=None, validator=not_none_validator)
    objs_rot: List = attr.ib(default=None, validator=not_none_validator)
    recs_pos: List = attr.ib(default=None, validator=not_none_validator)
    recs_rot: List = attr.ib(default=None, validator=not_none_validator)
    misplaced_count: int = attr.ib(default=None, validator=not_none_validator)
    objects_count: int = attr.ib(default=None, validator=not_none_validator)
    nav_recs_pos: List = attr.ib(default=None, validator=not_none_validator)
    recs_packers: List = attr.ib(default=None, validator=not_none_validator)
    oracle_paths: List = attr.ib(default=None)

    def reset(self):
        # reset mapping
        self.state_matrix = deepcopy(self.start_matrix)

        if "agent" not in self.recs_keys:
            # add agent as a receptacle for this mapping
            for mn in ["start_matrix", "end_matrix", "state_matrix"]:
                m = np.array(getattr(self, mn))
                col = np.zeros(shape=(1, m.shape[1]))
                m = np.concatenate([m, col], axis=0)
                setattr(self, mn, m)
            self.recs_rot.append([0.0] * 4)
            self.recs_pos.append([0.0] * 3)
            self.recs_keys.append("agent")

        # this should not happen
        assert not self.agent_has_object()

    def _init_episode_cache(self, cache):
        if cache == {}:
            new_cache = {
                "pos": {},
                "geo_dists": defaultdict(dict),
                "l2_dists": defaultdict(dict),
            }
            cache.update(new_cache)

    def get_object_goal_distance(self, task, episode_cache):
        """Returns all object-goal distances in a nice dictionary!"""
        is_cache_empty = episode_cache == {}
        self._init_episode_cache(episode_cache)
        obj_rec_mapping = self.get_correct_mapping()
        objs = list(obj_rec_mapping.keys())
        current_mapping = self.get_mapping("current")
        start_mapping = self.get_mapping("start")
        return_dict = {}
        for ok in objs:
            oid = task.obj_key_to_sim_obj_id[ok]

            gripped_object_id = task._sim.gripped_object_id
            if is_cache_empty or oid == gripped_object_id:
                obj_pos = task.get_translation(oid)

            obj_moved = False
            if not is_cache_empty and oid == gripped_object_id:
                obj_prev_pos = episode_cache["pos"][ok]
                obj_moved = not np.allclose(obj_pos, obj_prev_pos)

            geo_dists, l2_dists = [], []
            if is_cache_empty or obj_moved:
                for rk in obj_rec_mapping[ok]:
                    rid = task.obj_key_to_sim_obj_id[rk]
                    geo_dist = task.get_dist(oid, rid, "l2")
                    l2_dist = task.get_dist(oid, rid, "l2")

                    if l2_dist > geo_dist:
                        import pdb
                        pdb.set_trace()

                    geo_dists.append(abs(geo_dist))
                    l2_dists.append(abs(l2_dist))
                episode_cache["geo_dists"][ok] = geo_dists
                episode_cache["l2_dists"][ok] = l2_dists
                episode_cache["pos"][ok] = obj_pos
            else:
                geo_dists = episode_cache["geo_dists"][ok]
                l2_dists = episode_cache["l2_dists"][ok]

            return_dict[ok] = {
                "recs_keys": obj_rec_mapping[ok],
                "geo_dists": geo_dists,
                "l2_dists": l2_dists,
                "current_rec": current_mapping[ok],
                "start_rec": start_mapping[ok],
            }
        return return_dict

    def get_current_object_positions(self, task):
        objs_pos = []
        for ok in self.objs_keys:
            oid = task.obj_key_to_sim_obj_id[ok]
            objs_pos.append(task.get_translation(oid))
        return objs_pos

    def get_misplaced_objects(self, state_type="current"):
        state_mapping = self.get_mapping(state_type)
        correct_mapping = self.get_correct_mapping()
        misplaced_obj_keys = []
        for obj_key in self.objs_keys:
            if state_mapping[obj_key] not in correct_mapping[obj_key]:
                misplaced_obj_keys.append(obj_key)
        if state_type == "start":
            assert len(misplaced_obj_keys) == self.misplaced_count
        return misplaced_obj_keys

    def get_agent_object_distance(self, task, types=None):
        if types is None:
            types = ["l2", "geo"]
        agent_obj_mapping = {}
        for ok in self.objs_keys:
            agent_obj_mapping[ok] = {}
            oid = task.obj_key_to_sim_obj_id[ok]
            if "geo" in types:
                geo_dist = task._sim.get_or_dist(oid, "geo")
                agent_obj_mapping[ok]["geo_dist"] = geo_dist

            if "l2" in types:
                l2_dist = task._sim.get_or_dist(oid, "l2")
                agent_obj_mapping[ok]["l2_dist"] = l2_dist

        return agent_obj_mapping

    def get_agent_receptacle_distance(self, task, types=None):
        if types is None:
            types = ["l2", "geo"]
        agent_rec_mapping = {}
        for ok in self.recs_keys:
            agent_rec_mapping[ok] = {}
            if ok in ["floor", "agent"]:
                agent_rec_mapping[ok] = {
                    "geo_dist": -1,
                    "l2_dist": -1
                }
                continue
            oid = task.obj_key_to_sim_obj_id[ok]
            if "geo" in types:
                geo_dist = task._sim.get_or_dist(oid, "geo")
                agent_rec_mapping[ok]["geo_dist"] = geo_dist
            if "l2" in types:
                l2_dist = task._sim.get_or_dist(oid, "l2")
                agent_rec_mapping[ok]["l2_dist"] = l2_dist

        return agent_rec_mapping

    def get_mapping(self, state_type="current", task=None):
        if state_type == "current":
            matrix = self.state_matrix
        elif state_type == "start":
            matrix = self.start_matrix
        else:
            raise AssertionError
        obj_rec_mapping = OrderedDict()
        for obj_key in self.objs_keys:
            if task is not None:
                # return object-ids
                obj_rec_mapping[task.obj_key_to_sim_obj_id[obj_key]] = task.obj_key_to_sim_obj_id[self.get_rec(obj_key, matrix)]
            else:
                # return object-keys
                obj_rec_mapping[obj_key] = self.get_rec(obj_key, matrix)
        return obj_rec_mapping

    def get_correct_recs(self, obj_key):
        """get all possible final recs"""
        obj_ind = self.objs_keys.index(obj_key)
        rec_inds = np.argwhere(self.end_matrix[:, obj_ind] == 1).squeeze(-1)
        assert len(rec_inds.shape) == 1
        rec_keys = [self.recs_keys[rind] for rind in rec_inds]
        return rec_keys

    def get_rec(self, obj_key, state_matrix):
        """either start or current state"""
        obj_ind = self.objs_keys.index(obj_key)
        rec_inds = np.argwhere(state_matrix[:, obj_ind] == 1).squeeze(-1)
        assert len(rec_inds) == 1
        rec_key = self.recs_keys[rec_inds[0]]
        return rec_key

    def get_correct_mapping(self):
        obj_rec_mapping = {}
        for obj_key in self.objs_keys:
            mapped_rec_keys = self.get_correct_recs(obj_key)
            obj_rec_mapping[obj_key] = mapped_rec_keys
        return obj_rec_mapping

    def get_objects_ids(self, task):
        return [task.obj_key_to_sim_obj_id[key] for key in self.objs_keys]

    def get_objects_on_rec(self, task, rec_id):
        """Get all objects currently placed on the given receptacle"""
        rec_key = task.sim_obj_id_to_obj_key[rec_id]
        rec_idx = self.recs_keys.index(rec_key)
        obj_idxs = np.argwhere(self.state_matrix[rec_idx]).squeeze(axis=-1)
        obj_keys = [self.objs_keys[obj_idx] for obj_idx in obj_idxs]
        obj_ids = [task.obj_key_to_sim_obj_id[obj_key] for obj_key in obj_keys]
        return obj_ids

    def update_mapping(self, obj_id, update_type, task, rec_id=-1):
        """Update the state matrix with changes in environment"""
        obj_key = task.sim_obj_id_to_obj_key[obj_id]
        obj_ind = self.objs_keys.index(obj_key)
        agent_ind = self.recs_keys.index("agent")
        if update_type == "place":
            if rec_id == -1:
                rec_key = "floor"
            else:
                rec_key = task.sim_obj_id_to_obj_key[rec_id]
            rec_ind = self.recs_keys.index(rec_key)
            # assert object is held by agent
            assert self.state_matrix[agent_ind][obj_ind] == 1
            # remove object from agent
            self.state_matrix[agent_ind][obj_ind] = 0
            # put object on receptacle
            self.state_matrix[rec_ind][obj_ind] = 1
        elif update_type == "pick":
            assert rec_id == -1
            rec_key = self.get_rec(obj_key, self.state_matrix)
            rec_ind = self.recs_keys.index(rec_key)
            # assert object is placed on receptacle
            assert self.state_matrix[rec_ind][obj_ind] == 1
            # remove object from receptacle
            self.state_matrix[rec_ind][obj_ind] = 0
            # put object on agent
            self.state_matrix[agent_ind][obj_ind] = 1
        else:
            raise AssertionError

        assert not(self.agent_has_object() and update_type == "place")

    def agent_has_object(self):
        agent_ind = self.recs_keys.index("agent")
        has_object = self.state_matrix[agent_ind].sum() > 0
        return has_object

def load_replay_data(filter_path):
    if len(filter_path) != 0 and os.path.exists(filter_path):
        return pd.read_pickle(filter_path)
    return None


@registry.register_dataset(name="CosRearrangementDataset-v0")
class CosRearrangementDatasetV0(PointNavDatasetV1):
    r"""Class inherited from PointNavDataset that loads the Rearrangement dataset."""
    episodes: List[CosRearrangementEpisode]
    object_templates: Dict = attr.ib(default={}, validator=not_none_validator)
    scale_rot_sid_path: str

    def to_json(self) -> str:
        result = DatasetFloatJSONEncoder().encode(self)
        return result

    def __init__(self, config: Optional[Config] = None) -> None:
        super().__init__(config)

        if config is None:
            return

        checkpoint_file = config.CHECKPOINT_FILE
        if checkpoint_file is not None and os.path.isfile(checkpoint_file):
            df = pd.read_csv(checkpoint_file)
            df[["scene_id", "episode_id"]] = df["episode_id"].str.rsplit(pat="_", n=1, expand=True)

            done_episodes = defaultdict(set)
            for scene_id, episode_id in zip(df["scene_id"], df["episode_id"]):
                done_episodes[scene_id].add(episode_id)

            new_episodes = []
            for episode in self.episodes:
                scene_id = Path(episode.scene_id).stem
                if episode.episode_id not in done_episodes[scene_id]:
                    new_episodes.append(episode)

            self.episodes = new_episodes

    def from_json(
            self, json_str: str, scenes_dir: Optional[str] = None, filter_scenes_path=""
    ) -> None:
        deserialized = json.loads(json_str)
        self.replay_df = load_replay_data(filter_scenes_path)

        if CONTENT_SCENES_PATH_FIELD in deserialized:
            self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]

        if registry.mapping["debug"]:
            deserialized["episodes"] = deserialized["episodes"][:1]

        for i, episode in enumerate(deserialized["episodes"]):
            episode['episode_id'] = str(episode['episode_id'])

            if self.replay_df is not None and not self.replay_df[
                (self.replay_df['episode_id'] == episode['episode_id']) &
                (self.replay_df['scene_id'] == episode['scene_id'])
            ].empty:
                continue
            episode = CosRearrangementEpisode(**episode)
            episode.reset()
            self.episodes.append(episode)

        print(f"Episode Length: {len(self.episodes)}")
