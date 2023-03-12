from typing import Dict, List, Any

import argparse
from collections import defaultdict
import os
from pathlib import Path
import random
import sys

import numpy as np
import numba
import torch
import tqdm

cwd = os.getcwd()
pwd = os.path.dirname(cwd)
ppwd = os.path.dirname(pwd)

for dir in [cwd, pwd, ppwd]:
    sys.path.insert(1, dir)

from habitat.core.registry import registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.config.default import get_config
from habitat_baselines.utils.common import batch_obs
from habitat_baselines.utils.env_utils import construct_envs

from cos_eor.policy.rank import RankModule
from cos_eor.policy.nav import NavModule
from cos_eor.policy.oracle_rank import OracleRankModule
from cos_eor.policy.explore import ExploreModule
from cos_eor.policy.hie_policy import HiePolicy
from cos_eor.env.env import CosRearrangementRLEnv
from cos_eor.task.sensors import *
from cos_eor.task.measures import *

from PIL import Image
import matplotlib.pyplot as plt
def display_sample(rgb_obs, semantic_obs=np.array([]), depth_obs=np.array([])):
    from habitat_sim.utils.common import d3_40_colors_rgb

    rgb_img = Image.fromarray(rgb_obs, mode="RGB")

    arr = [rgb_img]
    titles = ["rgb"]
    if semantic_obs.size != 0:
        semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")
        arr.append(semantic_img)
        titles.append("semantic")

    if depth_obs.size != 0:
        depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")
        arr.append(depth_img)
        titles.append("depth")

    plt.figure(figsize=(12, 8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i + 1)
        ax.axis("off")
        ax.set_title(titles[i])
        plt.imshow(data)
    plt.savefig('plots.png')
    # plt.show(block=False)


dir_path = "./"
output_directory = "logs/baseline_1"
output_path = os.path.join(dir_path, output_directory)
if not os.path.exists(output_path):
    os.mkdir(output_path)

# TODO: Change for all!
config_yaml = './logs/baseline_1/configs/ihlen_0_int.yaml'
tag='ihlen_0_int'

config = get_config(config_yaml)
print(config.TASK_CONFIG.SEED, config.NUM_PROCESSES)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(config.TASK_CONFIG.SEED * config.NUM_PROCESSES)
np.random.seed(config.TASK_CONFIG.SEED * config.NUM_PROCESSES)
torch.manual_seed(config.TASK_CONFIG.SEED * config.NUM_PROCESSES)

out_dir = output_directory

config.defrost()

navmesh_file = Path(out_dir)/config.TASK_CONFIG.SIMULATOR.NAVMESH
navmesh_file.parent.mkdir(parents=True, exist_ok=True)
config.TASK_CONFIG.SIMULATOR.NAVMESH = str(navmesh_file)

config.TASK_CONFIG.DATASET.CHECKPOINT_FILE = None
config.freeze()
envs = construct_envs(
    config, get_env_class(config.ENV_NAME)
)
print("Environments Constructed!")
observations = envs.reset()
print("Observations Keys: ", observations[0].keys())

display_sample(observations[0]['rgb'], observations[0]['semantic'], observations[0]['depth'].squeeze())

# print("# Visible Objects: ", observations[0]['num_visible_objs'])
# print("# Visible Receptables: ", observations[0]['num_visible_recs'])

# print(observations[0]['cos_eor'].keys())
# print("Agent Position")
# print(observations[0]['cos_eor']['agent_pos'])
# print("Receptacle Positions")
# print(len(observations[0]['cos_eor']['recs_pos']))
# print("Object Positions")
# print(len(observations[0]['cos_eor']['objs_pos']))
# print("Instance ID count")
# print(observations[0]['cos_eor']['instance_id_count'])
# print("SID Class Map")
# print(observations[0]['cos_eor']['sid_class_map'])
# print("Current Mapping")
# print(observations[0]['cos_eor']['current_mapping'])
# print("Correct Mapping")
# print(observations[0]['cos_eor']['correct_mapping'])
# print(envs.action_spaces)