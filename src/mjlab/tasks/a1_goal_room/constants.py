"""Constants and helpers shared by the A1 goal-room task."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import mujoco
import numpy as np
import torch
import yaml

from mjlab import MJLAB_SRC_PATH
from mjlab.asset_zoo.robots.unitree_a1.a1_constants import get_spec

A1_TASK_DATA_ROOT = MJLAB_SRC_PATH / "tasks" / "a1_goal_room" / "data"
A1_WALKING_POLICY_CONFIG_PATH = A1_TASK_DATA_ROOT / "a1_superdog.yaml"


def _load_superdog_a1_config() -> dict[str, Any]:
  with A1_WALKING_POLICY_CONFIG_PATH.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
  assert isinstance(cfg, dict)
  return cfg


SUPERDOG_A1_CFG = _load_superdog_a1_config()
WALKING_POLICY_CHECKPOINT_PATH = A1_TASK_DATA_ROOT / str(
  SUPERDOG_A1_CFG["walking_policy_path"]
)

LOW_LEVEL_ACTOR_OBS_DIM = 45
LOW_LEVEL_ACTION_DIM = 12
LOW_LEVEL_HIDDEN_DIMS = (512, 256, 128)
LOW_LEVEL_ACTIVATION = "elu"

DEFAULT_ANGLES = tuple(float(x) for x in SUPERDOG_A1_CFG["default_angles"])
ACTION_SCALES = tuple(float(x) for x in SUPERDOG_A1_CFG["action_scales"])
CMD_SCALE = tuple(float(x) for x in SUPERDOG_A1_CFG["cmd_scale"])
ANG_VEL_SCALE = float(SUPERDOG_A1_CFG["ang_vel_scale"])
DOF_POS_SCALE = float(SUPERDOG_A1_CFG["dof_pos_scale"])
DOF_VEL_SCALE = float(SUPERDOG_A1_CFG["dof_vel_scale"])

LIDAR_RAYCAST_SENSOR_NAME = "a1_lidar"
LIDAR_NUM_RAYS = 40
LIDAR_MAX_RANGE = 3.0
LIDAR_ORIGIN_OFFSET = (0.15, 0.0, 0.12)
UPPER_POLICY_ACTOR_OBS_DIM = 47
UPPER_POLICY_CRITIC_OBS_DIM = 49
UPPER_POLICY_MAX_DISTANCE = 10.0
UPPER_POLICY_MAX_ANGULAR_VEL = 3.0
UPPER_POLICY_MAX_VX = 2.0
UPPER_POLICY_MAX_VY = 1.0
FOOT_GEOM_NAMES = (
  "FR_foot_collision",
  "FL_foot_collision",
  "RR_foot_collision",
  "RL_foot_collision",
)


@dataclass
class PlanarLidarPatternCfg:
  """Planar 2D lidar pattern with a shared origin and evenly spaced rays."""

  num_rays: int = LIDAR_NUM_RAYS
  origin_offset: tuple[float, float, float] = LIDAR_ORIGIN_OFFSET
  start_angle: float = 0.0
  end_angle: float = 2.0 * math.pi

  def generate_rays(
    self,
    mj_model: mujoco.MjModel | None,
    device: str,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    del mj_model
    if self.num_rays <= 0:
      raise ValueError(f"num_rays must be positive, got {self.num_rays}")

    angles = torch.linspace(
      self.start_angle,
      self.end_angle,
      self.num_rays + 1,
      device=device,
      dtype=torch.float32,
    )[:-1]
    local_offsets = torch.tensor(
      self.origin_offset,
      device=device,
      dtype=torch.float32,
    ).repeat(self.num_rays, 1)
    local_directions = torch.stack(
      [torch.cos(angles), torch.sin(angles), torch.zeros_like(angles)],
      dim=-1,
    )
    return local_offsets, local_directions


@lru_cache(maxsize=1)
def compute_root_height_for_default_angles() -> float:
  """Compute the floating-base height that places the feet on the ground."""
  spec = get_spec()
  model = spec.compile()
  data = mujoco.MjData(model)

  qpos = np.zeros(model.nq, dtype=np.float64)
  qpos[3] = 1.0
  qpos[7:] = np.array(DEFAULT_ANGLES, dtype=np.float64)
  data.qpos[:] = qpos
  mujoco.mj_forward(model, data)

  min_bottom = float("inf")
  for geom_name in FOOT_GEOM_NAMES:
    geom_id = model.geom(geom_name).id
    bottom = float(data.geom_xpos[geom_id][2] - model.geom_size[geom_id][0])
    min_bottom = min(min_bottom, bottom)
  return -min_bottom
