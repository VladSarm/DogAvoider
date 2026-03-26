"""Reward terms for the A1 goal-room task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.a1_goal_room.constants import (
  LIDAR_MAX_RANGE,
  UPPER_POLICY_MAX_VX,
  UPPER_POLICY_MAX_VY,
)
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def goal_distance(
  env: "ManagerBasedRlEnv",
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  target_pos_w = env.command_manager.get_command(command_name)
  return torch.norm(target_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2], dim=-1)


class distance_progress:
  """Reward progress towards the goal instead of absolute distance.

  The raw term returns ``(prev_distance - current_distance) / dt`` so that
  reward-manager scaling by ``dt`` turns the final contribution into:

    weight * (prev_distance - current_distance)

  This keeps the action-dependent signal from shrinking with the environment step size.
  """

  def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv"):
    del cfg
    self._prev_distance = torch.full((env.num_envs,), torch.nan, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._prev_distance[env_ids] = torch.nan

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    current_distance = goal_distance(env, command_name=command_name, asset_cfg=asset_cfg)
    is_first = torch.isnan(self._prev_distance)
    progress = torch.where(
      is_first,
      torch.zeros_like(current_distance),
      self._prev_distance - current_distance,
    )
    self._prev_distance[:] = current_distance
    return progress / env.step_dt


def reach_bonus(
  env: "ManagerBasedRlEnv",
  command_name: str,
  threshold: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  return (goal_distance(env, command_name=command_name, asset_cfg=asset_cfg) <= threshold).float()


def yaw_error(
  env: "ManagerBasedRlEnv",
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  target_pos_w = env.command_manager.get_command(command_name)
  goal_vec_w = target_pos_w - robot.data.root_link_pos_w
  goal_vec_b = quat_apply_inverse(robot.data.root_link_quat_w, goal_vec_w)
  return torch.abs(torch.atan2(goal_vec_b[:, 1], goal_vec_b[:, 0]))


def obstacle_collision(
  env: "ManagerBasedRlEnv",
  sensor_name: str,
  max_normal_z_abs: float = 0.5,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  assert sensor.data.normal is not None

  has_contact = sensor.data.found > 0
  is_side_contact = torch.abs(sensor.data.normal[..., 2]) < max_normal_z_abs
  return torch.any(has_contact & is_side_contact, dim=-1).float()


def constant_reward(env: "ManagerBasedRlEnv", value: float = 1.0) -> torch.Tensor:
  return torch.full((env.num_envs,), value, device=env.device)


def speed_magnitude_normalized(
  env: "ManagerBasedRlEnv",
  max_vx: float = UPPER_POLICY_MAX_VX,
  max_vy: float = UPPER_POLICY_MAX_VY,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward higher base linear speeds in XY, clipped by configured limits."""
  robot: Entity = env.scene[asset_cfg.name]
  lin_vel_xy = robot.data.root_link_lin_vel_b[:, :2]

  speed_x = torch.clamp(torch.abs(lin_vel_xy[:, 0]), 0.0, max_vx) / max_vx
  speed_y = torch.clamp(torch.abs(lin_vel_xy[:, 1]), 0.0, max_vy) / max_vy
  return 0.5 * (speed_x + speed_y)


def obstacle_proximity_penalty(
  env: "ManagerBasedRlEnv",
  sensor_name: str,
  *,
  threshold: float = 1.5,
  exponential_scale: float = 5.0,
  max_range: float = LIDAR_MAX_RANGE,
) -> torch.Tensor:
  """Penalize lidar rays that get close to obstacles.

  Each ray contributes a normalized exponential penalty in ``[0, 1]``:
  - 0 when the hit distance is at or beyond ``threshold``
  - 1 when the hit distance approaches zero

  The final term is the mean penalty over all rays so every ray contributes,
  while the scale stays stable when the number of rays changes.
  """

  scan = env.scene[sensor_name].data.distances
  invalid = ~torch.isfinite(scan) | (scan < 0.0)
  scan = torch.where(invalid, torch.full_like(scan, max_range), scan)
  scan = scan.clamp(0.0, max_range)

  close_mask = scan < threshold
  denom = torch.expm1(
    torch.tensor(exponential_scale * threshold, device=env.device, dtype=scan.dtype)
  )
  penalty = torch.expm1(exponential_scale * (threshold - scan.clamp(max=threshold))) / denom
  penalty = torch.where(close_mask, penalty, torch.zeros_like(penalty))
  return penalty.mean(dim=-1)
