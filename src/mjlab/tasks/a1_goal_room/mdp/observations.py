"""Observation helpers for the A1 goal-room task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import RayCastSensor
from mjlab.tasks.a1_goal_room.constants import (
  UPPER_POLICY_MAX_ANGULAR_VEL,
  UPPER_POLICY_MAX_DISTANCE,
  UPPER_POLICY_MAX_VX,
  UPPER_POLICY_MAX_VY,
)
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def goal_vector_b(
  env: "ManagerBasedRlEnv",
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  target_pos_w = env.command_manager.get_command(command_name)
  goal_vec_w = target_pos_w - robot.data.root_link_pos_w
  return quat_apply_inverse(robot.data.root_link_quat_w, goal_vec_w)


def a1_lidar_scan(
  env: "ManagerBasedRlEnv",
  sensor_name: str,
  max_range: float = 3.0,
) -> torch.Tensor:
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, RayCastSensor)
  scan = sensor.data.distances
  invalid = ~torch.isfinite(scan) | (scan < 0.0)
  scan = torch.where(invalid, torch.full_like(scan, max_range), scan)
  return scan.clamp_(0.0, max_range)


def normalized_lidar_scan(
  env: "ManagerBasedRlEnv",
  sensor_name: str,
  max_range: float,
) -> torch.Tensor:
  scan = a1_lidar_scan(env, sensor_name=sensor_name, max_range=max_range)
  return (scan / max_range) * 2.0 - 1.0


def goal_heading_trig(
  env: "ManagerBasedRlEnv",
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  goal_vec_b = goal_vector_b(env, command_name=command_name, asset_cfg=asset_cfg)
  heading = torch.atan2(goal_vec_b[:, 1], goal_vec_b[:, 0])
  return torch.stack((torch.sin(heading), torch.cos(heading)), dim=-1)


def goal_distance_normalized(
  env: "ManagerBasedRlEnv",
  command_name: str,
  max_distance: float = UPPER_POLICY_MAX_DISTANCE,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  goal_vec_b = goal_vector_b(env, command_name=command_name, asset_cfg=asset_cfg)
  distance = torch.norm(goal_vec_b[:, :2], dim=-1, keepdim=True)
  return (distance.clamp_(0.0, max_distance) / max_distance) * 2.0 - 1.0


def yaw_rate_normalized(
  env: "ManagerBasedRlEnv",
  max_angular_vel: float = UPPER_POLICY_MAX_ANGULAR_VEL,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  yaw_rate = robot.data.root_link_ang_vel_b[:, 2:3]
  return torch.clamp(yaw_rate, -max_angular_vel, max_angular_vel) / max_angular_vel


def linear_velocity_xy_normalized(
  env: "ManagerBasedRlEnv",
  max_vx: float = UPPER_POLICY_MAX_VX,
  max_vy: float = UPPER_POLICY_MAX_VY,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  lin_vel_xy = robot.data.root_link_lin_vel_b[:, :2]
  out = lin_vel_xy.clone()
  out[:, 0] = torch.clamp(out[:, 0], -max_vx, max_vx) / max_vx
  out[:, 1] = torch.clamp(out[:, 1], -max_vy, max_vy) / max_vy
  return out
