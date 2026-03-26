"""Termination helpers for the A1 goal-room task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def goal_reached(
  env: "ManagerBasedRlEnv",
  command_name: str,
  threshold: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  target_pos_w = env.command_manager.get_command(command_name)
  distance = torch.norm(target_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2], dim=-1)
  return distance <= threshold


def collision_detected(
  env: "ManagerBasedRlEnv",
  sensor_name: str,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return sensor.data.found.sum(dim=-1) > 0


def obstacle_collision_detected(
  env: "ManagerBasedRlEnv",
  sensor_name: str,
  max_normal_z_abs: float = 0.5,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  assert sensor.data.normal is not None

  has_contact = sensor.data.found > 0
  is_side_contact = torch.abs(sensor.data.normal[..., 2]) < max_normal_z_abs
  return torch.any(has_contact & is_side_contact, dim=-1)
