"""Action term that runs a frozen low-level walking policy for A1."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
from torch import nn

from mjlab.entity import Entity
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.sensor import BuiltinSensor
from mjlab.tasks.a1_goal_room.constants import (
  ACTION_SCALES,
  ANG_VEL_SCALE,
  CMD_SCALE,
  DEFAULT_ANGLES,
  DOF_POS_SCALE,
  DOF_VEL_SCALE,
  LOW_LEVEL_ACTION_DIM,
  LOW_LEVEL_ACTOR_OBS_DIM,
  LOW_LEVEL_HIDDEN_DIMS,
  WALKING_POLICY_CHECKPOINT_PATH,
)


@lru_cache(maxsize=None)
def _load_walking_policy_cached(
  checkpoint_path: str,
  device: str,
) -> nn.Module:
  checkpoint = torch.load(checkpoint_path, map_location=device)
  state_dict = checkpoint
  if isinstance(checkpoint, dict):
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))

  actor = nn.Sequential(
    nn.Linear(LOW_LEVEL_ACTOR_OBS_DIM, LOW_LEVEL_HIDDEN_DIMS[0]),
    nn.ELU(),
    nn.Linear(LOW_LEVEL_HIDDEN_DIMS[0], LOW_LEVEL_HIDDEN_DIMS[1]),
    nn.ELU(),
    nn.Linear(LOW_LEVEL_HIDDEN_DIMS[1], LOW_LEVEL_HIDDEN_DIMS[2]),
    nn.ELU(),
    nn.Linear(LOW_LEVEL_HIDDEN_DIMS[2], LOW_LEVEL_ACTION_DIM),
  )

  actor_state = {
    "0.weight": state_dict["actor.0.weight"],
    "0.bias": state_dict["actor.0.bias"],
    "2.weight": state_dict["actor.2.weight"],
    "2.bias": state_dict["actor.2.bias"],
    "4.weight": state_dict["actor.4.weight"],
    "4.bias": state_dict["actor.4.bias"],
    "6.weight": state_dict["actor.6.weight"],
    "6.bias": state_dict["actor.6.bias"],
  }
  actor.load_state_dict(actor_state, strict=True)
  actor.eval()
  actor.to(device)
  return actor


@dataclass(kw_only=True)
class FrozenWalkingPolicyActionCfg(ActionTermCfg):
  """High-level 3D commands mapped to 12 joint targets via a frozen policy."""

  checkpoint_path: str | None = None

  def build(self, env) -> "FrozenWalkingPolicyAction":
    return FrozenWalkingPolicyAction(self, env)


class FrozenWalkingPolicyAction(ActionTerm):
  """Runs the frozen SuperDog locomotion policy inside an action term."""

  cfg: FrozenWalkingPolicyActionCfg

  def __init__(self, cfg: FrozenWalkingPolicyActionCfg, env):
    super().__init__(cfg=cfg, env=env)
    self._robot: Entity = env.scene[cfg.entity_name]
    joint_ids, joint_names = self._robot.find_joints_by_actuator_names((".*",))
    self._joint_ids = torch.tensor(joint_ids, device=self.device, dtype=torch.long)
    self._joint_names = tuple(joint_names)
    self._action_dim = 3

    self._raw_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)
    self._velocity_commands = torch.zeros_like(self._raw_actions)
    self._last_low_level_action = torch.zeros(
      self.num_envs, len(self._joint_ids), device=self.device
    )

    self._default_angles = torch.tensor(DEFAULT_ANGLES, device=self.device).unsqueeze(0)
    self._action_scales = torch.tensor(ACTION_SCALES, device=self.device).unsqueeze(0)
    self._command_scale = torch.tensor(CMD_SCALE, device=self.device).unsqueeze(0)
    self._joint_targets = self._default_angles.repeat(self.num_envs, 1).clone()

    checkpoint_path = Path(cfg.checkpoint_path) if cfg.checkpoint_path else WALKING_POLICY_CHECKPOINT_PATH
    if not checkpoint_path.exists():
      raise FileNotFoundError(f"Frozen walking policy checkpoint not found: {checkpoint_path}")
    self._policy = _load_walking_policy_cached(str(checkpoint_path), self.device)

    imu_sensor = env.scene["robot/imu_ang_vel"]
    assert isinstance(imu_sensor, BuiltinSensor)
    self._imu_ang_vel = imu_sensor

  @property
  def action_dim(self) -> int:
    return self._action_dim

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  @property
  def low_level_action(self) -> torch.Tensor:
    return self._last_low_level_action

  @property
  def target_joint_pos(self) -> torch.Tensor:
    return self._joint_targets

  @property
  def velocity_commands(self) -> torch.Tensor:
    return self._velocity_commands

  @property
  def joint_names(self) -> tuple[str, ...]:
    return self._joint_names

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._raw_actions[env_ids] = 0.0
    self._velocity_commands[env_ids] = 0.0
    self._last_low_level_action[env_ids] = 0.0
    self._joint_targets[env_ids] = self._default_angles

  def process_actions(self, actions: torch.Tensor) -> None:
    actions = torch.clamp(actions.to(self.device), min=-1.0, max=1.0)
    self._raw_actions[:] = actions
    self._velocity_commands[:] = actions * self._command_scale

    base_ang_vel = self._imu_ang_vel.data
    projected_gravity = self._robot.data.projected_gravity_b
    joint_pos_rel = self._robot.data.joint_pos[:, self._joint_ids] - self._default_angles
    joint_vel = self._robot.data.joint_vel[:, self._joint_ids]

    low_level_obs = torch.cat(
      [
        torch.clamp(base_ang_vel * ANG_VEL_SCALE, min=-100.0, max=100.0),
        torch.clamp(projected_gravity, min=-100.0, max=100.0),
        torch.clamp(self._velocity_commands, min=-100.0, max=100.0),
        torch.clamp(joint_pos_rel * DOF_POS_SCALE, min=-100.0, max=100.0),
        torch.clamp(joint_vel * DOF_VEL_SCALE, min=-100.0, max=100.0),
        torch.clamp(self._last_low_level_action, min=-100.0, max=100.0),
      ],
      dim=-1,
    )
    assert low_level_obs.shape[1] == LOW_LEVEL_ACTOR_OBS_DIM

    with torch.no_grad():
      low_level_action = self._policy(low_level_obs)

    self._last_low_level_action[:] = low_level_action
    self._joint_targets[:] = self._default_angles + low_level_action * self._action_scales
    soft_limits = self._robot.data.soft_joint_pos_limits[:, self._joint_ids]
    self._joint_targets.clamp_(soft_limits[..., 0], soft_limits[..., 1])

  def apply_actions(self) -> None:
    self._robot.set_joint_position_target(self._joint_targets, joint_ids=self._joint_ids)
