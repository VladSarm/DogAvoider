"""Tests for the A1 goal-room scaffold task."""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import RayCastSensor
from mjlab.tasks.a1_goal_room.constants import (
  LIDAR_NUM_RAYS,
  LIDAR_RAYCAST_SENSOR_NAME,
  LOW_LEVEL_ACTOR_OBS_DIM,
  UPPER_POLICY_ACTOR_OBS_DIM,
  UPPER_POLICY_CRITIC_OBS_DIM,
  WALKING_POLICY_CHECKPOINT_PATH,
)
from mjlab.tasks.a1_goal_room.env_cfg import make_a1_goal_room_env_cfg
from mjlab.tasks.a1_goal_room.mdp.actions import _load_walking_policy_cached
from mjlab.tasks.a1_goal_room.mdp.observations import normalized_lidar_scan


def _device() -> str:
  return "cuda:0" if torch.cuda.is_available() else "cpu"


def _make_env(
  *,
  num_envs: int = 1,
  num_pillars: int = 2,
  seed: int = 0,
) -> ManagerBasedRlEnv:
  cfg = make_a1_goal_room_env_cfg(
    num_envs=num_envs,
    num_pillars=num_pillars,
    seed=seed,
  )
  return ManagerBasedRlEnv(cfg=cfg, device=_device())


def test_checkpoint_actor_output_shape() -> None:
  policy = _load_walking_policy_cached(str(WALKING_POLICY_CHECKPOINT_PATH), _device())
  obs = torch.zeros((2, LOW_LEVEL_ACTOR_OBS_DIM), device=_device())
  with torch.no_grad():
    action = policy(obs)
  assert action.shape == (2, 12)


def test_action_term_and_observation_shapes() -> None:
  env = _make_env(num_envs=1, num_pillars=2, seed=0)
  try:
    obs, _ = env.reset(seed=0)
    action_term = env.action_manager.get_term("high_level_command")
    assert action_term.action_dim == 3
    assert tuple(obs["actor"].shape) == (1, UPPER_POLICY_ACTOR_OBS_DIM)
    assert tuple(obs["critic"].shape) == (1, UPPER_POLICY_CRITIC_OBS_DIM)
    assert torch.allclose(obs["actor"][0, -3:], torch.zeros(3, device=env.device))
  finally:
    env.close()


def test_lidar_scan_shape_and_bounds() -> None:
  env = _make_env(num_envs=1, num_pillars=3, seed=0)
  try:
    env.reset(seed=0)
    lidar_sensor = env.scene[LIDAR_RAYCAST_SENSOR_NAME]
    assert isinstance(lidar_sensor, RayCastSensor)
    assert lidar_sensor.num_rays == LIDAR_NUM_RAYS
    lidar = normalized_lidar_scan(
      env,
      sensor_name=LIDAR_RAYCAST_SENSOR_NAME,
      max_range=3.0,
    )
    assert tuple(lidar.shape) == (1, LIDAR_NUM_RAYS)
    assert torch.all(lidar >= -1.0)
    assert torch.all(lidar <= 1.0)
  finally:
    env.close()


def test_task_uses_raycast_lidar_not_builtin_rangefinders() -> None:
  env = _make_env(num_envs=1, num_pillars=2, seed=0)
  try:
    obs, _ = env.reset(seed=0)
    assert tuple(obs["actor"].shape) == (1, UPPER_POLICY_ACTOR_OBS_DIM)
    assert tuple(obs["critic"].shape) == (1, UPPER_POLICY_CRITIC_OBS_DIM)
    sensor_names = [env.sim.mj_model.sensor(i).name for i in range(env.sim.mj_model.nsensor)]
    assert "robot/imu_ang_vel" in sensor_names
    assert "robot/root_angmom" in sensor_names
    assert not any(name.startswith("robot/lidar_") for name in sensor_names)
    assert LIDAR_RAYCAST_SENSOR_NAME in env.scene.sensors
  finally:
    env.close()


def test_goal_sampling_respects_room_and_pillars() -> None:
  env = _make_env(num_envs=2, num_pillars=4, seed=0)
  try:
    env.reset(seed=0)
    goal_term = env.command_manager.get_term("goal")
    for env_id in range(env.num_envs):
      local_goal = goal_term.target_pos_w[env_id, :2] - env.scene.env_origins[env_id, :2]
      interior_half = torch.tensor(goal_term.cfg.room_size, device=env.device) * 0.5
      interior_half -= goal_term.cfg.wall_thickness + goal_term.cfg.goal_radius
      assert torch.norm(local_goal) >= goal_term.cfg.min_goal_distance_from_start
      assert torch.all(torch.abs(local_goal) <= interior_half + 1e-6)

      pillar_centers = goal_term._pillar_centers_b[env_id]
      pillar_half_extents = goal_term._pillar_half_extents[env_id]
      if pillar_centers.numel() == 0:
        continue
      delta = torch.abs(local_goal.unsqueeze(0) - pillar_centers)
      clearance = pillar_half_extents + goal_term.cfg.goal_radius
      assert not bool(torch.all(delta <= clearance, dim=-1).any())
  finally:
    env.close()


def test_goal_room_env_smoke_step() -> None:
  env = _make_env(num_envs=1, num_pillars=2, seed=0)
  try:
    obs, _ = env.reset(seed=0)
    for _ in range(3):
      action = 2.0 * torch.rand((1, 3), device=env.device) - 1.0
      obs, reward, terminated, truncated, _ = env.step(action)
      assert tuple(obs["actor"].shape) == (1, UPPER_POLICY_ACTOR_OBS_DIM)
      assert tuple(obs["critic"].shape) == (1, UPPER_POLICY_CRITIC_OBS_DIM)
      assert reward.shape == (1,)
      assert terminated.shape == (1,)
      assert truncated.shape == (1,)
  finally:
    env.close()
