"""Tests for a1_constants.py."""

import mujoco

from mjlab.asset_zoo.robots.unitree_a1 import a1_constants
from mjlab.entity import Entity


def test_a1_entity_creation() -> None:
  entity = Entity(a1_constants.get_a1_robot_cfg())
  assert entity.num_actuators == 12
  assert entity.num_joints == 12
  assert entity.is_actuated
  assert not entity.is_fixed_base


def test_a1_model_has_expected_sites_and_sensors() -> None:
  model = Entity(a1_constants.get_a1_robot_cfg()).spec.compile()

  for site_name in ("imu", "FR", "FL", "RR", "RL"):
    assert model.site(site_name).name == site_name

  for sensor_name in ("imu_ang_vel", "imu_lin_vel", "imu_lin_acc", "root_angmom"):
    assert model.sensor(sensor_name).name == sensor_name

  assert model.nsensor == 4


def test_a1_keyframe_available() -> None:
  model = Entity(a1_constants.get_a1_robot_cfg()).spec.compile()
  key = model.key("init_state")
  assert isinstance(key.qpos[0], float)


def test_a1_home_pose_touches_ground_with_feet() -> None:
  model = Entity(a1_constants.get_a1_robot_cfg()).spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)

  for foot_name in ("FR_foot_collision", "FL_foot_collision", "RR_foot_collision", "RL_foot_collision"):
    geom_id = model.geom(foot_name).id
    foot_bottom = data.geom_xpos[geom_id][2] - model.geom_size[geom_id][0]
    assert abs(float(foot_bottom)) < 1e-6
