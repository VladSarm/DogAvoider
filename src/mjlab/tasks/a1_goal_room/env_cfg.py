"""Environment config for the A1 goal-room scaffold task."""

from __future__ import annotations

from copy import deepcopy

from mjlab.asset_zoo.robots import get_a1_robot_cfg
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers import TerminationTermCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, ObjRef, RayCastSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.a1_goal_room import mdp
from mjlab.tasks.a1_goal_room.constants import (
  LIDAR_MAX_RANGE,
  LIDAR_NUM_RAYS,
  LIDAR_RAYCAST_SENSOR_NAME,
  WALKING_POLICY_CHECKPOINT_PATH,
  PlanarLidarPatternCfg,
  compute_root_height_for_default_angles,
)
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import room_with_pillars
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.viewer import ViewerConfig


def make_a1_goal_room_env_cfg(
  *,
  num_envs: int = 1,
  num_pillars: int = 10,
  seed: int | None = None,
) -> ManagerBasedRlEnvCfg:
  """Create the direct-manager env used by ``my_train.py``."""

  robot_cfg = deepcopy(get_a1_robot_cfg())
  base_spec_fn = robot_cfg.spec_fn

  def _spec_without_keyframes():
    spec = base_spec_fn()
    while spec.keys:
      spec.delete(spec.keys[0])
    return spec

  robot_cfg.spec_fn = _spec_without_keyframes
  robot_cfg.init_state = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, compute_root_height_for_default_angles()),
    joint_pos={
      ".*_hip_joint": 0.0,
      ".*_thigh_joint": 0.8,
      ".*_calf_joint": -1.5,
    },
    joint_vel={".*": 0.0},
  )

  room_size = (10.0, 10.0)
  wall_thickness = 0.2
  pillar_size = (0.3, 0.3)
  pillar_height = 1.0
  goal_radius = 0.25
  goal_threshold = 0.35

  room_contact_sensor = ContactSensorCfg(
    name="room_collision",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=(
        "trunk_collision.*",
        "head_collision.*",
        "chin_collision",
        ".*_hip_collision",
        ".*_thigh_collision.*",
        ".*_calf_collision.*",
      ),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "normal"),
    reduce="none",
    num_slots=4,
  )

  lidar_sensor = RayCastSensorCfg(
    name=LIDAR_RAYCAST_SENSOR_NAME,
    frame=ObjRef(type="body", name="trunk", entity="robot"),
    ray_alignment="yaw",
    pattern=PlanarLidarPatternCfg(num_rays=LIDAR_NUM_RAYS),
    max_distance=LIDAR_MAX_RANGE,
    exclude_parent_body=True,
    include_geom_groups=(0,),
    debug_vis=False,
    viz=RayCastSensorCfg.VizCfg(
      show_rays=False,
      show_normals=False,
      hit_sphere_radius=0.12,
      hit_color=(0.1, 0.9, 0.2, 0.9),
      miss_color=(1.0, 0.2, 0.2, 0.35),
      hit_sphere_color=(0.2, 1.0, 1.0, 0.9),
    ),
  )

  scene = SceneCfg(
    num_envs=num_envs,
    env_spacing=12.0,
    terrain=TerrainEntityCfg(
      terrain_type="generator",
      terrain_generator=TerrainGeneratorCfg(
        seed=seed,
        curriculum=False,
        size=room_size,
        num_rows=1,
        num_cols=max(num_envs, 1),
        border_width=0.0,
        sub_terrains={
          "room": room_with_pillars(
            num_pillars=num_pillars,
            wall_thickness=wall_thickness,
            pillar_size=pillar_size,
            pillar_height=pillar_height,
            spawn_clearance=1.0,
          )
        },
      ),
    ),
    entities={"robot": robot_cfg},
    sensors=(room_contact_sensor, lidar_sensor),
  )

  observation_terms = {
    "lidar": ObservationTermCfg(
      func=mdp.normalized_lidar_scan,
      params={"sensor_name": LIDAR_RAYCAST_SENSOR_NAME, "max_range": LIDAR_MAX_RANGE},
    ),
    "yaw_rate": ObservationTermCfg(
      func=mdp.yaw_rate_normalized,
    ),
    "goal_heading": ObservationTermCfg(
      func=mdp.goal_heading_trig,
      params={"command_name": "goal"},
    ),
    "goal_distance": ObservationTermCfg(
      func=mdp.goal_distance_normalized,
      params={"command_name": "goal"},
    ),
    "last_action": ObservationTermCfg(func=envs_mdp.last_action),
  }

  critic_terms = {
    **observation_terms,
    "base_lin_vel_xy": ObservationTermCfg(
      func=mdp.linear_velocity_xy_normalized,
    ),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=observation_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  actions: dict[str, ActionTermCfg] = {
    "high_level_command": mdp.FrozenWalkingPolicyActionCfg(
      entity_name="robot",
      checkpoint_path=str(WALKING_POLICY_CHECKPOINT_PATH),
    )
  }

  commands: dict[str, CommandTermCfg] = {
    "goal": mdp.RandomRoomGoalCommandCfg(
      entity_name="robot",
      resampling_time_range=(1.0e6, 1.0e6),
      debug_vis=True,
      room_size=room_size,
      wall_thickness=wall_thickness,
      pillar_size=pillar_size,
      pillar_height=pillar_height,
      goal_radius=goal_radius,
      min_goal_distance_from_start=1.0,
      goal_reached_threshold=goal_threshold,
    )
  }

  events = {
    "reset_scene_to_default": EventTermCfg(func=envs_mdp.reset_scene_to_default, mode="reset"),
    "reset_base_yaw": EventTermCfg(
      func=envs_mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {"yaw": (-3.1415926535, 3.1415926535)},
        "velocity_range": {},
      },
    ),
  }

  rewards = {
    "goal_reached_reward": RewardTermCfg(
      func=mdp.rewards.reach_bonus,
      weight=200.0,
      params={"command_name": "goal", "threshold": goal_threshold},
    ),
    "collision_penalty": RewardTermCfg(
      func=mdp.rewards.obstacle_collision,
      weight=-1000.0,
      params={
        "sensor_name": room_contact_sensor.name,
        "max_normal_z_abs": 0.5,
      },
    ),
    "distance_progress_reward": RewardTermCfg(
      func=mdp.rewards.distance_progress,
      weight=100.0,
      params={"command_name": "goal"},
    ),
    "yaw_error_penalty": RewardTermCfg(
      func=mdp.rewards.yaw_error,
      weight=-3.0,
      params={"command_name": "goal"},
    ),
    # Reward higher linear speed up to the configured XY velocity limits.
    # With dt scaling at 50 Hz, weight +5.0 gives up to +0.1 per env step.
    "speed_reward": RewardTermCfg(
      func=mdp.rewards.speed_magnitude_normalized,
      weight=0.1,
    ),
    "obstacle_proximity_penalty": RewardTermCfg(
      func=mdp.rewards.obstacle_proximity_penalty,
      weight=-5.0,
      params={
        "sensor_name": LIDAR_RAYCAST_SENSOR_NAME,
        "threshold": 1.5,
        "exponential_scale": 5.0,
        "max_range": LIDAR_MAX_RANGE,
      },
    ),
    # Penalize jerky command changes. action_rate_l2 operates on raw high-level actions.
    "action_rate_penalty": RewardTermCfg(
      func=envs_mdp.action_rate_l2,
      weight=-1.0,
    ),
    # Reward manager scales by dt, so weight -5.0 gives -0.1 per env step at 50 Hz.
    "time_penalty": RewardTermCfg(
      func=mdp.rewards.constant_reward,
      weight=-5.0,
      params={"value": 1.0},
    ),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "goal_reached": TerminationTermCfg(
      func=mdp.terminations.goal_reached,
      params={"command_name": "goal", "threshold": goal_threshold},
    ),
    "room_collision": TerminationTermCfg(
      func=mdp.terminations.obstacle_collision_detected,
      params={
        "sensor_name": room_contact_sensor.name,
        "max_normal_z_abs": 0.5,
      },
    ),
    "bad_orientation": TerminationTermCfg(
      func=envs_mdp.bad_orientation,
      params={"limit_angle": 1.2},
    ),
    "root_too_low": TerminationTermCfg(
      func=envs_mdp.root_height_below_minimum,
      params={"minimum_height": 0.18},
    ),
  }

  return ManagerBasedRlEnvCfg(
    seed=seed,
    decimation=10,
    scene=scene,
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    episode_length_s=20.0,
    sim=SimulationCfg(
      mujoco=MujocoCfg(
        timestep=0.002,
        cone="elliptic",
        impratio=10,
        ccd_iterations=500,
        disableflags=("nativeccd",),
      ),
      contact_sensor_maxmatch=500,
    ),
    viewer=ViewerConfig(
      body_name="trunk",
      distance=2.2,
      elevation=-18.0,
      azimuth=160.0,
    ),
  )
