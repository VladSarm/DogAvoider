"""Unitree A1 constants."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import XmlPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets

##
# MJCF and assets.
##

A1_XML: Path = MJLAB_SRC_PATH / "asset_zoo" / "robots" / "unitree_a1" / "xmls" / "a1.xml"
assert A1_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, A1_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(A1_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Keyframe config.
##

INIT_STATE = EntityCfg.InitialStateCfg(
  joint_pos=None,
  joint_vel={".*": 0.0},
)


##
# Final config.
##

A1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(XmlPositionActuatorCfg(target_names_expr=(".*",)),),
  soft_joint_pos_limit_factor=0.9,
)


def get_a1_robot_cfg() -> EntityCfg:
  """Get a fresh A1 robot configuration instance."""
  return EntityCfg(
    init_state=INIT_STATE,
    spec_fn=get_spec,
    articulation=A1_ARTICULATION,
  )


A1_ACTION_SCALE: dict[str, float] = {
  "FR_hip_joint": 0.125,
  "FR_thigh_joint": 0.25,
  "FR_calf_joint": 0.25,
  "FL_hip_joint": 0.125,
  "FL_thigh_joint": 0.25,
  "FL_calf_joint": 0.25,
  "RR_hip_joint": 0.125,
  "RR_thigh_joint": 0.25,
  "RR_calf_joint": 0.25,
  "RL_hip_joint": 0.125,
  "RL_thigh_joint": 0.25,
  "RL_calf_joint": 0.25,
}


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_a1_robot_cfg())
  viewer.launch(robot.spec.compile())
