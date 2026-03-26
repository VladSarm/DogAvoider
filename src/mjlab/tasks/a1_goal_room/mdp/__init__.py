"""MDP terms for the A1 goal-room task."""

from mjlab.tasks.a1_goal_room.mdp import rewards as rewards
from mjlab.tasks.a1_goal_room.mdp import terminations as terminations
from mjlab.tasks.a1_goal_room.mdp.actions import (
  FrozenWalkingPolicyAction as FrozenWalkingPolicyAction,
)
from mjlab.tasks.a1_goal_room.mdp.actions import (
  FrozenWalkingPolicyActionCfg as FrozenWalkingPolicyActionCfg,
)
from mjlab.tasks.a1_goal_room.mdp.commands import (
  RandomRoomGoalCommand as RandomRoomGoalCommand,
)
from mjlab.tasks.a1_goal_room.mdp.commands import (
  RandomRoomGoalCommandCfg as RandomRoomGoalCommandCfg,
)
from mjlab.tasks.a1_goal_room.mdp.observations import (
  a1_lidar_scan as a1_lidar_scan,
)
from mjlab.tasks.a1_goal_room.mdp.observations import (
  goal_distance_normalized as goal_distance_normalized,
)
from mjlab.tasks.a1_goal_room.mdp.observations import (
  goal_heading_trig as goal_heading_trig,
)
from mjlab.tasks.a1_goal_room.mdp.observations import (
  goal_vector_b as goal_vector_b,
)
from mjlab.tasks.a1_goal_room.mdp.observations import (
  linear_velocity_xy_normalized as linear_velocity_xy_normalized,
)
from mjlab.tasks.a1_goal_room.mdp.observations import (
  normalized_lidar_scan as normalized_lidar_scan,
)
from mjlab.tasks.a1_goal_room.mdp.observations import (
  yaw_rate_normalized as yaw_rate_normalized,
)
