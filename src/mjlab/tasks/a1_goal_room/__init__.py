"""A1 goal navigation task in a room with pillars.

This package intentionally does not register a global task in mjlab's registry yet.
Import the env config directly from :mod:`mjlab.tasks.a1_goal_room.env_cfg`.
"""

from mjlab.tasks.a1_goal_room.env_cfg import (
  make_a1_goal_room_env_cfg as make_a1_goal_room_env_cfg,
)
from mjlab.tasks.a1_goal_room.models import DEFAULT_HIDDEN_DIMS as DEFAULT_HIDDEN_DIMS
from mjlab.tasks.a1_goal_room.models import (
  UPPER_POLICY_ACTION_DIM as UPPER_POLICY_ACTION_DIM,
)
from mjlab.tasks.a1_goal_room.models import ActorNet as ActorNet
from mjlab.tasks.a1_goal_room.models import CriticNet as CriticNet
from mjlab.tasks.a1_goal_room.models import (
  UpperPolicyModelBundle as UpperPolicyModelBundle,
)
from mjlab.tasks.a1_goal_room.models import (
  build_default_upper_policy_models as build_default_upper_policy_models,
)
from mjlab.tasks.a1_goal_room.td_actor_critic import (
  OnlineTDActorCriticAgent as OnlineTDActorCriticAgent,
)
from mjlab.tasks.a1_goal_room.td_actor_critic import (
  OnlineTDActorCriticCfg as OnlineTDActorCriticCfg,
)
from mjlab.tasks.a1_goal_room.td_actor_critic import ReplayBatch as ReplayBatch
from mjlab.tasks.a1_goal_room.td_actor_critic import ReplayBuffer as ReplayBuffer
