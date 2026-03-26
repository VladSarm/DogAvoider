"""Tests for default upper-level actor/critic architectures."""

from __future__ import annotations

import torch

from mjlab.tasks.a1_goal_room.constants import (
  UPPER_POLICY_ACTOR_OBS_DIM,
  UPPER_POLICY_CRITIC_OBS_DIM,
)
from mjlab.tasks.a1_goal_room.models import (
  DEFAULT_HIDDEN_DIMS,
  UPPER_POLICY_ACTION_DIM,
  ActorNet,
  CriticNet,
  build_default_upper_policy_models,
)


def test_default_actor_and_critic_output_shapes() -> None:
  actor = ActorNet()
  critic = CriticNet()

  actor_obs = torch.randn(5, UPPER_POLICY_ACTOR_OBS_DIM)
  critic_obs = torch.randn(5, UPPER_POLICY_CRITIC_OBS_DIM)

  actor_action = actor(actor_obs)
  sampled_action, log_prob = actor.sample(actor_obs)
  sampled_log_prob = actor.log_prob(actor_obs, sampled_action)
  critic_value = critic(critic_obs)

  assert actor_action.shape == (5, UPPER_POLICY_ACTION_DIM)
  assert sampled_action.shape == (5, UPPER_POLICY_ACTION_DIM)
  assert log_prob.shape == (5, 1)
  assert sampled_log_prob.shape == (5, 1)
  assert critic_value.shape == (5, 1)
  assert torch.all(actor_action <= 1.0)
  assert torch.all(actor_action >= -1.0)
  assert torch.all(sampled_action <= 1.0)
  assert torch.all(sampled_action >= -1.0)


def test_default_models_use_requested_hidden_dims() -> None:
  actor = ActorNet()
  critic = CriticNet()

  actor_linear_layers = [
    layer for layer in actor.backbone if isinstance(layer, torch.nn.Linear)
  ]
  critic_linear_layers = [
    layer for layer in critic.backbone if isinstance(layer, torch.nn.Linear)
  ]

  assert actor_linear_layers[0].out_features == DEFAULT_HIDDEN_DIMS[0]
  assert actor_linear_layers[1].out_features == DEFAULT_HIDDEN_DIMS[1]
  assert actor_linear_layers[2].out_features == DEFAULT_HIDDEN_DIMS[2]
  assert actor.mean_head.out_features == UPPER_POLICY_ACTION_DIM

  assert critic_linear_layers[0].out_features == DEFAULT_HIDDEN_DIMS[0]
  assert critic_linear_layers[1].out_features == DEFAULT_HIDDEN_DIMS[1]
  assert critic_linear_layers[2].out_features == DEFAULT_HIDDEN_DIMS[2]
  assert critic.value_head.out_features == 1


def test_build_default_upper_policy_models_returns_bundle() -> None:
  bundle = build_default_upper_policy_models()
  assert isinstance(bundle.actor, ActorNet)
  assert isinstance(bundle.critic, CriticNet)
