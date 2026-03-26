"""Unit tests for the model-free TD actor-critic module."""

from __future__ import annotations

from pathlib import Path

import torch

from mjlab.tasks.a1_goal_room.models import ActorNet, CriticNet
from mjlab.tasks.a1_goal_room.td_actor_critic import (
  OnlineTDActorCriticAgent,
  OnlineTDActorCriticCfg,
)


def test_online_td_actor_critic_agent_update_and_checkpoint(tmp_path: Path) -> None:
  torch.manual_seed(0)
  agent = OnlineTDActorCriticAgent(
    ActorNet(),
    CriticNet(),
    cfg=OnlineTDActorCriticCfg(
      batch_size=8,
      replay_capacity=64,
      min_replay_size=8,
      updates_per_step=1,
      random_exploration_steps=0,
    ),
    device="cpu",
  )

  actor_obs = torch.randn(8, 47)
  critic_obs = torch.randn(8, 49)
  actions = agent.act(actor_obs)
  next_actor_obs = torch.randn(8, 47)
  next_critic_obs = torch.randn(8, 49)
  reward = torch.randn(8)
  terminated = torch.zeros(8, dtype=torch.bool)
  truncated = torch.zeros(8, dtype=torch.bool)

  agent.observe(
    actor_obs,
    critic_obs,
    actions,
    reward,
    next_actor_obs,
    next_critic_obs,
    terminated,
    truncated,
  )

  assert agent.replay.size == 8
  assert "actor_loss" in agent.last_train_stats
  assert "critic_loss" in agent.last_train_stats
  assert "log_prob_mean" in agent.last_train_stats

  checkpoint_path = tmp_path / "agent.pt"
  agent.save_checkpoint(checkpoint_path)
  assert checkpoint_path.exists()

  restored = OnlineTDActorCriticAgent(
    ActorNet(),
    CriticNet(),
    cfg=OnlineTDActorCriticCfg(
      batch_size=8,
      replay_capacity=64,
      min_replay_size=8,
      updates_per_step=1,
      random_exploration_steps=0,
    ),
    device="cpu",
  )
  restored.load_checkpoint(checkpoint_path)
  assert restored.total_env_steps == agent.total_env_steps
  assert restored.total_updates == agent.total_updates
