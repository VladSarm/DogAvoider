"""Classical model-free TD actor-critic for the upper-level navigation policy.

This implementation intentionally keeps a replay buffer for convenience, so it is
an off-policy approximation of textbook actor-critic. The actual updates are
classical one-step TD:

  critic: MSE(V(z_t), r_t + gamma * (1 - done_t) * V_target(z_{t+1}))
  actor:  -log pi(a_t | x_t) * advantage_t.detach()
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ReplayBatch:
  actor_obs: torch.Tensor
  critic_obs: torch.Tensor
  actions: torch.Tensor
  rewards: torch.Tensor
  next_actor_obs: torch.Tensor
  next_critic_obs: torch.Tensor
  dones: torch.Tensor


class ReplayBuffer:
  """Simple fixed-size replay buffer stored on CPU."""

  def __init__(
    self,
    capacity: int,
    *,
    actor_obs_dim: int,
    critic_obs_dim: int,
    action_dim: int,
  ) -> None:
    if capacity < 1:
      raise ValueError("Replay buffer capacity must be positive.")
    self.capacity = int(capacity)
    self.actor_obs = torch.empty((capacity, actor_obs_dim), dtype=torch.float32)
    self.critic_obs = torch.empty((capacity, critic_obs_dim), dtype=torch.float32)
    self.actions = torch.empty((capacity, action_dim), dtype=torch.float32)
    self.rewards = torch.empty((capacity, 1), dtype=torch.float32)
    self.next_actor_obs = torch.empty((capacity, actor_obs_dim), dtype=torch.float32)
    self.next_critic_obs = torch.empty((capacity, critic_obs_dim), dtype=torch.float32)
    self.dones = torch.empty((capacity, 1), dtype=torch.float32)
    self._size = 0
    self._pos = 0

  @property
  def size(self) -> int:
    return self._size

  def add_batch(
    self,
    *,
    actor_obs: torch.Tensor,
    critic_obs: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_actor_obs: torch.Tensor,
    next_critic_obs: torch.Tensor,
    dones: torch.Tensor,
  ) -> None:
    batch_size = int(actor_obs.shape[0])
    indices = (torch.arange(batch_size) + self._pos) % self.capacity

    self.actor_obs[indices] = actor_obs.detach().to("cpu", dtype=torch.float32)
    self.critic_obs[indices] = critic_obs.detach().to("cpu", dtype=torch.float32)
    self.actions[indices] = actions.detach().to("cpu", dtype=torch.float32)
    self.rewards[indices] = rewards.detach().reshape(-1, 1).to("cpu", dtype=torch.float32)
    self.next_actor_obs[indices] = next_actor_obs.detach().to("cpu", dtype=torch.float32)
    self.next_critic_obs[indices] = next_critic_obs.detach().to("cpu", dtype=torch.float32)
    self.dones[indices] = dones.detach().reshape(-1, 1).to("cpu", dtype=torch.float32)

    self._pos = (self._pos + batch_size) % self.capacity
    self._size = min(self._size + batch_size, self.capacity)

  def sample(self, batch_size: int, *, device: str) -> ReplayBatch:
    if self._size < batch_size:
      raise ValueError(f"Cannot sample batch size {batch_size} from replay size {self._size}.")
    indices = torch.randint(0, self._size, (batch_size,))
    return ReplayBatch(
      actor_obs=self.actor_obs[indices].to(device),
      critic_obs=self.critic_obs[indices].to(device),
      actions=self.actions[indices].to(device),
      rewards=self.rewards[indices].to(device),
      next_actor_obs=self.next_actor_obs[indices].to(device),
      next_critic_obs=self.next_critic_obs[indices].to(device),
      dones=self.dones[indices].to(device),
    )


@dataclass(frozen=True)
class OnlineTDActorCriticCfg:
  discount: float = 0.99
  critic_tau: float = 0.02
  batch_size: int = 512
  replay_capacity: int = 100_000
  min_replay_size: int = 1_024
  updates_per_step: int = 2
  random_exploration_steps: int = 512
  actor_lr: float = 3.0e-4
  critic_lr: float = 1.0e-3
  actor_grad_clip: float | None = 2.0
  critic_grad_clip: float | None = 2.0
  advantage_epsilon: float = 1.0e-8


class OnlineTDActorCriticAgent:
  """Model-free TD actor-critic with replay-buffer sampling.

  This is not a textbook on-policy actor-critic because it keeps replay and samples
  off-policy batches. The update itself is nevertheless classical one-step TD:
  - critic: MSE TD target
  - actor: policy gradient using detached TD advantage
  """

  def __init__(
    self,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    *,
    cfg: OnlineTDActorCriticCfg,
    device: str,
  ) -> None:
    self.actor = actor.to(device)
    self.critic = critic.to(device)
    self.target_critic = copy.deepcopy(self.critic).to(device)
    self.target_critic.eval()
    for param in self.target_critic.parameters():
      param.requires_grad_(False)

    self.cfg = cfg
    self.device = device
    self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
    self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

    actor_input_dim = self.actor.input_dim
    critic_input_dim = self.critic.input_dim
    action_dim = self.actor.action_dim
    self.replay = ReplayBuffer(
      cfg.replay_capacity,
      actor_obs_dim=actor_input_dim,
      critic_obs_dim=critic_input_dim,
      action_dim=action_dim,
    )
    self.total_env_steps = 0
    self.total_updates = 0
    self.last_train_stats: dict[str, float] = {}

  @torch.no_grad()
  def act(self, actor_obs: torch.Tensor) -> torch.Tensor:
    if self.total_env_steps < self.cfg.random_exploration_steps:
      action_dim = self.actor.action_dim
      return 2.0 * torch.rand((actor_obs.shape[0], action_dim), device=actor_obs.device) - 1.0
    action, _ = self.actor.sample(actor_obs)
    return action

  def __call__(self, actor_obs: torch.Tensor) -> torch.Tensor:
    return self.act(actor_obs)

  def observe(
    self,
    actor_obs: torch.Tensor,
    critic_obs: torch.Tensor,
    actions: torch.Tensor,
    reward: torch.Tensor,
    next_actor_obs: torch.Tensor,
    next_critic_obs: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
  ) -> None:
    dones = terminated | truncated
    self.replay.add_batch(
      actor_obs=actor_obs,
      critic_obs=critic_obs,
      actions=actions,
      rewards=reward,
      next_actor_obs=next_actor_obs,
      next_critic_obs=next_critic_obs,
      dones=dones.float(),
    )
    self.total_env_steps += int(actor_obs.shape[0])

    if self.replay.size < self.cfg.min_replay_size:
      self.last_train_stats = {
        "replay_size": float(self.replay.size),
        "total_env_steps": float(self.total_env_steps),
        "total_updates": float(self.total_updates),
      }
      return

    stats_accumulator: dict[str, float] = {
      "actor_loss": 0.0,
      "critic_loss": 0.0,
      "advantage_mean": 0.0,
      "advantage_std": 0.0,
      "q_target_mean": 0.0,
      "value_mean": 0.0,
      "log_prob_mean": 0.0,
    }
    for _ in range(self.cfg.updates_per_step):
      batch = self.replay.sample(self.cfg.batch_size, device=self.device)
      train_stats = self._update_from_batch(batch)
      for key, value in train_stats.items():
        stats_accumulator[key] += value
      self.total_updates += 1

    for key in stats_accumulator:
      stats_accumulator[key] /= self.cfg.updates_per_step
    stats_accumulator["replay_size"] = float(self.replay.size)
    stats_accumulator["total_env_steps"] = float(self.total_env_steps)
    stats_accumulator["total_updates"] = float(self.total_updates)
    self.last_train_stats = stats_accumulator

  def _update_from_batch(self, batch: ReplayBatch) -> dict[str, float]:
    value = self.critic(batch.critic_obs)
    with torch.no_grad():
      target_value = self.target_critic(batch.next_critic_obs)
      td_target = batch.rewards + self.cfg.discount * (1.0 - batch.dones) * target_value

    critic_loss = F.mse_loss(value, td_target)

    self.critic_optimizer.zero_grad(set_to_none=True)
    critic_loss.backward()
    if self.cfg.critic_grad_clip is not None:
      torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.critic_grad_clip)
    self.critic_optimizer.step()

    with torch.no_grad():
      advantage = td_target - value
      advantage = (advantage - advantage.mean()) / (
        advantage.std(unbiased=False) + self.cfg.advantage_epsilon
      )

    log_prob = self.actor.log_prob(batch.actor_obs, batch.actions)
    actor_loss = -(log_prob * advantage).mean()

    self.actor_optimizer.zero_grad(set_to_none=True)
    actor_loss.backward()
    if self.cfg.actor_grad_clip is not None:
      torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.actor_grad_clip)
    self.actor_optimizer.step()

    self._soft_update_target_critic()
    return {
      "actor_loss": float(actor_loss.detach().item()),
      "critic_loss": float(critic_loss.detach().item()),
      "advantage_mean": float(advantage.detach().mean().item()),
      "advantage_std": float(advantage.detach().std(unbiased=False).item()),
      "q_target_mean": float(td_target.detach().mean().item()),
      "value_mean": float(value.detach().mean().item()),
      "log_prob_mean": float(log_prob.detach().mean().item()),
    }

  def _soft_update_target_critic(self) -> None:
    tau = self.cfg.critic_tau
    with torch.no_grad():
      for target_param, param in zip(
        self.target_critic.parameters(), self.critic.parameters(), strict=True
      ):
        target_param.data.lerp_(param.data, tau)

  def state_dict(self) -> dict:
    return {
      "actor": self.actor.state_dict(),
      "critic": self.critic.state_dict(),
      "target_critic": self.target_critic.state_dict(),
      "actor_optimizer": self.actor_optimizer.state_dict(),
      "critic_optimizer": self.critic_optimizer.state_dict(),
      "cfg": asdict(self.cfg),
      "total_env_steps": self.total_env_steps,
      "total_updates": self.total_updates,
    }

  def load_state_dict(self, state: dict) -> None:
    self.actor.load_state_dict(state["actor"])
    self.critic.load_state_dict(state["critic"])
    self.target_critic.load_state_dict(state["target_critic"])
    self.actor_optimizer.load_state_dict(state["actor_optimizer"])
    self.critic_optimizer.load_state_dict(state["critic_optimizer"])
    self.total_env_steps = int(state.get("total_env_steps", 0))
    self.total_updates = int(state.get("total_updates", 0))

  def save_checkpoint(self, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(self.state_dict(), path)

  def load_checkpoint(self, path: str | Path) -> None:
    state = torch.load(Path(path), map_location=self.device, weights_only=False)
    self.load_state_dict(state)
