"""Default upper-level actor and critic models for the A1 goal-room task."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.distributions import Normal

from mjlab.tasks.a1_goal_room.constants import (
  UPPER_POLICY_ACTOR_OBS_DIM,
  UPPER_POLICY_CRITIC_OBS_DIM,
)

UPPER_POLICY_ACTION_DIM = 3
DEFAULT_HIDDEN_DIMS = (256, 128, 32)


def _build_backbone(
  input_dim: int,
  *,
  hidden_dims: tuple[int, ...] = DEFAULT_HIDDEN_DIMS,
  activation_factory: type[torch.nn.Module] = torch.nn.LeakyReLU,
) -> torch.nn.Sequential:
  layers: list[torch.nn.Module] = []
  in_features = input_dim
  for hidden_dim in hidden_dims:
    layers.append(torch.nn.Linear(in_features, hidden_dim))
    layers.append(activation_factory())
    in_features = hidden_dim
  return torch.nn.Sequential(*layers)


def _atanh_clipped(x: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
  x = x.clamp(min=-1.0 + eps, max=1.0 - eps)
  return 0.5 * (torch.log1p(x) - torch.log1p(-x))


class ActorNet(torch.nn.Module):
  """Gaussian actor with tanh squashing for classical TD actor-critic."""

  def __init__(
    self,
    *,
    input_dim: int = UPPER_POLICY_ACTOR_OBS_DIM,
    action_dim: int = UPPER_POLICY_ACTION_DIM,
    hidden_dims: tuple[int, ...] = DEFAULT_HIDDEN_DIMS,
    log_std_bounds: tuple[float, float] = (-5.0, 2.0),
    init_log_std: float = -0.5,
  ) -> None:
    super().__init__()
    self.input_dim = input_dim
    self.action_dim = action_dim
    self.log_std_bounds = log_std_bounds
    self.backbone = _build_backbone(input_dim, hidden_dims=hidden_dims)
    last_hidden_dim = hidden_dims[-1] if hidden_dims else input_dim
    self.mean_head = torch.nn.Linear(last_hidden_dim, action_dim)
    self.log_std = torch.nn.Parameter(torch.full((action_dim,), init_log_std))

  def _features(self, obs: torch.Tensor) -> torch.Tensor:
    return self.backbone(obs)

  def _distribution(self, obs: torch.Tensor) -> tuple[Normal, torch.Tensor, torch.Tensor]:
    features = self._features(obs)
    mean = self.mean_head(features)
    log_std = self.log_std.clamp(*self.log_std_bounds).expand_as(mean)
    std = log_std.exp()
    return Normal(mean, std), mean, log_std

  @staticmethod
  def _squashed_log_prob(
    dist: Normal,
    pre_tanh_action: torch.Tensor,
    squashed_action: torch.Tensor,
  ) -> torch.Tensor:
    # Change-of-variables correction for tanh-squashed Gaussian.
    log_prob = dist.log_prob(pre_tanh_action)
    log_prob -= torch.log(1.0 - squashed_action.square() + 1.0e-6)
    return log_prob.sum(dim=-1, keepdim=True)

  def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dist, _, _ = self._distribution(obs)
    pre_tanh_action = dist.rsample()
    action = torch.tanh(pre_tanh_action)
    log_prob = self._squashed_log_prob(dist, pre_tanh_action, action)
    return action, log_prob

  def log_prob(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    dist, _, _ = self._distribution(obs)
    pre_tanh_action = _atanh_clipped(actions)
    return self._squashed_log_prob(dist, pre_tanh_action, actions)

  def mode(self, obs: torch.Tensor) -> torch.Tensor:
    _, mean, _ = self._distribution(obs)
    return torch.tanh(mean)

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    return self.mode(obs)


class CriticNet(torch.nn.Module):
  """State-value critic V(z)."""

  def __init__(
    self,
    *,
    input_dim: int = UPPER_POLICY_CRITIC_OBS_DIM,
    hidden_dims: tuple[int, ...] = DEFAULT_HIDDEN_DIMS,
  ) -> None:
    super().__init__()
    self.input_dim = input_dim
    self.backbone = _build_backbone(input_dim, hidden_dims=hidden_dims)
    last_hidden_dim = hidden_dims[-1] if hidden_dims else input_dim
    self.value_head = torch.nn.Linear(last_hidden_dim, 1)

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    return self.value_head(self.backbone(obs))


@dataclass(frozen=True)
class UpperPolicyModelBundle:
  actor: torch.nn.Module
  critic: torch.nn.Module


def build_default_upper_policy_models() -> UpperPolicyModelBundle:
  """Build the current default actor/critic pair.

  Both networks use the same hidden layout: ``256 -> 128 -> 32``.
  """

  return UpperPolicyModelBundle(
    actor=ActorNet(),
    critic=CriticNet(),
  )
