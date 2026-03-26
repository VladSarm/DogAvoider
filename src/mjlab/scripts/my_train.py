"""Direct ManagerBasedRlEnv scaffold for a custom upper-level algorithm."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import torch
import tyro
from torch.utils.tensorboard import SummaryWriter

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.a1_goal_room.env_cfg import make_a1_goal_room_env_cfg
from mjlab.tasks.a1_goal_room.models import build_default_upper_policy_models
from mjlab.tasks.a1_goal_room.td_actor_critic import (
  OnlineTDActorCriticAgent,
  OnlineTDActorCriticCfg,
)
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


@dataclass(frozen=True)
class MyTrainConfig:
  device: str | None = None
  num_envs: int = 512
  seed: int = 0
  viewer: Literal["none", "auto", "native", "viser"] = "none"
  num_pillars: int = 10
  action_mode: Literal["random", "fixed", "tdac"] = "tdac"
  fixed_vx: float = 0.4
  fixed_vy: float = 0.0
  fixed_yaw: float = 0.0
  max_steps: int = 100_000
  log_dir: str | None = None
  log_every: int = 1
  checkpoint_every: int = 500
  resume_checkpoint: str | None = None
  batch_size: int = 256
  replay_capacity: int = 300_000
  min_replay_size: int = 1_024
  updates_per_step: int = 4
  actor_lr: float = 2.0e-4
  critic_lr: float = 1.0e-3
  discount: float = 0.99
  critic_tau: float = 0.02
  random_exploration_steps: int = 512


class DirectEnvViewerAdapter:
  """Small adapter so mjlab viewers can run on a raw ManagerBasedRlEnv."""

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    *,
    policy=None,
    writer: SummaryWriter | None = None,
    log_every: int = 1,
    checkpoint_dir: Path | None = None,
    checkpoint_every: int = 0,
  ):
    self.env = env
    self.policy = policy
    self.writer = writer
    self.log_every = log_every
    self.checkpoint_dir = checkpoint_dir
    self.checkpoint_every = checkpoint_every
    self._cached_obs: dict[str, torch.Tensor] | None = None
    self._viewer_step = 0

  @property
  def num_envs(self) -> int:
    return self.env.num_envs

  @property
  def device(self) -> str:
    return self.env.device

  @property
  def cfg(self):
    return self.env.cfg

  @property
  def unwrapped(self) -> ManagerBasedRlEnv:
    return self.env

  def get_observations(self) -> torch.Tensor:
    if not self.env.obs_buf:
      obs, _ = self.env.reset()
      self._cached_obs = obs
    else:
      self._cached_obs = self.env.obs_buf
    assert self._cached_obs is not None
    return self._cached_obs["actor"]

  def reset(self):
    obs, extras = self.env.reset()
    self._cached_obs = obs
    return obs, extras

  def step(self, actions: torch.Tensor):
    if self._cached_obs is None:
      self.get_observations()
    assert self._cached_obs is not None
    step_start = time.perf_counter()
    actor_obs = self._cached_obs["actor"]
    critic_obs = self._cached_obs["critic"]
    next_obs, reward, terminated, truncated, extras = self.env.step(actions)
    if self.policy is not None and hasattr(self.policy, "observe"):
      self.policy.observe(
        actor_obs,
        critic_obs,
        actions,
        reward,
        next_obs["actor"],
        next_obs["critic"],
        terminated,
        truncated,
      )
    step_time_s = time.perf_counter() - step_start
    self._cached_obs = next_obs

    if self.writer is not None:
      _log_episode_summaries(self.writer, extras, self._viewer_step)
      _log_policy_stats(self.writer, self.policy, self._viewer_step)
      if self._viewer_step % max(self.log_every, 1) == 0:
        _log_step_scalars(
          self.writer,
          self.env,
          step=self._viewer_step,
          actor_obs=actor_obs,
          critic_obs=critic_obs,
          actions=actions,
          reward=reward,
          terminated=terminated,
          truncated=truncated,
          step_time_s=step_time_s,
        )
        self.writer.flush()

    if (
      self.checkpoint_dir is not None
      and self.checkpoint_every > 0
      and (self._viewer_step + 1) % self.checkpoint_every == 0
    ):
      checkpoint_path = _save_checkpoint(self.policy, self.checkpoint_dir, self._viewer_step + 1)
      if checkpoint_path is not None:
        print(f"[my_train] saved checkpoint: {checkpoint_path}")

    if self._viewer_step % 20 == 0:
      goal_term = self.env.command_manager.get_term("goal")
      mean_dist = goal_term.metrics["distance_to_goal"].mean().item()
      done_count = int(torch.count_nonzero(terminated | truncated).item())
      print(
        f"[my_train] step={self._viewer_step:04d} "
        f"reward_mean={reward.mean().item():+.4f} "
        f"done_count={done_count} "
        f"goal_dist_mean={mean_dist:.3f}"
      )

    self._viewer_step += 1
    return next_obs, reward, terminated, truncated, extras

  def close(self) -> None:
    self.env.close()


class RandomHighLevelPolicy:
  def __init__(self, num_envs: int, device: str):
    self._shape = (num_envs, 3)
    self._device = device

  def act(self, obs: torch.Tensor) -> torch.Tensor:
    del obs
    return 2.0 * torch.rand(self._shape, device=self._device) - 1.0

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
    del actor_obs, critic_obs, actions, reward, next_actor_obs, next_critic_obs, terminated, truncated

  def __call__(self, obs: torch.Tensor) -> torch.Tensor:
    return self.act(obs)


class FixedHighLevelPolicy:
  def __init__(self, num_envs: int, device: str, vx: float, vy: float, yaw: float):
    action = torch.tensor([vx, vy, yaw], device=device, dtype=torch.float32)
    self._action = action.clamp_(-1.0, 1.0).repeat(num_envs, 1)

  def act(self, obs: torch.Tensor) -> torch.Tensor:
    del obs
    return self._action

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
    del actor_obs, critic_obs, actions, reward, next_actor_obs, next_critic_obs, terminated, truncated

  def __call__(self, obs: torch.Tensor) -> torch.Tensor:
    return self.act(obs)


def _make_policy(cfg: MyTrainConfig, env: ManagerBasedRlEnv):
  if cfg.action_mode == "tdac":
    model_bundle = build_default_upper_policy_models()
    algo_cfg = OnlineTDActorCriticCfg(
      discount=cfg.discount,
      critic_tau=cfg.critic_tau,
      batch_size=cfg.batch_size,
      replay_capacity=cfg.replay_capacity,
      min_replay_size=cfg.min_replay_size,
      updates_per_step=cfg.updates_per_step,
      random_exploration_steps=cfg.random_exploration_steps,
      actor_lr=cfg.actor_lr,
      critic_lr=cfg.critic_lr,
    )
    agent = OnlineTDActorCriticAgent(
      model_bundle.actor,
      model_bundle.critic,
      cfg=algo_cfg,
      device=env.device,
    )
    if cfg.resume_checkpoint is not None:
      agent.load_checkpoint(cfg.resume_checkpoint)
    return agent
  if cfg.action_mode == "random":
    return RandomHighLevelPolicy(env.num_envs, env.device)
  return FixedHighLevelPolicy(
    env.num_envs,
    env.device,
    vx=cfg.fixed_vx,
    vy=cfg.fixed_vy,
    yaw=cfg.fixed_yaw,
  )


def _default_log_dir(cfg: MyTrainConfig) -> Path:
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  run_name = f"a1_goal_room_{cfg.action_mode}_{timestamp}"
  return Path("runs") / "my_train" / run_name


def _to_float(value: torch.Tensor | float | int) -> float:
  if isinstance(value, torch.Tensor):
    return float(value.detach().float().mean().item())
  return float(value)


def _log_episode_summaries(writer: SummaryWriter, extras: dict, step: int) -> None:
  log_dict = extras.get("log", {})
  if not isinstance(log_dict, dict):
    return
  for key, value in log_dict.items():
    writer.add_scalar(key, _to_float(value), step)


def _log_step_scalars(
  writer: SummaryWriter,
  env: ManagerBasedRlEnv,
  *,
  step: int,
  actor_obs: torch.Tensor,
  critic_obs: torch.Tensor,
  actions: torch.Tensor,
  reward: torch.Tensor,
  terminated: torch.Tensor,
  truncated: torch.Tensor,
  step_time_s: float,
) -> None:
  done = terminated | truncated
  goal_term = env.command_manager.get_term("goal")

  writer.add_scalar("Train/reward_mean", reward.mean().item(), step)
  writer.add_scalar("Train/reward_std", reward.std(unbiased=False).item(), step)
  writer.add_scalar("Train/reward_min", reward.min().item(), step)
  writer.add_scalar("Train/reward_max", reward.max().item(), step)
  writer.add_scalar("Train/done_count", int(torch.count_nonzero(done).item()), step)
  writer.add_scalar("Train/done_rate", done.float().mean().item(), step)
  writer.add_scalar("Train/terminated_rate", terminated.float().mean().item(), step)
  writer.add_scalar("Train/truncated_rate", truncated.float().mean().item(), step)
  writer.add_scalar("Train/episode_length_mean", env.episode_length_buf.float().mean().item(), step)
  writer.add_scalar("Train/env_steps_per_second", env.num_envs / max(step_time_s, 1e-9), step)
  writer.add_scalar("Train/sim_steps_per_second", (env.num_envs * env.cfg.decimation) / max(step_time_s, 1e-9), step)
  writer.add_scalar("Train/step_time_ms", step_time_s * 1000.0, step)

  writer.add_scalar("Action/mean", actions.mean().item(), step)
  writer.add_scalar("Action/std", actions.std(unbiased=False).item(), step)
  writer.add_scalar("Action/abs_mean", actions.abs().mean().item(), step)
  for index, name in enumerate(("vx", "vy", "yaw")):
    writer.add_scalar(f"Action/{name}_mean", actions[:, index].mean().item(), step)
    writer.add_scalar(f"Action/{name}_std", actions[:, index].std(unbiased=False).item(), step)

  writer.add_scalar("Obs/actor_mean", actor_obs.mean().item(), step)
  writer.add_scalar("Obs/actor_std", actor_obs.std(unbiased=False).item(), step)
  writer.add_scalar("Obs/critic_mean", critic_obs.mean().item(), step)
  writer.add_scalar("Obs/critic_std", critic_obs.std(unbiased=False).item(), step)
  writer.add_scalar("Obs/lidar_mean", actor_obs[:, :40].mean().item(), step)
  writer.add_scalar("Obs/lidar_min", actor_obs[:, :40].min().item(), step)
  writer.add_scalar("Obs/lidar_max", actor_obs[:, :40].max().item(), step)
  writer.add_scalar("Obs/yaw_rate_mean", actor_obs[:, 40].mean().item(), step)
  writer.add_scalar("Obs/goal_heading_sin_mean", actor_obs[:, 41].mean().item(), step)
  writer.add_scalar("Obs/goal_heading_cos_mean", actor_obs[:, 42].mean().item(), step)
  writer.add_scalar("Obs/goal_distance_norm_mean", actor_obs[:, 43].mean().item(), step)

  writer.add_scalar("Goal/distance_mean", goal_term.metrics["distance_to_goal"].mean().item(), step)
  writer.add_scalar("Goal/distance_min", goal_term.metrics["distance_to_goal"].min().item(), step)
  writer.add_scalar("Goal/distance_max", goal_term.metrics["distance_to_goal"].max().item(), step)
  writer.add_scalar("Goal/reached_rate", goal_term.metrics["goal_reached"].float().mean().item(), step)

  for term_name, term_value in goal_term.metrics.items():
    writer.add_scalar(f"Command/{term_name}", term_value.float().mean().item(), step)

  reward_terms = env.reward_manager.active_terms
  for index, term_name in enumerate(reward_terms):
    writer.add_scalar(
      f"RewardTerms/{term_name}",
      env.reward_manager._step_reward[:, index].mean().item(),
      step,
    )

  for term_name in env.termination_manager.active_terms:
    writer.add_scalar(
      f"Termination/{term_name}",
      env.termination_manager.get_term(term_name).float().mean().item(),
      step,
    )


def _log_policy_stats(writer: SummaryWriter, policy, step: int) -> None:
  stats = getattr(policy, "last_train_stats", None)
  if not isinstance(stats, dict):
    return
  for key, value in stats.items():
    writer.add_scalar(f"Algo/{key}", float(value), step)


def _save_checkpoint(policy, checkpoint_dir: Path, step: int) -> Path | None:
  if not hasattr(policy, "save_checkpoint"):
    return None
  checkpoint_path = checkpoint_dir / f"step_{step:08d}.pt"
  policy.save_checkpoint(checkpoint_path)
  latest_path = checkpoint_dir / "latest.pt"
  policy.save_checkpoint(latest_path)
  return checkpoint_path


def _run_loop(
  env: ManagerBasedRlEnv,
  policy,
  max_steps: int,
  *,
  writer: SummaryWriter,
  log_every: int,
  checkpoint_dir: Path,
  checkpoint_every: int,
) -> None:
  obs, _ = env.reset()
  print(f"[my_train] actor observation shape: {tuple(obs['actor'].shape)}")
  writer.add_text("run/device", str(env.device), 0)
  writer.add_text("run/action_mode", getattr(policy, "__class__", type(policy)).__name__, 0)
  writer.add_text("run/actor_obs_shape", str(tuple(obs["actor"].shape)), 0)
  writer.add_text("run/critic_obs_shape", str(tuple(obs["critic"].shape)), 0)

  for step in range(max_steps):
    step_start = time.perf_counter()
    actor_obs = obs["actor"]
    critic_obs = obs["critic"]
    actions = policy.act(actor_obs)
    next_obs, reward, terminated, truncated, extras = env.step(actions)
    policy.observe(
      actor_obs,
      critic_obs,
      actions,
      reward,
      next_obs["actor"],
      next_obs["critic"],
      terminated,
      truncated,
    )
    obs = next_obs
    step_time_s = time.perf_counter() - step_start
    _log_episode_summaries(writer, extras, step)
    _log_policy_stats(writer, policy, step)
    if step % max(log_every, 1) == 0 or step == max_steps - 1:
      _log_step_scalars(
        writer,
        env,
        step=step,
        actor_obs=actor_obs,
        critic_obs=critic_obs,
        actions=actions,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        step_time_s=step_time_s,
      )
      writer.flush()
    if checkpoint_every > 0 and (step + 1) % checkpoint_every == 0:
      checkpoint_path = _save_checkpoint(policy, checkpoint_dir, step + 1)
      if checkpoint_path is not None:
        print(f"[my_train] saved checkpoint: {checkpoint_path}")
    if step % 20 == 0 or step == max_steps - 1:
      goal_term = env.command_manager.get_term("goal")
      mean_dist = goal_term.metrics["distance_to_goal"].mean().item()
      done_count = int(torch.count_nonzero(terminated | truncated).item())
      print(
        f"[my_train] step={step:04d} "
        f"reward_mean={reward.mean().item():+.4f} "
        f"done_count={done_count} "
        f"goal_dist_mean={mean_dist:.3f} "
        f"log_dir={writer.log_dir}"
      )
  final_checkpoint = _save_checkpoint(policy, checkpoint_dir, max_steps)
  if final_checkpoint is not None:
    print(f"[my_train] saved final checkpoint: {final_checkpoint}")


def _run_viewer(
  env: ManagerBasedRlEnv,
  policy,
  viewer_name: str,
  *,
  writer: SummaryWriter,
  log_every: int,
  checkpoint_dir: Path,
  checkpoint_every: int,
) -> None:
  adapter = DirectEnvViewerAdapter(
    env,
    policy=policy,
    writer=writer,
    log_every=log_every,
    checkpoint_dir=checkpoint_dir,
    checkpoint_every=checkpoint_every,
  )
  adapter.reset()

  if viewer_name == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    viewer_name = "native" if has_display else "viser"

  if viewer_name == "native":
    NativeMujocoViewer(adapter, policy).run()
  elif viewer_name == "viser":
    ViserPlayViewer(adapter, policy).run()
  else:
    raise RuntimeError(f"Unsupported viewer backend: {viewer_name}")
  final_checkpoint = _save_checkpoint(policy, checkpoint_dir, adapter._viewer_step)
  if final_checkpoint is not None:
    print(f"[my_train] saved final checkpoint: {final_checkpoint}")


def main() -> None:
  configure_torch_backends()
  cfg = tyro.cli(MyTrainConfig)

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env_cfg = make_a1_goal_room_env_cfg(
    num_envs=cfg.num_envs,
    num_pillars=cfg.num_pillars,
    seed=cfg.seed,
  )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  log_dir = str(Path(cfg.log_dir) if cfg.log_dir is not None else _default_log_dir(cfg))
  checkpoint_dir = Path(log_dir) / "checkpoints"
  checkpoint_dir.mkdir(parents=True, exist_ok=True)
  writer = SummaryWriter(log_dir=log_dir)
  print(
    f"[my_train] device={device} num_envs={cfg.num_envs} "
    f"num_pillars={cfg.num_pillars} action_mode={cfg.action_mode} "
    f"tensorboard_log_dir={log_dir}"
  )
  (Path(log_dir) / "run_config.txt").write_text(repr(cfg) + "\n", encoding="utf-8")
  policy = _make_policy(cfg, env)

  try:
    if cfg.viewer == "none":
      _run_loop(
        env,
        policy,
        cfg.max_steps,
        writer=writer,
        log_every=cfg.log_every,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=cfg.checkpoint_every,
      )
    else:
      _run_viewer(
        env,
        policy,
        cfg.viewer,
        writer=writer,
        log_every=cfg.log_every,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=cfg.checkpoint_every,
      )
  finally:
    writer.close()
    env.close()


if __name__ == "__main__":
  main()
