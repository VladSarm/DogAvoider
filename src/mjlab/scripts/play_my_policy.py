"""Play a trained upper-level actor checkpoint in the A1 goal-room task."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.a1_goal_room.env_cfg import make_a1_goal_room_env_cfg
from mjlab.tasks.a1_goal_room.models import ActorNet
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


@dataclass(frozen=True)
class PlayConfig:
  checkpoint: str
  device: str | None = None
  num_envs: int = 1
  seed: int = 0
  num_pillars: int = 10
  viewer: Literal["none", "auto", "native", "viser"] = "viser"
  max_steps: int = 500


class DirectEnvViewerAdapter:
  """Small adapter so mjlab viewers can run on a raw ManagerBasedRlEnv."""

  def __init__(self, env: ManagerBasedRlEnv):
    self.env = env

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
      self.env.reset()
    return self.env.obs_buf["actor"]

  def reset(self):
    return self.env.reset()

  def step(self, actions: torch.Tensor):
    return self.env.step(actions)

  def close(self) -> None:
    self.env.close()


class CheckpointActorPolicy:
  """Deterministic mean-action wrapper for viewer playback."""

  def __init__(self, checkpoint_path: str | Path, device: str) -> None:
    self.device = device
    self.actor = ActorNet().to(device)
    state = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    actor_state = state["actor"] if isinstance(state, dict) and "actor" in state else state
    self.actor.load_state_dict(actor_state, strict=True)
    self.actor.eval()

  @torch.no_grad()
  def act(self, obs: torch.Tensor) -> torch.Tensor:
    return self.actor.mode(obs.to(self.device))

  def __call__(self, obs: torch.Tensor) -> torch.Tensor:
    return self.act(obs)


def _run_loop(env: ManagerBasedRlEnv, policy: CheckpointActorPolicy, max_steps: int) -> None:
  obs, _ = env.reset()
  print(f"[play_my_policy] actor observation shape: {tuple(obs['actor'].shape)}")
  for step in range(max_steps):
    action = policy.act(obs["actor"])
    obs, reward, terminated, truncated, _ = env.step(action)
    if step % 20 == 0 or step == max_steps - 1:
      done_count = int(torch.count_nonzero(terminated | truncated).item())
      print(
        f"[play_my_policy] step={step:04d} "
        f"reward_mean={reward.mean().item():+.4f} "
        f"done_count={done_count}"
      )


def _run_viewer(env: ManagerBasedRlEnv, policy: CheckpointActorPolicy, viewer_name: str) -> None:
  adapter = DirectEnvViewerAdapter(env)
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


def main() -> None:
  configure_torch_backends()
  cfg = tyro.cli(PlayConfig)

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env_cfg = make_a1_goal_room_env_cfg(
    num_envs=cfg.num_envs,
    num_pillars=cfg.num_pillars,
    seed=cfg.seed,
  )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  policy = CheckpointActorPolicy(cfg.checkpoint, device=device)
  print(
    f"[play_my_policy] device={device} num_envs={cfg.num_envs} "
    f"num_pillars={cfg.num_pillars} checkpoint={cfg.checkpoint}"
  )

  try:
    if cfg.viewer == "none":
      _run_loop(env, policy, cfg.max_steps)
    else:
      _run_viewer(env, policy, cfg.viewer)
  finally:
    env.close()


if __name__ == "__main__":
  main()
