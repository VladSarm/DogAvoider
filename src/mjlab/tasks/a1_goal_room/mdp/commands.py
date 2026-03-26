"""Goal command term for room navigation."""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import quat_apply_inverse


class RandomRoomGoalCommand(CommandTerm):
  cfg: "RandomRoomGoalCommandCfg"

  def __init__(self, cfg: "RandomRoomGoalCommandCfg", env):
    super().__init__(cfg, env)
    self._robot: Entity = env.scene[cfg.entity_name]
    self.target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
    self.distance_to_goal = torch.zeros(self.num_envs, device=self.device)
    self.goal_vector_b = torch.zeros(self.num_envs, 3, device=self.device)
    self._pillar_centers_b, self._pillar_half_extents = self._extract_pillars_from_terrain()

    self.metrics["distance_to_goal"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["goal_reached"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.target_pos_w

  def _extract_pillars_from_terrain(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    terrain = self._env.scene["terrain"]
    geom_ids = terrain.data.indexing.geom_ids.cpu().numpy()
    geom_type = self._env.sim.mj_model.geom_type[geom_ids]
    geom_size = self._env.sim.mj_model.geom_size[geom_ids]
    geom_pos = self._env.sim.mj_model.geom_pos[geom_ids]
    env_origins = self._env.scene.env_origins.cpu().numpy()

    pillar_half = np.array(
      [
        self.cfg.pillar_size[0] * 0.5,
        self.cfg.pillar_size[1] * 0.5,
        self.cfg.pillar_height * 0.5,
      ],
      dtype=np.float64,
    )
    room_half = np.array(self.cfg.room_size, dtype=np.float64) * 0.5

    pillar_centers_b: list[torch.Tensor] = []
    pillar_half_extents: list[torch.Tensor] = []
    for origin in env_origins:
      centers: list[list[float]] = []
      halfs: list[list[float]] = []
      for g_type, g_size, g_pos in zip(geom_type, geom_size, geom_pos, strict=False):
        if g_type != mujoco.mjtGeom.mjGEOM_BOX:
          continue
        if not np.allclose(g_size, pillar_half, atol=1e-4):
          continue
        local_xy = g_pos[:2] - origin[:2]
        if not np.all(np.abs(local_xy) <= room_half + 0.5):
          continue
        centers.append(local_xy.tolist())
        halfs.append(g_size[:2].tolist())
      if centers:
        pillar_centers_b.append(torch.tensor(centers, device=self.device))
        pillar_half_extents.append(torch.tensor(halfs, device=self.device))
      else:
        pillar_centers_b.append(torch.empty((0, 2), device=self.device))
        pillar_half_extents.append(torch.empty((0, 2), device=self.device))
    return pillar_centers_b, pillar_half_extents

  def _is_goal_valid(self, env_id: int, candidate_xy_b: torch.Tensor) -> bool:
    if torch.norm(candidate_xy_b) < self.cfg.min_goal_distance_from_start:
      return False

    interior_half = torch.tensor(self.cfg.room_size, device=self.device) * 0.5
    interior_half -= self.cfg.wall_thickness + self.cfg.goal_radius
    if torch.any(torch.abs(candidate_xy_b) > interior_half):
      return False

    pillar_centers = self._pillar_centers_b[env_id]
    if pillar_centers.numel() == 0:
      return True

    pillar_half_extents = self._pillar_half_extents[env_id]
    delta = torch.abs(candidate_xy_b.unsqueeze(0) - pillar_centers)
    clearance = pillar_half_extents + self.cfg.goal_radius
    intersects = torch.all(delta <= clearance, dim=-1)
    return not bool(intersects.any())

  def _fallback_goal_xy(self, env_id: int) -> torch.Tensor:
    interior_half = torch.tensor(self.cfg.room_size, device=self.device) * 0.5
    interior_half -= self.cfg.wall_thickness + self.cfg.goal_radius

    grid = torch.linspace(-1.0, 1.0, steps=11, device=self.device)
    candidates = []
    for x in grid:
      for y in grid:
        candidate = torch.stack((x * interior_half[0], y * interior_half[1]))
        if self._is_goal_valid(env_id, candidate):
          candidates.append(candidate)
    if not candidates:
      return torch.tensor((0.0, interior_half[1] * 0.5), device=self.device)
    scores = torch.stack([torch.norm(c) for c in candidates])
    return candidates[int(torch.argmax(scores))]

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    for env_id in env_ids.tolist():
      interior_half = torch.tensor(self.cfg.room_size, device=self.device) * 0.5
      interior_half -= self.cfg.wall_thickness + self.cfg.goal_radius
      goal_xy_b = None
      for _ in range(self.cfg.max_sampling_attempts):
        sample = (2.0 * torch.rand(2, device=self.device) - 1.0) * interior_half
        if self._is_goal_valid(env_id, sample):
          goal_xy_b = sample
          break
      if goal_xy_b is None:
        goal_xy_b = self._fallback_goal_xy(env_id)

      self.target_pos_w[env_id] = self._env.scene.env_origins[env_id]
      self.target_pos_w[env_id, :2] += goal_xy_b
      self.target_pos_w[env_id, 2] += self.cfg.goal_height

  def _update_command(self) -> None:
    root_pos_w = self._robot.data.root_link_pos_w
    root_quat_w = self._robot.data.root_link_quat_w
    goal_vec_w = self.target_pos_w - root_pos_w
    self.goal_vector_b = quat_apply_inverse(root_quat_w, goal_vec_w)
    self.distance_to_goal = torch.norm(goal_vec_w[:, :2], dim=-1)

  def _update_metrics(self) -> None:
    self._update_command()
    self.metrics["distance_to_goal"] = self.distance_to_goal
    self.metrics["goal_reached"] = (
      self.distance_to_goal <= self.cfg.goal_reached_threshold
    ).float()

  def _debug_vis_impl(self, visualizer) -> None:
    for env_id in visualizer.get_env_indices(self.num_envs):
      visualizer.add_sphere(
        center=self.target_pos_w[env_id],
        radius=self.cfg.goal_radius,
        color=self.cfg.viz.goal_color,
        label=f"goal_{env_id}",
      )


@dataclass(kw_only=True)
class RandomRoomGoalCommandCfg(CommandTermCfg):
  entity_name: str
  room_size: tuple[float, float] = (10.0, 10.0)
  wall_thickness: float = 0.2
  pillar_size: tuple[float, float] = (0.3, 0.3)
  pillar_height: float = 1.0
  goal_height: float = 0.05
  goal_radius: float = 0.25
  min_goal_distance_from_start: float = 1.0
  goal_reached_threshold: float = 0.35
  max_sampling_attempts: int = 1024

  @dataclass
  class VizCfg:
    goal_color: tuple[float, float, float, float] = (1.0, 0.45, 0.0, 0.65)

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env) -> RandomRoomGoalCommand:
    return RandomRoomGoalCommand(self, env)
