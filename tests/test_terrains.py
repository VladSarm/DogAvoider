"""Tests for terrain generation."""

import mujoco
import numpy as np

from mjlab.terrains.primitive_terrains import (
  BoxRoomWithPillarsTerrainCfg,
  BoxSteppingStonesTerrainCfg,
)

_CFG = BoxSteppingStonesTerrainCfg(
  proportion=1.0,
  size=(8.0, 8.0),
  stone_size_range=(0.2, 0.6),
  stone_distance_range=(0.05, 0.25),
  stone_height=0.2,
  stone_height_variation=0.05,
  stone_size_variation=0.05,
  displacement_range=0.1,
  floor_depth=2.0,
  platform_width=1.5,
  border_width=0.25,
)


def _generate_stones(
  cfg: BoxSteppingStonesTerrainCfg,
  difficulty: float,
  rng: np.random.Generator,
) -> list[tuple[float, float, float, float]]:
  """Generate terrain and return stone (cx, cy, half_x, half_y) tuples."""
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")
  output = cfg.function(difficulty=difficulty, spec=spec, rng=rng)

  center = cfg.size[0] / 2
  stones = []
  for geom_info in output.geometries:
    geom = geom_info.geom
    if geom is None:
      continue
    pos, size = geom.pos, geom.size
    # Skip platform, floor, and border geoms.
    is_platform = (
      np.isclose(pos[0], center)
      and np.isclose(pos[1], center)
      and np.isclose(size[0], cfg.platform_width / 2, atol=1e-4)
    )
    is_full_span = np.isclose(size[0], cfg.size[0] / 2) or np.isclose(
      size[1], cfg.size[1] / 2
    )
    if is_platform or is_full_span:
      continue
    stones.append((pos[0], pos[1], size[0], size[1]))
  return stones


def test_no_stone_centers_inside_platform():
  """No stone center should fall inside the platform."""
  center = _CFG.size[0] / 2
  p_half = _CFG.platform_width / 2
  p_min, p_max = center - p_half, center + p_half

  for difficulty in [0.0, 0.5, 1.0]:
    stones = _generate_stones(_CFG, difficulty, np.random.default_rng(42))
    for cx, cy, _, _ in stones:
      assert not (p_min <= cx <= p_max and p_min <= cy <= p_max), (
        f"Stone at ({cx:.3f}, {cy:.3f}) inside platform at difficulty={difficulty}"
      )


def test_stone_size_decreases_with_difficulty():
  """Average stone size should be smaller at higher difficulty."""
  sizes = {}
  for difficulty in [0.0, 1.0]:
    stones = _generate_stones(_CFG, difficulty, np.random.default_rng(42))
    sizes[difficulty] = np.mean([hx + hy for _, _, hx, hy in stones])

  assert sizes[0.0] > sizes[1.0]


def test_room_with_pillars_has_expected_geometry_count():
  cfg = BoxRoomWithPillarsTerrainCfg(
    size=(10.0, 10.0),
    num_pillars=7,
    wall_height=1.0,
    wall_thickness=0.2,
    pillar_size=(0.3, 0.3),
    pillar_height=1.0,
    spawn_clearance=1.0,
  )
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")
  output = cfg.function(difficulty=0.5, spec=spec, rng=np.random.default_rng(7))

  # 1 floor + 4 walls + N pillars.
  assert len(output.geometries) == 1 + 4 + 7
  assert np.allclose(output.origin, np.array([5.0, 5.0, 0.0]))


def test_room_with_pillars_keeps_center_clear():
  cfg = BoxRoomWithPillarsTerrainCfg(
    size=(10.0, 10.0),
    num_pillars=12,
    spawn_clearance=1.25,
    pillar_size=(0.3, 0.3),
  )
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")
  output = cfg.function(difficulty=0.0, spec=spec, rng=np.random.default_rng(42))

  center = np.array([cfg.size[0] / 2.0, cfg.size[1] / 2.0])
  half_diag = 0.5 * np.hypot(*cfg.pillar_size)
  for geom_info in output.geometries:
    geom = geom_info.geom
    if geom is None:
      continue
    if not np.allclose(geom.size[:2], np.array(cfg.pillar_size) / 2.0):
      continue
    if not np.isclose(geom.size[2], cfg.pillar_height / 2.0):
      continue
    pillar_xy = np.array(geom.pos[:2])
    assert np.linalg.norm(pillar_xy - center) >= cfg.spawn_clearance + half_diag


def test_room_with_pillars_default_geoms_are_raycast_visible():
  cfg = BoxRoomWithPillarsTerrainCfg(num_pillars=3)
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")
  output = cfg.function(difficulty=0.0, spec=spec, rng=np.random.default_rng(0))

  for geom_info in output.geometries:
    geom = geom_info.geom
    assert geom is not None
    # Terrain raycasts in mjlab usually query geom group 0.
    assert geom.group == 0
