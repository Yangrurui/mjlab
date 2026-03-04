"""Unitree G1 camera demo in Viser viewer.

Run with:
  uv run python scripts/demos/g1_camera_viser.py
  uv run python scripts/demos/g1_camera_viser.py --viewer native
  uv run python scripts/demos/g1_camera_viser.py --num-envs 4
"""

from __future__ import annotations

import os

import torch
import tyro

import mjlab
import mjlab.terrains as terrain_gen
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.sensor import CameraSensorCfg
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from mjlab.terrains import TerrainGeneratorCfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


def main(viewer: str = "viser", num_envs: int = 1) -> None:
  """Launch G1 with camera feeds shown in the Viser sidebar."""
  configure_torch_backends()
  device = "cuda:0" if torch.cuda.is_available() else "cpu"

  cfg = unitree_g1_flat_env_cfg(play=True)
  cfg.scene.num_envs = num_envs
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "generator"
  cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    num_rows=1,
    num_cols=1,
    border_width=0.5,
    curriculum=True,
    add_lights=True,
    sub_terrains={
      "pyramid_stairs": terrain_gen.BoxPyramidStairsTerrainCfg(
        proportion=1.0,
        step_height_range=(0.15, 0.15),
        step_width=0.32,
        platform_width=0.8,
        border_width=0.25,
      ),
    },
  )

  # Mount camera on robot torso so the view follows the robot motion.
  cam_cfg = CameraSensorCfg(
    name="g1_demo_cam",
    parent_body="robot/torso_link",
    pos=(0.28, 0.0, 0.18),
    quat=(1.0, 0.0, 0.0, 0.0),
    fovy=58.0,
    width=512,
    height=384,
    data_types=("rgb", "depth"),
    use_textures=True,
    use_shadows=True,
    enabled_geom_groups=(0, 1, 2),
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (cam_cfg,)

  print("=" * 60)
  print("Unitree G1 Camera Demo (Viser)")
  print("  Terrain: center platform + staircase terrain")
  print("  In Viser GUI: Controls -> Camera Feeds")
  print("  You can toggle frustum and adjust depth scale interactively")
  print("=" * 60)

  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  env = RslRlVecEnvWrapper(env)

  class ZeroPolicy:
    def __call__(self, obs) -> torch.Tensor:
      del obs
      return torch.zeros(env.unwrapped.action_space.shape, device=device)

  policy = ZeroPolicy()

  if viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
  else:
    resolved_viewer = viewer

  if resolved_viewer == "native":
    NativeMujocoViewer(env, policy).run()
  elif resolved_viewer == "viser":
    ViserPlayViewer(env, policy).run()
  else:
    raise ValueError(f"Unknown viewer: {viewer}")

  env.close()


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
