"""Demo: render Unitree G1 with CameraSensor.

Run with:
  uv run python scripts/demos/g1_camera_sensor.py
  uv run python scripts/demos/g1_camera_sensor.py --steps 300 --fps 20
  uv run python scripts/demos/g1_camera_sensor.py --save-prefix scripts/demos/g1
"""

from __future__ import annotations

import argparse
import time

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch

from mjlab.asset_zoo.robots import get_g1_robot_cfg
from mjlab.entity import EntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sensor import CameraSensorCfg
from mjlab.sim.sim import Simulation, SimulationCfg

SCENE_XML = """
<mujoco>
  <worldbody>
    <light name="key" pos="2 -2 5" dir="-0.3 0.3 -1"/>
    <light name="fill" pos="-2 2 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="8 8 0.1" rgba="0.55 0.55 0.55 1"/>
    <camera name="g1_view" pos="2.8 -2.2 1.6" quat="0.90 0.20 0.08 0.38"
            fovy="42" resolution="512 384"/>
  </worldbody>
</mujoco>
"""


def build_scene(device: str, width: int, height: int) -> tuple[Scene, Simulation]:
  world_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(SCENE_XML))
  robot_cfg = get_g1_robot_cfg()
  cam_cfg = CameraSensorCfg(
    name="g1_cam",
    camera_name="world/g1_view",
    width=width,
    height=height,
    data_types=("rgb", "depth"),
    use_shadows=True,
  )

  scene_cfg = SceneCfg(
    num_envs=1,
    env_spacing=2.0,
    entities={"world": world_cfg, "robot": robot_cfg},
    sensors=(cam_cfg,),
  )
  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim = Simulation(num_envs=1, cfg=SimulationCfg(), model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)
  if scene.sensor_context is not None:
    sim.set_sensor_context(scene.sensor_context)
  return scene, sim


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--steps", type=int, default=200, help="Simulation steps.")
  parser.add_argument("--fps", type=float, default=20.0, help="Display refresh rate.")
  parser.add_argument("--width", type=int, default=512, help="Camera image width.")
  parser.add_argument("--height", type=int, default=384, help="Camera image height.")
  parser.add_argument(
    "--device",
    type=str,
    default=None,
    help="Compute device, e.g. cpu or cuda:0. Auto-detect when omitted.",
  )
  parser.add_argument(
    "--save-prefix",
    type=str,
    default="scripts/demos/g1_render",
    help="Output path prefix for RGB/depth PNGs (set empty string to disable).",
  )
  args = parser.parse_args()

  if args.fps <= 0:
    raise ValueError("--fps must be > 0")
  if args.width <= 0 or args.height <= 0:
    raise ValueError("--width/--height must be > 0")

  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")

  scene, sim = build_scene(device=device, width=args.width, height=args.height)
  cam = scene["g1_cam"]

  plt.ion()
  fig, (ax_rgb, ax_depth) = plt.subplots(1, 2, figsize=(11, 4.5))
  rgb_im = ax_rgb.imshow(np.zeros((args.height, args.width, 3), dtype=np.uint8))
  depth_im = ax_depth.imshow(
    np.zeros((args.height, args.width), dtype=np.float32), cmap="magma", vmin=0, vmax=1
  )
  ax_rgb.set_title("G1 RGB")
  ax_depth.set_title("G1 Depth (normalized)")
  ax_rgb.axis("off")
  ax_depth.axis("off")
  fig.tight_layout()

  refresh_dt = 1.0 / args.fps
  last_refresh = 0.0

  for i in range(args.steps):
    # Keep the robot near its initialized pose; add a tiny root yaw sway so the
    # rendered view has visible changes over time.
    yaw = 0.06 * np.sin(i * 0.04)
    quat = torch.tensor(
      [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)],
      dtype=torch.float32,
      device=device,
    )
    sim.data.qpos[0, 3:7] = quat

    sim.forward()
    sim.sense()
    scene.update(dt=sim.mj_model.opt.timestep)

    now = time.perf_counter()
    if now - last_refresh < refresh_dt and i < args.steps - 1:
      continue
    last_refresh = now

    data = cam.data
    if data.rgb is None or data.depth is None:
      raise RuntimeError("Expected both RGB and depth from g1_cam.")

    rgb = data.rgb[0].detach().cpu().numpy()
    depth = data.depth[0, :, :, 0].detach().cpu().numpy()
    depth_max = max(np.percentile(depth, 98), 1e-6)
    depth_vis = np.clip(depth / depth_max, 0.0, 1.0)

    rgb_im.set_data(rgb)
    depth_im.set_data(depth_vis)
    ax_depth.set_title(f"G1 Depth (normalized), p98={depth_max:.2f}m")
    fig.canvas.draw_idle()
    plt.pause(0.001)

  data = cam.data
  if args.save_prefix and data.rgb is not None and data.depth is not None:
    rgb = data.rgb[0].detach().cpu().numpy()
    depth = data.depth[0, :, :, 0].detach().cpu().numpy()
    depth_max = max(np.percentile(depth, 98), 1e-6)
    depth_vis = np.clip(depth / depth_max, 0.0, 1.0)
    plt.imsave(f"{args.save_prefix}_rgb.png", rgb)
    plt.imsave(f"{args.save_prefix}_depth.png", depth_vis, cmap="magma")
    print(f"Saved: {args.save_prefix}_rgb.png, {args.save_prefix}_depth.png")

  print("Done. Close the figure window to exit.")
  plt.ioff()
  plt.show()


if __name__ == "__main__":
  main()
