"""Demo: visualize CameraSensor RGB and depth streams.

Run with:
  uv run python scripts/demos/camera_sensor.py
  uv run python scripts/demos/camera_sensor.py --steps 1500 --fps 30
  uv run python scripts/demos/camera_sensor.py --save-prefix scripts/demos/cam
"""

from __future__ import annotations

import argparse
import time

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch

from mjlab.entity import EntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sensor import CameraSensorCfg
from mjlab.sim.sim import Simulation, SimulationCfg

CAMERA_DEMO_XML = """
<mujoco>
  <option timestep="0.01"/>
  <worldbody>
    <light name="sun" pos="0 0 4" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.6 0.6 0.6 1"/>

    <camera name="overhead_cam" pos="2.2 -2.2 2.1" quat="0.83 0.34 0.14 0.42"
            fovy="45" resolution="320 240"/>

    <body name="slider" pos="0 0 0.2" mocap="true">
      <geom name="slider_geom" type="box" size="0.20 0.20 0.20"
            rgba="0.95 0.15 0.15 1"/>
    </body>

    <body name="static_box" pos="-0.8 0.8 0.2">
      <geom type="box" size="0.2 0.2 0.2" rgba="0.15 0.25 0.95 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def build_scene(device: str, width: int, height: int) -> tuple[Scene, Simulation]:
  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(CAMERA_DEMO_XML))
  cam_cfg = CameraSensorCfg(
    name="demo_cam",
    camera_name="world/overhead_cam",
    width=width,
    height=height,
    data_types=("rgb", "depth"),
  )
  scene_cfg = SceneCfg(
    num_envs=1,
    env_spacing=2.0,
    entities={"world": entity_cfg},
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
  parser.add_argument("--steps", type=int, default=800, help="Simulation steps.")
  parser.add_argument(
    "--fps",
    type=float,
    default=20.0,
    help="Display refresh rate for matplotlib.",
  )
  parser.add_argument("--width", type=int, default=320, help="Camera image width.")
  parser.add_argument("--height", type=int, default=240, help="Camera image height.")
  parser.add_argument(
    "--device",
    type=str,
    default=None,
    help="Compute device, e.g. cpu or cuda:0. Auto-detect when omitted.",
  )
  parser.add_argument(
    "--save-prefix",
    type=str,
    default=None,
    help="Optional output path prefix. Saves final RGB/Depth as PNG.",
  )
  args = parser.parse_args()

  if args.fps <= 0:
    raise ValueError("--fps must be > 0")
  if args.width <= 0 or args.height <= 0:
    raise ValueError("--width/--height must be > 0")

  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")

  scene, sim = build_scene(device=device, width=args.width, height=args.height)
  sensor = scene["demo_cam"]

  plt.ion()
  fig, (ax_rgb, ax_depth) = plt.subplots(1, 2, figsize=(10, 4))
  rgb_canvas = np.zeros((args.height, args.width, 3), dtype=np.uint8)
  depth_canvas = np.zeros((args.height, args.width), dtype=np.float32)

  rgb_im = ax_rgb.imshow(rgb_canvas)
  depth_im = ax_depth.imshow(depth_canvas, cmap="viridis", vmin=0.0, vmax=1.0)
  ax_rgb.set_title("RGB")
  ax_depth.set_title("Depth (normalized)")
  ax_rgb.axis("off")
  ax_depth.axis("off")
  fig.tight_layout()

  refresh_dt = 1.0 / args.fps
  last_refresh = 0.0

  for i in range(args.steps):
    t = i * 0.02
    sim.data.mocap_pos[0, 0, :] = torch.tensor(
      [0.8 * np.sin(t), 0.8 * np.cos(0.7 * t), 0.22],
      dtype=torch.float32,
      device=device,
    )
    sim.data.mocap_quat[0, 0, :] = torch.tensor(
      [1.0, 0.0, 0.0, 0.0],
      dtype=torch.float32,
      device=device,
    )

    sim.step()
    sim.sense()
    scene.update(dt=sim.mj_model.opt.timestep)

    now = time.perf_counter()
    if now - last_refresh < refresh_dt and i < args.steps - 1:
      continue
    last_refresh = now

    data = sensor.data
    if data.rgb is None or data.depth is None:
      raise RuntimeError("Camera is expected to provide both RGB and depth.")

    rgb = data.rgb[0].detach().cpu().numpy()
    depth = data.depth[0, :, :, 0].detach().cpu().numpy()

    # Robust depth normalization for visualization.
    depth_max = np.percentile(depth, 98)
    depth_max = max(depth_max, 1e-6)
    depth_vis = np.clip(depth / depth_max, 0.0, 1.0)

    rgb_im.set_data(rgb)
    depth_im.set_data(depth_vis)
    ax_depth.set_title(f"Depth (normalized), p98={depth_max:.2f}m")
    fig.canvas.draw_idle()
    plt.pause(0.001)

  data = sensor.data
  if data.rgb is not None and data.depth is not None and args.save_prefix is not None:
    rgb = data.rgb[0].detach().cpu().numpy()
    depth = data.depth[0, :, :, 0].detach().cpu().numpy()
    depth_max = max(np.percentile(depth, 98), 1e-6)
    depth_vis = np.clip(depth / depth_max, 0.0, 1.0)
    plt.imsave(f"{args.save_prefix}_rgb.png", rgb)
    plt.imsave(f"{args.save_prefix}_depth.png", depth_vis, cmap="viridis")
    print(f"Saved: {args.save_prefix}_rgb.png, {args.save_prefix}_depth.png")

  print("Done. Close the figure window to exit.")
  plt.ioff()
  plt.show()


if __name__ == "__main__":
  main()
