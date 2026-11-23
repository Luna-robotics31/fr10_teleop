#!/usr/bin/env python3
"""FR10 teleoperation simulator (Isaac Lab).

Scene: the FAIRINO FR10 with a parallel gripper bolted to a table, three cubes and a place
pad in front of it, and an operator camera. The robot is position controlled; joint targets
arrive from ``teleop_fr10.py --robot sim`` over ZeroMQ and joint states / camera images are
streamed back (see ``fr10/comm.py``).

Run it in the Isaac Lab environment:

    conda activate env_isaaclab
    python sim_fr10.py                 # with the Isaac Sim window
    python sim_fr10.py --headless      # no window (camera stream still works)

The world frame is the robot base frame: origin on the table top under the base, x forward.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="FR10 teleoperation simulator (Isaac Lab).")
parser.add_argument("--config", type=str, default=None, help="YAML configuration (default: config/default.yaml)")
parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                    help="override a configuration value, e.g. --set sim.physics_hz=240")
parser.add_argument("--no-camera", action="store_true", help="do not render/stream the operator camera")
parser.add_argument("--max-steps", type=int, default=0, help="stop after this many physics steps (0 = run until closed)")
AppLauncher.add_app_launcher_args(parser)
# One robot, contact-rich grasping: CPU physics is the fastest and most stable choice.
parser.set_defaults(device="cpu")
args = parser.parse_args()
if not args.no_camera:
    args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# --------------------------------------------------------------------------------------
# Everything below needs the running Omniverse app.
# --------------------------------------------------------------------------------------
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from fr10.config import Config, load_config, parse_override  # noqa: E402
from fr10.image_stream import JpegPublisher  # noqa: E402
from fr10.sim_bridge import SimBridge  # noqa: E402

FINGER_STROKE = 0.05  # finger joint value at "fully open" [m] (URDF upper limit)
ARM_JOINTS = ("j1", "j2", "j3", "j4", "j5", "j6")
FINGER_JOINTS = ("finger_left_joint", "finger_right_joint")

# name -> (rgb colour, x, y) on the table, robot base frame [m]
CUBES = {
    "cube_red": ((0.85, 0.10, 0.10), 0.55, -0.15),
    "cube_green": ((0.10, 0.65, 0.15), 0.70, -0.28),
    "cube_blue": ((0.10, 0.25, 0.85), 0.78, -0.05),
}
TABLE_COLOUR = (0.03, 0.035, 0.045)  # very dark charcoal: the bright cubes stand out clearly
PLACE_PAD = (0.62, 0.30)  # centre of the place pad [m]


def build_scene_cfg(cfg: Config) -> InteractiveSceneCfg:
    """Describe the scene. The table top is the plane z = 0 of the robot base frame."""
    table_h = cfg.sim.table_height
    cube = cfg.sim.cube_size
    home = cfg.robot.home_rad
    image_h, image_w = int(cfg.image.shape[0]), int(cfg.image.shape[1])
    eye_w = image_w // 2 if cfg.image.binocular else image_w

    def cube_cfg(name: str) -> RigidObjectCfg:
        colour, x, y = CUBES[name]
        return RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/" + name,
            spawn=sim_utils.CuboidCfg(
                size=(cube, cube, cube),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.10),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                # "max" makes this friction win against the default material of the fingers.
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.2, dynamic_friction=1.0, friction_combine_mode="max"
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=colour),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(x, y, 0.5 * cube)),
        )

    def camera_cfg(name: str) -> CameraCfg:
        return CameraCfg(
            prim_path="{ENV_REGEX_NS}/" + name,
            update_period=1.0 / cfg.sim.camera.rate_hz,
            height=image_h,
            width=eye_w,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=cfg.sim.camera.focal_length_mm, horizontal_aperture=20.955, clipping_range=(0.05, 50.0)
            ),
        )

    @configclass
    class FR10SceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(
            prim_path="/World/ground",
            spawn=sim_utils.GroundPlaneCfg(),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -table_h)),
        )
        light = AssetBaseCfg(
            prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.92, 0.92, 0.92))
        )
        table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/table",
            spawn=sim_utils.CuboidCfg(
                size=(1.7, 1.7, table_h),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=TABLE_COLOUR, roughness=0.9, metallic=0.0),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.45, 0.0, -0.5 * table_h)),
        )
        place_pad = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/place_pad",
            spawn=sim_utils.CuboidCfg(
                size=(0.24, 0.24, 0.004),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.85, 0.20)),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(PLACE_PAD[0], PLACE_PAD[1], 0.002)),
        )
        robot: ArticulationCfg = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/robot",
            spawn=sim_utils.UrdfFileCfg(
                asset_path=str(cfg.robot.urdf_path),
                usd_dir=str(cfg.robot.urdf_path.parent / "usd_cache"),
                fix_base=True,
                merge_fixed_joints=True,
                self_collision=False,
                # Gains are set by the actuator models below; nothing is baked into the USD.
                joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                    gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
                ),
                # A real FR10 compensates gravity in its servo loop and tracks position
                # commands rigidly; switching gravity off for the arm reproduces that.
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, max_depenetration_velocity=5.0),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False, solver_position_iteration_count=16, solver_velocity_iteration_count=1
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0),
                joint_pos={**{name: home[i] for i, name in enumerate(ARM_JOINTS)}, "finger_.*": FINGER_STROKE},
            ),
            actuators={
                # Stiff, well damped position servos (about 6 Hz bandwidth).
                "shoulder": ImplicitActuatorCfg(
                    joint_names_expr=["j[1-2]"], stiffness=16000.0, damping=800.0,
                    effort_limit_sim=2000.0, velocity_limit_sim=3.15,
                ),
                "elbow": ImplicitActuatorCfg(
                    joint_names_expr=["j3"], stiffness=4000.0, damping=200.0,
                    effort_limit_sim=1000.0, velocity_limit_sim=3.15,
                ),
                "wrist": ImplicitActuatorCfg(
                    joint_names_expr=["j[4-6]"], stiffness=1000.0, damping=40.0,
                    effort_limit_sim=300.0, velocity_limit_sim=3.2,
                ),
                # effort_limit_sim is the grip force of each finger [N].
                "gripper": ImplicitActuatorCfg(
                    joint_names_expr=["finger_.*"], stiffness=5000.0, damping=100.0,
                    effort_limit_sim=40.0, velocity_limit_sim=0.2,
                ),
            },
        )
        cube_red: RigidObjectCfg = cube_cfg("cube_red")
        cube_green: RigidObjectCfg = cube_cfg("cube_green")
        cube_blue: RigidObjectCfg = cube_cfg("cube_blue")
        if not args.no_camera:
            camera_left: CameraCfg = camera_cfg("camera_left")
            if cfg.image.binocular:
                camera_right: CameraCfg = camera_cfg("camera_right")

    return FR10SceneCfg(num_envs=1, env_spacing=4.0)


class FR10Simulation:
    """Owns the scene, applies teleop commands and publishes feedback."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.dt = 1.0 / cfg.sim.physics_hz
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=self.dt, device=args.device))

        eye = np.asarray(cfg.sim.camera.eye, dtype=float)
        target = np.asarray(cfg.sim.camera.target, dtype=float)
        self.sim.set_camera_view(eye.tolist(), target.tolist())
        self.view_yaw = math.atan2(target[1] - eye[1], target[0] - eye[0])

        self.scene = InteractiveScene(build_scene_cfg(cfg))
        self.sim.reset()

        self.robot = self.scene["robot"]
        self.arm_ids = [self.robot.joint_names.index(name) for name in ARM_JOINTS]
        self.finger_ids = [self.robot.joint_names.index(name) for name in FINGER_JOINTS]
        self.cubes = {name: self.scene[name] for name in CUBES}
        self.cameras = []
        if not args.no_camera:
            self.cameras.append(self.scene["camera_left"])
            if cfg.image.binocular:
                self.cameras.append(self.scene["camera_right"])
            self._place_cameras(eye, target)

        self.bridge = SimBridge(cfg, cfg.robot.home_rad, view_yaw=self.view_yaw)
        self.images = None
        if self.cameras:
            self.images = JpegPublisher(cfg.image.port, cfg.sim.camera.jpeg_quality)
        self._camera_period = 1.0 / cfg.sim.camera.rate_hz
        self._next_image_time = 0.0
        self.sim_time = 0.0
        self.steps = 0
        self.reset_scene()

    def _place_cameras(self, eye: np.ndarray, target: np.ndarray) -> None:
        forward = (target - eye) / np.linalg.norm(target - eye)
        right = np.cross(forward, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        # Stereo: two parallel cameras one interpupillary distance apart (left eye first).
        offsets = [-0.032, 0.032] if len(self.cameras) == 2 else [0.0]
        for camera, offset in zip(self.cameras, offsets):
            eyes = torch.tensor([(eye + offset * right).tolist()], dtype=torch.float32, device=self.sim.device)
            targets = torch.tensor([(target + offset * right).tolist()], dtype=torch.float32, device=self.sim.device)
            camera.set_world_poses_from_view(eyes, targets)




    def reset_scene(self) -> None:
        """Arm to the home pose, gripper open, cubes back to their start positions."""
        robot = self.robot
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = torch.zeros_like(joint_pos)
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        robot.set_joint_position_target(joint_pos)
        for cube in self.cubes.values():
            root_state = cube.data.default_root_state.clone()
            root_state[:, :3] += self.scene.env_origins
            cube.write_root_pose_to_sim(root_state[:, :7])
            cube.write_root_velocity_to_sim(torch.zeros_like(root_state[:, 7:]))
        self.scene.reset()
        self.scene.write_data_to_sim()

    def _apply(self, joint_target: np.ndarray, gripper: float) -> None:
        target = self.robot.data.joint_pos_target.clone()
        target[0, self.arm_ids] = torch.as_tensor(joint_target, dtype=target.dtype, device=target.device)
        target[0, self.finger_ids] = (1.0 - float(gripper)) * FINGER_STROKE
        self.robot.set_joint_position_target(target)

    def _publish_state(self) -> None:
        joint_pos = self.robot.data.joint_pos[0].cpu().numpy()
        joint_vel = self.robot.data.joint_vel[0].cpu().numpy()
        opening = float(np.mean(joint_pos[self.finger_ids]))
        objects = {}
        origin = self.scene.env_origins[0]
        for name, cube in self.cubes.items():
            pose = cube.data.root_state_w[0, :7].clone()
            pose[:3] -= origin
            objects[name] = [round(float(v), 5) for v in pose.cpu().numpy()]
        self.bridge.publish(
            self.sim_time,
            joint_pos[self.arm_ids],
            joint_vel[self.arm_ids],
            1.0 - opening / FINGER_STROKE,
            objects,
            
        )

    def _publish_image(self) -> None:
        frames = []
        for camera in self.cameras:
            rgb = camera.data.output["rgb"][0, ..., :3]
            frames.append(rgb.cpu().numpy())
        frame = frames[0] if len(frames) == 1 else np.concatenate(frames, axis=1)
        self.images.publish(np.ascontiguousarray(frame[:, :, ::-1]))  # RGB -> BGR

    def step(self) -> None:
        command = self.bridge.poll()
        if command.reset:
            self.reset_scene()
            print("[sim] scene reset", flush=True)
        self._apply(command.joint_target, command.gripper)
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.steps += 1
        self.sim_time += self.dt
        rendered = self.steps % self.cfg.sim.render_every == 0
        if rendered:
            self.sim.render()
        self.scene.update(self.dt)
        self._publish_state()
        if rendered and self.images is not None and self.sim_time >= self._next_image_time:
            self._next_image_time = self.sim_time + self._camera_period - 0.5 * self.dt
            self._publish_image()

    def run(self) -> None:
        cfg = self.cfg
        print(
            f"[sim] FR10 simulator running: physics {cfg.sim.physics_hz:.0f} Hz, "
            f"state tcp://*:{cfg.sim.state_port}, commands tcp://*:{cfg.sim.command_port}, "
            + (f"camera tcp://*:{cfg.image.port}" if self.images else "camera off"),
            flush=True,
        )
        print("[sim] now start the teleop program:  python teleop_fr10.py --robot sim", flush=True)
        wall_start = time.perf_counter()
        report_time, report_steps, connected = wall_start, 0, False
        while simulation_app.is_running():
            self.step()
            if args.max_steps and self.steps >= args.max_steps:
                break
            # Keep simulated time locked to wall-clock time.
            ahead = wall_start + self.steps * self.dt - time.perf_counter()
            if ahead > 0.0:
                time.sleep(ahead)
            elif ahead < -0.2:  # too slow to keep up: do not try to catch up in a burst
                wall_start = time.perf_counter() - self.steps * self.dt
            now = time.perf_counter()
            if self.bridge.teleop_connected != connected:
                connected = self.bridge.teleop_connected
                print("[sim] teleop program " + ("connected" if connected else "disconnected (holding position)"), flush=True)
            if now - report_time >= 10.0:
                factor = (self.steps - report_steps) * self.dt / (now - report_time)
                if factor < 0.9:
                    print(f"[sim] warning: running at {factor:.2f}x real time (lower sim.physics_hz or raise sim.render_every)", flush=True)
                report_time, report_steps = now, self.steps

    def close(self) -> None:
        self.bridge.close()
        if self.images is not None:
            self.images.close()


def main() -> None:
    overrides = dict(parse_override(text) for text in args.overrides)
    cfg = load_config(args.config, overrides)
    simulation = FR10Simulation(cfg)
    try:
        simulation.run()
    finally:
        simulation.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
