"""
Replay motion from a PKL (or CSV) file and output it to an NPZ file.

Usage (PKL):
  python csv_to_npz.py --input_file ./motions/my_motion.pkl --output_name my_motion --output_fps 50

Usage (CSV, unchanged):
  python csv_to_npz.py --input_file LAFAN/dance1_subject2.csv --input_fps 30 --frame_range 122 722 \
    --output_name dance1_subject2 --output_fps 50
"""

import argparse
import os
import pickle
import numpy as np

from isaaclab.app import AppLauncher

# args
parser = argparse.ArgumentParser(
    description="Replay motion from pkl/csv and output to npz."
)
parser.add_argument(
    "--input_file",
    type=str,
    required=True,
    help="Path to input motion file (.pkl or .csv).",
)
parser.add_argument(
    "--input_fps",
    type=int,
    default=None,
    help="FPS of the input motion. If not provided for PKL, will read from PKL['fps'].",
)
parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help="frame range: START END (both inclusive, 1-indexed). If not provided, all frames are loaded.",
)
parser.add_argument(
    "--output_name",
    type=str,
    required=True,
    help="The name for the output motion npz artifact.",
)
parser.add_argument(
    "--output_fps", type=int, default=50, help="The fps of the output motion."
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import (
    axis_angle_from_quat,
    quat_conjugate,
    quat_mul,
    quat_slerp,
)

# robot config
from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg()
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


class MotionLoader:
    def __init__(self, motion_file, input_fps, output_fps, device, frame_range):
        self.motion_file = motion_file
        self.cli_input_fps = input_fps  # may be None for PKL
        self.output_fps = output_fps
        self.device = device
        self.frame_range = frame_range
        self._load_motion()  # sets self.input_fps
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self._interpolate_motion()
        self._compute_velocities()
        self.current_idx = 0

    def _slice_range(self, arr):
        if self.frame_range is None:
            return arr
        start, end = self.frame_range
        # incoming is 1-indexed and inclusive; convert to 0-indexed slice
        return arr[start - 1 : end]

    def _load_motion(self):
        """Loads motion from NPZ (preferred) or CSV (legacy)."""
        import os
        import numpy as np
        import torch

        ext = os.path.splitext(self.motion_file)[1].lower()

        if ext == ".npz":
            # Load numeric arrays; allow_pickle=False is fine unless you saved objects
            data = np.load(self.motion_file, allow_pickle=False)

            # Required keys written above
            root_pos = np.asarray(data["root_pos"], dtype=np.float32)  # (N,3)
            root_rot_xyzw = np.asarray(data["root_rot"], dtype=np.float32)  # (N,4) xyzw
            dof_pos = np.asarray(data["dof_pos"], dtype=np.float32)  # (N,M)

            # FPS: CLI overrides file value; default to 30 if neither provided
            file_fps = float(data["fps"]) if "fps" in data else None
            self.input_fps = float(self.cli_input_fps or file_fps or 30)

            # Apply 1-indexed inclusive frame range if provided
            if self.frame_range is not None:
                s, e = self.frame_range
                root_pos = root_pos[s - 1 : e]
                root_rot_xyzw = root_rot_xyzw[s - 1 : e]
                dof_pos = dof_pos[s - 1 : e]

            # Isaac expects wxyz → convert xyzw → wxyz
            root_rot_wxyz = root_rot_xyzw[:, [3, 0, 1, 2]]

            # Build tensors on the chosen device
            self.motion_base_poss_input = torch.from_numpy(root_pos).to(self.device)
            self.motion_base_rots_input = torch.from_numpy(root_rot_wxyz).to(
                self.device
            )
            self.motion_dof_poss_input = torch.from_numpy(dof_pos).to(self.device)

        elif ext == ".csv":
            # Legacy path (unchanged): [tx,ty,tz, qx,qy,qz,qw, dofs...]
            if self.cli_input_fps is None:
                raise ValueError("--input_fps is required for CSV input.")
            self.input_fps = float(self.cli_input_fps)
            if self.frame_range is None:
                motion_np = np.loadtxt(self.motion_file, delimiter=",")
            else:
                motion_np = np.loadtxt(
                    self.motion_file,
                    delimiter=",",
                    skiprows=self.frame_range[0] - 1,
                    max_rows=self.frame_range[1] - self.frame_range[0] + 1,
                )
            motion = torch.from_numpy(motion_np).to(torch.float32).to(self.device)
            self.motion_base_poss_input = motion[:, :3]
            # CSV stored xyzw → convert to wxyz for Isaac
            self.motion_base_rots_input = motion[:, 3:7][:, [3, 0, 1, 2]]
            self.motion_dof_poss_input = motion[:, 7:]

        else:
            raise ValueError(
                "Unsupported input file type. Use .npz (preferred) or .csv"
            )

        self.input_frames = self.motion_base_poss_input.shape[0]
        if self.input_frames < 2:
            raise ValueError("Need at least 2 frames for interpolation/velocities.")
        self.duration = (self.input_frames - 1) * (1.0 / self.input_fps)
        print(
            f"Motion loaded ({self.motion_file}), frames: {self.input_frames}, "
            f"input_fps: {self.input_fps}, duration: {self.duration:.3f}s"
        )

    def _interpolate_motion(self):
        """Interpolates the motion to the output fps."""
        times = torch.arange(
            0.0, self.duration, self.output_dt, device=self.device, dtype=torch.float32
        )
        self.output_frames = times.shape[0]
        idx0, idx1, blend = self._compute_frame_blend(times)

        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[idx0],
            self.motion_base_poss_input[idx1],
            blend.unsqueeze(1),
        )
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[idx0], self.motion_base_rots_input[idx1], blend
        )
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[idx0],
            self.motion_dof_poss_input[idx1],
            blend.unsqueeze(1),
        )
        print(
            f"Motion interpolated → output_frames: {self.output_frames}, output_fps: {self.output_fps}"
        )

    def _compute_frame_blend(self, times: torch.Tensor):
        phase = times / self.duration
        idx0 = (phase * (self.input_frames - 1)).floor().long()
        idx1 = torch.minimum(
            idx0 + 1, torch.tensor(self.input_frames - 1, device=times.device)
        )
        blend = phase * (self.input_frames - 1) - idx0
        return idx0, idx1, blend

    def _lerp(self, a: torch.Tensor, b: torch.Tensor, t: torch.Tensor):
        return a * (1 - t) + b * t

    def _slerp(self, a: torch.Tensor, b: torch.Tensor, t: torch.Tensor):
        out = torch.zeros_like(a)
        for i in range(a.shape[0]):
            out[i] = quat_slerp(a[i], b[i], t[i])
        return out

    def _compute_velocities(self):
        dt = self.output_dt
        self.motion_base_lin_vels = torch.gradient(
            self.motion_base_poss, spacing=dt, dim=0
        )[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=dt, dim=0)[
            0
        ]
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, dt)

    def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        # rotations: (T,4) wxyz
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))  # (T-2,4)
        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)  # (T-2,3)
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)
        return omega

    def get_next_state(self):
        state = (
            self.motion_base_poss[self.current_idx : self.current_idx + 1],
            self.motion_base_rots[self.current_idx : self.current_idx + 1],
            self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
            self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
            self.motion_dof_poss[self.current_idx : self.current_idx + 1],
            self.motion_dof_vels[self.current_idx : self.current_idx + 1],
        )
        self.current_idx += 1
        reset_flag = False
        if self.current_idx >= self.output_frames:
            self.current_idx = 0
            reset_flag = True
        return state, reset_flag


def run_simulator(
    sim: sim_utils.SimulationContext, scene: InteractiveScene, joint_names: list[str]
):
    # Load motion
    motion = MotionLoader(
        motion_file=args_cli.input_file,
        input_fps=args_cli.input_fps,
        output_fps=args_cli.output_fps,
        device=sim.device,
        frame_range=tuple(args_cli.frame_range) if args_cli.frame_range else None,
    )

    robot = scene["robot"]
    robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]

    log = {
        "fps": [args_cli.output_fps],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }
    file_saved = False

    while simulation_app.is_running():
        (state, reset_flag) = motion.get_next_state()
        (
            motion_base_pos,
            motion_base_rot,
            motion_base_lin_vel,
            motion_base_ang_vel,
            motion_dof_pos,
            motion_dof_vel,
        ) = state

        # root (Isaac expects wxyz)
        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion_base_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        robot.write_root_state_to_sim(root_states)

        # joints
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = motion_dof_pos
        joint_vel[:, robot_joint_indexes] = motion_dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        sim.render()
        scene.update(sim.get_physics_dt())

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

        if not file_saved:
            log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
            log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
            log["body_pos_w"].append(robot.data.body_pos_w[0, :].cpu().numpy().copy())
            log["body_quat_w"].append(robot.data.body_quat_w[0, :].cpu().numpy().copy())
            log["body_lin_vel_w"].append(
                robot.data.body_lin_vel_w[0, :].cpu().numpy().copy()
            )
            log["body_ang_vel_w"].append(
                robot.data.body_ang_vel_w[0, :].cpu().numpy().copy()
            )

        if reset_flag and not file_saved:
            file_saved = True
            for k in (
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ):
                log[k] = np.stack(log[k], axis=0)
            np.savez("/tmp/motion.npz", **log)

            import wandb

            COLLECTION = args_cli.output_name
            run = wandb.init(project="csv_to_npz", name=COLLECTION)
            print(f"[INFO]: Logging motion to wandb: {COLLECTION}")
            REGISTRY = "motions"
            logged_artifact = run.log_artifact(
                artifact_or_path="/tmp/motion.npz", name=COLLECTION, type=REGISTRY
            )
            run.link_artifact(
                artifact=logged_artifact,
                target_path=f"wandb-registry-{REGISTRY}/{COLLECTION}",
            )
            print(f"[INFO]: Motion saved to wandb registry: {REGISTRY}/{COLLECTION}")


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO]: Setup complete...")

    run_simulator(
        sim,
        scene,
        joint_names=[
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ],
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
