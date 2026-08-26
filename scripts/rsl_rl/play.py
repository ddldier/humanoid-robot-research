"""Minimal WBT play script: checkpoint playback + alternating lateral pushes only."""

import argparse
import os
import pathlib
import sys

from isaaclab.app import AppLauncher
import cli_args  # isort: skip


# =============================================================================
# CLI
# =============================================================================

parser = argparse.ArgumentParser(
    description="Play WBT G1 with deterministic alternating lateral pushes."
)

parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--motion_file", type=str, required=True)

parser.add_argument(
    "--push_start",
    type=float,
    default=2.5,
    help="Seconds before first lateral push.",
)
parser.add_argument(
    "--push_every",
    type=float,
    default=2.0,
    help="Seconds between alternating pushes.",
)
parser.add_argument(
    "--push_dv",
    type=float,
    default=1.5,
    help="Magnitude of lateral delta velocity in robot heading frame [m/s].",
)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# =============================================================================
# Imports after Isaac Sim launch
# =============================================================================

import gymnasium as gym
import torch

import isaaclab.utils.math as math_utils

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from rsl_rl.runners import OnPolicyRunner

import whole_body_tracking.tasks  # noqa: F401


# =============================================================================
# Helpers
# =============================================================================

def get_policy_obs(obs):
    if isinstance(obs, tuple):
        obs = obs[0]

    if isinstance(obs, dict):
        obs = obs["policy"]

    if hasattr(obs, "dim"):
        while obs.dim() > 2:
            squeezed = False

            for dim in range(obs.dim()):
                if obs.shape[dim] == 1:
                    obs = obs.squeeze(dim)
                    squeezed = True
                    break

            if not squeezed:
                obs = obs.reshape(-1, obs.shape[-1])
                break

        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

    return obs


def get_policy_actions(actions, env):
    if isinstance(actions, tuple):
        actions = actions[0]

    if isinstance(actions, list):
        actions = actions[0]

    expected_dim = getattr(
        getattr(env.unwrapped, "action_manager", None),
        "total_action_dim",
        None,
    )

    if hasattr(actions, "dim"):
        while actions.dim() > 2:
            squeezed = False

            for dim in range(actions.dim()):
                if actions.shape[dim] == 1:
                    actions = actions.squeeze(dim)
                    squeezed = True
                    break

            if not squeezed:
                actions = actions.reshape(-1, actions.shape[-1])
                break

        if actions.dim() == 1:
            actions = actions.unsqueeze(0)

        if (
            actions.dim() == 2
            and expected_dim is not None
            and actions.shape[1] != expected_dim
            and actions.shape[0] == expected_dim
        ):
            actions = actions.transpose(0, 1).contiguous()

    return actions


# =============================================================================
# Main
# =============================================================================

@hydra_task_config(
    args_cli.task,
    "rsl_rl_cfg_entry_point",
)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlOnPolicyRunnerCfg,
):
    agent_cfg = cli_args.parse_rsl_rl_cfg(
        args_cli.task,
        args_cli,
    )

    env_cfg.scene.num_envs = args_cli.num_envs

    # -------------------------------------------------------------------------
    # Motion file
    # -------------------------------------------------------------------------

    motion_path = pathlib.Path(
        args_cli.motion_file
    ).expanduser().resolve()

    if not motion_path.is_file():
        raise FileNotFoundError(
            f"Motion file does not exist: {motion_path}"
        )

    env_cfg.commands.motion.motion_file = str(motion_path)

    # -------------------------------------------------------------------------
    # Deterministic evaluation.
    # Disable all training-time disturbances/randomization.
    # -------------------------------------------------------------------------

    if hasattr(env_cfg, "events"):
        for name in (
            "attack_reaction",
            "push_robot",

            # R3 新增：
            # 可视化时关闭训练阶段自动 lateral curriculum，
            # 避免与下面手动 --push_dv 测试叠加。
            "lateral_recovery_push",

            "physics_material",
            "add_joint_default_pos",
            "base_com",
            "body_mass",
            "actuator_gains",
        ):
            if hasattr(env_cfg.events, name):
                setattr(env_cfg.events, name, None)


    env_cfg.commands.motion.pose_range = {}
    env_cfg.commands.motion.velocity_range = {}
    env_cfg.commands.motion.joint_position_range = (0.0, 0.0)

    if hasattr(env_cfg, "observations") and hasattr(
        env_cfg.observations,
        "policy",
    ):
        env_cfg.observations.policy.enable_corruption = False

    print(f"[INFO] Motion: {motion_path}")
    print(
        "[PUSH] "
        f"start={args_cli.push_start:.2f}s, "
        f"every={args_cli.push_every:.2f}s, "
        f"dv=+/-{args_cli.push_dv:.2f} m/s"
    )

    # -------------------------------------------------------------------------
    # Local checkpoint
    # -------------------------------------------------------------------------

    log_root_path = os.path.abspath(
        os.path.join(
            "logs",
            "rsl_rl",
            agent_cfg.experiment_name,
        )
    )

    resume_path = get_checkpoint_path(
        log_root_path,
        agent_cfg.load_run,
        agent_cfg.load_checkpoint,
    )

    print(f"[INFO] Checkpoint: {resume_path}")

    # -------------------------------------------------------------------------
    # Environment
    # -------------------------------------------------------------------------

    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode=None,
    )

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env)

    # -------------------------------------------------------------------------
    # Policy
    # -------------------------------------------------------------------------

    ppo_runner = OnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=None,
        device=agent_cfg.device,
    )

    ppo_runner.load(resume_path)

    policy = ppo_runner.get_inference_policy(
        device=env.unwrapped.device
    )

    # -------------------------------------------------------------------------
    # Runtime
    # -------------------------------------------------------------------------

    robot = env.unwrapped.scene["robot"]

    control_dt = env.unwrapped.step_dt

    first_push_step = max(
        1,
        int(args_cli.push_start / control_dt),
    )

    push_interval_steps = max(
        1,
        int(args_cli.push_every / control_dt),
    )

    demo_step = 0
    push_count = 0
    next_push_step = first_push_step

    # -------------------------------------------------------------------------
    # Alternating heading-frame lateral delta-v.
    # -------------------------------------------------------------------------

    def apply_lateral_push(side_sign: float):
        root_state = robot.data.root_state_w.clone()

        heading_quat = math_utils.yaw_quat(
            root_state[:, 3:7]
        )

        local_dv = torch.zeros(
            (root_state.shape[0], 3),
            device=robot.device,
            dtype=root_state.dtype,
        )

        # +Y = robot left, -Y = robot right.
        local_dv[:, 1] = (
            side_sign * args_cli.push_dv
        )

        dv_world = math_utils.quat_apply(
            heading_quat,
            local_dv,
        )

        # Add disturbance to current linear velocity.
        root_state[:, 7:10] += dv_world

        robot.write_root_state_to_sim(root_state)

        direction = (
            "LEFT"
            if side_sign > 0.0
            else "RIGHT"
        )

        print(
            "\n"
            "========================================\n"
            f"[LATERAL PUSH] {direction}\n"
            f"delta-v = {args_cli.push_dv:.2f} m/s\n"
            "========================================\n"
        )

    # -------------------------------------------------------------------------
    # Initial observation
    # -------------------------------------------------------------------------

    obs = get_policy_obs(
        env.get_observations()
    )

    printed_shape = False

    # -------------------------------------------------------------------------
    # Simulation loop
    # -------------------------------------------------------------------------

    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)

            actions = get_policy_actions(
                actions,
                env,
            )

            if not printed_shape:
                print(
                    "[DEBUG] obs shape =",
                    tuple(obs.shape)
                    if hasattr(obs, "shape")
                    else type(obs),
                )
                print(
                    "[DEBUG] action shape =",
                    tuple(actions.shape)
                    if hasattr(actions, "shape")
                    else type(actions),
                )
                printed_shape = True

            if demo_step >= next_push_step:
                side_sign = (
                    +1.0
                    if push_count % 2 == 0
                    else -1.0
                )

                apply_lateral_push(side_sign)

                push_count += 1
                next_push_step += push_interval_steps

            step_out = env.step(actions)

            if isinstance(step_out, tuple):
                obs = step_out[0]
            else:
                obs = step_out

            obs = get_policy_obs(obs)

            demo_step += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()