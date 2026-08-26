"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion file.")
parser.add_argument("--export_only", action="store_true", help="Export ONNX and exit without entering the viewer loop.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import pathlib
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    if args_cli.wandb_path:
        import wandb

        run_path = args_cli.wandb_path

        api = wandb.Api()
        if "model" in args_cli.wandb_path:
            run_path = "/".join(args_cli.wandb_path.split("/")[:-1])
        wandb_run = api.run(run_path)
        # loop over files in the run
        files = [file.name for file in wandb_run.files() if "model" in file.name]
        # files are all model_xxx.pt find the largest filename
        if "model" in args_cli.wandb_path:
            file = args_cli.wandb_path.split("/")[-1]
        else:
            file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))

        wandb_file = wandb_run.file(str(file))
        wandb_file.download("./logs/rsl_rl/temp", replace=True)

        print(f"[INFO]: Loading model checkpoint from: {run_path}/{file}")
        resume_path = f"./logs/rsl_rl/temp/{file}"

        if args_cli.motion_file is None:
            art = next(
                (a for a in wandb_run.used_artifacts() if a.type == "motions"),
                None,
            )
            if art is None:
                print("[WARN] No motion artifact found in the run.")
            else:
                env_cfg.commands.motion.motion_file = str(
                    pathlib.Path(art.download()) / "motion.npz"
                )

    else:
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    if args_cli.motion_file is not None:
        motion_path = os.path.abspath(args_cli.motion_file)
        if not os.path.isfile(motion_path):
            raise FileNotFoundError(motion_path)
        env_cfg.commands.motion.motion_file = motion_path
        print(f"[INFO]: Using local motion file: {motion_path}")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    log_dir = os.path.dirname(resume_path)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    export_motion_policy_as_onnx(
        env.unwrapped,
        ppo_runner.alg.policy,
        normalizer=getattr(ppo_runner, "obs_normalizer", None),
        path=export_model_dir,
        filename="policy.onnx",
    )
    attach_onnx_metadata(env.unwrapped, args_cli.wandb_path if args_cli.wandb_path else "none", export_model_dir)
    print(f"[INFO]: ONNX exported to: {export_model_dir}/policy.onnx")
    if args_cli.export_only:
        env.close()
        return

    # helper: extract policy obs and make it compatible with current Isaac Lab / RSL-RL
    def get_policy_obs(obs_data):
        if isinstance(obs_data, tuple):
            obs_data = obs_data[0]

        if isinstance(obs_data, dict):
            obs_data = obs_data["policy"]

        if hasattr(obs_data, "dim"):
            while obs_data.dim() > 2:
                # remove extra singleton dims, e.g. [1, 1, 160] -> [1, 160]
                squeezed = False
                for dim in range(obs_data.dim()):
                    if obs_data.shape[dim] == 1:
                        obs_data = obs_data.squeeze(dim)
                        squeezed = True
                        break
                if not squeezed:
                    obs_data = obs_data.reshape(-1, obs_data.shape[-1])
                    break

            if obs_data.dim() == 1:
                obs_data = obs_data.unsqueeze(0)

        return obs_data

    # helper: make policy action shape [num_envs, action_dim]
    def get_policy_actions(action_data):
        if isinstance(action_data, tuple):
            action_data = action_data[0]

        if isinstance(action_data, list):
            action_data = action_data[0]

        expected_action_dim = None
        action_manager = getattr(env.unwrapped, "action_manager", None)
        if action_manager is not None:
            expected_action_dim = getattr(action_manager, "total_action_dim", None)

        if hasattr(action_data, "dim"):
            while action_data.dim() > 2:
                # remove extra singleton dims, e.g. [1, 1, 29] -> [1, 29]
                squeezed = False
                for dim in range(action_data.dim()):
                    if action_data.shape[dim] == 1:
                        action_data = action_data.squeeze(dim)
                        squeezed = True
                        break
                if not squeezed:
                    action_data = action_data.reshape(-1, action_data.shape[-1])
                    break

            if action_data.dim() == 1:
                action_data = action_data.unsqueeze(0)

            if action_data.dim() == 2 and expected_action_dim is not None:
                # correct: [num_envs, 29]
                if action_data.shape[1] == expected_action_dim:
                    pass
                # wrong: [29, num_envs] -> [num_envs, 29]
                elif action_data.shape[0] == expected_action_dim:
                    action_data = action_data.transpose(0, 1).contiguous()
                # flat but with extra shape, reshape to [1, 29]
                elif action_data.numel() == expected_action_dim:
                    action_data = action_data.reshape(1, expected_action_dim)

        return action_data

    # reset environment
    obs = get_policy_obs(env.get_observations())
    timestep = 0
    printed_debug_shape = False

    # simulate environment
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            actions = get_policy_actions(actions)

            if not printed_debug_shape:
                print(f"[DEBUG]: obs shape = {tuple(obs.shape) if hasattr(obs, 'shape') else type(obs)}")
                print(f"[DEBUG]: action shape = {tuple(actions.shape) if hasattr(actions, 'shape') else type(actions)}")
                printed_debug_shape = True

            step_out = env.step(actions)

            if isinstance(step_out, tuple) and len(step_out) == 5:
                obs, _, _, _, _ = step_out
            elif isinstance(step_out, tuple) and len(step_out) == 4:
                obs, _, _, _ = step_out
            else:
                obs = step_out

            obs = get_policy_obs(obs)

        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
