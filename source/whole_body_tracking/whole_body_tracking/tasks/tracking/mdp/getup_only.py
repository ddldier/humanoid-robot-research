from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from .getup import (
    GETUP_DEFAULT_JOINT_POS_BY_NAME,
    GETUP_DEFAULT_PROJECTED_GRAVITY,
    GETUP_DEFAULT_ROOT_HEIGHT,
    _root_height,
    standing_mask,
)


# =============================================================================
# GetUp-Only rewards
#
# Difference from the mixed punch+getup rewards:
#
# The mixed task intentionally switches get-up rewards OFF after the robot is
# standing because normal WBT tracking takes over.
#
# The GetUp-Only task has NO tracking reward. Therefore these rewards remain
# active after standing so that:
#
#   floor -> rise -> stand -> KEEP standing in the CSV default pose
#
# becomes the complete objective.
# =============================================================================


def getup_only_height_progress(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    floor_height: float = 0.25,
    target_height: float = GETUP_DEFAULT_ROOT_HEIGHT,
) -> torch.Tensor:
    """Dense 0..1 reward for root-height recovery, active in every state."""

    height = _root_height(
        env,
        asset_cfg,
    )

    progress = (
        (height - floor_height)
        / max(
            target_height - floor_height,
            1.0e-6,
        )
    )

    return torch.clamp(
        progress,
        0.0,
        1.0,
    )


def getup_only_orientation_to_default_exp(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_projected_gravity: tuple[float, float, float] = (
        GETUP_DEFAULT_PROJECTED_GRAVITY
    ),
    std: float = 0.55,
) -> torch.Tensor:
    """Reward root orientation toward the CSV default, active after standing too."""

    robot: Articulation = env.scene[
        asset_cfg.name
    ]

    target = torch.tensor(
        target_projected_gravity,
        device=robot.device,
        dtype=robot.data.projected_gravity_b.dtype,
    ).unsqueeze(0)

    error = torch.sum(
        torch.square(
            robot.data.projected_gravity_b
            - target
        ),
        dim=-1,
    )

    return torch.exp(
        -error
        / (std**2)
    )


def getup_only_default_joint_pose_exp(
    env,
    asset_cfg: SceneEntityCfg,
    target_joint_pos_by_name: dict[str, float],
    std: float = 0.65,
    pose_start_height: float = 0.35,
    target_height: float = GETUP_DEFAULT_ROOT_HEIGHT,
) -> torch.Tensor:
    """Move toward the CSV default pose, weak on floor and strong near standing."""

    robot: Articulation = env.scene[
        asset_cfg.name
    ]

    joint_pos = robot.data.joint_pos[
        :,
        asset_cfg.joint_ids,
    ]

    target = torch.tensor(
        [
            target_joint_pos_by_name[name]
            for name in asset_cfg.joint_names
        ],
        device=robot.device,
        dtype=joint_pos.dtype,
    ).unsqueeze(0)

    error = torch.mean(
        torch.square(
            joint_pos
            - target
        ),
        dim=-1,
    )

    height = _root_height(
        env,
        SceneEntityCfg("robot"),
    )

    height_blend = (
        (height - pose_start_height)
        / max(
            target_height - pose_start_height,
            1.0e-6,
        )
    )

    height_blend = torch.clamp(
        height_blend,
        0.0,
        1.0,
    )

    # Do not lock the limbs to the final stance while the robot still needs
    # hands/knees/legs to push against the ground.
    pose_weight = (
        0.10
        + 0.90 * height_blend
    )

    return (
        pose_weight
        * torch.exp(
            -error
            / (std**2)
        )
    )


def getup_only_feet_support_reward(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_threshold: float = 10.0,
    support_start_height: float = 0.38,
    target_height: float = GETUP_DEFAULT_ROOT_HEIGHT,
) -> torch.Tensor:
    """Reward foot support increasingly strongly as the body rises."""

    robot: Articulation = env.scene[
        asset_cfg.name
    ]

    sensor: ContactSensor = env.scene.sensors[
        sensor_cfg.name
    ]

    force = sensor.data.net_forces_w[
        :,
        sensor_cfg.body_ids,
    ]

    contact = (
        torch.linalg.vector_norm(
            force,
            dim=-1,
        )
        > contact_threshold
    )

    count = contact.sum(
        dim=-1
    )

    support = torch.zeros(
        env.scene.num_envs,
        device=robot.device,
        dtype=robot.data.root_pos_w.dtype,
    )

    # One foot is useful during transition.
    support += (
        (count == 1).float()
        * 0.30
    )

    # Final target is both feet supporting.
    support += (
        (count >= 2).float()
        * 1.00
    )

    height = _root_height(
        env,
        asset_cfg,
    )

    blend = (
        (height - support_start_height)
        / max(
            target_height - support_start_height,
            1.0e-6,
        )
    )

    blend = torch.clamp(
        blend,
        0.0,
        1.0,
    )

    return (
        blend
        * support
    )


def getup_only_stabilize_exp(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    start_height: float = 0.50,
    target_height: float = GETUP_DEFAULT_ROOT_HEIGHT,
    velocity_std: float = 1.0,
) -> torch.Tensor:
    """Reward low root velocity near/after standing without suppressing early motion."""

    robot: Articulation = env.scene[
        asset_cfg.name
    ]

    height = _root_height(
        env,
        asset_cfg,
    )

    blend = (
        (height - start_height)
        / max(
            target_height - start_height,
            1.0e-6,
        )
    )

    blend = torch.clamp(
        blend,
        0.0,
        1.0,
    )

    lin_error = torch.sum(
        torch.square(
            robot.data.root_lin_vel_b
        ),
        dim=-1,
    )

    ang_error = torch.sum(
        torch.square(
            robot.data.root_ang_vel_b
        ),
        dim=-1,
    )

    low_speed = torch.exp(
        -(
            lin_error
            + 0.25 * ang_error
        )
        / (velocity_std**2)
    )

    return (
        blend
        * low_speed
    )


def getup_only_success_hold_reward(
    env,
    asset_cfg: SceneEntityCfg,
    target_joint_pos_by_name: dict[str, float],
    joint_std: float = 0.50,
    velocity_std: float = 0.65,
    min_root_height: float = 0.64,
    max_projected_gravity_z: float = -0.75,
) -> torch.Tensor:
    """Strong final-state reward: stand, match CSV pose, and remain stable.

    This is deliberately active only after the robot satisfies the standing
    condition. It gives the standalone task a clear terminal-like attractor
    without terminating the episode.
    """

    robot: Articulation = env.scene[
        asset_cfg.name
    ]

    standing = standing_mask(
        env,
        SceneEntityCfg("robot"),
        min_root_height=min_root_height,
        max_projected_gravity_z=max_projected_gravity_z,
    ).float()

    joint_pos = robot.data.joint_pos[
        :,
        asset_cfg.joint_ids,
    ]

    target = torch.tensor(
        [
            target_joint_pos_by_name[name]
            for name in asset_cfg.joint_names
        ],
        device=robot.device,
        dtype=joint_pos.dtype,
    ).unsqueeze(0)

    joint_error = torch.mean(
        torch.square(
            joint_pos
            - target
        ),
        dim=-1,
    )

    pose_score = torch.exp(
        -joint_error
        / (joint_std**2)
    )

    lin_error = torch.sum(
        torch.square(
            robot.data.root_lin_vel_b
        ),
        dim=-1,
    )

    ang_error = torch.sum(
        torch.square(
            robot.data.root_ang_vel_b
        ),
        dim=-1,
    )

    stable_score = torch.exp(
        -(
            lin_error
            + 0.25 * ang_error
        )
        / (velocity_std**2)
    )

    return (
        standing
        * pose_score
        * stable_score
    )
