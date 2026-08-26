from __future__ import annotations

import math

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from whole_body_tracking.tasks.tracking.mdp.rewards import (
    leg_joint_velocity_outside_recovery_l2,
    lateral_root_velocity_deadzone_l2,
    motion_global_anchor_orientation_error_exp,
    motion_global_anchor_position_error_exp,
    motion_global_body_angular_velocity_error_exp,
    motion_global_body_linear_velocity_error_exp,
    motion_relative_body_orientation_error_exp,
    motion_relative_body_position_error_exp,
    quiet_double_support_exp,
    tail_balance_exp,
    tail_feet_slip_l2,
)


# =============================================================================
# Default posture from stable_default_pos(1).csv
#
# Values below are copied from the CSV "mean" row supplied by the user.
# This is the target posture used ONLY by the get-up reward.
# It is NOT added to the policy observation.
# =============================================================================

GETUP_DEFAULT_ROOT_HEIGHT = 0.70613724

# CSV mean root quaternion, WXYZ:
# [0.98170048, 0.000165305, 0.190428227, -0.001164311]
#
# To avoid forcing a global yaw, get-up orientation is rewarded using the
# equivalent projected-gravity direction in the robot body frame.
GETUP_DEFAULT_PROJECTED_GRAVITY = (
    0.373887302,
    0.000118875,
    -0.927474135,
)

GETUP_DEFAULT_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

GETUP_DEFAULT_JOINT_POS = [
    -0.998841822,
    -0.999644101,
    0.000569874,
    0.269701540,
    -0.268931359,
    0.000285700,
    0.172225922,
    -0.168453321,
    -0.004152125,
    0.905094087,
    0.902647316,
    -0.923300922,
    -0.924926341,
    -0.256087214,
    -0.253162414,
    0.134258151,
    -0.132709429,
    -0.062892899,
    0.064963907,
    0.099398121,
    -0.101900935,
    -0.232871160,
    -0.232416227,
    -0.227710605,
    0.226605475,
    -0.237971634,
    -0.233040035,
    0.579359174,
    -0.579387486,
]

# Always align targets by JOINT NAME instead of assuming the CSV column order
# equals IsaacLab's internal joint-id order.
GETUP_DEFAULT_JOINT_POS_BY_NAME = dict(
    zip(
        GETUP_DEFAULT_JOINT_NAMES,
        GETUP_DEFAULT_JOINT_POS,
        strict=True,
    )
)


# =============================================================================
# Instantaneous training gates
#
# IMPORTANT:
# This is NOT a deployment state machine.
# There is no persistent mode and no policy switching.
#
# At every reward evaluation we look only at the CURRENT physical pose:
#
#   sufficiently high + sufficiently upright -> normal WBT tracking reward
#   otherwise                               -> get-up reward
#
# The deployed controller is still:
#
#   observation(154-D) -> ONE policy -> action
# =============================================================================


def _root_height(
    env,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Root height relative to each environment origin."""

    robot: Articulation = env.scene[
        asset_cfg.name
    ]

    return (
        robot.data.root_pos_w[:, 2]
        - env.scene.env_origins[:, 2]
    )


def standing_mask(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_root_height: float = 0.64,
    max_projected_gravity_z: float = -0.75,
) -> torch.Tensor:
    """Return True when the current pose is standing enough for normal WBT.

    projected_gravity_b[:, 2] is near -1 when upright.
    The threshold deliberately allows the user's default forward lean.
    """

    robot: Articulation = env.scene[
        asset_cfg.name
    ]

    height = _root_height(
        env,
        asset_cfg,
    )

    gravity_z = (
        robot.data.projected_gravity_b[:, 2]
    )

    return (
        (height >= min_root_height)
        & (
            gravity_z
            <= max_projected_gravity_z
        )
    )


def fallen_mask(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_root_height: float = 0.64,
    max_projected_gravity_z: float = -0.75,
) -> torch.Tensor:
    """Complement of standing_mask()."""

    return ~standing_mask(
        env,
        asset_cfg,
        min_root_height,
        max_projected_gravity_z,
    )


def _standing_float(
    env,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    return standing_mask(
        env,
        asset_cfg,
    ).float()


# =============================================================================
# Standing-only wrappers around the PRE-R5 robust tracking rewards.
#
# These wrappers do not change the original reward equations or weights.
# They only multiply them by 0 when the robot is physically fallen.
# =============================================================================


def standing_motion_global_anchor_position_error_exp(
    env,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    return (
        _standing_float(env, asset_cfg)
        * motion_global_anchor_position_error_exp(
            env,
            command_name,
            std,
        )
    )


def standing_motion_global_anchor_orientation_error_exp(
    env,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    return (
        _standing_float(env, asset_cfg)
        * motion_global_anchor_orientation_error_exp(
            env,
            command_name,
            std,
        )
    )


def standing_motion_relative_body_position_error_exp(
    env,
    command_name: str,
    std: float,
    body_names: list[str] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    return (
        _standing_float(env, asset_cfg)
        * motion_relative_body_position_error_exp(
            env,
            command_name,
            std,
            body_names,
        )
    )


def standing_motion_relative_body_orientation_error_exp(
    env,
    command_name: str,
    std: float,
    body_names: list[str] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    return (
        _standing_float(env, asset_cfg)
        * motion_relative_body_orientation_error_exp(
            env,
            command_name,
            std,
            body_names,
        )
    )


def standing_motion_global_body_linear_velocity_error_exp(
    env,
    command_name: str,
    std: float,
    body_names: list[str] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    return (
        _standing_float(env, asset_cfg)
        * motion_global_body_linear_velocity_error_exp(
            env,
            command_name,
            std,
            body_names,
        )
    )


def standing_motion_global_body_angular_velocity_error_exp(
    env,
    command_name: str,
    std: float,
    body_names: list[str] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    return (
        _standing_float(env, asset_cfg)
        * motion_global_body_angular_velocity_error_exp(
            env,
            command_name,
            std,
            body_names,
        )
    )


def standing_quiet_double_support_exp(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    recovery_window_s: float = 1.40,
    contact_threshold: float = 10.0,
    tilt_std: float = 0.22,
    lateral_velocity_std: float = 0.22,
) -> torch.Tensor:
    return (
        _standing_float(env, asset_cfg)
        * quiet_double_support_exp(
            env,
            sensor_cfg,
            asset_cfg,
            recovery_window_s,
            contact_threshold,
            tilt_std,
            lateral_velocity_std,
        )
    )


def standing_leg_joint_velocity_outside_recovery_l2(
    env,
    asset_cfg: SceneEntityCfg,
    recovery_window_s: float = 1.40,
) -> torch.Tensor:
    return (
        _standing_float(
            env,
            SceneEntityCfg("robot"),
        )
        * leg_joint_velocity_outside_recovery_l2(
            env,
            asset_cfg,
            recovery_window_s,
        )
    )


def standing_lateral_root_velocity_deadzone_l2(
    env,
    asset_cfg: SceneEntityCfg,
    deadzone: float = 0.15,
    max_excess: float = 1.50,
) -> torch.Tensor:
    return (
        _standing_float(env, asset_cfg)
        * lateral_root_velocity_deadzone_l2(
            env,
            asset_cfg,
            deadzone,
            max_excess,
        )
    )


def standing_tail_balance_exp(
    env,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    start_phase: float = 0.82,
    contact_threshold: float = 10.0,
    tilt_std: float = 0.25,
    velocity_std: float = 0.55,
) -> torch.Tensor:
    return (
        _standing_float(env, asset_cfg)
        * tail_balance_exp(
            env,
            command_name,
            sensor_cfg,
            asset_cfg,
            start_phase,
            contact_threshold,
            tilt_std,
            velocity_std,
        )
    )


def standing_tail_feet_slip_l2(
    env,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    start_phase: float = 0.90,
    contact_threshold: float = 10.0,
) -> torch.Tensor:
    return (
        _standing_float(env, asset_cfg)
        * tail_feet_slip_l2(
            env,
            command_name,
            sensor_cfg,
            asset_cfg,
            start_phase,
            contact_threshold,
        )
    )



# =============================================================================
# Numerical-safety regularization for the GetUp task
#
# IsaacLab's standard action_rate_l2 is:
#   sum((action - prev_action)^2)
#
# It is intentionally unbounded. During early get-up fine-tuning, fallen-state
# observations are far outside the old punch policy's training distribution,
# so a transient very large action can create a gigantic negative reward and
# overflow the critic target.  This wrapper keeps the same idea but bounds the
# penalty.  It is used ONLY by Punch-GetUp-v0.
# =============================================================================


def safe_action_rate_l2(
    env,
    delta_clip: float = 3.0,
    total_clip: float = 100.0,
) -> torch.Tensor:
    """Bounded action-rate L2 penalty for GetUp fine-tuning.

    Returns a non-negative penalty. The negative sign still comes from the
    existing reward weight in RobustRewardsCfg (e.g. -0.04).

    This does NOT change action dimension and does NOT change PPO settings.
    """

    delta = (
        env.action_manager.action
        - env.action_manager.prev_action
    )

    # Protect the reward/critic from NaN/Inf even if a transient bad rollout
    # happens. The action term itself is separately clipped in getup_env_cfg.py.
    delta = torch.nan_to_num(
        delta,
        nan=0.0,
        posinf=delta_clip,
        neginf=-delta_clip,
    )

    delta = torch.clamp(
        delta,
        min=-delta_clip,
        max=delta_clip,
    )

    penalty = torch.sum(
        torch.square(delta),
        dim=1,
    )

    return torch.clamp(
        penalty,
        min=0.0,
        max=total_clip,
    )


# =============================================================================
# Get-up rewards
#
# These rewards are exactly zero while the robot is standing.
# While fallen, the policy is guided toward:
#
#   ground -> increase root height -> recover target body orientation
#          -> bring joints toward stable_default_pos CSV
#          -> establish foot support -> settle
#
# When it becomes standing again, these rewards switch off and the ORIGINAL
# WBT tracking rewards immediately become active again.
# =============================================================================


def getup_height_progress(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    floor_height: float = 0.25,
    target_height: float = GETUP_DEFAULT_ROOT_HEIGHT,
) -> torch.Tensor:
    """Dense 0..1 reward for raising the root toward the CSV default height."""

    fallen = fallen_mask(
        env,
        asset_cfg,
    ).float()

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

    progress = torch.clamp(
        progress,
        0.0,
        1.0,
    )

    return (
        fallen
        * progress
    )


def getup_orientation_to_default_exp(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_projected_gravity: tuple[float, float, float] = (
        GETUP_DEFAULT_PROJECTED_GRAVITY
    ),
    std: float = 0.55,
) -> torch.Tensor:
    """Reward torso/root orientation toward the CSV default without fixing yaw."""

    robot: Articulation = env.scene[
        asset_cfg.name
    ]

    fallen = fallen_mask(
        env,
        asset_cfg,
    ).float()

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

    return (
        fallen
        * torch.exp(
            -error
            / (std**2)
        )
    )


def getup_default_joint_pose_exp(
    env,
    asset_cfg: SceneEntityCfg,
    target_joint_pos_by_name: dict[str, float],
    std: float = 0.65,
    pose_start_height: float = 0.35,
    target_height: float = GETUP_DEFAULT_ROOT_HEIGHT,
) -> torch.Tensor:
    """Pull joints toward the CSV default pose mainly during the later rise.

    When lying low on the floor, arms/legs remain relatively free to push,
    roll and brace. The default-pose attraction gets stronger as root height
    increases.
    """

    robot: Articulation = env.scene[
        asset_cfg.name
    ]

    fallen = fallen_mask(
        env,
        SceneEntityCfg("robot"),
    ).float()

    joint_pos = robot.data.joint_pos[
        :,
        asset_cfg.joint_ids,
    ]

    # asset_cfg.joint_names is the manager-resolved order corresponding to
    # asset_cfg.joint_ids. Build the target in exactly that same order.
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

    # Keep only weak pose guidance while lying down, then strengthen it as the
    # pelvis/root rises.
    pose_weight = (
        0.15
        + 0.85 * height_blend
    )

    return (
        fallen
        * pose_weight
        * torch.exp(
            -error
            / (std**2)
        )
    )


def getup_feet_support_reward(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_threshold: float = 10.0,
    support_start_height: float = 0.40,
    target_height: float = GETUP_DEFAULT_ROOT_HEIGHT,
) -> torch.Tensor:
    """Encourage one/two-foot support more strongly as the robot rises."""

    robot: Articulation = env.scene[
        asset_cfg.name
    ]

    fallen = fallen_mask(
        env,
        asset_cfg,
    ).float()

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

    support += (
        (count == 1).float()
        * 0.35
    )

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
        fallen
        * blend
        * support
    )


def getup_stabilize_near_stand_exp(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    start_height: float = 0.52,
    target_height: float = GETUP_DEFAULT_ROOT_HEIGHT,
    velocity_std: float = 1.0,
) -> torch.Tensor:
    """Reward low root velocity only late in the get-up, not while lying down."""

    robot: Articulation = env.scene[
        asset_cfg.name
    ]

    fallen = fallen_mask(
        env,
        asset_cfg,
    ).float()

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
        fallen
        * blend
        * low_speed
    )


# =============================================================================
# Reset distribution: normal tracking samples + fallen get-up samples
#
# IMPORTANT:
# This function is NOT registered as EventTerm(mode="reset").
#
# ManagerBasedRLEnv resets managers in an order where command_manager.reset()
# happens after reset events. The WBT MotionCommand writes the robot pose during
# command reset, so an EventTerm-based fallen reset can be overwritten.
#
# G1PunchGetUpEnv._reset_idx() calls this function AFTER super()._reset_idx().
# =============================================================================


def reset_mixed_tracking_and_fallen_after_command(
    env,
    env_ids,
    fallen_probability: float = 0.25,
    root_height_range: tuple[float, float] = (
        0.50,
        0.58,
    ),
    xy_range: tuple[float, float] = (
        -0.10,
        0.10,
    ),
    joint_noise_range: tuple[float, float] = (
        -0.12,
        0.12,
    ),
    orientation_noise: float = 0.10,
):
    """Overwrite a subset of environments with safe fallen poses.

    The other environments are left exactly as the original WBT command reset
    produced them, so they continue normal punch tracking.

    The fallen orientations are +/- 90 degrees, not 180 degrees:
        pitch +90 deg : one horizontal front/back family
        pitch -90 deg : the opposite front/back family
        roll  +90 deg : left/right side family
        roll  -90 deg : the opposite side family

    180-degree roll/pitch would place the humanoid upside-down rather than
    lying horizontally and can create severe ground penetration / impulses.

    Returns:
        Tensor of environment ids that were converted to fallen starts.
    """

    robot: Articulation = env.scene["robot"]

    if env_ids is None:
        env_ids = torch.arange(
            env.scene.num_envs,
            device=robot.device,
            dtype=torch.long,
        )
    else:
        env_ids = torch.as_tensor(
            env_ids,
            device=robot.device,
            dtype=torch.long,
        )

    if env_ids.numel() == 0:
        return torch.empty(
            0,
            dtype=torch.long,
            device=robot.device,
        )

    choose_fallen = (
        torch.rand(
            env_ids.numel(),
            device=robot.device,
        )
        < fallen_probability
    )

    fallen_ids = env_ids[choose_fallen]

    if fallen_ids.numel() == 0:
        return fallen_ids

    n = fallen_ids.numel()

    # -------------------------------------------------------------------------
    # Resolve CSV joints by NAME.
    # -------------------------------------------------------------------------

    joint_ids, joint_names = robot.find_joints(
        GETUP_DEFAULT_JOINT_NAMES,
        preserve_order=True,
    )

    target_joint_pos = torch.tensor(
        [
            GETUP_DEFAULT_JOINT_POS_BY_NAME[name]
            for name in joint_names
        ],
        device=robot.device,
        dtype=robot.data.joint_pos.dtype,
    )

    # -------------------------------------------------------------------------
    # Root state.
    # -------------------------------------------------------------------------

    root_state = robot.data.root_state_w[
        fallen_ids
    ].clone()

    env_origins = env.scene.env_origins[
        fallen_ids
    ]

    root_state[:, 0] = (
        env_origins[:, 0]
        + torch.empty(
            n,
            device=robot.device,
        ).uniform_(
            xy_range[0],
            xy_range[1],
        )
    )

    root_state[:, 1] = (
        env_origins[:, 1]
        + torch.empty(
            n,
            device=robot.device,
        ).uniform_(
            xy_range[0],
            xy_range[1],
        )
    )

    # Start somewhat above the floor.  The body can settle naturally on the
    # first physics steps without being spawned deeply inside the ground.
    root_state[:, 2] = (
        env_origins[:, 2]
        + torch.empty(
            n,
            device=robot.device,
        ).uniform_(
            root_height_range[0],
            root_height_range[1],
        )
    )

    pose_type = torch.randint(
        low=0,
        high=4,
        size=(n,),
        device=robot.device,
    )

    roll = torch.zeros(
        n,
        device=robot.device,
    )
    pitch = torch.zeros(
        n,
        device=robot.device,
    )
    yaw = torch.empty(
        n,
        device=robot.device,
    ).uniform_(
        -math.pi,
        math.pi,
    )

    noise = torch.empty(
        n,
        device=robot.device,
    ).uniform_(
        -orientation_noise,
        orientation_noise,
    )

    # Horizontal front/back families: +/- 90 deg pitch.
    pitch = torch.where(
        pose_type == 0,
        torch.full_like(
            pitch,
            math.pi / 2.0,
        ) + noise,
        pitch,
    )

    pitch = torch.where(
        pose_type == 1,
        torch.full_like(
            pitch,
            -math.pi / 2.0,
        ) + noise,
        pitch,
    )

    # Horizontal side families: +/- 90 deg roll.
    roll = torch.where(
        pose_type == 2,
        torch.full_like(
            roll,
            math.pi / 2.0,
        ) + noise,
        roll,
    )

    roll = torch.where(
        pose_type == 3,
        torch.full_like(
            roll,
            -math.pi / 2.0,
        ) + noise,
        roll,
    )

    root_state[:, 3:7] = (
        math_utils.quat_from_euler_xyz(
            roll,
            pitch,
            yaw,
        )
    )

    # Phase 1 begins almost static.  Learn floor -> stand before adding violent
    # knock-down velocities.
    root_state[:, 7:13] = 0.0

    robot.write_root_state_to_sim(
        root_state,
        env_ids=fallen_ids,
    )

    # -------------------------------------------------------------------------
    # Joint state close to the user's CSV default pose.
    # -------------------------------------------------------------------------

    target = target_joint_pos.unsqueeze(
        0
    ).repeat(
        n,
        1,
    )

    joint_pos = (
        target
        + torch.empty_like(
            target
        ).uniform_(
            joint_noise_range[0],
            joint_noise_range[1],
        )
    )

    limits = robot.data.soft_joint_pos_limits[
        fallen_ids
    ][
        :,
        joint_ids,
        :
    ]

    joint_pos = torch.clamp(
        joint_pos,
        min=limits[..., 0],
        max=limits[..., 1],
    )

    joint_vel = torch.zeros_like(
        joint_pos
    )

    robot.write_joint_state_to_sim(
        joint_pos,
        joint_vel,
        joint_ids=joint_ids,
        env_ids=fallen_ids,
    )

    # Training-only bookkeeping.  Not an observation.
    if not hasattr(
        env,
        "_getup_reset_fallen",
    ):
        env._getup_reset_fallen = torch.zeros(
            env.scene.num_envs,
            dtype=torch.bool,
            device=robot.device,
        )

    env._getup_reset_fallen[
        env_ids
    ] = False

    env._getup_reset_fallen[
        fallen_ids
    ] = True

    return fallen_ids
