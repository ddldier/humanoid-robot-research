from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import whole_body_tracking.tasks.tracking.mdp as mdp

from .getup_env_cfg import (
    GetUpEventCfg,
    G1FlatWoStateEstimationPunchGetUpEnvCfg,
)
from .robust_env_cfg import (
    FEET,
    RobustRewardsCfg,
)


# =============================================================================
# Reward set: standalone self-get-up
# =============================================================================


@configclass
class GetUpOnlyRewardsCfg(RobustRewardsCfg):
    """100% self-get-up objective plus the old generic regularizers.

    The task-specific WBT/Robust rewards inherited from RobustRewardsCfg are
    disabled in the environment __post_init__ below.
    """

    getup_height = RewTerm(
        func=mdp.getup_only_height_progress,
        weight=6.00,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "floor_height": 0.25,
            "target_height": mdp.GETUP_DEFAULT_ROOT_HEIGHT,
        },
    )

    getup_orientation = RewTerm(
        func=mdp.getup_only_orientation_to_default_exp,
        weight=4.00,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_projected_gravity": (
                mdp.GETUP_DEFAULT_PROJECTED_GRAVITY
            ),
            "std": 0.55,
        },
    )

    getup_default_pose = RewTerm(
        func=mdp.getup_only_default_joint_pose_exp,
        weight=2.00,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=mdp.GETUP_DEFAULT_JOINT_NAMES,
            ),
            "target_joint_pos_by_name": (
                mdp.GETUP_DEFAULT_JOINT_POS_BY_NAME
            ),
            "std": 0.65,
            "pose_start_height": 0.35,
            "target_height": mdp.GETUP_DEFAULT_ROOT_HEIGHT,
        },
    )

    getup_feet_support = RewTerm(
        func=mdp.getup_only_feet_support_reward,
        weight=2.00,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=FEET,
            ),
            "asset_cfg": SceneEntityCfg("robot"),
            "contact_threshold": 10.0,
            "support_start_height": 0.38,
            "target_height": mdp.GETUP_DEFAULT_ROOT_HEIGHT,
        },
    )

    getup_stabilize = RewTerm(
        func=mdp.getup_only_stabilize_exp,
        weight=1.00,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "start_height": 0.50,
            "target_height": mdp.GETUP_DEFAULT_ROOT_HEIGHT,
            "velocity_std": 1.0,
        },
    )

    getup_success_hold = RewTerm(
        func=mdp.getup_only_success_hold_reward,
        weight=8.00,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=mdp.GETUP_DEFAULT_JOINT_NAMES,
            ),
            "target_joint_pos_by_name": (
                mdp.GETUP_DEFAULT_JOINT_POS_BY_NAME
            ),
            "joint_std": 0.50,
            "velocity_std": 0.65,
            "min_root_height": 0.64,
            "max_projected_gravity_z": -0.75,
        },
    )


# =============================================================================
# Environment
# =============================================================================


@configclass
class G1FlatWoStateEstimationGetUpOnlyEnvCfg(
    G1FlatWoStateEstimationPunchGetUpEnvCfg
):
    """Standalone 154-D self-get-up training task.

    Training:
        100% fallen reset
        -> raise root
        -> restore orientation
        -> feet support
        -> CSV default joint pose
        -> stable standing hold

    There is no punch tracking reward in this task.
    Observation/action shapes stay identical to model_46500.
    """

    rewards: GetUpOnlyRewardsCfg = GetUpOnlyRewardsCfg()
    events: GetUpEventCfg = GetUpEventCfg()

    # 100% of training resets start fallen.
    fallen_reset_probability: float = 1.00

    fallen_root_height_range: tuple[float, float] = (
        0.40,
        0.48,
    )

    fallen_xy_range: tuple[float, float] = (
        -0.15,
        0.15,
    )

    fallen_joint_noise_range: tuple[float, float] = (
        -0.20,
        0.20,
    )

    def __post_init__(self):
        super().__post_init__()

        # Keep the first standalone phase at 12 s.
        self.episode_length_s = 12.0

        # =====================================================================
        # Disable ALL punch/WBT/Robust task-specific rewards.
        #
        # Generic physical regularizers inherited from RobustRewardsCfg remain:
        #   action_rate_l2
        #   joint_limit
        #   joint_acceleration
        #   joint_torque
        #   termination_penalty
        # =====================================================================

        self.rewards.motion_global_anchor_pos = None
        self.rewards.motion_global_anchor_ori = None

        self.rewards.motion_body_pos = None
        self.rewards.motion_body_ori = None
        self.rewards.motion_body_lin_vel = None
        self.rewards.motion_body_ang_vel = None

        self.rewards.lower_body_pos = None
        self.rewards.lower_body_ori = None
        self.rewards.lower_body_lin_vel = None
        self.rewards.lower_body_ang_vel = None

        self.rewards.quiet_double_support = None
        self.rewards.recovery_step_sequence = None
        self.rewards.leg_joint_velocity_outside_recovery = None
        self.rewards.lateral_root_velocity = None
        self.rewards.tail_balance = None
        self.rewards.tail_feet_slip = None

        # Allow hands/knees/forearms/torso to contact ground during get-up.
        self.rewards.undesired_contacts = None

        # Fall-related WBT terminations stay disabled.
        self.terminations.anchor_pos = None
        self.terminations.anchor_ori = None
        self.terminations.ee_body_pos = None

        # No knockdown/push curriculum in the standalone primitive.
        self.events.lateral_recovery_push = None
        self.events.attack_reaction = None
        self.events.push_robot = None
