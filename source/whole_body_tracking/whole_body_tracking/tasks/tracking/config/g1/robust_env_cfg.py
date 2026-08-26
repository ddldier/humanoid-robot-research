from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import whole_body_tracking.tasks.tracking.mdp as mdp

from whole_body_tracking.tasks.tracking.tracking_env_cfg import (
    EventCfg,
    RewardsCfg,
)

from .flat_env_cfg import (
    G1FlatWoStateEstimationEnvCfg,
)


# =============================================================================
# Body groups
# =============================================================================

FEET = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
]

UPPER_BODY = [
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]

LOWER_BODY = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
]

# Regex patterns used only by the leg-velocity regularizer.
LEG_JOINTS = [
    ".*hip.*",
    ".*knee.*",
    ".*ankle.*",
]


# =============================================================================
# Rewards
# =============================================================================


@configclass
class RobustRewardsCfg(RewardsCfg):
    """154-D policy: firm stance + event-driven support-transition recovery.

    Design goal:
      - Normal state is intentionally stiff and quiet.
      - A short recovery window opens only after a known lateral push.
      - Small disturbances may recover without stepping.
      - If one foot steps and lands but the robot remains unstable, only the
        OTHER foot receives the second-step transition reward.
    """

    # -------------------------------------------------------------------------
    # Global anchor:
    # Stronger than the previous soft-leg experiments.
    # -------------------------------------------------------------------------

    motion_global_anchor_pos = RewTerm(
        func=mdp.motion_global_anchor_position_error_exp,
        weight=0.45,
        params={
            "command_name": "motion",
            "std": 0.3,
        },
    )

    motion_global_anchor_ori = RewTerm(
        func=mdp.motion_global_anchor_orientation_error_exp,
        weight=0.80,
        params={
            "command_name": "motion",
            "std": 0.4,
        },
    )

    # -------------------------------------------------------------------------
    # Upper body:
    # Keep punch quality.
    # -------------------------------------------------------------------------

    motion_body_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=1.00,
        params={
            "command_name": "motion",
            "std": 0.3,
            "body_names": UPPER_BODY,
        },
    )

    motion_body_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=1.00,
        params={
            "command_name": "motion",
            "std": 0.4,
            "body_names": UPPER_BODY,
        },
    )

    motion_body_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=0.80,
        params={
            "command_name": "motion",
            "std": 1.0,
            "body_names": UPPER_BODY,
        },
    )

    motion_body_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=0.80,
        params={
            "command_name": "motion",
            "std": 3.14,
            "body_names": UPPER_BODY,
        },
    )

    # -------------------------------------------------------------------------
    # Lower body:
    # Much stronger than the previous 0.25-0.60 settings.
    #
    # This is the main "make the legs firm again" change.
    # Still slightly below 1.0 so a real recovery step remains possible.
    # -------------------------------------------------------------------------

    lower_body_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=0.85,
        params={
            "command_name": "motion",
            "std": 0.3,
            "body_names": LOWER_BODY,
        },
    )

    lower_body_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=0.85,
        params={
            "command_name": "motion",
            "std": 0.4,
            "body_names": LOWER_BODY,
        },
    )

    lower_body_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=0.65,
        params={
            "command_name": "motion",
            "std": 1.0,
            "body_names": LOWER_BODY,
        },
    )

    lower_body_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=0.65,
        params={
            "command_name": "motion",
            "std": 3.14,
            "body_names": LOWER_BODY,
        },
    )

    # -------------------------------------------------------------------------
    # Normal-state support:
    # Outside recovery windows, require quiet two-foot support.
    # This term switches OFF for 1.4 s after a curriculum push.
    # -------------------------------------------------------------------------

    quiet_double_support = RewTerm(
        func=mdp.quiet_double_support_exp,
        weight=0.85,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=FEET,
            ),
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=FEET,
            ),

            # fallback only;
            # R4 event now supplies dynamic per-env recovery windows.
            "recovery_window_s": 1.40,

            "contact_threshold": 10.0,
            "tilt_std": 0.22,
            "lateral_velocity_std": 0.22,
        },
    )


    # -------------------------------------------------------------------------
    # Recovery sequence:
    # No continuous "capture position" reward anymore.
    #
    # Instead:
    # first foot lift -> land -> if still unstable -> OTHER foot lift -> land
    # -> stable double support.
    # -------------------------------------------------------------------------

    recovery_step_sequence = RewTerm(
        func=mdp.recovery_step_sequence_reward,
        weight=1.30,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=FEET,
            ),
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=FEET,
            ),

            "recovery_window_s": 1.40,

            "contact_threshold": 10.0,
            "min_air_time": 0.05,
            "min_no_step_recovery_time": 0.20,

            "stable_lateral_speed": 0.25,
            "stable_tilt": 0.18,

            "first_liftoff_bonus": 0.20,
            "first_land_bonus": 0.50,

            "alternate_liftoff_bonus": 0.30,
            "alternate_land_bonus": 0.65,

            "complete_hold_reward": 1.00,

            # 最多奖励 6 次交替 step。
            # 不代表 robot 不能走第 7 步，只是不再额外刷 step reward。
            "max_rewarded_steps": 6,
        },
    )


    # -------------------------------------------------------------------------
    # Stop continuous loose-leg motion OUTSIDE recovery.
    # -------------------------------------------------------------------------

    leg_joint_velocity_outside_recovery = RewTerm(
        func=mdp.leg_joint_velocity_outside_recovery_l2,
        weight=-0.015,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=LEG_JOINTS,
            ),
            "recovery_window_s": 1.40,
        },
    )

    # Persistent left/right drift is always bad, but a dead-zone preserves
    # small natural motion from the reference punch.
    lateral_root_velocity = RewTerm(
        func=mdp.lateral_root_velocity_deadzone_l2,
        weight=-0.12,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot"
            ),
            "deadzone": 0.15,
            "max_excess": 1.50,
        },
    )


    # -------------------------------------------------------------------------
    # Existing late-motion stabilization, now without a hard both-feet gate.
    # -------------------------------------------------------------------------

    tail_balance = RewTerm(
        func=mdp.tail_balance_exp,
        weight=0.65,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=FEET,
            ),
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=FEET,
            ),
            "start_phase": 0.82,
            "contact_threshold": 10.0,
            "tilt_std": 0.25,
            "velocity_std": 0.55,
        },
    )

    tail_feet_slip = RewTerm(
        func=mdp.tail_feet_slip_l2,
        weight=-0.10,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=FEET,
            ),
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=FEET,
            ),
            "start_phase": 0.90,
            "contact_threshold": 10.0,
        },
    )

    # Action-rate penalty is brought close to the original -0.05.
    # It is not fully restored so fast recovery actions are still possible.
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.04,
    )

    joint_acceleration = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.0e-7,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*"],
            )
        },
    )

    joint_torque = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2.0e-6,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*"],
            )
        },
    )

    termination_penalty = RewTerm(
        func=mdp.is_terminated,
        weight=-5.0,
    )


# =============================================================================
# Events
# =============================================================================


@configclass
class RobustEventCfg(EventCfg):
    """Corrective 2000-iteration curriculum.

    The first ~300 PPO iterations have NO lateral push so the inherited soft
    lower-body behavior can first recover a firm stance.

    Then disturbance magnitude increases automatically inside the same run.
    """

    # Narrow randomization for this corrective run.
    # We will widen it again after the support-transition behavior is healthy.
    body_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=".*",
            ),
            "mass_distribution_params": (
                0.95,
                1.05,
            ),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )

    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*"],
            ),
            "stiffness_distribution_params": (
                0.95,
                1.05,
            ),
            "damping_distribution_params": (
                0.95,
                1.05,
            ),
            "operation": "scale",
        },
    )

    # Automatic curriculum:
    #   0-300 iter    : no push
    #   300-800       : 0.35-0.55
    #   800-1400      : 0.50-0.75
    #   1400-2000     : 0.65-0.95
    #
    # The step thresholds assume PPO num_steps_per_env = 24.
    lateral_recovery_push = EventTerm(
        func=mdp.curriculum_lateral_push_by_setting_velocity,
        mode="interval",

        # 大扰动需要充分恢复时间。
        # 不要 2~4 秒又撞一下。
        interval_range_s=(
            6.0,
            9.0,
        ),

        params={
            # 15% scheduled events 不发生 push。
            "apply_probability": 0.85,

            "asset_cfg": SceneEntityCfg(
                "robot"
            ),
        },
    )


    # Keep a very small amount of the synthetic punch-reaction exposure so
    # yesterday's punch robustness is not completely forgotten.
    # Real contact feedback should be added in a later dedicated phase.
    attack_reaction = EventTerm(
        func=mdp.phase_push_by_setting_velocity,
        mode="interval",
        interval_range_s=(
            0.10,
            0.18,
        ),
        params={
            "command_name": "motion",
            "phase_range": (
                0.40,
                0.62,
            ),
            "probability": 0.05,
            "velocity_range": {
                "forward": (
                    -0.10,
                    -0.04,
                ),
                "lateral": (
                    -0.02,
                    0.02,
                ),
                "vertical": (
                    -0.01,
                    0.01,
                ),
                "roll": (
                    -0.02,
                    0.02,
                ),
                "pitch": (
                    -0.04,
                    0.02,
                ),
                "yaw": (
                    -0.03,
                    0.03,
                ),
            },
            "asset_cfg": SceneEntityCfg(
                "robot"
            ),
        },
    )


# =============================================================================
# Environment
# =============================================================================


@configclass
class G1FlatWoStateEstimationRobustEnvCfg(
    G1FlatWoStateEstimationEnvCfg
):
    """154-D WBT policy with firm support and support-transition recovery."""

    rewards: RobustRewardsCfg = RobustRewardsCfg()
    events: RobustEventCfg = RobustEventCfg()

    def __post_init__(self):
        super().__post_init__()

        # ---------------------------------------------------------------------
        # Slightly longer timeout for multi-step recovery.
        # ---------------------------------------------------------------------
        self.episode_length_s = 12.0

        # ---------------------------------------------------------------------
        # Corrective-run initialization:
        # much narrower than the old very aggressive random initialization.
        #
        # Reason:
        # the inherited checkpoint is already soft/unstable. We first want it
        # to relearn a clean nominal stance, not start every episode in a large
        # recovery state.
        # ---------------------------------------------------------------------

        self.commands.motion.pose_range = {
            "x": (
                -0.05,
                0.05,
            ),
            "y": (
                -0.05,
                0.05,
            ),
            "z": (
                -0.015,
                0.015,
            ),
            "roll": (
                -0.10,
                0.10,
            ),
            "pitch": (
                -0.10,
                0.10,
            ),
            "yaw": (
                -0.15,
                0.15,
            ),
        }

        self.commands.motion.velocity_range = {
            "x": (
                -0.30,
                0.30,
            ),
            "y": (
                -0.30,
                0.30,
            ),
            "z": (
                -0.10,
                0.10,
            ),
            "roll": (
                -0.30,
                0.30,
            ),
            "pitch": (
                -0.30,
                0.30,
            ),
            "yaw": (
                -0.45,
                0.45,
            ),
        }

        self.commands.motion.joint_position_range = (
            -0.08,
            0.08,
        )

        # Disable the old broad world-frame push.
        # Only the controlled heading-frame curriculum push is used here.
        self.events.push_robot = None

        # ---------------------------------------------------------------------
        # Termination:
        # tighter than the previous soft-leg experiment, but still looser than
        # the original baseline so a legitimate step is not killed too early.
        # ---------------------------------------------------------------------

        self.terminations.anchor_pos.params[
            "threshold"
        ] = 0.45

        self.terminations.anchor_ori.params[
            "threshold"
        ] = 1.15

        self.terminations.ee_body_pos.params[
            "threshold"
        ] = 0.50

        self.terminations.ee_body_pos.params[
            "body_names"
        ] = FEET
