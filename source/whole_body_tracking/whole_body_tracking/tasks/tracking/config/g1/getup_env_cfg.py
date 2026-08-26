from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import whole_body_tracking.tasks.tracking.mdp as mdp

from .robust_env_cfg import (
    FEET,
    G1FlatWoStateEstimationRobustEnvCfg,
    RobustEventCfg,
    RobustRewardsCfg,
)


# =============================================================================
# Get-up rewards
# =============================================================================


@configclass
class GetUpRewardsCfg(RobustRewardsCfg):
    """Original Robust rewards + fallen-only get-up shaping."""

    getup_height = RewTerm(
        func=mdp.getup_height_progress,
        weight=2.00,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot"
            ),
            "floor_height": 0.25,
            "target_height": (
                mdp.GETUP_DEFAULT_ROOT_HEIGHT
            ),
        },
    )

    getup_orientation = RewTerm(
        func=mdp.getup_orientation_to_default_exp,
        weight=2.00,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot"
            ),
            "target_projected_gravity": (
                mdp.GETUP_DEFAULT_PROJECTED_GRAVITY
            ),
            "std": 0.55,
        },
    )

    getup_default_pose = RewTerm(
        func=mdp.getup_default_joint_pose_exp,
        weight=1.00,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=(
                    mdp.GETUP_DEFAULT_JOINT_NAMES
                ),
            ),
            "target_joint_pos_by_name": (
                mdp.GETUP_DEFAULT_JOINT_POS_BY_NAME
            ),
            "std": 0.65,
            "pose_start_height": 0.35,
            "target_height": (
                mdp.GETUP_DEFAULT_ROOT_HEIGHT
            ),
        },
    )

    getup_feet_support = RewTerm(
        func=mdp.getup_feet_support_reward,
        weight=0.80,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=FEET,
            ),
            "asset_cfg": SceneEntityCfg(
                "robot"
            ),
            "contact_threshold": 10.0,
            "support_start_height": 0.40,
            "target_height": (
                mdp.GETUP_DEFAULT_ROOT_HEIGHT
            ),
        },
    )

    getup_stabilize = RewTerm(
        func=mdp.getup_stabilize_near_stand_exp,
        weight=0.50,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot"
            ),
            "start_height": 0.52,
            "target_height": (
                mdp.GETUP_DEFAULT_ROOT_HEIGHT
            ),
            "velocity_std": 1.0,
        },
    )


# =============================================================================
# Events
# =============================================================================


@configclass
class GetUpEventCfg(RobustEventCfg):
    """Keep the inherited Robust events.

    Fallen initialization is intentionally NOT an EventTerm(mode="reset").
    It is applied after MotionCommand reset by G1PunchGetUpEnv._reset_idx().
    """

    pass


# =============================================================================
# Environment
# =============================================================================


@configclass
class G1FlatWoStateEstimationPunchGetUpEnvCfg(
    G1FlatWoStateEstimationRobustEnvCfg
):
    """ONE 154-D policy: punch tracking while standing + get-up while fallen."""

    rewards: GetUpRewardsCfg = GetUpRewardsCfg()
    events: GetUpEventCfg = GetUpEventCfg()

    # -------------------------------------------------------------------------
    # Phase-1 reset distribution.
    #
    # Start with 25% fallen samples instead of immediately forcing 50%.
    # This keeps most rollouts close to model_46500's original distribution
    # while the critic learns the new get-up value targets.
    #
    # After this is numerically stable and get-up starts learning, increase:
    #   0.25 -> 0.40 -> 0.50
    # -------------------------------------------------------------------------

    fallen_reset_probability: float = 0.25

    fallen_root_height_range: tuple[
        float,
        float,
    ] = (
        0.50,
        0.58,
    )

    fallen_xy_range: tuple[
        float,
        float,
    ] = (
        -0.10,
        0.10,
    )

    fallen_joint_noise_range: tuple[
        float,
        float,
    ] = (
        -0.12,
        0.12,
    )

    fallen_orientation_noise: float = 0.10

    def __post_init__(self):
        super().__post_init__()

        # ---------------------------------------------------------------------
        # 1) Keep the original 12 s episode for the first stable fine-tune.
        #
        # Do not simultaneously change task distribution AND return horizon.
        # Once get-up is stable, 16 s can be tested later.
        # ---------------------------------------------------------------------

        self.episode_length_s = 12.0

        # ---------------------------------------------------------------------
        # 2) Standing-only WBT tracking.
        #
        # We preserve the original weights and params and replace funcs only.
        # ---------------------------------------------------------------------

        self.rewards.motion_global_anchor_pos.func = (
            mdp.standing_motion_global_anchor_position_error_exp
        )

        self.rewards.motion_global_anchor_ori.func = (
            mdp.standing_motion_global_anchor_orientation_error_exp
        )

        self.rewards.motion_body_pos.func = (
            mdp.standing_motion_relative_body_position_error_exp
        )

        self.rewards.motion_body_ori.func = (
            mdp.standing_motion_relative_body_orientation_error_exp
        )

        self.rewards.motion_body_lin_vel.func = (
            mdp.standing_motion_global_body_linear_velocity_error_exp
        )

        self.rewards.motion_body_ang_vel.func = (
            mdp.standing_motion_global_body_angular_velocity_error_exp
        )

        self.rewards.lower_body_pos.func = (
            mdp.standing_motion_relative_body_position_error_exp
        )

        self.rewards.lower_body_ori.func = (
            mdp.standing_motion_relative_body_orientation_error_exp
        )

        self.rewards.lower_body_lin_vel.func = (
            mdp.standing_motion_global_body_linear_velocity_error_exp
        )

        self.rewards.lower_body_ang_vel.func = (
            mdp.standing_motion_global_body_angular_velocity_error_exp
        )

        self.rewards.quiet_double_support.func = (
            mdp.standing_quiet_double_support_exp
        )

        self.rewards.leg_joint_velocity_outside_recovery.func = (
            mdp.standing_leg_joint_velocity_outside_recovery_l2
        )

        self.rewards.lateral_root_velocity.func = (
            mdp.standing_lateral_root_velocity_deadzone_l2
        )

        self.rewards.tail_balance.func = (
            mdp.standing_tail_balance_exp
        )

        self.rewards.tail_feet_slip.func = (
            mdp.standing_tail_feet_slip_l2
        )

        # Old lateral-push stepping sequence is unrelated to floor get-up.
        self.rewards.recovery_step_sequence = None

        # ---------------------------------------------------------------------
        # 3) GetUp-specific numerical guard for action-rate reward.
        #
        # Keep the ORIGINAL reward weight from RobustRewardsCfg.
        # Only replace the unbounded function with a bounded equivalent.
        # ---------------------------------------------------------------------

        self.rewards.action_rate_l2.func = (
            mdp.safe_action_rate_l2
        )

        self.rewards.action_rate_l2.params = {
            "delta_clip": 3.0,
            "total_clip": 100.0,
        }

        # ---------------------------------------------------------------------
        # 4) Ground contact is necessary during get-up.
        # ---------------------------------------------------------------------

        self.rewards.undesired_contacts = None

        # ---------------------------------------------------------------------
        # 5) Falling must NOT terminate a get-up episode.
        # ---------------------------------------------------------------------

        self.terminations.anchor_pos = None
        self.terminations.anchor_ori = None
        self.terminations.ee_body_pos = None

        # ---------------------------------------------------------------------
        # 6) Phase 1: no extra knock-down pushes yet.
        # ---------------------------------------------------------------------

        self.events.lateral_recovery_push = None
        self.events.attack_reaction = None

        # ---------------------------------------------------------------------
        # 7) Action safety.
        #
        # JointPositionActionCfg officially supports per-joint clipping.
        # This prevents a transient OOD fallen-state actor output from sending
        # an absurd position target into the simulator.
        #
        # This does NOT change the action dimension.
        # ---------------------------------------------------------------------

        self.actions.joint_pos.clip = {
            ".*": (
                -3.0,
                3.0,
            )
        }

        # ---------------------------------------------------------------------
        # 8) Observation value clipping.
        #
        # Observation DIMENSIONS remain exactly unchanged (policy = 154-D).
        # We only prevent extreme post-contact velocities / raw previous action
        # values from feeding huge numbers into the old model_46500 networks.
        # ---------------------------------------------------------------------

        self.observations.policy.base_ang_vel.clip = (
            -20.0,
            20.0,
        )

        self.observations.policy.joint_vel.clip = (
            -30.0,
            30.0,
        )

        self.observations.policy.actions.clip = (
            -3.0,
            3.0,
        )

        self.observations.critic.base_lin_vel.clip = (
            -20.0,
            20.0,
        )

        self.observations.critic.base_ang_vel.clip = (
            -20.0,
            20.0,
        )

        self.observations.critic.joint_vel.clip = (
            -30.0,
            30.0,
        )

        self.observations.critic.actions.clip = (
            -3.0,
            3.0,
        )
