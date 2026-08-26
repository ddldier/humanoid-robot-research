from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# =============================================================================
# Original WBT motion-tracking rewards
# =============================================================================


def _get_body_indexes(
    command: MotionCommand,
    body_names: list[str] | None,
) -> list[int]:
    return [
        i
        for i, name in enumerate(command.cfg.body_names)
        if (body_names is None) or (name in body_names)
    ]


def motion_global_anchor_position_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(
        command_name
    )
    error = torch.sum(
        torch.square(
            command.anchor_pos_w
            - command.robot_anchor_pos_w
        ),
        dim=-1,
    )
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(
        command_name
    )
    error = (
        quat_error_magnitude(
            command.anchor_quat_w,
            command.robot_anchor_quat_w,
        )
        ** 2
    )
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    body_names: list[str] | None = None,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(
        command_name
    )
    body_indexes = _get_body_indexes(
        command,
        body_names,
    )
    error = torch.sum(
        torch.square(
            command.body_pos_relative_w[:, body_indexes]
            - command.robot_body_pos_w[:, body_indexes]
        ),
        dim=-1,
    )
    return torch.exp(
        -error.mean(-1) / std**2
    )


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    body_names: list[str] | None = None,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(
        command_name
    )
    body_indexes = _get_body_indexes(
        command,
        body_names,
    )
    error = (
        quat_error_magnitude(
            command.body_quat_relative_w[:, body_indexes],
            command.robot_body_quat_w[:, body_indexes],
        )
        ** 2
    )
    return torch.exp(
        -error.mean(-1) / std**2
    )


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    body_names: list[str] | None = None,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(
        command_name
    )
    body_indexes = _get_body_indexes(
        command,
        body_names,
    )
    error = torch.sum(
        torch.square(
            command.body_lin_vel_w[:, body_indexes]
            - command.robot_body_lin_vel_w[:, body_indexes]
        ),
        dim=-1,
    )
    return torch.exp(
        -error.mean(-1) / std**2
    )


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    body_names: list[str] | None = None,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(
        command_name
    )
    body_indexes = _get_body_indexes(
        command,
        body_names,
    )
    error = torch.sum(
        torch.square(
            command.body_ang_vel_w[:, body_indexes]
            - command.robot_body_ang_vel_w[:, body_indexes]
        ),
        dim=-1,
    )
    return torch.exp(
        -error.mean(-1) / std**2
    )


def feet_contact_time(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[
        sensor_cfg.name
    ]
    first_air = contact_sensor.compute_first_air(
        env.step_dt,
        env.physics_dt,
    )[:, sensor_cfg.body_ids]
    last_contact_time = (
        contact_sensor.data.last_contact_time[
            :, sensor_cfg.body_ids
        ]
    )
    reward = torch.sum(
        (last_contact_time < threshold)
        * first_air,
        dim=-1,
    )
    return reward


# =============================================================================
# Recovery-state helpers
#
# IMPORTANT:
# These tensors are training-only bookkeeping attached to the environment.
# They are NOT observations and do NOT change the 154-D actor input.
# =============================================================================


def _ensure_recovery_state(
    env: ManagerBasedRLEnv,
) -> None:
    """Ensure all recovery bookkeeping tensors exist.

    These tensors are training-only state.
    They are NOT policy observations and do not change the 154-D actor input.

    Each tensor is checked independently because events.py and rewards.py
    may access recovery state in different orders.
    """

    robot = env.scene["robot"]

    num_envs = env.scene.num_envs
    device = robot.device

    if not hasattr(
        env,
        "_wbt_recovery_start_step",
    ):
        env._wbt_recovery_start_step = torch.full(
            (num_envs,),
            -1,
            dtype=torch.long,
            device=device,
        )

    if not hasattr(
        env,
        "_wbt_recovery_until_step",
    ):
        env._wbt_recovery_until_step = torch.full(
            (num_envs,),
            -1,
            dtype=torch.long,
            device=device,
        )

    if not hasattr(
        env,
        "_wbt_recovery_stage",
    ):
        env._wbt_recovery_stage = torch.zeros(
            num_envs,
            dtype=torch.long,
            device=device,
        )

    # Legacy R3 bookkeeping.
    # Keeping it is harmless and avoids compatibility problems.
    if not hasattr(
        env,
        "_wbt_recovery_first_foot",
    ):
        env._wbt_recovery_first_foot = torch.full(
            (num_envs,),
            -1,
            dtype=torch.long,
            device=device,
        )

    # R4 multi-step bookkeeping.
    if not hasattr(
        env,
        "_wbt_recovery_last_foot",
    ):
        env._wbt_recovery_last_foot = torch.full(
            (num_envs,),
            -1,
            dtype=torch.long,
            device=device,
        )

    if not hasattr(
        env,
        "_wbt_recovery_step_count",
    ):
        env._wbt_recovery_step_count = torch.zeros(
            num_envs,
            dtype=torch.long,
            device=device,
        )

    if not hasattr(
        env,
        "_wbt_last_push_dv",
    ):
        env._wbt_last_push_dv = torch.zeros(
            num_envs,
            dtype=torch.float32,
            device=device,
        )



def _recovery_active_mask(
    env: ManagerBasedRLEnv,
    recovery_window_s: float = 1.40,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return active recovery mask.

    Prefer the dynamic per-env recovery end time generated by the R4
    curriculum event. Fall back to the old fixed window for compatibility.
    """

    _ensure_recovery_state(env)

    current_step = int(
        getattr(env, "common_step_counter", 0)
    )

    start_step = env._wbt_recovery_start_step

    elapsed_steps = (
        torch.full_like(
            start_step,
            current_step,
        )
        - start_step
    )

    if hasattr(
        env,
        "_wbt_recovery_until_step",
    ):
        until_step = (
            env._wbt_recovery_until_step
        )

        active = (
            (start_step >= 0)
            & (current_step <= until_step)
        )

    else:
        window_steps = max(
            1,
            int(round(
                recovery_window_s
                / float(env.step_dt)
            )),
        )

        active = (
            (start_step >= 0)
            & (elapsed_steps >= 0)
            & (elapsed_steps <= window_steps)
        )

    return active, elapsed_steps



# =============================================================================
# Stable support outside the recovery window
# =============================================================================


def quiet_double_support_exp(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    recovery_window_s: float = 1.40,
    contact_threshold: float = 10.0,
    tilt_std: float = 0.22,
    lateral_velocity_std: float = 0.22,
) -> torch.Tensor:
    """Reward a firm, quiet two-foot stance OUTSIDE recovery windows.

    This is the main anti-"soft legs" term.

    Normal state:
        both feet should support the robot,
        torso should be upright,
        lateral root velocity should be small.

    During the short recovery window:
        this reward is disabled so the policy is free to lift a foot.
    """
    active, _ = _recovery_active_mask(
        env,
        recovery_window_s,
    )

    robot = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[
        sensor_cfg.name
    ]

    contact_force = (
        contact_sensor.data.net_forces_w[
            :, sensor_cfg.body_ids
        ]
    )

    contact_state = (
        torch.linalg.vector_norm(
            contact_force,
            dim=-1,
        )
        > contact_threshold
    )

    both_feet_contact = torch.all(
        contact_state,
        dim=-1,
    ).float()

    tilt_error = torch.sum(
        torch.square(
            robot.data.projected_gravity_b[:, :2]
        ),
        dim=-1,
    )

    upright_reward = torch.exp(
        -tilt_error
        / (tilt_std**2)
    )

    lateral_velocity = (
        robot.data.root_lin_vel_b[:, 1]
    )

    lateral_reward = torch.exp(
        -torch.square(lateral_velocity)
        / (lateral_velocity_std**2)
    )

    outside_recovery = (~active).float()

    return (
        outside_recovery
        * both_feet_contact
        * upright_reward
        * lateral_reward
    )


def leg_joint_velocity_outside_recovery_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    recovery_window_s: float = 1.40,
) -> torch.Tensor:
    """Penalize restless leg motion only outside recovery windows.

    During a recovery window the term turns off, so rapid swing steps remain
    possible. Outside recovery it discourages the continuous loose-leg sway
    seen in the previous policy.
    """
    active, _ = _recovery_active_mask(
        env,
        recovery_window_s,
    )

    robot = env.scene[asset_cfg.name]

    joint_vel = robot.data.joint_vel[
        :, asset_cfg.joint_ids
    ]

    penalty = torch.mean(
        torch.square(joint_vel),
        dim=-1,
    )

    return (
        (~active).float()
        * penalty
    )


def lateral_root_velocity_deadzone_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    deadzone: float = 0.15,
    max_excess: float = 1.50,
) -> torch.Tensor:
    """Penalize persistent lateral drift without exploding at huge pushes."""

    robot = env.scene[
        asset_cfg.name
    ]

    lateral_velocity = torch.abs(
        robot.data.root_lin_vel_b[:, 1]
    )

    excess = torch.relu(
        lateral_velocity
        - deadzone
    )

    # Important for dv up to 4 m/s:
    # prevent this single term from dominating the reward scale.
    excess = torch.clamp(
        excess,
        max=max_excess,
    )

    return torch.square(
        excess
    )


# =============================================================================
# Event-driven support-transition recovery reward
# =============================================================================


def recovery_step_sequence_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    recovery_window_s: float = 1.40,
    contact_threshold: float = 10.0,
    min_air_time: float = 0.05,
    min_no_step_recovery_time: float = 0.20,

    stable_lateral_speed: float = 0.25,
    stable_tilt: float = 0.18,

    first_liftoff_bonus: float = 0.20,
    first_land_bonus: float = 0.50,

    alternate_liftoff_bonus: float = 0.30,
    alternate_land_bonus: float = 0.65,

    complete_hold_reward: float = 1.00,

    max_rewarded_steps: int = 6,
) -> torch.Tensor:
    """Reward repeated alternating support transitions after a push.

    This is reward bookkeeping only.
    It does NOT control the robot and is NOT part of policy observations.

    Logic:

        small disturbance
            -> both feet recover directly
            -> complete

        large disturbance
            -> first swing foot lifts
            -> lands

        if still unstable
            -> ONLY the opposite foot is eligible for next step reward

        after it lands
            -> if still unstable
            -> previous foot becomes eligible again

        repeat:
            L -> R -> L -> R ...

        until:
            both feet contact
            lateral velocity is small
            torso is upright

    Maximum rewarded steps prevents reward hacking.
    The policy may physically take more steps if needed, but extra steps no
    longer produce transition bonuses.
    """

    _ensure_recovery_state(env)

    robot = env.scene[
        asset_cfg.name
    ]

    # Make sure new R4 tensors exist even if reward is evaluated before
    # the first curriculum event.
    n = env.scene.num_envs
    device = robot.device

    if not hasattr(
        env,
        "_wbt_recovery_last_foot",
    ):
        env._wbt_recovery_last_foot = torch.full(
            (n,),
            -1,
            dtype=torch.long,
            device=device,
        )

    if not hasattr(
        env,
        "_wbt_recovery_step_count",
    ):
        env._wbt_recovery_step_count = torch.zeros(
            n,
            dtype=torch.long,
            device=device,
        )

    active, elapsed_steps = (
        _recovery_active_mask(
            env,
            recovery_window_s,
        )
    )

    reward = torch.zeros(
        n,
        device=device,
        dtype=robot.data.root_pos_w.dtype,
    )

    if not torch.any(active):
        return reward

    contact_sensor: ContactSensor = (
        env.scene.sensors[
            sensor_cfg.name
        ]
    )

    contact_force = (
        contact_sensor.data.net_forces_w[
            :,
            sensor_cfg.body_ids,
        ]
    )

    contact_state = (
        torch.linalg.vector_norm(
            contact_force,
            dim=-1,
        )
        > contact_threshold
    )

    current_air_time = (
        contact_sensor.data.current_air_time[
            :,
            sensor_cfg.body_ids,
        ]
    )

    both_feet_contact = torch.all(
        contact_state,
        dim=-1,
    )

    lateral_speed = torch.abs(
        robot.data.root_lin_vel_b[:, 1]
    )

    tilt = torch.sqrt(
        torch.sum(
            torch.square(
                robot.data.projected_gravity_b[
                    :, :2
                ]
            ),
            dim=-1,
        )
    )

    stable = (
        both_feet_contact
        & (
            lateral_speed
            <= stable_lateral_speed
        )
        & (
            tilt
            <= stable_tilt
        )
    )

    stage = env._wbt_recovery_stage
    last_foot = env._wbt_recovery_last_foot
    step_count = env._wbt_recovery_step_count

    min_no_step_steps = max(
        1,
        int(round(
            min_no_step_recovery_time
            / float(env.step_dt)
        )),
    )

    # ============================================================
    # Stage 0:
    # no step yet
    # ============================================================

    stage0 = (
        active
        & (stage == 0)
    )

    # Small perturbation can recover without stepping.
    no_step_complete = (
        stage0
        & (
            elapsed_steps
            >= min_no_step_steps
        )
        & stable
    )

    stage[
        no_step_complete
    ] = 5

    # Detect first real swing foot.
    stage0 = (
        active
        & (stage == 0)
    )

    air_mask = (
        current_air_time
        >= min_air_time
    )

    first_liftoff = (
        stage0
        & torch.any(
            air_mask,
            dim=-1,
        )
    )

    if torch.any(first_liftoff):
        env_ids = torch.nonzero(
            first_liftoff,
            as_tuple=False,
        ).squeeze(-1)

        chosen_foot = torch.argmax(
            current_air_time[
                env_ids
            ],
            dim=-1,
        )

        last_foot[
            env_ids
        ] = chosen_foot

        stage[
            env_ids
        ] = 1

        reward[
            env_ids
        ] += first_liftoff_bonus

    # ============================================================
    # Stage 1:
    # current swing foot is airborne.
    # Wait for it to land.
    # ============================================================

    stage1 = (
        active
        & (stage == 1)
    )

    if torch.any(stage1):
        env_ids = torch.nonzero(
            stage1,
            as_tuple=False,
        ).squeeze(-1)

        foot_ids = last_foot[
            env_ids
        ]

        landed = contact_state[
            env_ids,
            foot_ids,
        ]

        landed_env_ids = env_ids[
            landed
        ]

        if landed_env_ids.numel() > 0:

            old_count = step_count[
                landed_env_ids
            ].clone()

            step_count[
                landed_env_ids
            ] += 1

            first_step = (
                old_count == 0
            )

            later_step = (
                old_count > 0
            )

            if torch.any(first_step):
                ids = landed_env_ids[
                    first_step
                ]

                reward[
                    ids
                ] += first_land_bonus

            if torch.any(later_step):
                ids = landed_env_ids[
                    later_step
                ]

                reward[
                    ids
                ] += alternate_land_bonus

            # After every landing:
            # either recovery is complete,
            # or we wait for the OPPOSITE foot.
            stage[
                landed_env_ids
            ] = 2

    # ============================================================
    # Stage 2:
    # last step has landed.
    #
    # If stable -> finish.
    # If unstable -> only opposite foot can start next rewarded step.
    # ============================================================

    stage2 = (
        active
        & (stage == 2)
    )

    stage2_complete = (
        stage2
        & stable
    )

    stage[
        stage2_complete
    ] = 5

    stage2 = (
        active
        & (stage == 2)
        & (~stable)
        & (
            step_count
            < max_rewarded_steps
        )
    )

    if torch.any(stage2):
        env_ids = torch.nonzero(
            stage2,
            as_tuple=False,
        ).squeeze(-1)

        previous_foot = last_foot[
            env_ids
        ]

        # Exactly two selected feet:
        # 0 -> 1
        # 1 -> 0
        opposite_foot = (
            1 - previous_foot
        )

        opposite_airborne = (
            current_air_time[
                env_ids,
                opposite_foot,
            ]
            >= min_air_time
        )

        next_step_env_ids = env_ids[
            opposite_airborne
        ]

        if next_step_env_ids.numel() > 0:

            previous_for_selected = (
                last_foot[
                    next_step_env_ids
                ]
            )

            opposite_selected = (
                1
                - previous_for_selected
            )

            last_foot[
                next_step_env_ids
            ] = opposite_selected

            stage[
                next_step_env_ids
            ] = 1

            reward[
                next_step_env_ids
            ] += alternate_liftoff_bonus

    # ============================================================
    # Stage 5:
    # stable recovery complete.
    # Earlier recovery receives this reward for more remaining steps.
    # ============================================================

    complete = (
        active
        & (stage == 5)
        & stable
    )

    reward[
        complete
    ] += complete_hold_reward

    return reward



# =============================================================================
# Late-motion stabilization terms
# =============================================================================


def tail_balance_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    start_phase: float = 0.82,
    contact_threshold: float = 10.0,
    tilt_std: float = 0.25,
    velocity_std: float = 0.55,
) -> torch.Tensor:
    """Reward upright and low-speed recovery near the end of the motion.

    IMPORTANT:
    Unlike the old R1 implementation, this does NOT multiply by
    both_feet_contact. A legitimate swing step is therefore not zeroed out.
    """
    command: MotionCommand = env.command_manager.get_term(
        command_name
    )

    robot = env.scene[asset_cfg.name]

    total_steps = max(
        int(command.motion.time_step_total) - 1,
        1,
    )

    phase = (
        command.time_steps.float()
        / float(total_steps)
    )

    phase_gate = (
        phase >= start_phase
    ).float()

    tilt_error = torch.sum(
        torch.square(
            robot.data.projected_gravity_b[:, :2]
        ),
        dim=-1,
    )

    upright_reward = torch.exp(
        -tilt_error
        / tilt_std**2
    )

    linear_speed_error = torch.sum(
        torch.square(
            robot.data.root_lin_vel_b
        ),
        dim=-1,
    )

    angular_speed_error = torch.sum(
        torch.square(
            robot.data.root_ang_vel_b
        ),
        dim=-1,
    )

    low_speed_reward = torch.exp(
        -(
            linear_speed_error
            + 0.25 * angular_speed_error
        )
        / velocity_std**2
    )

    return (
        phase_gate
        * upright_reward
        * low_speed_reward
    )


def tail_feet_slip_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    start_phase: float = 0.90,
    contact_threshold: float = 10.0,
) -> torch.Tensor:
    """Penalize sliding feet during the final settling phase.

    Swinging feet are not penalized because the penalty is multiplied by
    current contact state.
    """
    command: MotionCommand = env.command_manager.get_term(
        command_name
    )

    robot = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[
        sensor_cfg.name
    ]

    total_steps = max(
        int(command.motion.time_step_total) - 1,
        1,
    )

    phase = (
        command.time_steps.float()
        / float(total_steps)
    )

    phase_gate = (
        phase >= start_phase
    ).float()

    contact_force = (
        contact_sensor.data.net_forces_w[
            :, sensor_cfg.body_ids
        ]
    )

    contact_state = (
        torch.linalg.vector_norm(
            contact_force,
            dim=-1,
        )
        > contact_threshold
    ).float()

    feet_velocity_xy = (
        robot.data.body_lin_vel_w[
            :, asset_cfg.body_ids, :2
        ]
    )

    slip_error = torch.sum(
        torch.square(
            feet_velocity_xy
        ),
        dim=-1,
    )

    slip_error = torch.sum(
        contact_state
        * slip_error,
        dim=-1,
    )

    return (
        phase_gate
        * slip_error
    )
