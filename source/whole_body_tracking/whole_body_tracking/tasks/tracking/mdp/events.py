from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# =============================================================================
# Existing project events
# =============================================================================


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal[
        "uniform",
        "log_uniform",
        "gaussian",
    ] = "uniform",
):
    """Randomize calibrated joint default positions."""
    asset: Articulation = env.scene[
        asset_cfg.name
    ]

    # Save nominal value for export.
    asset.data.default_joint_pos_nominal = torch.clone(
        asset.data.default_joint_pos[0]
    )

    if env_ids is None:
        env_ids = torch.arange(
            env.scene.num_envs,
            device=asset.device,
        )

    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)
    else:
        joint_ids = torch.tensor(
            asset_cfg.joint_ids,
            dtype=torch.int,
            device=asset.device,
        )

    if pos_distribution_params is not None:
        pos = (
            asset.data.default_joint_pos
            .to(asset.device)
            .clone()
        )

        pos = _randomize_prop_by_op(
            pos,
            pos_distribution_params,
            env_ids,
            joint_ids,
            operation=operation,
            distribution=distribution,
        )[env_ids][:, joint_ids]

        if (
            env_ids != slice(None)
            and joint_ids != slice(None)
        ):
            env_ids = env_ids[:, None]

        asset.data.default_joint_pos[
            env_ids,
            joint_ids,
        ] = pos

        env.action_manager.get_term(
            "joint_pos"
        )._offset[
            env_ids,
            joint_ids,
        ] = pos


def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[
        str,
        tuple[float, float],
    ],
    asset_cfg: SceneEntityCfg,
):
    """Randomize rigid-body CoM."""
    asset: Articulation = env.scene[
        asset_cfg.name
    ]

    if env_ids is None:
        env_ids = torch.arange(
            env.scene.num_envs,
            device="cpu",
        )
    else:
        env_ids = env_ids.cpu()

    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(
            asset.num_bodies,
            dtype=torch.int,
            device="cpu",
        )
    else:
        body_ids = torch.tensor(
            asset_cfg.body_ids,
            dtype=torch.int,
            device="cpu",
        )

    range_list = [
        com_range.get(
            key,
            (0.0, 0.0),
        )
        for key in [
            "x",
            "y",
            "z",
        ]
    ]

    ranges = torch.tensor(
        range_list,
        device="cpu",
    )

    rand_samples = math_utils.sample_uniform(
        ranges[:, 0],
        ranges[:, 1],
        (
            len(env_ids),
            3,
        ),
        device="cpu",
    ).unsqueeze(1)

    coms = (
        asset.root_physx_view
        .get_coms()
        .clone()
    )

    coms[
        :,
        body_ids,
        :3,
    ] += rand_samples

    asset.root_physx_view.set_coms(
        coms,
        env_ids,
    )


def phase_push_by_setting_velocity(
    env,
    env_ids,
    command_name: str,
    phase_range: tuple[
        float,
        float,
    ],
    probability: float,
    velocity_range: dict[
        str,
        tuple[float, float],
    ],
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot"
    ),
):
    """Apply one heading-frame punch reaction per sampled motion cycle."""
    asset: Articulation = env.scene[
        asset_cfg.name
    ]

    command: MotionCommand = (
        env.command_manager.get_term(
            command_name
        )
    )

    if env_ids is None:
        env_ids = torch.arange(
            env.scene.num_envs,
            device=asset.device,
            dtype=torch.long,
        )
    else:
        env_ids = torch.as_tensor(
            env_ids,
            device=asset.device,
            dtype=torch.long,
        )

    if env_ids.numel() == 0:
        return

    total_steps = max(
        int(
            command.motion.time_step_total
        )
        - 1,
        1,
    )

    phase = (
        command.time_steps[
            env_ids
        ].float()
        / float(total_steps)
    )

    in_phase = (
        (phase >= phase_range[0])
        & (phase <= phase_range[1])
    )

    candidates_mask = (
        in_phase
        & (
            ~command.punch_reaction_decided[
                env_ids
            ]
        )
    )

    candidate_env_ids = env_ids[
        candidates_mask
    ]

    if candidate_env_ids.numel() == 0:
        return

    command.punch_reaction_decided[
        candidate_env_ids
    ] = True

    if probability <= 0.0:
        return

    selected_mask = (
        torch.rand(
            candidate_env_ids.numel(),
            device=asset.device,
        )
        < probability
    )

    selected_env_ids = (
        candidate_env_ids[
            selected_mask
        ]
    )

    if selected_env_ids.numel() == 0:
        return

    keys = [
        "forward",
        "lateral",
        "vertical",
        "roll",
        "pitch",
        "yaw",
    ]

    range_list = [
        velocity_range.get(
            key,
            (0.0, 0.0),
        )
        for key in keys
    ]

    ranges = torch.tensor(
        range_list,
        device=asset.device,
        dtype=torch.float32,
    )

    delta_local = (
        math_utils.sample_uniform(
            ranges[:, 0],
            ranges[:, 1],
            (
                selected_env_ids.numel(),
                6,
            ),
            device=asset.device,
        )
    )

    delta_lin_heading = (
        delta_local[:, :3]
    )

    delta_ang_heading = (
        delta_local[:, 3:]
    )

    root_quat_w = (
        asset.data.root_quat_w[
            selected_env_ids
        ]
    )

    heading_quat = math_utils.yaw_quat(
        root_quat_w
    )

    delta_lin_w = math_utils.quat_apply(
        heading_quat,
        delta_lin_heading,
    )

    delta_ang_w = math_utils.quat_apply(
        heading_quat,
        delta_ang_heading,
    )

    root_velocity_w = (
        asset.data.root_vel_w[
            selected_env_ids
        ].clone()
    )

    root_velocity_w[:, :3] += (
        delta_lin_w
    )

    root_velocity_w[:, 3:] += (
        delta_ang_w
    )

    asset.write_root_velocity_to_sim(
        root_velocity_w,
        env_ids=selected_env_ids,
    )

    command.punch_reaction_applied[
        selected_env_ids
    ] = True


# =============================================================================
# NEW: curriculum lateral push + recovery-window bookkeeping
#
# This does NOT modify observations.
# It only:
#   1) applies a heading-frame lateral delta-v,
#   2) records when recovery started so rewards can evaluate support transition.
# =============================================================================


def curriculum_lateral_push_by_setting_velocity(
    env,
    env_ids,
    apply_probability: float = 0.85,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """R4-DV4 30k curriculum.

    Assumes PPO num_steps_per_env = 24.

    New-run curriculum:

        0 - 3000 iter:
            0.60 - 1.40 m/s

        3000 - 6000:
            0.90 - 1.80

        6000 - 9000:
            1.20 - 2.30

        9000 - 12000:
            1.50 - 2.80

        12000 - 15000:
            1.80 - 3.20

        15000 - 18000:
            2.10 - 3.60

        18000 - 22000:
            2.40 - 4.00

        22000 - 26000:
            high-intensity mixed distribution

        26000+:
            final full-range mixed distribution

    This function only changes training disturbances and recovery bookkeeping.
    It does NOT change the 154-D policy observation.
    """

    asset: Articulation = env.scene[asset_cfg.name]

    current_step = int(
        getattr(env, "common_step_counter", 0)
    )

    # ------------------------------------------------------------------
    # Resolve environment IDs.
    # ------------------------------------------------------------------

    if env_ids is None:
        env_ids = torch.arange(
            env.scene.num_envs,
            device=asset.device,
            dtype=torch.long,
        )
    else:
        env_ids = torch.as_tensor(
            env_ids,
            device=asset.device,
            dtype=torch.long,
        )

    if env_ids.numel() == 0:
        return

    # ------------------------------------------------------------------
    # Preserve some completely clean / no-push episodes.
    # ------------------------------------------------------------------

    apply_mask = (
        torch.rand(
            env_ids.numel(),
            device=asset.device,
        )
        < apply_probability
    )

    env_ids = env_ids[apply_mask]

    if env_ids.numel() == 0:
        return

    num_envs = env_ids.numel()

    magnitude = torch.empty(
        num_envs,
        device=asset.device,
    )

    # ------------------------------------------------------------------
    # 24 simulation/control steps collected per PPO iteration.
    #
    # 3000  -> 72,000
    # 6000  -> 144,000
    # 9000  -> 216,000
    # 12000 -> 288,000
    # 15000 -> 360,000
    # 18000 -> 432,000
    # 22000 -> 528,000
    # 26000 -> 624,000
    # 30000 -> 720,000
    # ------------------------------------------------------------------

    if current_step < 72_000:
        current_range = (0.60, 1.40)
        retain_range = (0.40, 0.80)

    elif current_step < 144_000:
        current_range = (0.90, 1.80)
        retain_range = (0.50, 1.00)

    elif current_step < 216_000:
        current_range = (1.20, 2.30)
        retain_range = (0.60, 1.20)

    elif current_step < 288_000:
        current_range = (1.50, 2.80)
        retain_range = (0.70, 1.40)

    elif current_step < 360_000:
        current_range = (1.80, 3.20)
        retain_range = (0.80, 1.60)

    elif current_step < 432_000:
        current_range = (2.10, 3.60)
        retain_range = (0.90, 1.80)

    elif current_step < 528_000:
        current_range = (2.40, 4.00)
        retain_range = (1.00, 2.00)

    else:
        current_range = None
        retain_range = None

    # ------------------------------------------------------------------
    # Standard curriculum stages:
    #
    # 75% = current challenge band
    # 25% = easier retained band
    #
    # This prevents catastrophic forgetting.
    # ------------------------------------------------------------------

    if current_range is not None:

        hard_mask = (
            torch.rand(
                num_envs,
                device=asset.device,
            )
            < 0.75
        )

        easy_mask = ~hard_mask

        if torch.any(hard_mask):
            magnitude[hard_mask].uniform_(
                current_range[0],
                current_range[1],
            )

        if torch.any(easy_mask):
            magnitude[easy_mask].uniform_(
                retain_range[0],
                retain_range[1],
            )

    # ------------------------------------------------------------------
    # 22000 - 26000:
    #
    # High-difficulty reinforcement.
    #
    # 15% : 0.6 - 1.4
    # 20% : 1.4 - 2.4
    # 25% : 2.4 - 3.2
    # 40% : 3.2 - 4.0
    # ------------------------------------------------------------------

    elif current_step < 624_000:

        r = torch.rand(
            num_envs,
            device=asset.device,
        )

        mask_1 = r < 0.15
        mask_2 = (r >= 0.15) & (r < 0.35)
        mask_3 = (r >= 0.35) & (r < 0.60)
        mask_4 = r >= 0.60

        if torch.any(mask_1):
            magnitude[mask_1].uniform_(
                0.60,
                1.40,
            )

        if torch.any(mask_2):
            magnitude[mask_2].uniform_(
                1.40,
                2.40,
            )

        if torch.any(mask_3):
            magnitude[mask_3].uniform_(
                2.40,
                3.20,
            )

        if torch.any(mask_4):
            magnitude[mask_4].uniform_(
                3.20,
                4.00,
            )

    # ------------------------------------------------------------------
    # 26000+:
    #
    # Final consolidation distribution.
    #
    # Still biased toward strong disturbances, but preserves everything.
    #
    # 15% : 0.6 - 1.2
    # 20% : 1.2 - 2.2
    # 25% : 2.2 - 3.2
    # 40% : 3.2 - 4.0
    # ------------------------------------------------------------------

    else:

        r = torch.rand(
            num_envs,
            device=asset.device,
        )

        mask_1 = r < 0.15
        mask_2 = (r >= 0.15) & (r < 0.35)
        mask_3 = (r >= 0.35) & (r < 0.60)
        mask_4 = r >= 0.60

        if torch.any(mask_1):
            magnitude[mask_1].uniform_(
                0.60,
                1.20,
            )

        if torch.any(mask_2):
            magnitude[mask_2].uniform_(
                1.20,
                2.20,
            )

        if torch.any(mask_3):
            magnitude[mask_3].uniform_(
                2.20,
                3.20,
            )

        if torch.any(mask_4):
            magnitude[mask_4].uniform_(
                3.20,
                4.00,
            )

    # ------------------------------------------------------------------
    # Random LEFT / RIGHT disturbance.
    # ------------------------------------------------------------------

    sign = torch.where(
        torch.rand(
            num_envs,
            device=asset.device,
        )
        < 0.5,

        -torch.ones(
            num_envs,
            device=asset.device,
        ),

        torch.ones(
            num_envs,
            device=asset.device,
        ),
    )

    delta_local = torch.zeros(
        num_envs,
        3,
        device=asset.device,
        dtype=asset.data.root_pos_w.dtype,
    )

    delta_local[:, 1] = (
        sign * magnitude
    )

    root_quat_w = asset.data.root_quat_w[
        env_ids
    ]

    heading_quat = math_utils.yaw_quat(
        root_quat_w
    )

    delta_world = math_utils.quat_apply(
        heading_quat,
        delta_local,
    )

    root_velocity_w = (
        asset.data.root_vel_w[
            env_ids
        ].clone()
    )

    root_velocity_w[:, :3] += delta_world

    asset.write_root_velocity_to_sim(
        root_velocity_w,
        env_ids=env_ids,
    )

    # ==================================================================
    # Recovery bookkeeping.
    # ==================================================================

    # ==================================================================
    # Recovery bookkeeping.
    #
    # IMPORTANT:
    # R3 reward code may already have created some of these tensors.
    # Therefore each tensor must be initialized independently.
    # Do NOT guard the entire block only with _wbt_recovery_start_step.
    # ==================================================================

    n = env.scene.num_envs

    if not hasattr(
        env,
        "_wbt_recovery_start_step",
    ):
        env._wbt_recovery_start_step = torch.full(
            (n,),
            -1,
            dtype=torch.long,
            device=asset.device,
        )

    if not hasattr(
        env,
        "_wbt_recovery_until_step",
    ):
        env._wbt_recovery_until_step = torch.full(
            (n,),
            -1,
            dtype=torch.long,
            device=asset.device,
        )

    if not hasattr(
        env,
        "_wbt_recovery_stage",
    ):
        env._wbt_recovery_stage = torch.zeros(
            n,
            dtype=torch.long,
            device=asset.device,
        )

    if not hasattr(
        env,
        "_wbt_recovery_last_foot",
    ):
        env._wbt_recovery_last_foot = torch.full(
            (n,),
            -1,
            dtype=torch.long,
            device=asset.device,
        )

    if not hasattr(
        env,
        "_wbt_recovery_step_count",
    ):
        env._wbt_recovery_step_count = torch.zeros(
            n,
            dtype=torch.long,
            device=asset.device,
        )

    if not hasattr(
        env,
        "_wbt_last_push_dv",
    ):
        env._wbt_last_push_dv = torch.zeros(
            n,
            dtype=torch.float32,
            device=asset.device,
        )


    recovery_window_s = torch.clamp(
        1.45 + 0.30 * magnitude,
        min=1.55,
        max=2.80,
    )

    recovery_steps = torch.ceil(
        recovery_window_s
        / float(env.step_dt)
    ).long()

    env._wbt_recovery_start_step[
        env_ids
    ] = current_step

    env._wbt_recovery_until_step[
        env_ids
    ] = (
        current_step
        + recovery_steps
    )

    env._wbt_recovery_stage[
        env_ids
    ] = 0

    env._wbt_recovery_last_foot[
        env_ids
    ] = -1

    env._wbt_recovery_step_count[
        env_ids
    ] = 0

    env._wbt_last_push_dv[
        env_ids
    ] = magnitude

