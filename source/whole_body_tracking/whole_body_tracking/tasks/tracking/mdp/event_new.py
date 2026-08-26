# =============================================================================
# 文件: events.py
# 描述: 定义训练环境中使用的各种随机化事件和扰动事件。
#       包括关节默认位置随机化、刚体质心随机化、基于运动相位的速度脉冲扰动，
#       以及针对 R5 机器人的课程式侧向推挤（附带恢复窗口书签）。
#       所有恢复相关的张量均为训练专用书签，不参与策略观测（154维输入不变）。
# =============================================================================

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
    """
    随机化关节默认位置（标定值）。
    该函数用于在环境初始化或重置时，为指定关节添加随机偏移，以增强策略的鲁棒性。
    它会修改 `asset.data.default_joint_pos` 的值，同时更新动作管理器中的偏移量，
    确保后续控制指令基于新的默认位置。
    Args:
        env: 基于管理器的环境实例。
        env_ids: 要应用随机化的环境索引，若为 None 则应用于所有环境。
        asset_cfg: 指定目标机器人（Articulation）的场景实体配置。
        pos_distribution_params: 随机化参数（范围或标准差），格式为 (min, max) 或 (mean, std)，
            取决于 distribution 类型。若为 None 则不修改。
        operation: 随机化操作类型，'add'（加性）、'scale'（缩放）、'abs'（绝对值）。
        distribution: 随机分布类型，'uniform'、'log_uniform'、'gaussian'。
    Returns:
        无返回值，直接修改环境状态。
    """
    asset: Articulation = env.scene[
        asset_cfg.name
    ]

    # 保存名义默认位置（用于导出或重置）
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
        # 根据指定操作和分布，生成随机值并应用到选中的关节和环境
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

        # 更新默认关节位置
        asset.data.default_joint_pos[
            env_ids,
            joint_ids,
        ] = pos

        # 同步更新动作管理器中的偏移量，使控制目标跟随新的默认位置
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
    """
    随机化刚体质心（CoM）位置。
    为指定刚体在局部坐标系中添加随机偏移（x, y, z 方向），以模拟质量分布不确定性。
    该操作直接修改物理引擎中的质心数据。
    Args:
        env: 环境实例。
        env_ids: 要应用的环境索引，若为 None 则应用于所有环境。
        com_range: 字典，键为 'x', 'y', 'z'，值为对应方向的随机范围 (min, max)。
        asset_cfg: 指定目标机器人的场景实体配置。
    Returns:
        无返回值。
    """
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

    # 从字典中提取 x, y, z 的范围
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

    # 采样均匀分布偏移量
    rand_samples = math_utils.sample_uniform(
        ranges[:, 0],
        ranges[:, 1],
        (
            len(env_ids),
            3,
        ),
        device="cpu",
    ).unsqueeze(1)

    # 获取当前质心并添加偏移
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

    # 写回物理引擎
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
    """
    在运动周期的特定相位施加一次脉冲式速度扰动（常用于模拟碰撞或外部冲击）。

    该函数根据当前运动命令（如行走）的进度（相位）决定是否施加扰动。
    它会检查是否已经在该周期内决策过，避免重复施加，并记录施加状态。
    Args:
        env: 环境实例。
        env_ids: 要应用的环境索引。
        command_name: 运动命令的名称（用于获取 MotionCommand 实例）。
        phase_range: 扰动施加的相位区间 [min, max]，0~1 表示一个完整运动周期。
        probability: 在满足相位的环境中，实际施加扰动的概率。
        velocity_range: 字典，包含 'forward', 'lateral', 'vertical', 'roll', 'pitch', 'yaw'，
            对应局部坐标系下的线速度和角速度增量范围 (min, max)。
        asset_cfg: 指定目标机器人的场景实体配置。
    Returns:
        无返回值。
    """
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
    # 计算当前相位（步数比例）
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

    # 只对还未决策过的环境考虑施加
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

    # 标记已决策，防止重复
    command.punch_reaction_decided[
        candidate_env_ids
    ] = True

    if probability <= 0.0:
        return

    # 按概率选取实际施加扰动的环境
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

    # 从 velocity_range 提取各方向范围
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

    # 采样局部坐标系下的增量
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

    # 将局部增量转换到世界坐标系（基于当前航向角）
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

    # 累加到当前速度并写入仿真
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

    # 标记已施加扰动
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
    """
    R4-DV4 30k 课程式侧向推挤。
    该函数实现了针对 R5 机器人的训练课程，通过逐步增加侧向推挤强度，
    引导策略学会从大扰动中恢复。同时维护恢复状态记录，用于后续奖励计算。
    推挤强度随时间（训练步数）逐步增加，并在后期引入混合分布，增强泛化性。
    该函数仅修改训练时的扰动和恢复书签，不改变策略观测（154维）。
    Args:
        env: 环境实例。
        env_ids: 要应用推挤的环境索引，若为 None 则应用于所有环境。
        apply_probability: 每个环境中实际施加推挤的概率（0~1）。
        asset_cfg: 指定目标机器人的场景实体配置。
    Returns:
        无返回值。
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 获取当前训练步数（由环境维护）
    current_step = int(
        getattr(env, "common_step_counter", 0)
    )

    # ------------------------------------------------------------------
    # 解析环境 ID
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
    # 按概率保留一部分环境不施加推挤，以保留基础步态训练
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
    # 课程阶段：根据当前训练步数决定推挤速度大小（横向速度增量 m/s）
    # 时间节点：3000 次 PPO 迭代 ≈ 72,000 环境步（步长 0.02s，24步/迭代）
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
    # 标准课程阶段：75% 当前挑战区间，25% 保留较易区间（避免遗忘）
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
    # 22000 - 26000 迭代（528,000 - 624,000 步）：
    # 高强度强化阶段，混合分布偏向强扰动
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
    # 26000+ 迭代（624,000+ 步）：
    # 最终巩固分布，仍偏向强扰动但覆盖更广
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
    # 随机左/右侧向推挤（在机器人航向坐标系中 +Y 或 -Y）
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

    # 局部坐标系下的速度增量（仅横向）
    delta_local = torch.zeros(
        num_envs,
        3,
        device=asset.device,
        dtype=asset.data.root_pos_w.dtype,
    )

    delta_local[:, 1] = (
        sign * magnitude
    )

    # 转换到世界坐标系
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

    # 叠加到当前速度
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
    # 恢复书签维护
    #
    # 重要提示：
    # R3 奖励代码可能已经创建了其中一些张量，
    # 因此每个张量必须独立初始化，不能仅以 _wbt_recovery_start_step 为守卫。
    # 这些张量用于记录推挤后的恢复窗口、阶段、步数等，供奖励函数使用。
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

    # 动态恢复窗口：较大推挤需要更多时间完成支撑转移
    recovery_window_s = torch.clamp(
        1.45 + 0.30 * magnitude,
        min=1.55,
        max=2.80,
    )

    recovery_steps = torch.ceil(
        recovery_window_s
        / float(env.step_dt)
    ).long()

    # 记录恢复开始和结束步数
    env._wbt_recovery_start_step[
        env_ids
    ] = current_step

    env._wbt_recovery_until_step[
        env_ids
    ] = (
        current_step
        + recovery_steps
    )

    # 重置恢复阶段和步数状态
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