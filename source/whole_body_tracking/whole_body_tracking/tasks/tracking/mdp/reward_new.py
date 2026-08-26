# =============================================================================
# 文件: rewards.py
# 描述: 定义机器人运动跟踪和恢复训练所使用的奖励函数。
#       包含原始的运动跟踪奖励（位置、姿态、速度误差的指数奖励），
#       以及为 R5 机器人设计的恢复状态辅助函数、恢复缩放奖励、
#       稳定支撑奖励、事件驱动的支撑转移奖励和运动末期稳定项。
#       所有恢复相关的张量均为训练专用书签，不参与策略观测（154维输入不变）。
# =============================================================================

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
# 原始 WBT 运动跟踪奖励函数
# 这些函数计算机器人状态与参考运动之间的误差，并以指数形式转换为奖励。
# =============================================================================


def _get_body_indexes(
    command: MotionCommand,
    body_names: list[str] | None,
) -> list[int]:
    """
    根据身体名称列表，从命令中获取对应的身体索引列表。

    辅助函数，用于筛选需要计算奖励的身体部位。

    Args:
        command: 运动命令对象，包含身体名称配置。
        body_names: 需要筛选的身体名称列表，若为 None 则返回所有身体索引。

    Returns:
        身体索引列表（list[int]）。
    """
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
    """
    计算全局坐标系下锚点位置误差的指数奖励。
    奖励 = exp(-||anchor_pos - robot_anchor_pos||^2 / std^2)，
    值域在 (0, 1]，误差越小奖励越接近 1。
    Args:
        env: 强化学习环境实例。
        command_name: 命令名称，用于获取对应的 MotionCommand。
        std: 高斯核的标准差，控制奖励对误差的敏感度。
    Returns:
        形状为 (num_envs,) 的张量，每个环境一个奖励值。
    """
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
    """
    计算全局坐标系下锚点姿态误差的指数奖励。

    奖励 = exp(-(姿态误差角^2) / std^2)，
    姿态误差角通过四元数误差大小计算。

    Args:
        env: 环境实例。
        command_name: 命令名称。
        std: 标准差。

    Returns:
        奖励张量，形状 (num_envs,)。
    """
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
    """
    计算相对坐标系下身体位置误差的指数奖励（相对于锚点）。
    奖励 = exp(-mean(||body_pos_relative - robot_body_pos||^2) / std^2)，
    先计算每个身体的平方误差，再对所有身体求平均。
    Args:
        env: 环境实例。
        command_name: 命令名称。
        std: 标准差。
        body_names: 可选，指定计算哪些身体，若为 None 则包含所有身体。
    Returns:
        奖励张量，形状 (num_envs,)。
    """
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
    """
    计算相对坐标系下身体姿态误差的指数奖励（相对于锚点）。
    奖励 = exp(-mean(姿态误差角^2) / std^2)。
    Args:
        env: 环境实例。
        command_name: 命令名称。
        std: 标准差。
        body_names: 可选，指定身体。
    Returns:
        奖励张量，形状 (num_envs,)。
    """
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
    """
    计算全局坐标系下身体线速度误差的指数奖励。
    奖励 = exp(-mean(||body_lin_vel - robot_body_lin_vel||^2) / std^2)。
    Args:
        env: 环境实例。
        command_name: 命令名称。
        std: 标准差。
        body_names: 可选，指定身体。

    Returns:
        奖励张量，形状 (num_envs,)。
    """
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
    """
    计算全局坐标系下身体角速度误差的指数奖励。

    奖励 = exp(-mean(||body_ang_vel - robot_body_ang_vel||^2) / std^2)。

    Args:
        env: 环境实例。
        command_name: 命令名称。
        std: 标准差。
        body_names: 可选，指定身体。

    Returns:
        奖励张量，形状 (num_envs,)。
    """
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
    """
    计算脚部接触时间的奖励：当脚部在较短时间内从空中落地时给予奖励。
    使用接触传感器，计算每只脚上一次接触时间与当前时间的差值，
    若差值小于阈值，则奖励首次离地时间（first_air）的累加和。
    鼓励脚部快速触地，减少悬空时间。
    Args:
        env: 环境实例。
        sensor_cfg: 接触传感器的场景实体配置。
        threshold: 接触时间阈值（秒）。
    Returns:
        奖励张量，形状 (num_envs,)。
    """
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
# 恢复状态辅助函数
#
# 重要提示：
# 这些张量是环境上附加的训练专用书签，不属于观测，因此不会改变 154 维的策略输入。
# =============================================================================
def _ensure_recovery_state(
    env: ManagerBasedRLEnv,
) -> None:
    """
    确保所有恢复书签张量已存在。
    这些张量仅在训练期间使用，用于追踪恢复过程（如推挤后的恢复窗口、步态阶段等）。
    由于 events.py 和 rewards.py 可能以不同顺序访问恢复状态，每个张量都独立检查。
    注意：这些张量不参与策略观测。
    """
    robot = env.scene["robot"]
    num_envs = env.scene.num_envs
    device = robot.device
    # 恢复开始步数（-1 表示无激活恢复）
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

    # 恢复结束步数（动态窗口）
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

    # 恢复阶段（0~5，用于步态序列奖励）
    if not hasattr(
        env,
        "_wbt_recovery_stage",
    ):
        env._wbt_recovery_stage = torch.zeros(
            num_envs,
            dtype=torch.long,
            device=device,
        )

    # 旧版 R3 书签（保留以兼容）
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

    # R4 多步书签：记录上一次落足脚和步数计数
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

    # 上一次推挤的强度（用于动态窗口）
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
    """
    返回当前激活的恢复掩码和已逝去步数。
    优先使用 R4 课程事件生成的动态每环境恢复结束步数。
    若不存在，则回退到固定窗口（recovery_window_s）以保持兼容性。
    Args:
        env: 环境实例。
        recovery_window_s: 固定恢复窗口时长（秒），仅在动态窗口不可用时使用。
    Returns:
        active: 布尔张量，指示哪些环境处于恢复窗口内。
        elapsed_steps: 距离恢复开始已逝去的环境步数。
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
# 恢复窗口之外的稳定支撑奖励
#
# 这些项鼓励在非恢复期间保持稳固的双足支撑和直立姿态。
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
    """
    在恢复窗口之外奖励稳定的双足支撑。
    此奖励是防止“软腿”的主要项。
    正常状态要求：
        - 双脚着地
        - 躯干直立
        - 横向速度较小
    在恢复窗口内，此奖励被禁用，以便策略可以自由抬脚。
    Args:
        env: 环境实例。
        sensor_cfg: 接触传感器配置。
        asset_cfg: 机器人实体配置。
        recovery_window_s: 恢复窗口时长。
        contact_threshold: 接触力阈值（N），超过则认为接触。
        tilt_std: 倾斜角标准差（用于高斯核）。
        lateral_velocity_std: 横向速度标准差。

    Returns:
        奖励张量，形状 (num_envs,)。
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
    """
    在恢复窗口之外惩罚腿部关节速度的二次方（L2 惩罚）。
    在恢复窗口内，该惩罚关闭，以便策略可以进行快速的摆动步。
    窗口外，它抑制之前策略中出现的持续松腿摆动。

    Args:
        env: 环境实例。
        asset_cfg: 机器人实体配置。
        recovery_window_s: 恢复窗口时长。

    Returns:
        惩罚项张量（正值，可被奖励函数减掉），形状 (num_envs,)。
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
    """
    惩罚持续横向漂移，但避免在大推挤时该项过度主导奖励。
    计算横向速度绝对值与死区的差值，并截断到 max_excess 以防止梯度爆炸。
    Args:
        env: 环境实例。
        asset_cfg: 机器人实体配置。
        deadzone: 死区阈值（m/s），低于该值不惩罚。
        max_excess: 超出部分的截断上限。
    Returns:
        惩罚项张量，形状 (num_envs,)。
    """
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

    # 对于高达 4 m/s 的 delta-v，防止该项占主导
    excess = torch.clamp(
        excess,
        max=max_excess,
    )

    return torch.square(
        excess
    )


# =============================================================================
# 基于事件的支撑转移恢复奖励
#
# 使用阶段机奖励推挤后重复交替支撑转移（步态序列）。
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
    """
    奖励推挤后重复的交替支撑转移（步态序列）。

    该函数仅用于奖励计算，不控制机器人，也不属于策略观测。

    逻辑概述：
        - 小扰动：双脚直接恢复 → 完成
        - 大扰动：
            1. 第一只摆动脚抬起 → 获得抬脚奖励
            2. 该脚落地 → 获得落地奖励
            3. 如果仍不稳定，仅允许对侧脚开始下一步（阶段转移）
            4. 重复 L→R→L→R...
            5. 直到稳定（双足着地、横向速度小、躯干直立）

    最大奖励步数限制防止奖励利用，策略可以物理上走更多步，但额外步数不再产生奖励。

    Args:
        env: 环境实例。
        sensor_cfg: 接触传感器配置。
        asset_cfg: 机器人实体配置。
        recovery_window_s: 恢复窗口时长。
        contact_threshold: 接触力阈值。
        min_air_time: 最小空中时间（秒），用于确认有效抬脚。
        min_no_step_recovery_time: 无步恢复所需的最短时间（秒），小扰动时直接完成。
        stable_lateral_speed: 稳定横向速度阈值（m/s）。
        stable_tilt: 稳定倾斜阈值（弧度）。
        first_liftoff_bonus: 首次抬脚奖励。
        first_land_bonus: 首次落地奖励。
        alternate_liftoff_bonus: 交替抬脚奖励。
        alternate_land_bonus: 交替落地奖励。
        complete_hold_reward: 完成恢复后的持续奖励。
        max_rewarded_steps: 最大奖励步数。

    Returns:
        奖励张量，形状 (num_envs,)。
    """
    _ensure_recovery_state(env)

    robot = env.scene[
        asset_cfg.name
    ]

    # 确保 R4 张量存在（即使奖励在第一次课程事件前被评估）
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
    # 阶段 0：尚未开始步态
    # ============================================================

    stage0 = (
        active
        & (stage == 0)
    )

    # 小扰动可在不抬脚的情况下直接恢复
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

    # 检测第一次真正的摆动抬脚
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
    # 阶段 1：当前摆动脚已抬离地面，等待其落地
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

            # 每次落地后：要么恢复完成，要么等待对侧脚
            stage[
                landed_env_ids
            ] = 2

    # ============================================================
    # 阶段 2：上一步已落地
    # 如果稳定 → 完成；如果不稳定 → 仅允许对侧脚开始下一步
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

        # 假设只有两只脚：0 ↔ 1
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
    # 阶段 5：稳定恢复完成
    # 较早完成的恢复获得该奖励的剩余步数更多
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
# 运动末期稳定项
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
    """
    在运动末期（相位 >= start_phase）奖励直立和低速恢复。

    重要：与旧版 R1 不同，此处不乘以双足接触条件，因此合法的摆动步不会被置零。

    Args:
        env: 环境实例。
        command_name: 命令名称。
        sensor_cfg: 接触传感器配置（此处未使用，保留接口）。
        asset_cfg: 机器人实体配置。
        start_phase: 激活此奖励的起始相位（0~1）。
        contact_threshold: 接触阈值（未使用）。
        tilt_std: 倾斜标准差。
        velocity_std: 速度标准差。

    Returns:
        奖励张量，形状 (num_envs,)。
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
    """
    在运动末期（相位 >= start_phase）惩罚脚部滑动。

    只有当前接触的脚才会被惩罚，摆动脚不受影响。

    Args:
        env: 环境实例。
        command_name: 命令名称。
        sensor_cfg: 接触传感器配置。
        asset_cfg: 机器人实体配置。
        start_phase: 激活此惩罚的起始相位。
        contact_threshold: 接触力阈值。

    Returns:
        惩罚项张量（正值），形状 (num_envs,)。
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