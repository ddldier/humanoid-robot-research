# Whole Body Tracking V2 - Enhanced Reward System

这是[BeyondMimic Whole Body Tracking](https://github.com/HybridRobotics/whole_body_tracking)项目的**增强版本**，包含了重要的奖励函数和事件系统改进。

## 🆕 与V1版本的主要区别

### 核心改进

**服务器1版本 (whole_body_tracking_custom):**
- rewards.py: 82行 (基础版本)
- 基础的运动跟踪奖励

**服务器2版本 (这个仓库):**
- rewards.py: **1091行** (增强版本) ⭐
- reward_new.py: 31,203字节 (实验性新奖励)
- 大幅扩展的奖励函数系统

### 新增奖励函数

#### 1. 恢复状态奖励系统
- `_ensure_recovery_state()` - 恢复状态保证
- `_recovery_active_mask()` - 恢复激活掩码
- `recovery_step_sequence_reward()` - 恢复步序奖励

#### 2. 稳定性和平衡奖励
- `quiet_double_support_exp()` - 安静双支撑奖励
- `tail_balance_exp()` - 尾部平衡奖励
- `tail_feet_slip_l2()` - 防止脚部滑动

#### 3. 运动控制奖励
- `leg_joint_velocity_outside_recovery_l2()` - 腿部关节速度控制
- `lateral_root_velocity_deadzone_l2()` - 侧向速度死区控制

#### 4. 原始运动跟踪奖励（保留）
- `motion_global_anchor_position_error_exp()` - 全局锚点位置误差
- `motion_global_anchor_orientation_error_exp()` - 全局锚点方向误差
- `motion_relative_body_position_error_exp()` - 相对身体位置误差
- `motion_relative_body_orientation_error_exp()` - 相对身体方向误差
- `motion_global_body_linear_velocity_error_exp()` - 全局线速度误差
- `motion_global_body_angular_velocity_error_exp()` - 全局角速度误差
- `feet_contact_time()` - 脚部接触时间

### 文件变化对比

| 文件 | 服务器1 | 服务器2 | 说明 |
|------|---------|---------|------|
| rewards.py | 82行 | **1,091行** | 大幅扩展 |
| events.py | 833行 | 818行 | 略有调整 |
| reward_new.py | - | **新增** | 实验性奖励 |
| event_new.py | - | **新增** | 实验性事件 |

### 新增备份文件（用于追溯）
- `rewards.py.before_robust` - robust特性添加前的版本
- `events.py.before_robust` - robust特性添加前的版本
- `commands.py.before_one_shot_push` - one-shot push特性前的版本

## 技术改进

### 1. 恢复机制增强
为R5机器人设计的恢复状态辅助函数，支持：
- 自动恢复状态检测
- 恢复缩放奖励
- 稳定支撑奖励
- 事件驱动的支撑转移

### 2. 中文注释
新版本包含详细的中文注释，便于理解：
```python
# 描述: 定义机器人运动跟踪和恢复训练所使用的奖励函数。
#       包含原始的运动跟踪奖励（位置、姿态、速度误差的指数奖励），
#       以及为 R5 机器人设计的恢复状态辅助函数
```

### 3. 模块化设计
- 清晰的功能分区
- 辅助函数封装
- 更好的代码可读性

## 使用建议

如果你需要：
- **基础运动跟踪** → 使用 whole_body_tracking_custom (服务器1版本)
- **高级恢复和稳定性控制** → 使用这个仓库 (服务器2版本)
- **实验新奖励函数** → 查看 reward_new.py 和 event_new.py

## 相关项目

- [whole_body_tracking_custom](https://github.com/ddldier/whole_body_tracking_custom) - 服务器1基础版本
- [G9_tail_direct_v2_handoff](https://github.com/ddldier/G9_tail_direct_v2_handoff) - G9部署项目
- [g9_deployment_tools](https://github.com/ddldier/g9_deployment_tools) - 部署工具集

## 原项目

**上游项目**: [HybridRobotics/whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking)

## 许可证

MIT License (继承自原项目)
