# Humanoid Robot Research

人形机器人运动跟踪和控制研究项目，基于BeyondMimic框架。

## 📚 项目概述

这个仓库包含了人形机器人全身运动跟踪的完整研究代码，支持复杂动作学习（后空翻、飞踢、舞蹈等）和sim2real部署。

## 🔬 版本说明

本仓库包含两个主要版本：

### 主分支 (main) - V2增强版
- **rewards.py**: 1091行 - 扩展的奖励函数系统
- **新增功能**:
  - 恢复状态奖励系统（R5机器人）
  - 稳定性和平衡控制奖励
  - 实验性reward_new.py和event_new.py
  - 详细的中文注释
- **适用**: 高级研究、鲁棒性控制、生产部署

### V1分支 (v1-basic) - 基础版
- **rewards.py**: 82行 - 精简的基础版本
- **包含**: 7种动作库完整实验数据
- **适用**: 学习、基础研究、快速实验

## 🎯 主要功能

### 运动跟踪奖励函数

#### V2版本新增 (16个函数)
1. **恢复状态系统**
   - `_ensure_recovery_state()` - 恢复状态保证
   - `_recovery_active_mask()` - 恢复激活掩码
   - `recovery_step_sequence_reward()` - 恢复步序奖励

2. **稳定性控制**
   - `quiet_double_support_exp()` - 安静双支撑
   - `tail_balance_exp()` - 尾部平衡
   - `tail_feet_slip_l2()` - 防滑控制

3. **运动优化**
   - `leg_joint_velocity_outside_recovery_l2()` - 关节速度控制
   - `lateral_root_velocity_deadzone_l2()` - 侧向速度死区

#### 基础跟踪奖励
- `motion_global_anchor_position_error_exp()` - 全局锚点位置
- `motion_global_anchor_orientation_error_exp()` - 全局锚点方向
- `motion_relative_body_position_error_exp()` - 相对身体位置
- `motion_relative_body_orientation_error_exp()` - 相对身体方向
- `motion_global_body_linear_velocity_error_exp()` - 全局线速度
- `motion_global_body_angular_velocity_error_exp()` - 全局角速度

## 🚀 快速开始

### 环境要求
- Python 3.10
- NVIDIA Isaac Sim 4.5.0
- Isaac Lab 2.1.0
- PyTorch
- CUDA支持的GPU

### 安装
```bash
# 克隆仓库
git clone https://github.com/ddldier/humanoid-robot-research.git
cd humanoid-robot-research

# 安装依赖（需要先安装Isaac Lab）
# 详见Isaac Lab官方文档
```

### 训练
```bash
# 使用V2增强版训练
python scripts/rsl_rl/train.py --task=Isaac-Velocity-Flat-G1-v0

# 使用本地动作数据训练
python scripts/rsl_rl/train_local_motion.py --motion=your_motion.npz
```

### 评估
```bash
# 播放训练好的策略
python scripts/rsl_rl/play.py --task=Isaac-Velocity-Flat-G1-v0 --checkpoint=model.pt

# 导出ONNX模型
python scripts/rsl_rl/play_export_local_motion.py --checkpoint=model.pt
```

## 📁 项目结构

```
humanoid-robot-research/
├── source/                              # 源代码
│   └── whole_body_tracking/
│       └── tasks/tracking/
│           ├── mdp/                     # MDP定义
│           │   ├── rewards.py          # V2: 1091行奖励函数
│           │   ├── reward_new.py       # 实验性新奖励
│           │   ├── events.py           # 事件处理
│           │   ├── event_new.py        # 实验性新事件
│           │   ├── commands.py         # 命令生成
│           │   └── terminations.py     # 终止条件
│           └── config/                  # 环境配置
│               └── g1/                  # G1机器人配置
│                   ├── flat_env_cfg.py
│                   ├── flat_env_cfg_wo_state.py
│                   ├── robust_env_cfg.py
│                   ├── getup_env_cfg.py
│                   └── getup_only_env_cfg.py
├── scripts/                             # 训练和评估脚本
│   ├── rsl_rl/
│   │   ├── train.py                    # 训练脚本
│   │   ├── play.py                     # 播放脚本
│   │   ├── train_local_motion.py       # 本地动作训练
│   │   └── play_export_local_motion.py # 导出脚本
│   ├── csv_to_npz_g9_exactfps.py       # 动作转换工具
│   └── validate_motion_npz.py          # 动作验证工具
├── README.md                            # 本文件
└── README_V2.md                         # V2版本详细说明
```

## 🔬 研究成果

### 支持的动作
- 后空翻 (Backflip)
- 飞踢 (Flying Kick)
- 舞蹈动作 (Dance)
- 出拳 (Punch)
- 侧踢拳击 (Sidekick Boxing)
- G9右飞踢

### 技术特点
- Sim2Real部署流程
- Isaac Sim ↔ MuJoCo跨仿真器验证
- ONNX模型导出
- 无状态估计版本支持

## 📊 版本对比

| 特性 | V1基础版 | V2增强版 |
|------|----------|----------|
| rewards.py | 82行 | 1,091行 |
| 奖励函数 | 8个 | 16个 |
| 恢复系统 | 基础 | 增强 |
| 稳定性控制 | 基础 | 高级 |
| 中文注释 | 无 | 详细 |
| 实验功能 | 无 | reward_new.py, event_new.py |

## 📖 相关项目

- [humanoid-robot-deployment](https://github.com/ddldier/humanoid-robot-deployment) - 部署工具和资源
- [HybridRobotics/whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking) - 上游项目

## 📝 引用

如果使用了BeyondMimic相关内容，请引用：
```bibtex
@article{beyondmimic2024,
  title={BeyondMimic: Versatile Humanoid Control with Guided Diffusion},
  journal={arXiv preprint arXiv:2508.08241},
  year={2024}
}
```

## 📄 许可证

MIT License

## 👥 贡献者

- 王浩然 (ddldier)

---

**GitHub**: https://github.com/ddldier/humanoid-robot-research  
**相关**: [humanoid-robot-deployment](https://github.com/ddldier/humanoid-robot-deployment)
