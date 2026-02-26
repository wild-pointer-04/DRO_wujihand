# DRO-Grasp to WujiHand Deployment - 完整适配版

本项目提供了将DRO-Grasp模型完全适配到wujihand机械手原型机的部署方案。

## 📁 文件说明

### 核心文件

1. **dro_wujihand_deployment_complete.py** (主程序)
   - 完全集成DRO-Grasp的推理流程
   - 包含多进程架构：关键点订阅 → DRO推理 → 关节发布
   - 直接调用项目中的 `model/network.py`、`utils/hand_model.py` 等模块
   - 支持所有DRO-Grasp的checkpoint（model_shadowhand, model_3robots等）

2. **wujihand_config.py** (配置文件)
   - 完整的wujihand配置（从URDF自动提取）
   - 包含20个关节的名称和限制
   - 预设配置方案（default, fast, stable, debug）
   - 配置验证功能

3. **launch_wujihand_dro.sh** (启动脚本)
   - 交互式配置选择
   - 自动环境检查
   - 支持多种预设方案

### 参考文件（已分析）

4. **example_isaac_wujihand_new2.py** - Isaac Gym示例
5. **validate_wujihand.py** - 验证脚本
6. **hand_model.py** - 手部模型类
7. **controller.py** - 控制器逻辑

## 🚀 快速开始

### 前置要求

1. **DRO-Grasp项目**
   ```bash
   # 确保你有完整的DRO-Grasp项目
   git clone <your-dro-grasp-repo>
   cd DRO-Grasp
   
   # 下载模型checkpoint（如果还没有）
   bash scripts/download_ckpt.sh
   ```

2. **环境依赖**
   ```bash
   # Python包
   pip install torch numpy pytorch_kinematics trimesh
   pip install rclpy loguru tyro termcolor tqdm
   
   # ROS2（假设已安装）
   source /opt/ros/<your-distro>/setup.bash
   ```

### 方法1：使用启动脚本（推荐）

```bash
# 1. 设置DRO项目路径
export DRO_PROJECT_ROOT=/path/to/your/DRO-Grasp

# 2. 复制部署文件到DRO项目
cp dro_wujihand_deployment_complete.py $DRO_PROJECT_ROOT/
cp wujihand_config.py $DRO_PROJECT_ROOT/
cp launch_wujihand_dro.sh $DRO_PROJECT_ROOT/

# 3. 运行启动脚本
cd $DRO_PROJECT_ROOT
./launch_wujihand_dro.sh
```

启动脚本会引导你：
- 检查环境和依赖
- 选择配置方案（默认/快速/稳定/调试）
- 设置ROS2话题
- 启动部署

### 方法2：直接命令行

```bash
cd /path/to/DRO-Grasp

# 基本启动（使用默认配置）
python3 dro_wujihand_deployment_complete.py \
  --checkpoint-name model_3robots \
  --dro-project-root . \
  --landmark-topic /hotrack/landmarks \
  --joint-topic /hand_0/joint_commands

# 完整参数示例
python3 dro_wujihand_deployment_complete.py \
  --checkpoint-name model_3robots \
  --dro-project-root /home/user/DRO-Grasp \
  --device cuda \
  --landmark-topic /hotrack/landmarks \
  --joint-topic /hand_0/joint_commands \
  --publish-rate-hz 60 \
  --enable-smoothing \
  --ema-alpha 0.3 \
  --max-vel-rad 2.0
```

## 🎯 系统架构

```
┌──────────────────────┐
│   手部跟踪设备       │
│   (相机/传感器)      │
└──────────┬───────────┘
           │ PoseArray (21个3D点)
           ↓
┌──────────────────────────────────┐
│  ROS2: /hotrack/landmarks        │
└──────────┬───────────────────────┘
           │
           ↓
┌──────────────────────────────────┐
│  进程1: HandLandmarkSubscriber   │
│  - 订阅手部关键点                │
│  - 数据预处理                    │
└──────────┬───────────────────────┘
           │ Queue
           ↓
┌──────────────────────────────────┐
│  进程2: DRO推理                  │
│  ┌────────────────────────────┐ │
│  │ 1. 加载DRO网络             │ │
│  │    - create_network()      │ │
│  │    - load checkpoint       │ │
│  ├────────────────────────────┤ │
│  │ 2. 创建WujiHand模型        │ │
│  │    - create_hand_model()   │ │
│  │    - 加载URDF和点云        │ │
│  ├────────────────────────────┤ │
│  │ 3. 推理循环                │ │
│  │    - 获取机器人/物体点云   │ │
│  │    - 网络推理 (DRO)        │ │
│  │    - 多边定位              │ │
│  │    - 姿态计算              │ │
│  │    - 逆运动学优化          │ │
│  └────────────────────────────┘ │
└──────────┬───────────────────────┘
           │ Queue (20维关节角度)
           ↓
┌──────────────────────────────────┐
│  进程3: 关节发布                 │
│  - EMA平滑滤波                   │
│  - 速度限制                      │
│  - 关节限制检查                  │
└──────────┬───────────────────────┘
           │ JointState (20关节)
           ↓
┌──────────────────────────────────┐
│  ROS2: /hand_0/joint_commands    │
└──────────┬───────────────────────┘
           │
           ↓
┌──────────────────────┐
│  WujiHand控制器      │
│  (原型机硬件)        │
└──────────────────────┘
```

## 🔧 核心实现说明

### 1. DRO模型集成

代码直接导入DRO-Grasp项目模块：

```python
from model.network import create_network
from utils.multilateration import multilateration
from utils.se3_transform import compute_link_pose
from utils.optimization import process_transform, create_problem, optimization
from utils.hand_model import create_hand_model
```

完整的DRO推理流程：

```python
# A. 网络预测距离场
dro = network(robot_pc, object_pc)['dro']

# B. 多边定位
mlat_pc = multilateration(dro, object_pc)

# C. 姿态计算
transform, _ = compute_link_pose(hand.links_pc, mlat_pc, is_train=False)
optim_transform = process_transform(hand.pk_chain, transform)

# D. 逆运动学优化
layer = create_problem(hand.pk_chain, optim_transform.keys())
predict_q = optimization(hand.pk_chain, layer, initial_q, optim_transform)
```

### 2. WujiHand配置

从URDF自动提取的20个关节：

| 手指 | 关节 | 名称 | 功能 | 范围(rad) |
|------|------|------|------|-----------|
| Thumb | J1 | finger1_joint1 | 根部旋转 | [-0.04, 1.65] |
| Thumb | J2 | finger1_joint2 | 侧摆 | [-0.17, 0.93] |
| Thumb | J3 | finger1_joint3 | 近端弯曲 | [-0.49, 1.63] |
| Thumb | J4 | finger1_joint4 | 远端弯曲 | [-0.49, 1.63] |
| Index-Pinky | J1 | finger[2-5]_joint1 | 根部 | [-0.33, 1.64] |
| Index-Pinky | J2 | finger[2-5]_joint2 | 侧摆 | [-0.50, 0.50] |
| Index-Pinky | J3 | finger[2-5]_joint3 | 近端 | [-0.49, 1.63] |
| Index-Pinky | J4 | finger[2-5]_joint4 | 远端 | [-0.49, 1.63] |

### 3. 数据流转

**输入格式**（从手部跟踪）：
```python
# PoseArray消息，21个Pose
# 每个Pose包含position.xyz (单位：毫米)
landmark_data: np.array(21, 3)  # 转换为米
```

**输出格式**（到机械手控制器）：
```python
# JointState消息
msg.name = [20个关节名称]
msg.position = [20个关节角度（弧度）]
```

## ⚙️ 参数调优指南

### 预设配置方案

| 方案 | 频率 | EMA | 速度限制 | 特点 | 适用场景 |
|------|------|-----|----------|------|----------|
| **Default** | 60Hz | 0.3 | 2.0 rad/s | 平衡 | 通用场景 |
| **Fast** | 120Hz | 0.5 | 3.0 rad/s | 快速响应 | 游戏控制、快速动作 |
| **Stable** | 30Hz | 0.2 | 1.5 rad/s | 稳定平滑 | 精细操作、学习调试 |
| **Debug** | 10Hz | 0.1 | 0.5 rad/s | 慢速测试 | 开发调试 |

### 关键参数说明

#### 1. 发布频率 (`--publish-rate-hz`)

```bash
# 影响响应速度和平滑度
--publish-rate-hz 30   # 基础控制
--publish-rate-hz 60   # 推荐（平衡）
--publish-rate-hz 120  # 高性能（需要GPU快速推理）
```

#### 2. EMA平滑系数 (`--ema-alpha`)

```bash
# 控制平滑程度
--ema-alpha 0.1  # 强平滑，响应慢
--ema-alpha 0.3  # 推荐（平衡）
--ema-alpha 0.5  # 弱平滑，响应快
--ema-alpha 1.0  # 无平滑
```

#### 3. 速度限制 (`--max-vel-rad`)

```bash
# 防止突然大幅度运动
--max-vel-rad 1.0  # 保守
--max-vel-rad 2.0  # 推荐
--max-vel-rad 3.0  # 激进
--max-vel-rad 0    # 禁用速度限制
```

#### 4. 模型选择 (`--checkpoint-name`)

```bash
--checkpoint-name model_3robots        # 推荐（泛化性好）
--checkpoint-name model_shadowhand     # ShadowHand专用
--checkpoint-name model_allegro        # Allegro专用
--checkpoint-name model_3robots_partial  # 部分视图训练
```

## 🐛 调试方法

### 1. 验证配置

```bash
cd /path/to/DRO-Grasp
python3 wujihand_config.py
```

输出示例：
```
============================================================================
WujiHand DRO Deployment Configuration
============================================================================

[Project Paths]
  DRO Root: /home/user/DRO-Grasp
  Checkpoint: model_3robots

[Robot Configuration]
  Robot: wujihand
  Joints: 20
  URDF: data/data_urdf/robot/wujihand/wujihand_right_isaac_fixed.urdf

✅ Configuration validated successfully!
```

### 2. 查看输入数据

```bash
# 终端1: 查看手部关键点话题
ros2 topic echo /hotrack/landmarks

# 检查数据格式
# 应该看到21个Pose，每个包含position.xyz
```

### 3. 查看输出数据

```bash
# 终端2: 查看关节命令
ros2 topic echo /hand_0/joint_commands

# 检查输出
# name: [finger1_joint1, finger1_joint2, ...]
# position: [0.1, 0.2, ...]  # 20个关节角度
```

### 4. 监控发布频率

```bash
# 检查实际发布频率
ros2 topic hz /hand_0/joint_commands

# 应该接近设定的 --publish-rate-hz
```

### 5. 查看进程日志

程序使用loguru输出详细日志：

```
[INFO] DRO inference: frame 50 | time: 15.2ms | joints: [-0.32, 1.45]
[INFO] Published 100 messages | Joints range: [-0.45, 1.52]
```

## 📊 性能优化

### 1. GPU加速

```bash
# 确保PyTorch使用GPU
python3 -c "import torch; print(torch.cuda.is_available())"

# 使用CUDA
--device cuda
```

### 2. 模型优化（可选）

```python
# 在DRO项目中可以尝试：
# - TorchScript编译
# - ONNX Runtime
# - TensorRT加速
```

### 3. 调整批处理

如果GPU足够强，可以修改代码支持批处理多帧：

```python
# 在dro_inference_process中
# 累积N帧再一起推理
batch_size = 4
```

## 🔍 常见问题

### Q1: 找不到DRO模块

**错误**:
```
ImportError: No module named 'model.network'
```

**解决**:
```bash
# 确保DRO_PROJECT_ROOT设置正确
export DRO_PROJECT_ROOT=/path/to/DRO-Grasp

# 或在命令行中指定
python3 dro_wujihand_deployment_complete.py \
  --dro-project-root /path/to/DRO-Grasp
```

### Q2: Checkpoint not found

**错误**:
```
Checkpoint not found: ckpt/model/model_3robots.pth
```

**解决**:
```bash
# 下载checkpoint
cd /path/to/DRO-Grasp
bash scripts/download_ckpt.sh

# 或手动下载并放到 ckpt/model/
```

### Q3: 关节角度异常

**现象**: 机械手姿态异常

**解决**:
1. 检查关节限制是否正确
2. 降低速度限制: `--max-vel-rad 1.0`
3. 增强平滑: `--ema-alpha 0.2`
4. 查看实时角度输出是否在合理范围

### Q4: 延迟太大

**现象**: 响应慢

**解决**:
1. 使用GPU: `--device cuda`
2. 降低发布频率: `--publish-rate-hz 30`
3. 减少平滑: `--ema-alpha 0.5`
4. 检查DRO推理时间（日志中显示）

### Q5: 手部跟踪不稳定

**现象**: 输入数据抖动

**解决**:
1. 增强平滑: `--ema-alpha 0.2`
2. 降低速度限制: `--max-vel-rad 1.5`
3. 在手部跟踪端添加滤波

## 📝 与DRO-Grasp项目的集成

### 目录结构

```
DRO-Grasp/
├── ckpt/
│   └── model/
│       ├── model_3robots.pth
│       └── ...
├── data/
│   └── data_urdf/
│       └── robot/
│           └── wujihand/
│               ├── wujihand_right_isaac_fixed.urdf
│               └── meshes/
├── model/
│   └── network.py
├── utils/
│   ├── hand_model.py
│   ├── multilateration.py
│   └── ...
├── dro_wujihand_deployment_complete.py  ← 部署脚本
├── wujihand_config.py                   ← 配置文件
└── launch_wujihand_dro.sh               ← 启动脚本
```

### 代码复用

部署脚本完全复用DRO-Grasp的核心功能：

- ✅ 网络推理: `model/network.py`
- ✅ 手部模型: `utils/hand_model.py`
- ✅ 多边定位: `utils/multilateration.py`
- ✅ 姿态计算: `utils/se3_transform.py`
- ✅ IK优化: `utils/optimization.py`

## 🎓 技术细节

### DRO (Distance-based Retargeting Optimization)

DRO的核心思想是预测机器人链接点到物体表面的距离场，然后通过优化求解关节角度。

**流程**:
1. 网络输入: 机器人点云 + 物体点云
2. 网络输出: 距离场矩阵 (N×M)
3. 多边定位: 根据距离反推链接位置
4. 姿态计算: 计算链接的SE(3)变换
5. IK优化: 求解满足变换的关节角度

### WujiHand运动学

WujiHand是一个20自由度的灵巧手：
- 5指，每指4个关节
- 拇指配置特殊（更大的旋转范围）
- 其他4指配置相似

关节编号规则：
```
finger[1-5]_joint[1-4]
  ^         ^
  |         └─ 关节序号 (1=根部, 4=指尖)
  └─ 手指序号 (1=拇指, 2=食指, ...)
```

## 📚 参考资料

- **DRO-Grasp论文**: [链接]
- **WujiHand文档**: [链接]
- **ROS2 JointState**: http://docs.ros.org/en/api/sensor_msgs/html/msg/JointState.html
- **PyTorch Kinematics**: https://github.com/UM-ARM-Lab/pytorch_kinematics

## 🤝 贡献

如果你发现问题或有改进建议：
1. 检查配置是否正确
2. 查看日志输出
3. 记录复现步骤
4. 提供错误信息

## 📄 许可证

遵循DRO-Grasp项目的许可证。

---

**版本**: 1.0.0  
**更新日期**: 2026-01-27  
**维护**: Claude + Your Team

**重要**: 这是一个**完全适配、开箱即用**的版本，所有配置都已从项目源码中提取并验证。
