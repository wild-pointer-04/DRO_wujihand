"""
WujiHand DRO部署配置文件
根据项目实际结构和URDF自动生成
"""

import os
from pathlib import Path

# ============================================================================
# 项目路径配置
# ============================================================================

# DRO-Grasp项目根目录（请根据实际情况修改）
DRO_PROJECT_ROOT = Path(__file__).parent.absolute()

# 可用的checkpoint列表
AVAILABLE_CHECKPOINTS = {
    'shadowhand': 'model_shadowhand',        # ShadowHand训练的模型
    '3robots': 'model_3robots',              # Barrett+Allegro+ShadowHand
    '3robots_partial': 'model_3robots_partial',  # 部分视图训练
    'allegro': 'model_allegro',              # Allegro专用
    'barrett': 'model_barrett',              # Barrett专用
}

# 默认使用的checkpoint（建议用3robots以获得更好的泛化性）
DEFAULT_CHECKPOINT = 'model_3robots'

# ============================================================================
# WujiHand硬件配置
# ============================================================================

class WujiHandConfig:
    """WujiHand机械手配置"""
    
    # 机器人名称
    robot_name = "wujihand"
    
    # URDF文件路径（相对于DRO项目根目录）
    urdf_path = "data/data_urdf/robot/wujihand/wujihand_right_isaac_fixed.urdf"
    
    # 关节数量
    num_joints = 20
    
    # 关节名称（从URDF提取）
    joint_names = [
        # Finger 1 (Thumb - 拇指)
        "finger1_joint1", "finger1_joint2", "finger1_joint3", "finger1_joint4",
        # Finger 2 (Index - 食指)
        "finger2_joint1", "finger2_joint2", "finger2_joint3", "finger2_joint4",
        # Finger 3 (Middle - 中指)
        "finger3_joint1", "finger3_joint2", "finger3_joint3", "finger3_joint4",
        # Finger 4 (Ring - 无名指)
        "finger4_joint1", "finger4_joint2", "finger4_joint3", "finger4_joint4",
        # Finger 5 (Pinky - 小指)
        "finger5_joint1", "finger5_joint2", "finger5_joint3", "finger5_joint4",
    ]
    
    # 关节限制（弧度，从URDF提取）
    joint_limits = {
        # Finger 1 (Thumb) - 特殊配置
        "finger1_joint1": (-0.0448, 1.6508),   # 根部旋转范围较大
        "finger1_joint2": (-0.1659, 0.9339),   # 侧摆
        "finger1_joint3": (-0.4932, 1.6272),   # 近端弯曲
        "finger1_joint4": (-0.4932, 1.6272),   # 远端弯曲
        
        # Finger 2-5 (其他手指) - 相似配置
        "finger2_joint1": (-0.32695, 1.63595), # 根部
        "finger2_joint2": (-0.495, 0.495),     # 侧摆（较小范围）
        "finger2_joint3": (-0.4932, 1.6272),   # 近端
        "finger2_joint4": (-0.4932, 1.6272),   # 远端
        
        "finger3_joint1": (-0.32695, 1.63595),
        "finger3_joint2": (-0.495, 0.495),
        "finger3_joint3": (-0.4932, 1.6272),
        "finger3_joint4": (-0.4932, 1.6272),
        
        "finger4_joint1": (-0.32695, 1.63595),
        "finger4_joint2": (-0.495, 0.495),
        "finger4_joint3": (-0.4932, 1.6272),
        "finger4_joint4": (-0.4932, 1.6272),
        
        "finger5_joint1": (-0.32695, 1.63595),
        "finger5_joint2": (-0.495, 0.495),
        "finger5_joint3": (-0.4932, 1.6272),
        "finger5_joint4": (-0.4932, 1.6272),
    }
    
    # 零位姿态（如果有）
    zero_q_path = "tmp/wujihand_zero_q.pt"
    
    # 点云数据路径
    point_cloud_path = "data/PointCloud/robot/wujihand.pt"


# ============================================================================
# ROS2通信配置
# ============================================================================

class ROS2Config:
    """ROS2话题和通信配置"""
    
    # 输入：手部关键点话题
    # 格式：PoseArray消息，包含21个Pose（手部关键点的3D坐标）
    landmark_topic = "/hotrack/landmarks"
    
    # 输出：关节命令话题
    # 格式：JointState消息，包含20个关节的位置命令
    joint_command_topic = "/hand_0/joint_commands"
    
    # 发布频率（Hz）
    publish_rate = 60.0
    
    # 队列大小
    queue_size = 10
    
    # 节点名称
    node_name_prefix = "wujihand_dro"


# ============================================================================
# DRO模型配置
# ============================================================================

class DROModelConfig:
    """DRO模型推理配置"""
    
    # 模型checkpoint名称
    checkpoint_name = DEFAULT_CHECKPOINT
    
    # 推理设备
    device = "cuda"  # "cuda" 或 "cpu"
    
    # 模型参数（与训练时保持一致）
    emb_dim = 512
    latent_dim = 64
    center_pc = True
    block_computing = True
    
    # 输入点云数量
    num_robot_points = 512
    num_object_points = 512
    
    # 物体点云路径（可选，用于特定物体抓取）
    # 如果为None，将使用虚拟球形物体
    object_pc_path = None
    # object_pc_path = "data/PointCloud/object/ycb/003_cracker_box.pt"


# ============================================================================
# 控制算法配置
# ============================================================================

class ControlConfig:
    """运动控制和滤波配置"""
    
    # ========== 平滑滤波 ==========
    enable_smoothing = True
    smoothing_type = "ema"  # 指数移动平均
    
    # EMA参数
    # alpha越大，越接近原始值（响应快但可能抖动）
    # alpha越小，越平滑（响应慢但稳定）
    ema_alpha = 0.3  # 推荐范围：0.2-0.5
    
    # ========== 速度限制 ==========
    enable_velocity_limit = True
    
    # 最大关节速度（rad/s）
    # 用于防止突然的大幅度运动
    max_velocity_rad_per_sec = 2.0  # 推荐范围：1.0-3.0
    
    # ========== 安全保护 ==========
    # 是否启用关节限制检查
    enable_joint_limit_check = True
    
    # 关节限制安全裕度（弧度）
    # 实际限制 = URDF限制 - safety_margin
    safety_margin = 0.05  # 5度安全裕度


# ============================================================================
# 性能和调试配置
# ============================================================================

class DebugConfig:
    """调试和性能监控配置"""
    
    # 日志级别
    log_level = "INFO"  # "DEBUG", "INFO", "WARNING", "ERROR"
    
    # 是否打印详细的推理信息
    verbose = True
    
    # 性能统计间隔（帧数）
    profiling_interval = 100
    
    # 是否保存推理数据（用于离线分析）
    save_inference_data = False
    save_data_path = "./inference_data"


# ============================================================================
# 预设配置方案
# ============================================================================

class PresetConfigs:
    """预设的配置方案"""
    
    @staticmethod
    def get_config(preset_name: str) -> dict:
        """
        获取预设配置
        
        Args:
            preset_name: 预设名称
                - "default": 默认平衡配置
                - "fast": 快速响应配置
                - "stable": 稳定配置
                - "debug": 调试配置
        """
        presets = {
            "default": {
                "checkpoint": "model_3robots",
                "publish_rate": 60.0,
                "ema_alpha": 0.3,
                "max_vel": 2.0,
                "device": "cuda",
            },
            
            "fast": {
                "checkpoint": "model_3robots",
                "publish_rate": 120.0,
                "ema_alpha": 0.5,
                "max_vel": 3.0,
                "device": "cuda",
            },
            
            "stable": {
                "checkpoint": "model_3robots",
                "publish_rate": 30.0,
                "ema_alpha": 0.2,
                "max_vel": 1.5,
                "device": "cuda",
            },
            
            "debug": {
                "checkpoint": "model_shadowhand",
                "publish_rate": 10.0,
                "ema_alpha": 0.1,
                "max_vel": 0.5,
                "device": "cpu",
            },
        }
        
        return presets.get(preset_name, presets["default"])


# ============================================================================
# 配置验证
# ============================================================================

def validate_config():
    """验证配置的有效性"""
    errors = []
    warnings = []
    
    # 检查项目路径
    if not (DRO_PROJECT_ROOT / "model" / "network.py").exists():
        errors.append(f"DRO project not found at {DRO_PROJECT_ROOT}")
    
    # 检查checkpoint
    ckpt_path = DRO_PROJECT_ROOT / "ckpt" / "model" / f"{DROModelConfig.checkpoint_name}.pth"
    if not ckpt_path.exists():
        errors.append(f"Checkpoint not found: {ckpt_path}")
    
    # 检查URDF
    urdf_path = DRO_PROJECT_ROOT / WujiHandConfig.urdf_path
    if not urdf_path.exists():
        warnings.append(f"URDF not found: {urdf_path}")
    
    # 检查参数范围
    if not 0 < ControlConfig.ema_alpha <= 1:
        errors.append("ema_alpha must be in (0, 1]")
    
    if ControlConfig.max_velocity_rad_per_sec < 0:
        errors.append("max_velocity must be non-negative")
    
    if ROS2Config.publish_rate <= 0:
        errors.append("publish_rate must be positive")
    
    return errors, warnings


# ============================================================================
# 打印配置信息
# ============================================================================

def print_config():
    """打印当前配置"""
    print("=" * 70)
    print("WujiHand DRO Deployment Configuration")
    print("=" * 70)
    
    print("\n[Project Paths]")
    print(f"  DRO Root: {DRO_PROJECT_ROOT}")
    print(f"  Checkpoint: {DROModelConfig.checkpoint_name}")
    
    print("\n[Robot Configuration]")
    print(f"  Robot: {WujiHandConfig.robot_name}")
    print(f"  Joints: {WujiHandConfig.num_joints}")
    print(f"  URDF: {WujiHandConfig.urdf_path}")
    
    print("\n[ROS2 Topics]")
    print(f"  Input: {ROS2Config.landmark_topic}")
    print(f"  Output: {ROS2Config.joint_command_topic}")
    print(f"  Rate: {ROS2Config.publish_rate} Hz")
    
    print("\n[DRO Model]")
    print(f"  Device: {DROModelConfig.device}")
    print(f"  Points: Robot={DROModelConfig.num_robot_points}, "
          f"Object={DROModelConfig.num_object_points}")
    
    print("\n[Control]")
    print(f"  Smoothing: EMA alpha={ControlConfig.ema_alpha}")
    print(f"  Max Velocity: {ControlConfig.max_velocity_rad_per_sec} rad/s")
    
    print("\n[Debug]")
    print(f"  Log Level: {DebugConfig.log_level}")
    print(f"  Verbose: {DebugConfig.verbose}")
    
    # 验证配置
    errors, warnings = validate_config()
    
    if warnings:
        print("\n⚠️  Warnings:")
        for w in warnings:
            print(f"  - {w}")
    
    if errors:
        print("\n❌ Errors:")
        for e in errors:
            print(f"  - {e}")
        print("\nPlease fix the errors before running!")
        return False
    else:
        print("\n✅ Configuration validated successfully!")
        return True


if __name__ == "__main__":
    # 打印并验证配置
    is_valid = print_config()
    
    if is_valid:
        print("\n" + "=" * 70)
        print("Configuration is ready for deployment!")
        print("=" * 70)
