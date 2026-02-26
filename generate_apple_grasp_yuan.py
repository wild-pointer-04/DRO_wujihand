"""
safe_grasp_executor.py
功能：
1. 使用 DRO 算法计算抓苹果的最优姿态
2. [新增] 在终端详细打印计算出的关节角度
3. 使用线性插值 (Lerp) 平滑控制 WujiHand 进行抓取
"""

import torch
import numpy as np
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from loguru import logger
import sys

# 引入项目模块
# 确保你在 DRO-Grasp 根目录下运行
from model.network import create_network
from utils.hand_model import create_hand_model
from utils.multilateration import multilateration
from utils.se3_transform import compute_link_pose
from utils.optimization import process_transform, create_problem, optimization
from types import SimpleNamespace

# ================= 配置区域 =================
# 1. 路径配置
CHECKPOINT_PATH = "ckpt/model/model_3robots.pth"
# 你的苹果点云路径 (请确认路径正确)
OBJECT_PC_PATH = "data/PointCloud/object/contactdb/apple.pt" 
DEVICE = "cuda"

# 2. 动作配置
INTERPOLATION_TIME = 3.0  # 抓取动作耗时 (秒) - 越长越慢越安全
CONTROL_RATE = 30.0       # 控制频率 (Hz)
# ===========================================

# 机械手关节名称 (带前缀，必须与真机一致)
JOINT_NAMES = [
    "right_finger1_joint1", "right_finger1_joint2", "right_finger1_joint3", "right_finger1_joint4",
    "right_finger2_joint1", "right_finger2_joint2", "right_finger2_joint3", "right_finger2_joint4",
    "right_finger3_joint1", "right_finger3_joint2", "right_finger3_joint3", "right_finger3_joint4",
    "right_finger4_joint1", "right_finger4_joint2", "right_finger4_joint3", "right_finger4_joint4",
    "right_finger5_joint1", "right_finger5_joint2", "right_finger5_joint3", "right_finger5_joint4",
]

# 定义一个"张开手"的姿态作为起点 (全0通常是平伸状态)
OPEN_HAND_POSE = np.zeros(20) 

class SafeGrasper(Node):
    def __init__(self):
        super().__init__('safe_grasper')
        self.pub = self.create_publisher(JointState, '/right_hand/joint_commands', 10)
        logger.info("ROS2 Publisher initialized on /right_hand/joint_commands")

    def publish_joints(self, joints):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        msg.position = joints.tolist()
        self.pub.publish(msg)

def calculate_grasp_pose():
    """使用 DRO 算法计算目标姿态 (固定底座版)"""
    logger.info("Calculating optimal grasp pose (Fixed Base Mode)...")
    device = torch.device(DEVICE)
    
    # 1. 加载模型
    hand = create_hand_model('wujihand', device)
    net_config = SimpleNamespace(emb_dim=512, latent_dim=64, pretrain=None, center_pc=True, block_computing=True)
    network = create_network(net_config, mode='validate').to(device)
    network.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    network.eval()
    
    # 2. 加载物体
    try:
        object_data = torch.load(OBJECT_PC_PATH, map_location=device)
        if isinstance(object_data, dict): object_data = object_data['pc']
        object_pc = object_data[:512, :3]
        
        # [关键调整] 调整苹果位置
        # 对于固定底座，物体必须出现在"手心正好能抓到"的地方
        # WujiHand 坐标系通常 X轴朝前。我们把苹果放在手心前方 8-10cm 处
        object_pc[:, 0] += 0.09  
        object_pc[:, 1] += 0.0   # 左右居中
        object_pc[:, 2] += 0.02  # 稍微抬高一点(如果手心朝上)
        
        logger.info(f"Loaded object: {OBJECT_PC_PATH}")
    except Exception as e:
        logger.warning(f"Using virtual sphere (Load failed: {e})")
        object_pc = torch.randn(512, 3, device=device) * 0.04
        object_pc[:, 0] += 0.09

    # 3. 迭代优化 (带约束)
    initial_q = hand.get_initial_q()
    # 先把底座归零
    initial_q[:6] = 0.0 
    
    for i in range(15): # 增加迭代次数以适应硬约束
        robot_pc = hand.get_transformed_links_pc(initial_q)[:, :3]
        with torch.no_grad():
            dro = network(robot_pc.unsqueeze(0), object_pc.unsqueeze(0))['dro']
        
        mlat_pc = multilateration(dro, object_pc.unsqueeze(0))
        transform, _ = compute_link_pose(hand.links_pc, mlat_pc, is_train=False)
        optim_transform = process_transform(hand.pk_chain, transform)
        layer = create_problem(hand.pk_chain, optim_transform.keys())
        initial_q = optimization(hand.pk_chain, layer, initial_q.unsqueeze(0), optim_transform)[0]
        
        # [核心修正]: 每一轮优化后，强制把虚拟底座(前6维)按回 0
        # 这迫使手指去适应物体，而不是让手腕飞过去适应物体
        initial_q[:6] = 0.0 
    
    # 提取结果
    final_joints = initial_q.cpu().numpy()
    if len(final_joints) > 20: final_joints = final_joints[6:] 
    
    # 安全裁剪
    final_joints = np.clip(final_joints, -0.5, 1.6)
    
    return final_joints

def main():
    rclpy.init()
    node = SafeGrasper()
    
    # 1. 算出目标去哪 (Target)
    try:
        target_pose = calculate_grasp_pose()
    except Exception as e:
        logger.error(f"Calculation failed: {e}")
        return

    # =========================================================
    # [新增] 打印结果区域
    # =========================================================
    print("\n" + "="*60)
    print("🎯 CALCULATED OPTIMAL GRASP POSE (Radians)")
    print("="*60)
    print(f"{'Index':<5} | {'Joint Name':<25} | {'Angle'}")
    print("-" * 60)
    
    for i, (name, angle) in enumerate(zip(JOINT_NAMES, target_pose)):
        print(f"{i:<5} | {name:<25} | {angle:.4f}")
        
    print("-" * 60)
    print("\n[Raw Array for Copying]:")
    # 设置打印选项，确保打印完整数组不省略
    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    print(target_pose)
    print("="*60 + "\n")
    # =========================================================

    # 2. 准备开始 (Start)
    start_pose = OPEN_HAND_POSE 
    
    logger.info("Ready to execute grasp.")
    logger.info(f"Interpolation time: {INTERPOLATION_TIME}s")
    
    try:
        input("Press Enter to START GRASPING >>> ") # 安全锁
    except KeyboardInterrupt:
        print("\nCancelled.")
        return
    
    # 3. 线性插值循环 (Linear Interpolation Loop)
    steps = int(INTERPOLATION_TIME * CONTROL_RATE)
    dt = 1.0 / CONTROL_RATE
    
    logger.info("Executing...")
    for step in range(steps + 1):
        # 计算进度 t (0.0 -> 1.0)
        t = step / steps
        
        # 核心公式： 当前 = 起点 + (终点 - 起点) * t
        current_cmd = start_pose + (target_pose - start_pose) * t
        
        # 发送命令
        node.publish_joints(current_cmd)
        
        # 保持频率
        time.sleep(dt)
        
        if step % 10 == 0:
            logger.info(f"Progress: {t*100:.1f}%")

    logger.success("Grasp completed! Holding position...")
    
    # 4. 保持姿态 (Hold)
    try:
        while rclpy.ok():
            node.publish_joints(target_pose)
            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("Releasing...")
        
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()