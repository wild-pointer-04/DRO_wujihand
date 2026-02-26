"""
execute_perfect_grasp.py
功能：
1. 使用"归一化+反向偏移"的正确逻辑计算抓取姿态
2. 打印建议的苹果摆放位置
3. 等待用户确认摆放
4. 通过 ROS2 控制真机执行抓取
"""

import torch
import numpy as np
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from loguru import logger
from types import SimpleNamespace

# 引入项目模块
from model.network import create_network
from utils.hand_model import create_hand_model
from utils.multilateration import multilateration
from utils.se3_transform import compute_link_pose
from utils.optimization import process_transform, create_problem, optimization

# ================= 配置区域 =================
CHECKPOINT_PATH = "ckpt/model/model_3robots.pth" 
OBJECT_PC_PATH = "data/PointCloud/object/contactdb/apple.pt" 
DEVICE = "cuda"

# 动作配置
INTERPOLATION_TIME = 3.0  # 动作耗时(秒)
CONTROL_RATE = 30.0       # 频率(Hz)

JOINT_NAMES = [
    "right_finger1_joint1", "right_finger1_joint2", "right_finger1_joint3", "right_finger1_joint4",
    "right_finger2_joint1", "right_finger2_joint2", "right_finger2_joint3", "right_finger2_joint4",
    "right_finger3_joint1", "right_finger3_joint2", "right_finger3_joint3", "right_finger3_joint4",
    "right_finger4_joint1", "right_finger4_joint2", "right_finger4_joint3", "right_finger4_joint4",
    "right_finger5_joint1", "right_finger5_joint2", "right_finger5_joint3", "right_finger5_joint4",
]

class SafeGrasper(Node):
    def __init__(self):
        super().__init__('safe_grasper')
        self.pub = self.create_publisher(JointState, '/right_hand/joint_commands', 10)
        logger.info("ROS2 Connection Established.")

    def publish_joints(self, joints):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        msg.position = joints.tolist()
        self.pub.publish(msg)

def calculate_grasp_pose():
    """
    计算逻辑：物体归零，手腕后退初始化，获取最佳相对姿态
    """
    logger.info("Calculating optimal grasp pose...")
    device = torch.device(DEVICE)
    
    # 1. 加载模型
    hand = create_hand_model('wujihand', device)
    net_config = SimpleNamespace(emb_dim=512, latent_dim=64, pretrain=None, center_pc=True, block_computing=True)
    network = create_network(net_config, mode='validate').to(device)
    network.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    network.eval()
    
    # 2. 加载物体并归一化
    try:
        object_data = torch.load(OBJECT_PC_PATH, map_location=device)
        if isinstance(object_data, dict): object_data = object_data['pc']
        object_pc = object_data[:512, :3]
        # 强制归一化到 (0,0,0)
        centroid = torch.mean(object_pc, dim=0)
        object_pc = object_pc - centroid
    except Exception as e:
        logger.warning(f"Using virtual sphere (Load failed: {e})")
        object_pc = torch.randn(512, 3, device=device) * 0.04

    # 3. 初始化：把手放在物体后方
    initial_q = hand.get_initial_q()
    # 设置基座: 后退 12cm, 稍微抬高
    initial_base = torch.tensor([-0.12, 0.0, 0.02, 0.0, 0.0, 0.0], device=device)
    initial_q[:6] = initial_base
    initial_q[6:] = 0.1 # 手指微屈

    # 4. 推理与优化
    robot_pc = hand.get_transformed_links_pc(initial_q)[:, :3]
    with torch.no_grad():
        dro = network(robot_pc.unsqueeze(0), object_pc.unsqueeze(0))['dro']
    
    mlat_pc = multilateration(dro, object_pc.unsqueeze(0))
    transform, _ = compute_link_pose(hand.links_pc, mlat_pc, is_train=False)
    optim_transform = process_transform(hand.pk_chain, transform)
    layer = create_problem(hand.pk_chain, optim_transform.keys())
    final_q = optimization(hand.pk_chain, layer, initial_q.unsqueeze(0), optim_transform)[0]
    
    # 5. 返回结果
    final_base = final_q[:6].cpu().numpy()
    final_fingers = final_q[6:].cpu().numpy()
    final_fingers = np.clip(final_fingers, -0.5, 1.6) # 安全裁剪
    
    return final_base, final_fingers

def main():
    rclpy.init()
    node = SafeGrasper()
    
    # 1. 计算姿态
    try:
        base_pos, target_pose = calculate_grasp_pose()
    except Exception as e:
        logger.error(f"Calculation failed: {e}")
        return

    # 2. 打印摆放位置建议
    print("\n" + "="*60)
    print("📢 【请调整苹果位置】")
    print("算法认为最佳抓取点是：")
    print(f"  手腕前方 X: {-base_pos[0]:.4f} m (约 {int(-base_pos[0]*100)} cm)")
    print(f"  手腕右方 Y: {-base_pos[1]:.4f} m (约 {int(-base_pos[1]*100)} cm)")
    print(f"  手腕上方 Z: {-base_pos[2]:.4f} m (约 {int(-base_pos[2]*100)} cm)")
    print("="*60)
    
    print("\n[Target Joints]:")
    print(target_pose)
    
    # 3. 确认执行
    logger.info(f"Ready to execute over {INTERPOLATION_TIME}s.")
    try:
        input(">>> 请将苹果摆好，然后按 [Enter] 键开始抓取... <<<")
    except KeyboardInterrupt:
        print("\nCancelled.")
        return
    
    # 4. 执行抓取
    start_pose = np.zeros(20) # 假设从平手开始
    steps = int(INTERPOLATION_TIME * CONTROL_RATE)
    dt = 1.0 / CONTROL_RATE
    
    logger.info("Grasping...")
    for step in range(steps + 1):
        if not rclpy.ok(): break
        t = step / steps
        current_cmd = start_pose + (target_pose - start_pose) * t
        node.publish_joints(current_cmd)
        time.sleep(dt)

    logger.success("Grasp completed! Holding position...")
    
    # 5. 保持姿态
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