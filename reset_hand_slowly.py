import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import numpy as np
import time
from loguru import logger

# ================= 配置 =================
# 动作耗时 (秒) - 虽然是盲发，给2秒缓冲比瞬间跳变安全
DURATION = 2.0  
CONTROL_RATE = 30.0

# 关节名称 (和你提供的代码保持完全一致)
JOINT_NAMES = [
    "right_finger1_joint1", "right_finger1_joint2", "right_finger1_joint3", "right_finger1_joint4",
    "right_finger2_joint1", "right_finger2_joint2", "right_finger2_joint3", "right_finger2_joint4",
    "right_finger3_joint1", "right_finger3_joint2", "right_finger3_joint3", "right_finger3_joint4",
    "right_finger4_joint1", "right_finger4_joint2", "right_finger4_joint3", "right_finger4_joint4",
    "right_finger5_joint1", "right_finger5_joint2", "right_finger5_joint3", "right_finger5_joint4",
]

class ForceResetter(Node):
    def __init__(self):
        super().__init__('force_resetter')
        # 和你成功的脚本一样的发布话题
        self.pub = self.create_publisher(JointState, '/right_hand/joint_commands', 10)
        logger.info("Publisher initialized on /right_hand/joint_commands")

    def publish_joints(self, joints):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        msg.position = joints.tolist()
        self.pub.publish(msg)

def main():
    rclpy.init()
    node = ForceResetter()
    
    logger.info("Force resetting hand to 0...")

    # 1. 设定起点和终点
    # 既然是盲发，我们假设手大概在半握状态 (0.8)，这样插值会比从 0 插值更顺滑
    # 如果你不在乎顺滑度，这一步其实无所谓，主要是为了给电机一个缓冲
    start_pose = np.ones(20) * 0.8  
    target_pose = np.zeros(20)      # 目标全是 0
    
    # 2. 执行循环
    steps = int(DURATION * CONTROL_RATE)
    dt = 1.0 / CONTROL_RATE
    
    try:
        for step in range(steps + 1):
            if not rclpy.ok(): break
            
            # 计算进度
            t = step / steps
            
            # 盲插值：从 0.8 慢慢变到 0.0
            current_cmd = start_pose + (target_pose - start_pose) * t
            
            node.publish_joints(current_cmd)
            time.sleep(dt)
            
            if step % 10 == 0:
                print(f"Resetting... {int(t*100)}%", end="\r")

    except KeyboardInterrupt:
        pass

    # 3. 强制锁定归零
    # 循环结束后，再连续发 1 秒的 0，确保彻底张开
    logger.success("\nLocking to 0.0 position...")
    end_time = time.time() + 1.0
    while time.time() < end_time and rclpy.ok():
        node.publish_joints(target_pose)
        time.sleep(0.05)

    logger.success("Done! Hand should be open.")
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()