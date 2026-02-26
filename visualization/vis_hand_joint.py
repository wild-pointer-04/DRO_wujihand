"""
Visualizes hand joint motion within joint range (upper & lower limits).
With Coordinate Frames (Axes) Visualization.
UPDATED: Applies 'contact_point_orthogonal_rotation' from HandModel to visualize the CORRECTED frames.
"""

import os
import sys
import numpy as np
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
import time
import argparse
import torch
import viser
# 确保引用的是正确的 create_hand_model
from utils.hand_model import create_hand_model

parser = argparse.ArgumentParser()
parser.add_argument('--robot_name', type=str, default='shadowhand')
parser.add_argument('--axis_len', type=float, default=0.03, help="Length of the coordinate axes in meters")
args = parser.parse_args()
robot_name = args.robot_name

# 1. 加载模型
hand = create_hand_model(robot_name)
device = hand.device 
print(f"Running visualization on device: {device}")

pk_chain = hand.pk_chain
lower, upper = pk_chain.get_joint_limits()

# 2. 将 Limits 转换到正确的设备上
lower = torch.tensor(lower, dtype=torch.float32, device=device)
upper = torch.tensor(upper, dtype=torch.float32, device=device)

# 启动 Viser
server = viser.ViserServer(host='127.0.0.1', port=8080)

# 获取初始姿态
canonical_q = hand.get_canonical_q().to(device)

# 获取并显示初始 Mesh
canonical_trimesh = hand.get_trimesh_q(canonical_q)["visual"]

server.scene.add_mesh_simple(
    robot_name,
    canonical_trimesh.vertices,
    canonical_trimesh.faces,
    color=(102, 192, 255),
    opacity=0.5
)

def draw_axes(server, transforms, axis_len=0.02):
    """
    辅助函数：遍历所有Link并画出坐标轴
    """
    for link_name, transform in transforms.items():
        # 获取变换矩阵
        if hasattr(transform, 'get_matrix'):
            matrix = transform.get_matrix()
        else:
            matrix = transform
            
        if isinstance(matrix, torch.Tensor):
            matrix = matrix.squeeze().detach().cpu().numpy()
        
        # 提取位置和旋转
        pos = matrix[:3, 3]
        rot_urdf = matrix[:3, :3]
        
        # ==========================================================
        # [关键修改] 应用 HandModel 中的修正矩阵
        # ==========================================================
        rot_final = rot_urdf # 默认是 URDF 原始旋转
        
        if hasattr(hand, 'contact_point_orthogonal_rotation'):
            # 检查该 link 是否有对应的修正矩阵
            if link_name in hand.contact_point_orthogonal_rotation:
                correction = hand.contact_point_orthogonal_rotation[link_name]
                if isinstance(correction, torch.Tensor):
                    correction = correction.cpu().numpy()
                
                # 应用旋转：World_Rot = URDF_Rot @ Correction_Rot
                rot_final = rot_urdf @ correction

        # 辅助函数：构造 add_line_segments 需要的 (1, 2, 3) 数组
        def get_line_points(start, end):
            return np.array([[start, end]], dtype=np.float32)

        # 1. 绘制 X 轴 (Red)
        x_end = pos + rot_final @ np.array([axis_len, 0, 0])
        server.scene.add_line_segments(
            f"{link_name}_axis_x",
            points=get_line_points(pos, x_end),
            colors=(1.0, 0.0, 0.0), # Red
            line_width=2.0
        )
        
        # 2. 绘制 Y 轴 (Green)
        y_end = pos + rot_final @ np.array([0, axis_len, 0])
        server.scene.add_line_segments(
            f"{link_name}_axis_y",
            points=get_line_points(pos, y_end),
            colors=(0.0, 1.0, 0.0), # Green
            line_width=2.0
        )

        # 3. 绘制 Z 轴 (Blue) - ★★★ 这是修正后的探测方向 ★★★
        z_end = pos + rot_final @ np.array([0, 0, axis_len])
        server.scene.add_line_segments(
            f"{link_name}_axis_z",
            points=get_line_points(pos, z_end),
            colors=(0.0, 0.0, 1.0), # Blue
            line_width=2.0
        )

def update(q):
    q = q.to(device)

    # 1. 更新 Mesh
    trimesh = hand.get_trimesh_q(q)["visual"]
    server.scene.add_mesh_simple(
        robot_name,
        trimesh.vertices,
        trimesh.faces,
        color=(102, 192, 255),
        opacity=0.5 
    )
    
    # 2. 计算正运动学 (FK)
    if len(q.shape) == 1:
        q_input = q.unsqueeze(0)
    else:
        q_input = q
        
    transforms = pk_chain.forward_kinematics(q_input)
    
    # 3. 绘制坐标轴 (现在会应用修正了)
    draw_axes(server, transforms, axis_len=args.axis_len)

# 创建 GUI 滑块
gui_joints = []
lower_cpu = lower.cpu().numpy()
upper_cpu = upper.cpu().numpy()

# 初始化位置
init_q = lower * 0.75 + upper * 0.25 
init_q_cpu = init_q.cpu().numpy()

for i, joint_name in enumerate(hand.get_joint_orders()):
    val = 0.0 if i < 6 else float(init_q_cpu[i])
    slider = server.gui.add_slider(
        label=joint_name,
        min=float(lower_cpu[i]),
        max=float(upper_cpu[i]),
        step=float(upper_cpu[i] - lower_cpu[i]) / 100,
        initial_value=val,
    )
    gui_joints.append(slider)

# 绑定回调
def on_slider_update(_):
    q_vals = torch.tensor([gui.value for gui in gui_joints], dtype=torch.float32, device=device)
    update(q_vals)

for slider in gui_joints:
    slider.on_update(on_slider_update)

on_slider_update(None)

while True:
    time.sleep(1)