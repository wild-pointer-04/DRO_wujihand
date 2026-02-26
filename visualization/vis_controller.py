import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
import time
import argparse
import trimesh
import torch
import viser
import numpy as np

from utils.hand_model import create_hand_model
from utils.controller import controller
from utils.vis_utils import vis_vector

def vis_controller_result(robot_name='shadowhand', object_name=None):
    # 保持原有的 dataset 可视化逻辑不变
    dataset_path = os.path.join(ROOT_DIR, f'data/CMapDataset_filtered/cmap_dataset.pt')
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        return
        
    metadata = torch.load(dataset_path)['metadata']
    metadata = [m for m in metadata if (object_name is None or m[1] == object_name) and m[2] == robot_name]

    server = viser.ViserServer(host='127.0.0.1', port=8080)

    if len(metadata) == 0:
        print("No metadata found.")
        return

    slider = server.gui.add_slider(
        label='robot',
        min=0,
        max=len(metadata) - 1,
        step=1,
        initial_value=0
    )
    slider.on_update(lambda gui: on_update(gui.target.value))

    hand = create_hand_model(robot_name)

    def on_update(idx):
        q, object_name, _ = metadata[idx]
        # 确保 q 在正确的 device
        q = q.to(hand.device)
        outer_q, inner_q = controller(robot_name, q)

        name = object_name.split('+')
        object_path = os.path.join(ROOT_DIR, f'data/data_urdf/object/{name[0]}/{name[1]}/{name[1]}.stl')
        if os.path.exists(object_path):
            object_trimesh = trimesh.load_mesh(object_path)
            server.scene.add_mesh_simple(
                'object',
                object_trimesh.vertices,
                object_trimesh.faces,
                color=(239, 132, 167),
                opacity=0.75
            )

        # 原始
        robot_trimesh = hand.get_trimesh_q(q)["visual"]
        server.scene.add_mesh_simple(
            'origin',
            robot_trimesh.vertices,
            robot_trimesh.faces,
            color=(102, 192, 255),
            opacity=0.5
        )
        # 外张
        robot_trimesh = hand.get_trimesh_q(outer_q)["visual"]
        server.scene.add_mesh_simple(
            'outer',
            robot_trimesh.vertices,
            robot_trimesh.faces,
            color=(255, 149, 71),
            opacity=0.5
        )
        # 内收
        robot_trimesh = hand.get_trimesh_q(inner_q)["visual"]
        server.scene.add_mesh_simple(
            'inner',
            robot_trimesh.vertices,
            robot_trimesh.faces,
            color=(255, 111, 190),
            opacity=0.5
        )

    while True:
        time.sleep(1)


def vis_hand_direction(robot_name='shadowhand'):
    """
    Revised visualization: Uses HandModel's corrected rotation matrix to show
    the TRUE contact direction (Z-axis), instead of the raw URDF link direction.
    """
    server = viser.ViserServer(host='127.0.0.1', port=8080)

    # 加载模型
    hand = create_hand_model(robot_name)
    device = hand.device
    
    # 初始姿态
    q = hand.get_canonical_q().to(device)
    joint_orders = hand.get_joint_orders()
    lower, upper = hand.pk_chain.get_joint_limits()

    # 显示手部 Mesh
    canonical_trimesh = hand.get_trimesh_q(q)["visual"]
    server.scene.add_mesh_simple(
        robot_name,
        canonical_trimesh.vertices,
        canonical_trimesh.faces,
        color=(102, 192, 255),
        opacity=0.8
    )

    # === 核心修改：使用 HandModel 的修正矩阵来画箭头 ===
    hand.update_status(q) # 更新 FK
    transforms = hand.frame_status # 获取所有 link 的位姿
    
    for link_name, transform in transforms.items():
        # 我们只画那些我们在 hand_model 里定义了修正矩阵的 link (即指尖/指肚)
        if link_name not in hand.contact_point_orthogonal_rotation:
            continue
            
        # 1. 获取 Link 的当前位姿矩阵 (World Frame)
        matrix = transform.get_matrix()[0] # (4, 4)
        pos = matrix[:3, 3]
        rot_urdf = matrix[:3, :3]
        
        # 2. 获取修正矩阵 (Correction)
        correction = hand.contact_point_orthogonal_rotation[link_name] # (3, 3)
        
        # 3. 计算修正后的最终旋转 (Corrected Frame)
        # rot_final = rot_urdf @ correction
        rot_final = rot_urdf @ correction
        
        # 4. 提取 Z 轴 (Blue Axis) - 这就是我们在 hand_model 里定义的“抓取/接触方向”
        # column 2 is Z axis (0, 1, 2)
        z_axis_dir = rot_final[:, 2] 

        # 转换到 numpy 用于绘图
        pos_np = pos.detach().cpu().numpy()
        dir_np = z_axis_dir.detach().cpu().numpy()

        # 5. 画出箭头 (绿色表示修正后的抓取方向)
        print(f"Drawing vector for {link_name}")
        vec_mesh = vis_vector(
            pos_np,
            vector=dir_np,
            length=0.04,  # 稍微画长一点
            cyliner_r=0.002,
            color=(0, 255, 0) # Green for "Go" / Correct Direction
        )
        server.scene.add_mesh_trimesh(f"{link_name}_dir", vec_mesh, visible=True)

    # === GUI 滑块逻辑 ===
    # 转换 limits 到 CPU list
    lower_cpu = torch.tensor(lower, device=device).cpu().numpy()
    upper_cpu = torch.tensor(upper, device=device).cpu().numpy()
    
    # 初始 Q
    init_q = lower_cpu * 0.75 + upper_cpu * 0.25
    init_q[:6] = 0
    current_q = list(init_q)

    def update(joint_idx, joint_q):
        current_q[joint_idx] = joint_q
        q_tensor = torch.tensor(current_q, dtype=torch.float32, device=device)
        
        # 更新 Mesh
        trimesh_data = hand.get_trimesh_q(q_tensor)["visual"]
        server.scene.add_mesh_simple(
            robot_name,
            trimesh_data.vertices,
            trimesh_data.faces,
            color=(102, 192, 255),
            opacity=0.8
        )
        
        # 更新箭头方向 (需要重新计算 FK)
        hand.update_status(q_tensor)
        transforms = hand.frame_status
        
        for link_name, transform in transforms.items():
            if link_name not in hand.contact_point_orthogonal_rotation:
                continue
                
            matrix = transform.get_matrix()[0]
            pos = matrix[:3, 3]
            rot_urdf = matrix[:3, :3]
            correction = hand.contact_point_orthogonal_rotation[link_name]
            rot_final = rot_urdf @ correction
            z_axis_dir = rot_final[:, 2] 

            pos_np = pos.detach().cpu().numpy()
            dir_np = z_axis_dir.detach().cpu().numpy()
            
            # 重新生成箭头 mesh
            vec_mesh = vis_vector(
                pos_np,
                vector=dir_np,
                length=0.04,
                cyliner_r=0.002,
                color=(0, 255, 0)
            )
            server.scene.add_mesh_trimesh(f"{link_name}_dir", vec_mesh, visible=True)

    # 添加滑块
    for i, joint_name in enumerate(joint_orders):
        # 简单起见，给所有关节都加滑块
        slider = server.gui.add_slider(
            label=joint_name,
            min=float(lower_cpu[i]),
            max=float(upper_cpu[i]),
            step=(float(upper_cpu[i]) - float(lower_cpu[i])) / 100,
            initial_value=float(current_q[i]),
        )
        # Python 闭包坑，需要默认参数锁定 i
        slider.on_update(lambda gui, idx=i: update(idx, gui.target.value))

    while True:
        time.sleep(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot_name', default='shadowhand', type=str)
    parser.add_argument('--controller', action='store_true')
    args = parser.parse_args()

    if args.controller:
        vis_controller_result(args.robot_name)
    else:
        vis_hand_direction(args.robot_name)