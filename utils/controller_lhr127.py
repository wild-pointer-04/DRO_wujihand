import os
import sys
import time
import json
import trimesh
import torch
import viser

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.hand_model import create_hand_model
from utils.rotation import q_rot6d_to_q_euler


def get_link_dir(robot_name, joint_name):
    # 0. 【通用屏蔽】
    if joint_name.startswith('virtual') or joint_name == 'world':
        return None

    if robot_name == 'allegro':
        if joint_name in ['joint_0.0', 'joint_4.0', 'joint_8.0', 'joint_13.0']: return None
        link_dir = torch.tensor([0, 0, 1], dtype=torch.float32)
    elif robot_name == 'barrett':
        if joint_name in ['bh_j11_joint', 'bh_j21_joint']: return None
        link_dir = torch.tensor([-1, 0, 0], dtype=torch.float32)
    elif robot_name == 'ezgripper':
        link_dir = torch.tensor([1, 0, 0], dtype=torch.float32)
    elif robot_name == 'robotiq_3finger':
        if joint_name in ['gripper_fingerB_knuckle', 'gripper_fingerC_knuckle']: return None
        link_dir = torch.tensor([0, 0, -1], dtype=torch.float32)
    elif robot_name == 'shadowhand':
        if joint_name in ['WRJ2', 'WRJ1']: return None
        if joint_name != 'THJ5': link_dir = torch.tensor([0, 0, 1], dtype=torch.float32)
        else: link_dir = torch.tensor([1, 0, 0], dtype=torch.float32)
    elif robot_name == 'leaphand':
        if joint_name in ['13']: return None
        if joint_name in ['0', '4', '8']: link_dir = torch.tensor([1, 0, 0], dtype=torch.float32)
        elif joint_name in ['1', '5', '9', '12', '14']: link_dir = torch.tensor([0, 1, 0], dtype=torch.float32)
        else: link_dir = torch.tensor([0, -1, 0], dtype=torch.float32)

    elif robot_name == 'omnihand':
        if any(x in joint_name for x in ['wrist', 'palm', 'metacarpal', 'roll', 'cmc']): return None
        if 'thumb' in joint_name: return torch.tensor([0, 0, 1], dtype=torch.float32)
        if 'proximal' in joint_name: return torch.tensor([1, 0, 0], dtype=torch.float32)
        return torch.tensor([0, 0, 1], dtype=torch.float32)
    
    # === [WujiHand 最终修正] ===
    elif robot_name == 'wujihand':
        # 1. 屏蔽侧摆 (Joint 2)
        if 'joint2' in joint_name:
            return None
            
        # 2. 大拇指 (Finger 1) 修正：
        # 根据可视化，之前的向量导致绿色箭头朝外。
        # 我们现在全部取反，强制绿色箭头指向手心/手指弯曲方向。
        if 'finger1' in joint_name:
            if 'joint1' in joint_name:
                # 根部：试用 [0, 1, 0] 或 [1, 0, 0]。
                # 既然上一版 [-1, 0, 0] 是突出来的，我们改成 [1, 0, 0]
                return torch.tensor([1, 0, 0], dtype=torch.float32)
            else:
                # 指尖 (Joint 3, 4)：
                # 上一版 [1, 0, 0] 指向外侧，我们改成 [-1, 0, 0]
                return torch.tensor([-1, 0, 0], dtype=torch.float32)

        # 3. 其他手指 (Finger 2-5): 
        # 图片显示这些手指的箭头是正确的（指向手心），保持不变
        return torch.tensor([1, 0, 0], dtype=torch.float32)

    return None


def controller(robot_name, q_para):
    q_batch = torch.atleast_2d(q_para)

    hand = create_hand_model(robot_name, device=q_batch.device)
    joint_orders = hand.get_joint_orders()
    pk_chain = hand.pk_chain
    
    if q_batch.shape[-1] != len(pk_chain.get_joint_parameter_names()):
        q_batch = q_rot6d_to_q_euler(q_batch)
    status = pk_chain.forward_kinematics(q_batch)

    outer_q_batch = []
    inner_q_batch = []
    for batch_idx in range(q_batch.shape[0]):
        joint_dots = {}
        for frame_name in pk_chain.get_frame_names():
            frame = pk_chain.find_frame(frame_name)
            joint = frame.joint
            link_dir = get_link_dir(robot_name, joint.name)
            if link_dir is None:
                continue

            frame_transform = status[frame_name].get_matrix()[batch_idx]
            axis_dir = frame_transform[:3, :3] @ joint.axis
            link_dir = frame_transform[:3, :3] @ link_dir
            normal_dir = torch.cross(axis_dir, link_dir, dim=0)
            axis_origin = frame_transform[:3, 3]
            origin_dir = -axis_origin / torch.norm(axis_origin)
            joint_dots[joint.name] = torch.dot(normal_dir, origin_dir)

        q = q_batch[batch_idx]
        lower_q, upper_q = hand.pk_chain.get_joint_limits()
        outer_q, inner_q = q.clone(), q.clone()
        
        for joint_name, dot in joint_dots.items():
            if joint_name not in joint_orders: continue
            idx = joint_orders.index(joint_name)
            
            # --- WujiHand 强力抓取逻辑 ---
            if robot_name == 'wujihand':
                # 🚀 增加抓取步长 (Gain)，解决“抓不紧”的问题
                # 之前的 0.05 太小了，改成 0.15 甚至 0.2
                
                step_close = 0.0
                step_open = 0.08

                if 'joint1' in joint_name: 
                    step_close = 0.05  # 根部稍慢，为了包络
                elif 'joint3' in joint_name or 'joint4' in joint_name: 
                    step_close = 0.20  # 指尖极速闭合，扣死物体！
                else:
                    step_close = 0.15

                # 判定逻辑：
                # 只要物体在手心一侧 (dot > -0.3)，就全力闭合
                if dot >= -0.3: 
                    target = upper_q[idx]
                    # 给一个 Overdrive (超调)，确保电机输出最大力矩去抵达目标
                    outer_q[idx] += step_close * (target - outer_q[idx])
                    inner_q[idx] += step_close * (target - inner_q[idx])
                else:
                    target = lower_q[idx]
                    outer_q[idx] += step_open * (target - outer_q[idx])
                    inner_q[idx] += step_open * 0.5 * (target - inner_q[idx])

            # --- 其他机器人 (保持不变) ---
            elif robot_name == 'robotiq_3finger':  
                outer_q[idx] += 0.25 * ((outer_q[idx] - lower_q[idx]) if dot <= 0 else (outer_q[idx] - upper_q[idx]))
                inner_q[idx] += 0.15 * ((inner_q[idx] - upper_q[idx]) if dot <= 0 else (inner_q[idx] - lower_q[idx]))
            
            elif robot_name == 'omnihand':
                target = upper_q[idx] if dot >= 0 else lower_q[idx]
                outer_q[idx] += 0.25 * (target - outer_q[idx])
                inner_q[idx] += 0.15 * (target - inner_q[idx])

            else: 
                outer_q[idx] += 0.25 * ((lower_q[idx] - outer_q[idx]) if dot >= 0 else (upper_q[idx] - outer_q[idx]))
                inner_q[idx] += 0.15 * ((upper_q[idx] - inner_q[idx]) if dot >= 0 else (lower_q[idx] - inner_q[idx]))
                
        outer_q_batch.append(outer_q)
        inner_q_batch.append(inner_q)

    outer_q_batch = torch.stack(outer_q_batch, dim=0)
    inner_q_batch = torch.stack(inner_q_batch, dim=0)

    if q_para.ndim == 2:  # batch
        return outer_q_batch.to(q_para.device), inner_q_batch.to(q_para.device)
    else:
        return outer_q_batch[0].to(q_para.device), inner_q_batch[0].to(q_para.device)