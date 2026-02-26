# 添加了 omnihand 适配的 controller 函数

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
    if joint_name.startswith('virtual'):
        return None

    if robot_name == 'allegro':
        if joint_name in ['joint_0.0', 'joint_4.0', 'joint_8.0', 'joint_13.0']:
            return None
        link_dir = torch.tensor([0, 0, 1], dtype=torch.float32)
    elif robot_name == 'barrett':
        if joint_name in ['bh_j11_joint', 'bh_j21_joint']:
            return None
        link_dir = torch.tensor([-1, 0, 0], dtype=torch.float32)
    elif robot_name == 'ezgripper':
        link_dir = torch.tensor([1, 0, 0], dtype=torch.float32)
    elif robot_name == 'robotiq_3finger':
        if joint_name in ['gripper_fingerB_knuckle', 'gripper_fingerC_knuckle']:
            return None
        link_dir = torch.tensor([0, 0, -1], dtype=torch.float32)
    elif robot_name == 'shadowhand':
        if joint_name in ['WRJ2', 'WRJ1']:
            return None
        if joint_name != 'THJ5':
            link_dir = torch.tensor([0, 0, 1], dtype=torch.float32)
        else:
            link_dir = torch.tensor([1, 0, 0], dtype=torch.float32)
    elif robot_name == 'leaphand':
        if joint_name in ['13']:
            return None
        if joint_name in ['0', '4', '8']:
            link_dir = torch.tensor([1, 0, 0], dtype=torch.float32)
        elif joint_name in ['1', '5', '9', '12', '14']:
            link_dir = torch.tensor([0, 1, 0], dtype=torch.float32)
        else:
            link_dir = torch.tensor([0, -1, 0], dtype=torch.float32)

    elif robot_name == 'omnihand':
        # 1. 【强力过滤名单】
        # wrist/palm: 手腕手掌
        # metacarpal: 掌骨 (食指根部错误的来源)
        # roll: 侧摆关节 (会导致手指扭曲)
        # cmc: 腕掌关节 (大拇指根部指向肉里的来源)
        # 只要名字里带这些，统统屏蔽，不画箭头！
        if any(x in joint_name for x in ['wrist', 'palm', 'metacarpal', 'roll', 'cmc']):
            return None

        # 2. 【大拇指修正】 (Thumb)
        if 'thumb' in joint_name:
            return torch.tensor([0, 0, 1], dtype=torch.float32)

        # 3. 【四指根部】 (Proximal)
        # 这个保持 [1,0,0] 不变，因为你之前验证过这能修好四指扭曲
        if 'proximal' in joint_name:
            return torch.tensor([1, 0, 0], dtype=torch.float32)

        # 4. 【四指中段/末端】 (PIP/Distal)
        # 保持 [0,0,1] 不变
        return torch.tensor([0, 0, 1], dtype=torch.float32)

    elif robot_name == 'wujihand':
        # 1. 屏蔽侧摆 (Joint 2)
        if 'joint2' in joint_name:
            return None

        # ===========================================================
        # A 组：食指 (Finger 2) - 【已确认正确】
        # ===========================================================
        if 'finger2' in joint_name:
            if 'joint1' in joint_name:
                return torch.tensor([0., 0., 1.]) 
            else:
                return torch.tensor([0., 1., 0.]) 

        # ===========================================================
        # B 组：中/无/小 (Finger 3, 4, 5) - 【已确认正确】
        # ===========================================================
        elif any(f in joint_name for f in ['finger3', 'finger4', 'finger5']):
            if 'joint1' in joint_name:
                return torch.tensor([0., 0., 1.]) 
            else:
                return torch.tensor([0., -1., 0.])

        # ===========================================================
        # C 组：大拇指 (Finger 1) - 【尝试 Z 轴】
        # ===========================================================
        elif 'finger1' in joint_name:
            # 既然 X 和 Y 都会顺着手指或指向旁边，那肯定就是 Z 了
            # 我们先试 [0, 0, 1]
            # 如果指向指背，改成 [0, 0, -1] 即可
            return torch.tensor([0., 0., 1.])
        
        else:
             return torch.tensor([0., -1., 0.])
        

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
            idx = joint_orders.index(joint_name)
            if robot_name == 'robotiq_3finger':  # open -> upper, close -> lower
                outer_q[idx] += 0.25 * ((outer_q[idx] - lower_q[idx]) if dot <= 0 else (outer_q[idx] - upper_q[idx]))
                inner_q[idx] += 0.15 * ((inner_q[idx] - upper_q[idx]) if dot <= 0 else (inner_q[idx] - lower_q[idx]))
            
            #加了 omnihand 的控制逻辑
            if robot_name == 'omnihand':
                # 看到物体(dot>0) -> 去抓紧(upper)
                target = upper_q[idx] if dot >= 0 else lower_q[idx]
                
                outer_q[idx] += 0.25 * (target - outer_q[idx])
                inner_q[idx] += 0.15 * (target - inner_q[idx])
            #结束

            else:  # open -> lower, close -> upper
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