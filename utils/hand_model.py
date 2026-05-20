import os
import sys
import json
import math
import random
import numpy as np
import torch
import trimesh
import pytorch_kinematics as pk

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.func_utils import farthest_point_sampling
from utils.mesh_utils import load_link_geometries
from utils.rotation import *


class HandModel:
    def __init__(
        self,
        robot_name,
        urdf_path,
        meshes_path,
        links_pc_path,
        device,
        link_num_points=512
    ):
        self.robot_name = robot_name
        self.urdf_path = urdf_path
        self.meshes_path = meshes_path
        self.device = device

        # 加载运动学链
        self.pk_chain = pk.build_chain_from_urdf(open(urdf_path).read()).to(dtype=torch.float32, device=device)
        self.dof = len(self.pk_chain.get_joint_parameter_names())
        
        # 加载点云数据
        if os.path.exists(links_pc_path):
            links_pc_data = torch.load(links_pc_path, map_location=device)
            self.links_pc = links_pc_data['filtered']
            self.links_pc_original = links_pc_data['original']
        else:
            self.links_pc = None
            self.links_pc_original = None

        # 加载 Mesh
        self.meshes = load_link_geometries(robot_name, self.urdf_path, self.pk_chain.get_link_names())

        self.vertices = {}
        # 加载不需要处理的 link 列表
        removed_links_path = os.path.join(ROOT_DIR, 'data_utils/removed_links.json')
        if os.path.exists(removed_links_path):
            removed_links = json.load(open(removed_links_path))[robot_name]
        else:
            removed_links = []

        for link_name, link_mesh in self.meshes.items():
            if link_name in removed_links:
                continue
            v = link_mesh.sample(link_num_points)
            self.vertices[link_name] = v

        self.frame_status = None

        # =================================================================
        # [WujiHand 修正逻辑]
        # 目标：统一修正坐标系，将默认探测方向 Z轴(蓝) 旋转到 指腹方向 X轴(红)
        # =================================================================
        self.contact_point_orthogonal_rotation = {}

        if self.robot_name == 'wujihand':
            # 定义旋转矩阵：绕 Y 轴旋转 +90 度
            # 效果：Z轴(0,0,1) -> X轴(1,0,0)
            rot_matrix = torch.tensor([
                [0, 0, 1],   
                [0, 1, 0],   
                [-1, 0, 0]   
            ], dtype=torch.float32, device=device)
            
            # 应用到所有有效 Link (大拇指和四指逻辑一致)
            for link_name in self.vertices.keys():
                self.contact_point_orthogonal_rotation[link_name] = rot_matrix
        
        elif self.robot_name == 'shadowhand':
            # Shadowhand 默认不做额外旋转，给个单位矩阵防止报错
            identity_rot = torch.eye(3, dtype=torch.float32, device=device)
            for link_name in self.vertices.keys():
                self.contact_point_orthogonal_rotation[link_name] = identity_rot

    def get_joint_orders(self):
        return [joint.name for joint in self.pk_chain.get_joints()]

    def update_status(self, q):
        if q.shape[-1] != self.dof:
            q = q_rot6d_to_q_euler(q)
        self.frame_status = self.pk_chain.forward_kinematics(q.to(self.device))

    def get_transformed_links_pc(self, q=None, links_pc=None):
        """
        Use robot link pc & q value to get point cloud.
        """
        if q is None:
            q = torch.zeros(self.dof, dtype=torch.float32, device=self.device)
        self.update_status(q)
        if links_pc is None:
            links_pc = self.links_pc

        all_pc_se3 = []
        for link_index, (link_name, link_pc) in enumerate(links_pc.items()):
            if not torch.is_tensor(link_pc):
                link_pc = torch.tensor(link_pc, dtype=torch.float32, device=q.device)
            n_link = link_pc.shape[0]
            se3 = self.frame_status[link_name].get_matrix()[0].to(q.device)
            homogeneous_tensor = torch.ones(n_link, 1, device=q.device)
            link_pc_homogeneous = torch.cat([link_pc.to(q.device), homogeneous_tensor], dim=1)
            link_pc_se3 = (link_pc_homogeneous @ se3.T)[:, :3]
            index_tensor = torch.full([n_link, 1], float(link_index), device=q.device)
            link_pc_se3_index = torch.cat([link_pc_se3, index_tensor], dim=1)
            all_pc_se3.append(link_pc_se3_index)
        all_pc_se3 = torch.cat(all_pc_se3, dim=0)

        return all_pc_se3

    def get_sampled_pc(self, q=None, num_points=512):
        if q is None:
            q = self.get_canonical_q()

        sampled_pc = self.get_transformed_links_pc(q, self.vertices)
        return farthest_point_sampling(sampled_pc, num_points)

    def get_canonical_q(self):
        """ For visualization purposes only. """
        lower, upper = self.pk_chain.get_joint_limits()
        # 将 list 转换为 tensor 以支持运算
        lower = torch.tensor(lower, dtype=torch.float32, device=self.device)
        upper = torch.tensor(upper, dtype=torch.float32, device=self.device)

        canonical_q = lower * 0.75 + upper * 0.25
        canonical_q[:6] = 0
        return canonical_q

    def get_paper_canonical_q(self, q):
        """
        Paper Section III-A: keep the wrist 6D pose from q,
        set all finger joints (index 6:) to exactly 0 (fully open hand).

        This produces P^B (canonical configuration) with the SAME wrist pose
        as P^A (grasp configuration), ensuring index consistency for contrastive learning.
        """
        canonical_q = q.clone()
        canonical_q[6:] = 0.0
        return canonical_q

    def get_initial_q(self, q=None, max_angle: float = math.pi / 6):
        """
        Compute the robot initial joint value q based on the target grasp.
        """
        if q is None:  # random sample root rotation and joint values
            q_initial = torch.zeros(self.dof, dtype=torch.float32, device=self.device)

            q_initial[3:6] = (torch.rand(3) * 2 - 1) * torch.pi
            q_initial[5] /= 2

            lower_joint_limits, upper_joint_limits = self.pk_chain.get_joint_limits()
            # 转为 tensor
            lower_joint_limits = torch.tensor(lower_joint_limits, dtype=torch.float32, device=self.device)
            upper_joint_limits = torch.tensor(upper_joint_limits, dtype=torch.float32, device=self.device)
            
            lower_limits = lower_joint_limits[6:]
            upper_limits = upper_joint_limits[6:]
            portion = random.uniform(0.65, 0.85)
            q_initial[6:] = lower_limits * portion + upper_limits * (1 - portion)
        else:
            if len(q) == self.dof:
                q = q_euler_to_q_rot6d(q)
            q_initial = q.clone()

            # compute random initial rotation
            direction = - q_initial[:3] / torch.norm(q_initial[:3])
            angle = torch.tensor(random.uniform(0, max_angle), device=q.device)  # sample rotation angle
            axis = torch.randn(3).to(q.device)  # sample rotation axis
            axis -= torch.dot(axis, direction) * direction  # ensure orthogonality
            axis = axis / torch.norm(axis)
            random_rotation = axisangle_to_matrix(axis, angle).to(q.device)
            rotation_matrix = random_rotation @ rot6d_to_matrix(q_initial[3:9])
            q_initial[3:9] = matrix_to_rot6d(rotation_matrix)

            # compute random initial joint values
            lower_joint_limits, upper_joint_limits = self.pk_chain.get_joint_limits()
            lower_joint_limits = torch.tensor(lower_joint_limits, dtype=torch.float32, device=self.device)
            upper_joint_limits = torch.tensor(upper_joint_limits, dtype=torch.float32, device=self.device)
            
            lower_limits = lower_joint_limits[6:]
            upper_limits = upper_joint_limits[6:]
            
            portion = random.uniform(0.65, 0.85)
            q_initial[9:] = lower_limits * portion + upper_limits * (1 - portion)

            q_initial = q_rot6d_to_q_euler(q_initial)

        return q_initial

    def get_trimesh_q(self, q):
        """ Return the hand trimesh object corresponding to the input joint value q. """
        self.update_status(q)

        scene = trimesh.Scene()
        for link_name in self.vertices:
            mesh_transform_matrix = self.frame_status[link_name].get_matrix()[0].cpu().numpy()
            scene.add_geometry(self.meshes[link_name].copy().apply_transform(mesh_transform_matrix))

        vertices = []
        faces = []
        vertex_offset = 0
        for geom in scene.geometry.values():
            if isinstance(geom, trimesh.Trimesh):
                vertices.append(geom.vertices)
                faces.append(geom.faces + vertex_offset)
                vertex_offset += len(geom.vertices)
        if len(vertices) > 0:
            all_vertices = np.vstack(vertices)
            all_faces = np.vstack(faces)
        else:
            all_vertices = np.array([])
            all_faces = np.array([])

        parts = {}
        for link_name in self.meshes:
            mesh_transform_matrix = self.frame_status[link_name].get_matrix()[0].cpu().numpy()
            part_mesh = self.meshes[link_name].copy().apply_transform(mesh_transform_matrix)
            parts[link_name] = part_mesh

        return_dict = {
            'visual': trimesh.Trimesh(vertices=all_vertices, faces=all_faces),
            'parts': parts
        }
        return return_dict

    def get_trimesh_se3(self, transform, index):
        """ Return the hand trimesh object corresponding to the input transform. """
        scene = trimesh.Scene()
        for link_name in transform:
            mesh_transform_matrix = transform[link_name][index].cpu().numpy()
            scene.add_geometry(self.meshes[link_name].copy().apply_transform(mesh_transform_matrix))

        vertices = []
        faces = []
        vertex_offset = 0
        for geom in scene.geometry.values():
            if isinstance(geom, trimesh.Trimesh):
                vertices.append(geom.vertices)
                faces.append(geom.faces + vertex_offset)
                vertex_offset += len(geom.vertices)
        
        if len(vertices) > 0:
            all_vertices = np.vstack(vertices)
            all_faces = np.vstack(faces)
        else:
            all_vertices = np.array([])
            all_faces = np.array([])

        return trimesh.Trimesh(vertices=all_vertices, faces=all_faces)


# =========================================================
# 必须保留这个函数，外部文件通过这个函数创建 HandModel 实例
# =========================================================
def create_hand_model(
    robot_name,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    num_points=512
):
    json_path = os.path.join(ROOT_DIR, 'data/data_urdf/robot/urdf_assets_meta.json')
    if not os.path.exists(json_path):
         print(f"Warning: Meta file not found at {json_path}")
         
    urdf_assets_meta = json.load(open(json_path))
    urdf_path = os.path.join(ROOT_DIR, urdf_assets_meta['urdf_path'][robot_name])
    meshes_path = os.path.join(ROOT_DIR, urdf_assets_meta['meshes_path'][robot_name])
    links_pc_path = os.path.join(ROOT_DIR, f'data/PointCloud/robot/{robot_name}.pt')
    
    hand_model = HandModel(robot_name, urdf_path, meshes_path, links_pc_path, device, num_points)
    return hand_model