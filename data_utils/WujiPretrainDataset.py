"""
Paper-accurate pretraining dataset for WujiHand Stage 1.

Implements the exact procedure from DRO-Grasp paper Section III-A:
  - P^A (grasp configuration): FK with sampled finger joint angles
  - P^B (canonical configuration): SAME wrist 6D pose, all finger joints at 0

Key property: Index consistency is guaranteed because the same pre-sampled
link mesh vertices are transformed by FK for both P^A and P^B.
Point i always corresponds to the same physical location on the hand mesh.

Supports both offline data (from generate_wujihand_pretrain_data.py) and
online generation mode (no pre-saved data needed).
"""

import os
import sys
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.hand_model import create_hand_model
from utils.func_utils import farthest_point_sampling


class WujiPretrainDataset(Dataset):
    """
    Pretraining dataset that generates (P^A, P^B) pairs for contrastive learning.

    Args:
        num_samples: Number of synthetic samples to use (or generate on-the-fly)
        use_synthetic_data: If True, load from MultiDex_filtered/wujihand/wujihand.pt
                           If False, generate data on-the-fly (no pre-saved data needed)
        num_points: Total hand points after FPS (paper uses 512)
        link_num_points: Points sampled per link mesh (before FPS)
        device: torch device
    """

    def __init__(
        self,
        num_samples=50000,
        use_synthetic_data=True,
        num_points=512,
        link_num_points=512,
        device=None,
    ):
        self.num_samples = num_samples
        self.use_synthetic_data = use_synthetic_data
        self.num_points = num_points
        self.link_num_points = link_num_points
        self.device = device if device is not None else torch.device('cpu')

        self.hand = create_hand_model(
            'wujihand',
            device=self.device,
            num_points=link_num_points
        )

        if use_synthetic_data:
            dataset_path = os.path.join(
                ROOT_DIR, 'data', 'MultiDex_filtered', 'wujihand', 'wujihand.pt'
            )
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(
                    f"Synthetic data not found at {dataset_path}. "
                    f"Run: python scripts/generate_wujihand_pretrain_data.py first."
                )
            dataset = torch.load(dataset_path, map_location='cpu')
            self.metadata = dataset['metadata']
            if len(self.metadata) > num_samples:
                self.metadata = self.metadata[:num_samples]
            print(f"Loaded {len(self.metadata)} grasp configurations from {dataset_path}")
        else:
            # Online generation mode: no metadata needed
            self.metadata = [None] * num_samples
            print(f"Online generation mode: {num_samples} samples will be generated on-the-fly")

        # Pre-compute joint limits for online sampling
        lower, upper = self.hand.pk_chain.get_joint_limits()
        self.joint_lower = torch.tensor(lower, dtype=torch.float32)
        self.joint_upper = torch.tensor(upper, dtype=torch.float32)

    def _sample_grasp_q(self):
        """Generate a random grasp configuration (same logic as generate script)."""
        q = torch.zeros(self.hand.dof, dtype=torch.float32)

        # Wrist 6D pose
        q[0:3] = (torch.rand(3) * 2 - 1) * 0.3
        axis = torch.randn(3)
        axis = axis / torch.norm(axis)
        angle = (torch.rand(1) * 2 - 1) * torch.pi
        q[3:6] = axis * angle

        # Finger joints: synergistic sampling
        finger_lower = self.joint_lower[6:]
        finger_upper = self.joint_upper[6:]

        for finger_idx in range(5):
            base = finger_idx * 4
            closure = torch.rand(1).item() * 0.8 + 0.1

            # Joint 1 (root flexion)
            j1_low, j1_high = finger_lower[base].item(), finger_upper[base].item()
            q[6 + base] = j1_low + closure * (j1_high - j1_low)
            q[6 + base] += torch.randn(1).item() * 0.15 * (j1_high - j1_low)

            # Joint 2 (abduction): near zero
            j2_low, j2_high = finger_lower[base + 1].item(), finger_upper[base + 1].item()
            q[6 + base + 1] = torch.randn(1).item() * 0.05 * (j2_high - j2_low)

            # Joint 3 (proximal flexion)
            j3_low, j3_high = finger_lower[base + 2].item(), finger_upper[base + 2].item()
            q[6 + base + 2] = j3_low + closure * (j3_high - j3_low)
            q[6 + base + 2] += torch.randn(1).item() * 0.15 * (j3_high - j3_low)

            # Joint 4 (distal flexion)
            j4_low, j4_high = finger_lower[base + 3].item(), finger_upper[base + 3].item()
            distal_ratio = closure * (0.7 + 0.3 * torch.rand(1).item())
            q[6 + base + 3] = j4_low + distal_ratio * (j4_high - j4_low)
            q[6 + base + 3] += torch.randn(1).item() * 0.1 * (j4_high - j4_low)

        q[6:] = torch.clamp(q[6:], finger_lower, finger_upper)
        return q

    def _get_robot_pc(self, q, apply_fps=True):
        """
        Transform pre-sampled link vertices by FK to get world-coordinate point cloud.

        Uses self.hand.vertices (pre-sampled mesh points) which guarantees:
        - Point i = same link + same barycentric coordinate → index consistency
        """
        # Transform all link vertices by FK
        all_pc = self.hand.get_transformed_links_pc(q, self.hand.vertices)
        pc_xyz = all_pc[:, :3]  # (total_link_points, 3)

        if apply_fps and self.num_points < pc_xyz.shape[0]:
            pc_xyz, _ = farthest_point_sampling(pc_xyz, self.num_points)

        return pc_xyz

    def __getitem__(self, index):
        if self.use_synthetic_data:
            grasp_q, _, _ = self.metadata[index]
            grasp_q = grasp_q.to(self.device)
        else:
            grasp_q = self._sample_grasp_q().to(self.device)

        # P^A: grasp configuration point cloud
        robot_pc_1 = self._get_robot_pc(grasp_q)  # (N, 3)

        # P^B: canonical (fully open) configuration — same wrist, fingers at 0
        canonical_q = self.hand.get_paper_canonical_q(grasp_q)
        robot_pc_2 = self._get_robot_pc(canonical_q)  # (N, 3)

        return {
            'robot_pc_1': robot_pc_1,
            'robot_pc_2': robot_pc_2,
        }

    def __len__(self):
        return self.num_samples


def create_wujihand_dataloader(
    num_samples=50000,
    use_synthetic_data=True,
    num_points=512,
    link_num_points=512,
    batch_size=16,
    num_workers=8,
    device=None,
):
    """
    Create a DataLoader for WujiHand Stage 1 pretraining.

    Args:
        num_samples: Total training samples per epoch
        use_synthetic_data: Load pre-generated data (True) or generate on-the-fly (False)
        num_points: Hand point cloud size (paper: 512)
        link_num_points: Points sampled per link mesh
        batch_size: Batch size
        num_workers: DataLoader workers
        device: torch device

    Returns:
        torch.utils.data.DataLoader
    """
    dataset = WujiPretrainDataset(
        num_samples=num_samples,
        use_synthetic_data=use_synthetic_data,
        num_points=num_points,
        link_num_points=link_num_points,
        device=device,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )
    return dataloader


if __name__ == '__main__':
    # Quick test
    print("Testing WujiPretrainDataset...")
    ds = WujiPretrainDataset(
        num_samples=8,
        use_synthetic_data=False,  # online generation for testing
        num_points=512,
        link_num_points=64,  # fewer points for quick test
    )
    batch = ds[0]
    print(f"  robot_pc_1 (P^A, grasp):   {batch['robot_pc_1'].shape}")
    print(f"  robot_pc_2 (P^B, canonical): {batch['robot_pc_2'].shape}")
    print(f"  Wrist diff (should be ~0): {(batch['robot_pc_1'].mean(0) - batch['robot_pc_2'].mean(0)).norm():.4f}")
    print("Done!")
