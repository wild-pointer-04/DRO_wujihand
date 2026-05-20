"""
Generate synthetic grasp configurations for WujiHand Stage 1 pretraining.

Algorithm (paper Section III-A):
- Random wrist 6D pose (translation + rotation)
- Synergistic finger joint sampling (flexion joints bend together)
- Saves in MultiDex format compatible with existing PretrainDataset.

Usage:
    python scripts/generate_wujihand_pretrain_data.py --num_samples 50000
"""

import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.hand_model import create_hand_model


def sample_wrist_pose(hand, device):
    """
    Sample a random wrist 6D pose.
    Virtual joints: [tx, ty, tz, roll, pitch, yaw] (indices 0-5)
    Returns tensor of shape (dof,) with first 6 elements set.
    """
    q = torch.zeros(hand.dof, dtype=torch.float32, device=device)

    # Translation: random within a reasonable range
    q[0:3] = (torch.rand(3, device=device) * 2 - 1) * 0.3  # [-0.3, 0.3]

    # Rotation: random orientation (axis-angle sampling)
    axis = torch.randn(3, device=device)
    axis = axis / torch.norm(axis)
    angle = (torch.rand(1, device=device) * 2 - 1) * torch.pi  # [-pi, pi]
    q[3:6] = axis * angle

    return q


def sample_finger_joints(hand, device):
    """
    Synergistic sampling of finger joints (indices 6:).

    WujiHand joint structure (per finger, 4 joints each):
      - Joint 1: root flexion (axis ~Y)
      - Joint 2: abduction (axis ~Z for fingers 2-5, axis ~-Y for finger1)
      - Joint 3: proximal flexion (axis ~Y)
      - Joint 4: distal flexion (axis ~Y)

    Closure strategy:
      - Flexion joints (1, 3, 4): coordinated closure with ratio α ∈ [0.1, 0.9]
      - Abduction joints (2): small noise near 0
    """
    lower, upper = hand.pk_chain.get_joint_limits()
    lower = torch.tensor(lower, dtype=torch.float32, device=device)
    upper = torch.tensor(upper, dtype=torch.float32, device=device)

    finger_lower = lower[6:]  # (20,)
    finger_upper = upper[6:]  # (20,)

    q_fingers = torch.zeros(20, dtype=torch.float32, device=device)

    for finger_idx in range(5):
        base = finger_idx * 4

        # Sample a common closure ratio for this finger's flexion joints
        closure_ratio = torch.rand(1, device=device).item() * 0.8 + 0.1  # [0.1, 0.9]

        # Joint 1 (root flexion): primary closure
        j1_low, j1_high = finger_lower[base].item(), finger_upper[base].item()
        q_fingers[base] = j1_low + closure_ratio * (j1_high - j1_low)
        q_fingers[base] += torch.randn(1, device=device).item() * 0.15 * (j1_high - j1_low)

        # Joint 2 (abduction): stay near 0 with small noise
        j2_low, j2_high = finger_lower[base + 1].item(), finger_upper[base + 1].item()
        j2_range = j2_high - j2_low
        q_fingers[base + 1] = torch.randn(1, device=device).item() * 0.05 * j2_range

        # Joint 3 (proximal flexion): coordinated with joint 1
        j3_low, j3_high = finger_lower[base + 2].item(), finger_upper[base + 2].item()
        q_fingers[base + 2] = j3_low + closure_ratio * (j3_high - j3_low)
        q_fingers[base + 2] += torch.randn(1, device=device).item() * 0.15 * (j3_high - j3_low)

        # Joint 4 (distal flexion): slightly less closure (coupled to proximal)
        j4_low, j4_high = finger_lower[base + 3].item(), finger_upper[base + 3].item()
        distal_ratio = closure_ratio * (0.7 + 0.3 * torch.rand(1, device=device).item())
        q_fingers[base + 3] = j4_low + distal_ratio * (j4_high - j4_low)
        q_fingers[base + 3] += torch.randn(1, device=device).item() * 0.1 * (j4_high - j4_low)

    # Clamp to joint limits
    q_fingers = torch.clamp(q_fingers, finger_lower, finger_upper)

    return q_fingers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=50000,
                        help='Number of synthetic grasp configurations to generate')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    print(f"Loading WujiHand model...")
    hand = create_hand_model('wujihand', device=device)
    print(f"  DOF: {hand.dof}")
    print(f"  Joint names: {hand.get_joint_orders()}")
    print(f"  Link vertices: {list(hand.vertices.keys())}")

    output_dir = os.path.join(ROOT_DIR, 'data', 'MultiDex_filtered', 'wujihand')
    os.makedirs(output_dir, exist_ok=True)

    metadata = []
    print(f"\nGenerating {args.num_samples} synthetic grasp configurations...")
    for _ in tqdm(range(args.num_samples)):
        q = sample_wrist_pose(hand, device)
        q[6:] = sample_finger_joints(hand, device)
        metadata.append((q.cpu(), 'synthetic', 'wujihand'))

    output_path = os.path.join(output_dir, 'wujihand.pt')
    data = {
        'info': {
            'robot_name': 'wujihand',
            'num_samples': args.num_samples,
            'generation_method': 'synergistic_sampling',
        },
        'metadata': metadata,
    }
    torch.save(data, output_path)
    print(f"\nSaved {args.num_samples} grasp configurations to: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")


if __name__ == '__main__':
    main()
