"""
Training script for WujiHand Stage 1: Configuration-Invariant Pretraining.

Implements the exact procedure from DRO-Grasp paper Section III-A:
  1. Load WujiHand encoder (Static Graph CNN, DGCNN-based)
  2. Create paper-accurate pretraining dataset (P^A=grasp, P^B=canonical)
  3. Train with contrastive loss (dist2weight + InfoNCE)
  4. Save frozen encoder for Stage 2

Usage:
    # Option A: Use pre-generated synthetic data (recommended)
    python scripts/generate_wujihand_pretrain_data.py --num_samples 50000
    python scripts/train_wujihand_pretrain.py

    # Option B: Online data generation (no pre-generated data needed)
    python scripts/train_wujihand_pretrain.py --online

    # Option C: With Hydra config override
    python scripts/train_wujihand_pretrain.py --config-name pretrain_wujihand
"""

import os
import sys
import argparse
import warnings
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from model.module import PretrainingModule
from model.network import create_encoder_network
from data_utils.WujiPretrainDataset import create_wujihand_dataloader


def main():
    parser = argparse.ArgumentParser(description='WujiHand Stage 1 Pretraining')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_points', type=int, default=512,
                        help='Total hand points (paper: N_R = 512)')
    parser.add_argument('--link_num_points', type=int, default=512,
                        help='Points sampled per link mesh')
    parser.add_argument('--num_samples', type=int, default=50000,
                        help='Samples per epoch')
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--emb_dim', type=int, default=512)
    parser.add_argument('--save_every_n_epoch', type=int, default=5)
    parser.add_argument('--gpu', type=int, nargs='+', default=[0],
                        help='GPU device IDs')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--online', action='store_true',
                        help='Use online data generation (no pre-saved data needed)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: output/pretrain_wujihand)')
    parser.add_argument('--name', type=str, default='pretrain_wujihand',
                        help='Experiment name for wandb logging')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable wandb logging')
    args = parser.parse_args()

    use_synthetic = not args.online

    # Validate data availability
    if use_synthetic:
        data_path = os.path.join(
            ROOT_DIR, 'data', 'MultiDex_filtered', 'wujihand', 'wujihand.pt'
        )
        if not os.path.exists(data_path):
            print(f"[WARNING] Synthetic data not found at: {data_path}")
            print("Switching to online generation mode.")
            print("To pre-generate data, run: python scripts/generate_wujihand_pretrain_data.py")
            use_synthetic = False

    output_dir = args.output_dir or os.path.join(
        ROOT_DIR, 'output', 'pretrain_wujihand'
    )
    save_dir = os.path.join(output_dir, 'state_dict')
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 70)
    print("WujiHand Stage 1: Configuration-Invariant Pretraining")
    print("=" * 70)
    print(f"  Mode: {'Pre-generated data' if use_synthetic else 'Online generation'}")
    print(f"  Hand points (N_R): {args.num_points}")
    print(f"  Link points: {args.link_num_points}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Samples/epoch: {args.num_samples}")
    print(f"  Max epochs: {args.max_epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Embedding dim: {args.emb_dim}")
    print(f"  GPU: {args.gpu}")
    print(f"  Output: {output_dir}")
    print("=" * 70)

    pl.seed_everything(args.seed)

    # Create dataloader
    dataloader = create_wujihand_dataloader(
        num_samples=args.num_samples,
        use_synthetic_data=use_synthetic,
        num_points=args.num_points,
        link_num_points=args.link_num_points,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"\nDataloader created: {len(dataloader)} batches/epoch")

    # Create encoder (train from scratch, no pretrain)
    encoder = create_encoder_network(emb_dim=args.emb_dim)
    print(f"Encoder created: Static Graph CNN (DGCNN, K=32)")

    # Create Lightning training config
    class TrainingCfg:
        lr = args.lr
        temperature = args.temperature
        save_dir = save_dir
        save_every_n_epoch = args.save_every_n_epoch

    # Create LightningModule
    model = PretrainingModule(cfg=TrainingCfg(), encoder=encoder)

    # Setup logger
    if args.no_wandb:
        logger = None
    else:
        logger = WandbLogger(
            name=args.name,
            save_dir=output_dir,
            project='DROGrasp-Pretrain',
        )

    # Create trainer
    trainer = pl.Trainer(
        logger=logger,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=args.gpu,
        max_epochs=args.max_epochs,
        log_every_n_steps=5,
    )

    print(f"\nStarting training...")
    print(f"  Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print(f"  Logging: {'WandB' if not args.no_wandb else 'None'}")
    print(f"  Checkpoints: {save_dir}")
    print()

    trainer.fit(model, dataloader)

    # Save final checkpoint
    final_path = os.path.join(save_dir, f'epoch_{args.max_epochs}.pth')
    if not os.path.exists(final_path):
        # Trainer may have saved it; if not, save explicitly
        final_path = os.path.join(save_dir, 'final.pth')
        torch.save(encoder.state_dict(), final_path)
        print(f"Final encoder saved to: {final_path}")

    print("\n" + "=" * 70)
    print("Stage 1 Pretraining Complete!")
    print(f"Encoder saved to: {save_dir}")
    print()
    print("Next steps for Stage 2:")
    print(f"  1. Set model.pretrain = '{os.path.relpath(final_path, ROOT_DIR)}' in config")
    print("  2. Train the full DRO-Grasp network with frozen encoder")
    print("=" * 70)


if __name__ == '__main__':
    torch.set_float32_matmul_precision("high")
    torch.autograd.set_detect_anomaly(True)
    torch.cuda.empty_cache()
    torch.multiprocessing.set_sharing_strategy("file_system")
    warnings.simplefilter(action='ignore', category=FutureWarning)
    main()
