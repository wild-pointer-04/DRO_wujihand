import os
import sys
import time
import warnings
import numpy as np
from tqdm import tqdm
from termcolor import cprint
from types import SimpleNamespace
import torch

# 添加项目根目录到路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from model.network import create_network
from data_utils.CMapDataset import create_dataloader
from utils.multilateration import multilateration
from utils.se3_transform import compute_link_pose
from utils.optimization import process_transform, create_problem, optimization
from utils.hand_model import create_hand_model
from validation.validate_utils import validate_isaac

# === 配置区域 ===
gpu = 0
device = torch.device(f'cuda:{gpu}')
# 确保你有这个权重文件
ckpt_name = 'model_shadowhand'  # 可选: ‘model_3robots‘, 'model_3robots_partial', 'model_allegro', 'model_barrett', 'model_shadowhand'
batch_size = 10  # 如果显存不够可以调小
# =================

def main():
    # 1. 创建并加载网络
    cprint(f"=> Loading network checkpoint: {ckpt_name}...", 'cyan')
    network = create_network(
        SimpleNamespace(**{
            'emb_dim': 512,
            'latent_dim': 64,
            'pretrain': None,
            'center_pc': True,
            'block_computing': True
        }),
        mode='validate'
    ).to(device)
    
    ckpt_path = f"ckpt/model/{ckpt_name}.pth"
    if not os.path.exists(ckpt_path):
        cprint(f"Error: Checkpoint not found at {ckpt_path}", 'red')
        return

    network.load_state_dict(torch.load(ckpt_path, map_location=device))
    network.eval()

    # 2. 准备数据加载器 (只包含原始的3个机器人)
    cprint("=> Creating dataloader for [barrett, allegro, shadowhand]...", 'cyan')
    dataloader = create_dataloader(
        SimpleNamespace(**{
            'batch_size': batch_size,
            'robot_names': ['wujihand'], #['barrett', 'allegro', 'shadowhand'], # 这里不包含 omnihand，因为我们测试的是旧模型
            'debug_object_names': ['contactdb+apple'],
            'object_pc_type': 'random' if ckpt_name != 'model_3robots_partial' else 'partial',
            'num_workers': 8 # 可以根据你的CPU核心数调整
        }),
        is_train=False
    )

    global_robot_name = None
    hand = None
    all_success_q = []
    time_list = []
    success_num = 0
    total_num = 0

    cprint("=> Start validating...", 'cyan')
    
    for i, data in enumerate(dataloader):
        robot_name = data['robot_name'] # 当前 batch 是哪个机器人
        object_name = data['object_name']

        # 如果换机器人了，打印上一个机器人的统计结果
        if robot_name != global_robot_name:
            if global_robot_name is not None:
                # 统计逻辑
                if len(all_success_q) > 0:
                    all_success_q_tensor = torch.cat(all_success_q, dim=0)
                    diversity_std = torch.std(all_success_q_tensor, dim=0).mean()
                else:
                    diversity_std = 0.0
                    
                times = np.array(time_list)
                time_mean = np.mean(times) if len(times) > 0 else 0
                time_std = np.std(times) if len(times) > 0 else 0

                success_rate = success_num / total_num * 100 if total_num > 0 else 0
                cprint(f"\n[{global_robot_name} Summary]", 'magenta', attrs=['bold'])
                cprint(f"Success Rate: {success_num}/{total_num} ({success_rate:.2f}%)", 'yellow')
                cprint(f"Diversity Std: {diversity_std:.3f}", 'cyan')
                cprint(f"Optimization Time: {time_mean:.2f} ± {time_std:.2f} s", 'blue')

                # 重置计数器
                all_success_q = []
                time_list = []
                success_num = 0
                total_num = 0
            
            # 初始化新机器人的模型
            hand = create_hand_model(robot_name, device)
            global_robot_name = robot_name

        # === 核心推理循环 ===
        predict_q_list = []
        # 使用 tqdm 显示进度
        for data_idx in tqdm(range(batch_size), desc=f"{robot_name}/{object_name}", leave=False):
            initial_q = data['initial_q'][data_idx: data_idx + 1].to(device)
            robot_pc = data['robot_pc'][data_idx: data_idx + 1].to(device)
            object_pc = data['object_pc'][data_idx: data_idx + 1].to(device)

            # A. 网络预测
            with torch.no_grad():
                dro = network(robot_pc, object_pc)['dro'].detach()

            # B. 多边定位 & 姿态计算
            mlat_pc = multilateration(dro, object_pc)
            transform, _ = compute_link_pose(hand.links_pc, mlat_pc, is_train=False)
            optim_transform = process_transform(hand.pk_chain, transform)

            # C. 逆运动学优化 (Optimization)
            layer = create_problem(hand.pk_chain, optim_transform.keys())
            start_time = time.time()
            predict_q = optimization(hand.pk_chain, layer, initial_q, optim_transform)
            print(f"Predict Q: {predict_q}")
            end_time = time.time()
            time_list.append(end_time - start_time)

            predict_q_list.append(predict_q)

        predict_q_batch = torch.cat(predict_q_list, dim=0)
        
        # D. Isaac Gym 仿真验证
        # 这一步会调用 Isaac Gym，可能会比较慢，且需要图形界面或 Headless 设置正确
        success, isaac_q = validate_isaac(robot_name, object_name, predict_q_batch, gpu=gpu)
        
        succ_num = success.sum().item() if success is not None else 0
        if success is not None:
            success_q = predict_q_batch[success]
            all_success_q.append(success_q)

        # 实时打印当前 batch 结果
        cprint(f"  Result: {succ_num}/{batch_size} ({succ_num / batch_size * 100:.0f}%)", 'green' if succ_num > 0 else 'red')
        success_num += succ_num
        total_num += batch_size

    # === 打印最后一个机器人的结果 ===
    if global_robot_name is not None:
        if len(all_success_q) > 0:
            all_success_q_tensor = torch.cat(all_success_q, dim=0)
            diversity_std = torch.std(all_success_q_tensor, dim=0).mean()
        else:
            diversity_std = 0.0
        
        times = np.array(time_list)
        time_mean = np.mean(times) if len(times) > 0 else 0
        time_std = np.std(times) if len(times) > 0 else 0

        success_rate = success_num / total_num * 100 if total_num > 0 else 0
        cprint(f"\n[{global_robot_name} Summary]", 'magenta', attrs=['bold'])
        cprint(f"Success Rate: {success_num}/{total_num} ({success_rate:.2f}%)", 'yellow')
        cprint(f"Diversity Std: {diversity_std:.3f}", 'cyan')
        cprint(f"Optimization Time: {time_mean:.2f} ± {time_std:.2f} s", 'blue')


if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)
    torch.set_num_threads(8)
    main()