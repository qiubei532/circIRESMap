#!/usr/bin/env python3
"""
circIRESMap configuration file.
All hyperparameters have been optimized through a 4-stage search.
"""

import os
import sys
import csv
import random
import numpy as np
import torch

try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2147483647)

RANDOM_SEED = 42

N_FOLDS = 5

FASTA_PREFIX = 'data/circRNA'

OUTPUT_DIR = 'results'

PREFERRED_GPU_INDEX = 0

# ==================== Stage 1: Training hyperparameters ====================
BEST_LR = 1e-3
BEST_FOCAL_GAMMA = 1.5
BEST_FOCAL_ALPHA = 0.5

# ==================== Stage 2: Data & sampling parameters ====================
BEST_WINDOW_SIZE = 151
BEST_N_BAGS = 9
BEST_BATCH_SIZE = 128

# ==================== Stage 3: Model structure parameters ====================
BEST_NUM_FILTERS = 128
BEST_BIGRU_HIDDEN_SIZE = 128
BEST_BIGRU_NUM_LAYERS = 2

# ==================== Stage 4: DAMSR parameters ====================
BEST_DAMSR_LAYERS = 1
BEST_DAMSR_NUM_HEADS = 2
BEST_CONTEXT_POOL_GAMMA = 0.97

# ==================== Training budget ====================
MAX_EPOCHS = 100
EARLY_STOP_PATIENCE = 15

USE_CIRCULAR = True


def set_random_seed(seed=RANDOM_SEED):
    """设置所有随机种子，确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_runtime_device() -> torch.device:
    """根据配置选择运行设备，并固定当前 CUDA 设备。"""
    if not torch.cuda.is_available():
        return torch.device('cpu')

    gpu_index = PREFERRED_GPU_INDEX
    env_gpu_index = os.getenv('DEEPCIP_GPU_INDEX')
    if env_gpu_index is not None:
        try:
            gpu_index = int(env_gpu_index)
        except ValueError:
            print(f"警告: DEEPCIP_GPU_INDEX={env_gpu_index} 不是整数，回退到 {PREFERRED_GPU_INDEX}")
            gpu_index = PREFERRED_GPU_INDEX

    gpu_count = torch.cuda.device_count()
    if gpu_index < 0 or gpu_index >= gpu_count:
        print(f"警告: GPU 索引 {gpu_index} 超出范围 [0, {gpu_count - 1}]，回退到 0")
        gpu_index = 0

    torch.cuda.set_device(gpu_index)
    return torch.device(f'cuda:{gpu_index}')


def print_gpu_info(device: torch.device):
    """打印当前可见 GPU 与实际使用 GPU 信息。"""
    print("\n" + "=" * 80)
    print("GPU 信息")
    print("=" * 80)

    if device.type != 'cuda':
        print("当前使用设备: CPU（未检测到可用 CUDA）")
        return

    print(f"当前使用设备: {device}")
    print(f"可见 GPU 数量: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total_memory_gb = props.total_memory / (1024 ** 3)
        marker = " <== 使用中" if i == device.index else ""
        print(
            f"  GPU {i}: {props.name}, 显存 {total_memory_gb:.2f} GB, "
            f"算力 {props.major}.{props.minor}{marker}"
        )

    print(f"PyTorch CUDA 版本: {torch.version.cuda}")