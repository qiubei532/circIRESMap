#!/usr/bin/env python3
"""
Data loading, preprocessing, and result output utilities.
"""

import os
import json
import random
import numpy as np
import pandas as pd
from typing import List, Set, Dict


def load_sequences(fasta_prefix: str) -> Dict[str, str]:
    """加载序列文件"""
    seq_file = f'{fasta_prefix}_seq.fasta'
    sequences = {}
    
    with open(seq_file, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith('>'):
                seq_id = line[1:]
                seq = lines[i + 1].strip()
                sequences[seq_id] = seq
    
    return sequences


def load_annotations(fasta_file: str, sequences: Dict[str, str]) -> Dict[str, np.ndarray]:
    """加载标注文件"""
    labels = {}
    
    with open(fasta_file, 'r') as f:
        lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if line.startswith('>'):
                parts = line[1:].split()
                if len(parts) == 1:
                    seq_id = parts[0]
                    if seq_id in sequences:
                        seq = sequences[seq_id]
                        labels[seq_id] = np.zeros(len(seq))
        
        for line in lines:
            line = line.strip()
            if line.startswith('>'):
                parts = line[1:].split()
                if len(parts) == 3:
                    seq_id = parts[0]
                    start = int(parts[1]) - 1
                    end = int(parts[2])
                    if seq_id in labels:
                        labels[seq_id][start:end] = 1
    
    return labels


def load_id_mapping(fasta_prefix: str) -> Dict[str, Dict]:
    """加载 ID 映射表"""
    mapping_file = f'{fasta_prefix}_id_mapping.tsv'
    
    try:
        df = pd.read_csv(mapping_file, sep='\t')
        mapping = {}
        
        for _, row in df.iterrows():
            unique_id = row['unique_id']
            mapping[unique_id] = {
                'circBase_ID': row.get('circBase ID', 'NA'),
                'circRNADb_ID': row.get('circRNADb ID', 'NA'),
                'circAtlas_ID': row.get('circAtlas ID', 'NA')
            }
        
        return mapping
    except FileNotFoundError:
        print(f"警告: 找不到 ID 映射文件 {mapping_file}")
        return {}


def extract_window_circular(seq: str, center: int, window_size: int, use_circular: bool = True) -> str:
    """提取窗口（支持环形截取）"""
    half = window_size // 2
    seq_len = len(seq)
    
    if use_circular:
        chars = []
        for i in range(window_size):
            pos = (center - half + i) % seq_len
            chars.append(seq[pos])
        return ''.join(chars)
    else:
        start = center - half
        end = center + half + 1
        
        if start < 0:
            left_pad = '0' * (-start)
            window = left_pad + seq[0:end]
        elif end > seq_len:
            right_pad = '0' * (end - seq_len)
            window = seq[start:seq_len] + right_pad
        else:
            window = seq[start:end]
        
        return window


def load_data_from_fasta(fasta_prefix: str):
    """从 FASTA 文件加载数据"""
    print("=" * 80)
    print("步骤 1: 从 FASTA 文件加载数据")
    print("=" * 80)
    
    print(f"\n加载序列: {fasta_prefix}_seq.fasta")
    sequences = load_sequences(fasta_prefix)
    print(f"  加载 {len(sequences)} 个 circRNA 序列")
    
    print(f"\n加载训练集标注: {fasta_prefix}_train.fasta")
    train_labels = load_annotations(f'{fasta_prefix}_train.fasta', sequences)
    print(f"  加载 {len(train_labels)} 个训练集 circRNA")
    
    print(f"\n加载测试集标注: {fasta_prefix}_test.fasta")
    test_labels = load_annotations(f'{fasta_prefix}_test.fasta', sequences)
    print(f"  加载 {len(test_labels)} 个测试集 circRNA")
    
    print(f"\n加载 ID 映射: {fasta_prefix}_id_mapping.tsv")
    id_mapping = load_id_mapping(fasta_prefix)
    if id_mapping:
        print(f"  加载 {len(id_mapping)} 个 ID 映射")
    
    return sequences, train_labels, test_labels, id_mapping


def create_kfold_splits(train_ids: List[str], n_folds: int, random_seed: int):
    """创建 K 折交叉验证划分"""
    print("\n" + "=" * 80)
    print(f"步骤 2: 创建 {n_folds} 折交叉验证划分")
    print("=" * 80)
    
    random.seed(random_seed)
    train_ids_list = list(train_ids)
    random.shuffle(train_ids_list)
    
    fold_size = len(train_ids_list) // n_folds
    folds = []
    
    for i in range(n_folds):
        start_idx = i * fold_size
        if i == n_folds - 1:
            val_ids = set(train_ids_list[start_idx:])
        else:
            val_ids = set(train_ids_list[start_idx:start_idx + fold_size])
        
        fold_train_ids = set(train_ids_list) - val_ids
        folds.append((fold_train_ids, val_ids))
        
        print(f"Fold {i+1}: 训练 {len(fold_train_ids)} circRNA, 验证 {len(val_ids)} circRNA")
    
    return folds


def extract_samples(circrna_ids: Set[str], 
                   sequences: Dict[str, str],
                   labels: Dict[str, np.ndarray],
                   id_mapping: Dict[str, Dict],
                   window_size: int, 
                   use_circular: bool):
    """从指定的 circRNA 中提取样本"""
    samples = []
    
    for unique_id in circrna_ids:
        if unique_id not in sequences or unique_id not in labels:
            continue
        
        seq = sequences[unique_id]
        label_array = labels[unique_id]
        
        bio_ids = id_mapping.get(unique_id, {})
        circbase_id = bio_ids.get('circBase_ID', 'NA')
        circrnadb_id = bio_ids.get('circRNADb_ID', 'NA')
        circatlas_id = bio_ids.get('circAtlas_ID', 'NA')
        
        for pos in range(len(seq)):
            window = extract_window_circular(seq, pos, window_size, use_circular)
            label = int(label_array[pos])
            
            samples.append({
                'unique_id': unique_id,
                'circBase_ID': circbase_id,
                'circRNADb_ID': circrnadb_id,
                'circAtlas_ID': circatlas_id,
                'position': pos + 1,
                'window': window,
                'label': label
            })
    
    return samples


def extract_pos_neg_samples(circrna_ids: Set[str], 
                            sequences: Dict[str, str],
                            labels: Dict[str, np.ndarray],
                            window_size: int, 
                            use_circular: bool):
    """分别提取正样本和负样本"""
    pos_windows = []
    neg_windows = []
    
    for unique_id in circrna_ids:
        if unique_id not in sequences or unique_id not in labels:
            continue
        
        seq = sequences[unique_id]
        label_array = labels[unique_id]
        
        for pos in range(len(seq)):
            window = extract_window_circular(seq, pos, window_size, use_circular)
            
            if label_array[pos] == 1:
                pos_windows.append(window)
            else:
                neg_windows.append(window)
    
    return pos_windows, neg_windows


def to_one_hot(sequences, window_size):
    """将序列转换为 one-hot 编码"""
    seq_data = []
    for seq in sequences:
        mat = np.zeros((window_size, 4))
        for j, nucleotide in enumerate(seq):
            if nucleotide == 'A':
                mat[j][0] = 1.0
            elif nucleotide == 'C':
                mat[j][1] = 1.0
            elif nucleotide == 'G':
                mat[j][2] = 1.0
            elif nucleotide == 'U' or nucleotide == 'T':
                mat[j][3] = 1.0
            elif nucleotide == 'N':
                mat[j] = 0.25
            elif nucleotide == '0':
                mat[j] = 0.0
        seq_data.append(mat)
    
    return np.array(seq_data)


def build_site_prediction_dataframe(samples, pred_probs, dataset_name, fold=None, threshold=0.5):
    """将逐位点预测组织为表格，便于后续拼接回 circRNA。"""
    pred_probs = np.asarray(pred_probs).flatten()
    if len(samples) != len(pred_probs):
        raise ValueError(f"样本数与预测数不一致: {len(samples)} vs {len(pred_probs)}")

    records = []
    for sample, pred_prob in zip(samples, pred_probs):
        pred_prob = float(pred_prob)
        records.append({
            'dataset': dataset_name,
            'fold': fold,
            'unique_id': sample['unique_id'],
            'circBase_ID': sample.get('circBase_ID', 'NA'),
            'circRNADb_ID': sample.get('circRNADb_ID', 'NA'),
            'circAtlas_ID': sample.get('circAtlas_ID', 'NA'),
            'position': int(sample['position']),
            'true_label': int(sample['label']),
            'pred_prob': pred_prob,
            'pred_label': int(pred_prob > threshold)
        })

    return pd.DataFrame(records)


def build_circrna_prediction_records(site_df: pd.DataFrame, sequences: Dict[str, str]):
    """按 circRNA 聚合逐位点预测，重建完整预测轨迹。"""
    circrna_records = []

    for unique_id, group in site_df.groupby('unique_id', sort=False):
        group = group.sort_values('position').reset_index(drop=True)
        sequence = sequences.get(unique_id, '')
        pred_probs = [float(prob) for prob in group['pred_prob'].tolist()]
        pred_labels = [int(label) for label in group['pred_label'].tolist()]
        true_labels = [int(label) for label in group['true_label'].tolist()]

        fold_value = group['fold'].iloc[0]
        if pd.isna(fold_value):
            fold_value = None
        else:
            fold_value = int(fold_value)

        circrna_records.append({
            'dataset': group['dataset'].iloc[0],
            'fold': fold_value,
            'unique_id': unique_id,
            'circBase_ID': group['circBase_ID'].iloc[0],
            'circRNADb_ID': group['circRNADb_ID'].iloc[0],
            'circAtlas_ID': group['circAtlas_ID'].iloc[0],
            'seq_length': len(sequence) if sequence else len(group),
            'sequence': sequence,
            'positions': [int(pos) for pos in group['position'].tolist()],
            'true_label_sequence': ''.join(str(label) for label in true_labels),
            'pred_prob_sequence': pred_probs,
            'pred_label_sequence': ''.join(str(label) for label in pred_labels),
            'positive_site_count': int(sum(true_labels)),
            'predicted_positive_count': int(sum(pred_labels))
        })

    return circrna_records


def save_prediction_outputs(site_df: pd.DataFrame,
                            sequences: Dict[str, str],
                            output_dir: str,
                            file_prefix: str,
                            description: str):
    """保存逐位点结果与重建后的 circRNA 级结果。"""
    os.makedirs(output_dir, exist_ok=True)

    site_file = os.path.join(output_dir, f'{file_prefix}_site_predictions.csv')
    site_df.to_csv(site_file, index=False)

    circrna_records = build_circrna_prediction_records(site_df, sequences)
    circrna_file = os.path.join(output_dir, f'{file_prefix}_circrna_predictions.json')
    with open(circrna_file, 'w', encoding='utf-8') as f:
        json.dump(circrna_records, f, indent=2, ensure_ascii=False)

    circrna_summary = pd.DataFrame([
        {
            'dataset': record['dataset'],
            'fold': record['fold'],
            'unique_id': record['unique_id'],
            'circBase_ID': record['circBase_ID'],
            'circRNADb_ID': record['circRNADb_ID'],
            'circAtlas_ID': record['circAtlas_ID'],
            'seq_length': record['seq_length'],
            'positive_site_count': record['positive_site_count'],
            'predicted_positive_count': record['predicted_positive_count']
        }
        for record in circrna_records
    ])
    summary_file = os.path.join(output_dir, f'{file_prefix}_circrna_summary.csv')
    circrna_summary.to_csv(summary_file, index=False)

    print(f"已保存{description}逐位点结果: {site_file}")
    print(f"已保存{description}circRNA级结果: {circrna_file}")
    print(f"已保存{description}circRNA摘要: {summary_file}")

    return {
        'site_file': site_file,
        'circrna_file': circrna_file,
        'summary_file': summary_file
    }
