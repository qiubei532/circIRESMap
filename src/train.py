#!/usr/bin/env python3
"""
Training utilities: BaggingSampler, train_with_cross_validation, train_final_model_and_test.
"""

import os
import json
import copy
import numpy as np
import pandas as pd
import torch
# import torch.nn as nn

from .config import (
    RANDOM_SEED, MAX_EPOCHS, EARLY_STOP_PATIENCE, OUTPUT_DIR,
    BEST_N_BAGS, get_runtime_device
)
from .model import FocalLoss, build_cnn_model
from .data_utils import (
    extract_pos_neg_samples, extract_samples, to_one_hot,
    build_site_prediction_dataframe, save_prediction_outputs
)


class BaggingSampler:
    """
    Balanced Bagging 采样器

    特点：
    1. 使用全部正样本 + 指定的负样本子集
    2. 正负比例更平衡（约 1:1）
    3. 每个 epoch 遍历所有样本
    """

    def __init__(self, pos_onehot, neg_onehot_subset, batch_size=128):
        """
        Args:
            pos_onehot: 正样本 one-hot 编码 (已预处理)
            neg_onehot_subset: 负样本子集 one-hot 编码 (已预处理)
            batch_size: batch 大小
        """
        self.pos_onehot = pos_onehot
        self.neg_onehot = neg_onehot_subset
        self.batch_size = batch_size

        # 合并所有样本
        self.X_all = np.concatenate([pos_onehot, neg_onehot_subset], axis=0)
        self.y_all = np.concatenate([
            np.ones((len(pos_onehot), 1), dtype=np.float32),
            np.zeros((len(neg_onehot_subset), 1), dtype=np.float32)
        ], axis=0)

        # 计算 batch 数
        self.n_samples = len(self.X_all)
        self.batches_per_epoch = self.n_samples // batch_size
        if self.n_samples % batch_size > 0:
            self.batches_per_epoch += 1

    def get_epoch_batches(self):
        """获取一个 epoch 的所有 batch"""
        # 打乱所有样本
        indices = np.random.permutation(self.n_samples)

        batches = []
        for batch_idx in range(self.batches_per_epoch):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, self.n_samples)
            batch_indices = indices[start_idx:end_idx]

            X_batch = self.X_all[batch_indices]
            y_batch = self.y_all[batch_indices]

            batches.append((
                torch.from_numpy(X_batch),
                torch.from_numpy(y_batch)
            ))

        return batches


def train_with_cross_validation(sequences,
                                train_labels,
                                folds,
                                id_mapping,
                                window_size,
                                use_circular,
                                output_dir=None,
                                lr=1e-3,
                                focal_gamma=2.0,
                                focal_alpha=0.5,
                                max_epochs=MAX_EPOCHS,
                                patience=EARLY_STOP_PATIENCE,
                                run_name='default',
                                n_bags=BEST_N_BAGS,
                                batch_size=128,
                                num_filters=128,
                                bigru_hidden_size=128,
                                bigru_num_layers=2,
                                damsr_layers=1,
                                damsr_num_heads=2,
                                context_pool_gamma=0.97):
    """
    5 折交叉验证训练 - CNN + DAMSR + BiGRU 双分支（轻量化版本）

    训练策略：
    - 将负样本分成 n_bags 份（不重复）
    - 训练 n_bags 个独立的 CNN + DAMSR + BiGRU 网络
    - 每个模型使用：全部正样本 + 1/n_bags 负样本
    - 预测时取所有模型输出的平均值

    模型创新（轻量化适配短序列）：
    - DAMSR（NeurIPS 2023）：1层2头，多尺度衰减注意力，gamma固定
    - BiGRU：1层双向，递归累积全局上下文
    - 双分支互补：DAMSR显式注意力 + BiGRU隐式递归
    """
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.metrics import recall_score, f1_score, matthews_corrcoef, confusion_matrix, roc_auc_score, average_precision_score

    print("\n" + "=" * 80)
    print("5 折交叉验证训练 - CNN + DAMSR + BiGRU 双分支（轻量化）")
    print("=" * 80)

    print(f"训练策略: Balanced Bagging")
    print(f"  - 负样本分成 {n_bags} 份（不重复，100% 利用）")
    print(f"  - 训练 {n_bags} 个独立模型")
    print(f"  - 每个模型：全部正样本 + 1/{n_bags} 负样本")
    print(f"  - 预测时取 {n_bags} 个模型的平均")
    print(f"\n模型架构: CNN + DAMSR + BiGRU 双分支")
    print(f"  - CNN: 提取局部motif（7bp卷积核）, num_filters={num_filters}")
    print(f"  - DAMSR: {damsr_layers}层{damsr_num_heads}头，多尺度衰减注意力（gamma固定）")
    print(f"  - BiGRU: {bigru_num_layers}层双向, hidden_size={bigru_hidden_size}")
    print(f"\n当前超参数:")
    print(f"  - run_name: {run_name}")
    print(f"  - lr: {lr}")
    print(f"  - focal_gamma: {focal_gamma}")
    print(f"  - focal_alpha: {focal_alpha}")
    print(f"  - max_epochs: {max_epochs}")
    print(f"  - patience: {patience}")
    print(f"  - num_filters: {num_filters}")
    print(f"  - bigru_hidden_size: {bigru_hidden_size}")
    print(f"  - bigru_num_layers: {bigru_num_layers}")

    device = get_runtime_device()
    print(f"  - 使用设备: {device}")

    if output_dir is None:
        output_dir = OUTPUT_DIR

    fold_results = []
    cv_site_prediction_frames = []

    for fold_idx, (fold_train_ids, fold_val_ids) in enumerate(folds, 1):
        print(f"\n{'='*80}")
        print(f"Fold {fold_idx}/{len(folds)}")
        print(f"{'='*80}")

        # 准备数据
        print(f"\n[1/4] 准备数据...")

        pos_windows, neg_windows = extract_pos_neg_samples(
            fold_train_ids, sequences, train_labels, window_size, use_circular
        )
        print(f"  训练集: 正样本 {len(pos_windows)}, 负样本 {len(neg_windows)}")

        # 预处理 one-hot 编码（只做一次）
        print(f"  预处理正样本 one-hot 编码...")
        pos_onehot = to_one_hot(pos_windows, window_size).astype(np.float32)
        print(f"  预处理负样本 one-hot 编码...")
        neg_onehot = to_one_hot(neg_windows, window_size).astype(np.float32)

        # 将负样本分成 n_bags 份（不重复）
        print(f"\n  将负样本分成 {n_bags} 份...")
        np.random.seed(RANDOM_SEED + fold_idx)  # 每个 fold 使用不同的划分
        neg_indices = np.random.permutation(len(neg_windows))

        neg_subsets = []
        subset_size = len(neg_windows) // n_bags
        for i in range(n_bags):
            start_idx = i * subset_size
            if i == n_bags - 1:
                # 最后一份包含剩余所有
                end_idx = len(neg_windows)
            else:
                end_idx = start_idx + subset_size
            neg_subsets.append(neg_indices[start_idx:end_idx])

        for i, subset in enumerate(neg_subsets):
            ratio = len(pos_windows) / len(subset)
            print(f"    Bag {i+1}: {len(subset)} 负样本, 正负比例 1:{1/ratio:.2f}")

        # 准备验证集
        val_samples = extract_samples(
            fold_val_ids, sequences, train_labels, id_mapping, window_size, use_circular
        )
        X_val_seq = [s['window'] for s in val_samples]
        y_val = np.array([s['label'] for s in val_samples], dtype=np.float32)
        X_val = to_one_hot(X_val_seq, window_size).astype(np.float32)

        print(f"  验证集: {len(y_val)} 样本, 正样本 {y_val.sum():.0f}, 负样本 {len(y_val)-y_val.sum():.0f}")

        X_val_tensor = torch.from_numpy(X_val)
        y_val_tensor = torch.from_numpy(y_val).unsqueeze(1)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # 训练 n_bags 个模型
        print(f"\n[2/4] 训练 {n_bags} 个 CNN-DAMSR 模型...")

        models = []
        best_model_states = []
        criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction='none')

        for bag_idx in range(n_bags):
            print(f"\n  --- 训练模型 {bag_idx+1}/{n_bags} (CNN + DAMSR + BiGRU 轻量化) ---")

            # 固定随机种子（确保可复现性和稳定性）
            seed = RANDOM_SEED + fold_idx * 100 + bag_idx
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)

            # 创建该 bag 的采样器
            neg_subset_onehot = neg_onehot[neg_subsets[bag_idx]]
            sampler = BaggingSampler(pos_onehot, neg_subset_onehot, batch_size=batch_size)
            print(f"    样本数: {sampler.n_samples} (正:{len(pos_onehot)}, 负:{len(neg_subset_onehot)})")
            print(f"    每 epoch batch 数: {sampler.batches_per_epoch}")

            # 构建模型
            model = build_cnn_model(
                window_size=window_size,
                num_filters=num_filters,
                bigru_hidden_size=bigru_hidden_size,
                bigru_num_layers=bigru_num_layers,
                damsr_layers=damsr_layers,
                damsr_num_heads=damsr_num_heads,
                context_pool_gamma=context_pool_gamma
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            print(f"    模型架构: CNN (共享) + DAMSR (衰减注意力) + BiGRU (递归) 双分支")
            print(f"    DAMSR: {damsr_layers}层, {damsr_num_heads}头, gamma={context_pool_gamma:.2f}, 特征{num_filters*2}维 | BiGRU: {bigru_num_layers}层双向, 特征{bigru_hidden_size*2}维")
            print(f"    特征融合: {num_filters*2 + bigru_hidden_size*2}维 → 256 → 128 → 1")

            # 训练
            best_aupr = 0
            best_epoch = 0
            patience_counter = 0
            best_state = None

            for epoch in range(max_epochs):
                batches = sampler.get_epoch_batches()

                model.train()
                train_loss = 0
                batch_count = 0

                for X_batch, y_batch in batches:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch).mean()
                    loss.backward()

                    # 梯度裁剪（提升训练稳定性）
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    optimizer.step()

                    train_loss += loss.item()
                    batch_count += 1

                    if batch_count % 50 == 0:
                        print(f"      Batch {batch_count}/{sampler.batches_per_epoch} - Loss: {train_loss/batch_count:.4f}", end='\r')

                # 验证（单模型）
                model.eval()
                all_preds = []

                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(device)
                        outputs = model(X_batch)
                        all_preds.extend(outputs.cpu().numpy())

                all_preds = np.array(all_preds).flatten()
                val_aupr = average_precision_score(y_val, all_preds)

                print(f"    Epoch {epoch+1}/{max_epochs} - Loss: {train_loss/batch_count:.4f} - Val AUPR: {val_aupr:.4f}", end='')

                # Early stopping
                if val_aupr > best_aupr:
                    best_aupr = val_aupr
                    best_epoch = epoch + 1
                    best_state = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                    print(f" ✓")
                else:
                    patience_counter += 1
                    print()
                    if patience_counter >= patience:
                        print(f"    Early stopping at epoch {epoch+1}")
                        break

            print(f"    最佳 Epoch: {best_epoch}, AUPR: {best_aupr:.4f}")

            # 恢复最佳状态
            model.load_state_dict(best_state)
            models.append(model)
            best_model_states.append(best_state)

        # 集成评估
        print(f"\n[3/4] 集成评估...")

        for model in models:
            model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)

                # 所有模型预测取平均
                outputs_sum = torch.zeros(X_batch.size(0), 1).to(device)
                for model in models:
                    outputs_sum += model(X_batch)
                outputs = outputs_sum / len(models)

                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(y_batch.numpy())

        y_pred_prob = np.array(all_preds).flatten()
        y_val_final = np.array(all_labels).flatten()
        y_pred = (y_pred_prob > 0.5).astype(int)

        # 计算指标
        val_acc = (y_pred == y_val_final).mean()
        val_auc = roc_auc_score(y_val_final, y_pred_prob)
        val_aupr = average_precision_score(y_val_final, y_pred_prob)

        tn, fp, fn, tp = confusion_matrix(y_val_final, y_pred).ravel()
        recall = recall_score(y_val_final, y_pred)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = f1_score(y_val_final, y_pred)
        mcc = matthews_corrcoef(y_val_final, y_pred)

        print(f"\n[4/4] 验证集最终结果 ({n_bags} 个CNN-DAMSR模型集成):")
        print(f"    Accuracy: {val_acc:.4f}")
        print(f"    Recall (Sn): {recall:.4f}")
        print(f"    Specificity (Sp): {specificity:.4f}")
        print(f"    F1-Score: {f1:.4f}")
        print(f"    MCC: {mcc:.4f}")
        print(f"    AUC: {val_auc:.4f}")
        print(f"    AUPR: {val_aupr:.4f} ⭐")

        fold_site_df = build_site_prediction_dataframe(
            val_samples,
            y_pred_prob,
            dataset_name='cv_validation',
            fold=fold_idx
        )
        cv_site_prediction_frames.append(fold_site_df)
        save_prediction_outputs(
            fold_site_df,
            sequences,
            output_dir,
            f'cv_fold_{fold_idx}',
            f'Fold {fold_idx} 验证集'
        )

        fold_results.append({
            'fold': fold_idx,
            'model': 'CNN-DAMSR',
            'run_name': run_name,
            'lr': lr,
            'focal_gamma': focal_gamma,
            'focal_alpha': focal_alpha,
            'max_epochs': max_epochs,
            'patience': patience,
            'n_models': n_bags,
            'accuracy': val_acc,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1,
            'mcc': mcc,
            'auc': val_auc,
            'aupr': val_aupr
        })

    # 总结
    print(f"\n{'='*80}")
    print("交叉验证总结 - CNN + DAMSR")
    print(f"{'='*80}\n")

    results_df = pd.DataFrame(fold_results)
    print(results_df.to_string(index=False))

    print(f"\n平均结果:")
    print(f"  Accuracy: {results_df['accuracy'].mean():.4f} ± {results_df['accuracy'].std():.4f}")
    print(f"  Recall (Sn): {results_df['recall'].mean():.4f} ± {results_df['recall'].std():.4f}")
    print(f"  Specificity (Sp): {results_df['specificity'].mean():.4f} ± {results_df['specificity'].std():.4f}")
    print(f"  F1-Score: {results_df['f1_score'].mean():.4f} ± {results_df['f1_score'].std():.4f}")
    print(f"  MCC: {results_df['mcc'].mean():.4f} ± {results_df['mcc'].std():.4f}")
    print(f"  AUC: {results_df['auc'].mean():.4f} ± {results_df['auc'].std():.4f}")
    print(f"  AUPR: {results_df['aupr'].mean():.4f} ± {results_df['aupr'].std():.4f} ⭐")

    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, 'cv_results.csv')
    results_df.to_csv(results_file, index=False)
    print(f"\n已保存交叉验证结果: {results_file}")

    if cv_site_prediction_frames:
        cv_site_df = pd.concat(cv_site_prediction_frames, ignore_index=True)
        save_prediction_outputs(
            cv_site_df,
            sequences,
            output_dir,
            'cv_oof',
            '交叉验证 OOF'
        )

    return results_df


def train_final_model_and_test(sequences, train_labels, test_labels, id_mapping,
                                window_size, use_circular,
                                n_bags, batch_size, lr, focal_gamma, focal_alpha,
                                num_filters, bigru_hidden_size, bigru_num_layers,
                                damsr_layers, damsr_num_heads, context_pool_gamma,
                                max_epochs=MAX_EPOCHS,
                                patience=EARLY_STOP_PATIENCE,
                                output_dir=OUTPUT_DIR):
    """
    在全部训练集上训练最终模型，并在独立测试集上评估

    模型: CNN + DAMSR + Balanced Bagging
    """
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.metrics import classification_report, confusion_matrix, recall_score, f1_score, matthews_corrcoef, roc_auc_score, average_precision_score

    print(f"\n{'='*80}")
    print("训练最终模型 - CNN + DAMSR + BiGRU 双分支（轻量化）")
    print(f"{'='*80}")

    print(f"训练策略: Balanced Bagging ({n_bags} 个双分支模型)")
    print(f"模型架构: CNN + DAMSR + BiGRU 双分支融合（轻量化）")
    print(f"  - 分支1: DAMSR ({damsr_layers}层{damsr_num_heads}头, 多尺度衰减注意力) → 中心加权池化特征")
    print(f"  - 分支2: BiGRU ({bigru_num_layers}层双向, 递归累积) → 中心位置特征")
    print(f"  - 创新: 显式注意力 + 隐式递归的互补建模")

    device = get_runtime_device()

    # 准备训练数据
    print(f"\n[1/5] 准备训练数据...")
    train_ids = list(train_labels.keys())

    np.random.seed(RANDOM_SEED)
    np.random.shuffle(train_ids)
    split_idx = int(0.9 * len(train_ids))
    final_train_ids = set(train_ids[:split_idx])
    final_val_ids = set(train_ids[split_idx:])

    print(f"  训练集 circRNA: {len(final_train_ids)}")
    print(f"  验证集 circRNA: {len(final_val_ids)}")

    pos_windows, neg_windows = extract_pos_neg_samples(
        final_train_ids, sequences, train_labels, window_size, use_circular
    )
    print(f"  训练样本: 正样本 {len(pos_windows)}, 负样本 {len(neg_windows)}")

    # 预处理 one-hot 编码
    print(f"  预处理正样本 one-hot 编码...")
    pos_onehot = to_one_hot(pos_windows, window_size).astype(np.float32)
    print(f"  预处理负样本 one-hot 编码...")
    neg_onehot = to_one_hot(neg_windows, window_size).astype(np.float32)

    # 将负样本分成 n_bags 份
    print(f"\n  将负样本分成 {n_bags} 份...")
    neg_indices = np.random.permutation(len(neg_windows))

    neg_subsets = []
    subset_size = len(neg_windows) // n_bags
    for i in range(n_bags):
        start_idx = i * subset_size
        if i == n_bags - 1:
            end_idx = len(neg_windows)
        else:
            end_idx = start_idx + subset_size
        neg_subsets.append(neg_indices[start_idx:end_idx])

    for i, subset in enumerate(neg_subsets):
        ratio = len(pos_windows) / len(subset)
        print(f"    Bag {i+1}: {len(subset)} 负样本, 正负比例 1:{1/ratio:.2f}")

    # 验证集
    val_samples = extract_samples(
        final_val_ids, sequences, train_labels, id_mapping, window_size, use_circular
    )
    X_val_seq = [s['window'] for s in val_samples]
    y_val = np.array([s['label'] for s in val_samples], dtype=np.float32)
    X_val = to_one_hot(X_val_seq, window_size).astype(np.float32)

    X_val_tensor = torch.from_numpy(X_val)
    y_val_tensor = torch.from_numpy(y_val).unsqueeze(1)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 测试集
    print(f"\n[2/5] 准备独立测试集...")
    test_ids = set(test_labels.keys())
    test_samples = extract_samples(
        test_ids, sequences, test_labels, id_mapping, window_size, use_circular
    )

    X_test_seq = [s['window'] for s in test_samples]
    y_test = np.array([s['label'] for s in test_samples], dtype=np.float32)
    X_test = to_one_hot(X_test_seq, window_size).astype(np.float32)

    print(f"  测试样本: {len(y_test)}, 正样本 {y_test.sum():.0f}, 负样本 {len(y_test)-y_test.sum():.0f}")

    X_test_tensor = torch.from_numpy(X_test)
    y_test_tensor = torch.from_numpy(y_test).unsqueeze(1)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 训练 n_bags 个模型
    print(f"\n[3/5] 训练 {n_bags} 个 CNN-DAMSR 模型...")

    models = []
    best_model_states = []
    criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction='none')

    for bag_idx in range(n_bags):
        print(f"\n  --- 训练模型 {bag_idx+1}/{n_bags} (CNN + DAMSR + BiGRU 轻量化) ---")

        # 固定随机种子（确保可复现性和稳定性）
        seed = RANDOM_SEED + 1000 + bag_idx  # 使用不同的种子偏移
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        # 创建该 bag 的采样器
        neg_subset_onehot = neg_onehot[neg_subsets[bag_idx]]
        sampler = BaggingSampler(pos_onehot, neg_subset_onehot, batch_size=batch_size)
        print(f"    样本数: {sampler.n_samples} (正:{len(pos_onehot)}, 负:{len(neg_subset_onehot)})")

        # 构建模型
        model = build_cnn_model(window_size=window_size,
                                num_filters=num_filters,
                                bigru_hidden_size=bigru_hidden_size,
                                bigru_num_layers=bigru_num_layers,
                                damsr_layers=damsr_layers,
                                damsr_num_heads=damsr_num_heads,
                                context_pool_gamma=context_pool_gamma)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # 训练
        best_aupr = 0
        best_epoch = 0
        patience_counter = 0
        best_state = None

        for epoch in range(max_epochs):
            batches = sampler.get_epoch_batches()

            model.train()
            train_loss = 0
            batch_count = 0

            for X_batch, y_batch in batches:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch).mean()
                loss.backward()

                # 梯度裁剪（提升训练稳定性）
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                train_loss += loss.item()
                batch_count += 1

                if batch_count % 50 == 0:
                    print(f"      Batch {batch_count}/{sampler.batches_per_epoch} - Loss: {train_loss/batch_count:.4f}", end='\r')

            # 验证（单模型）
            model.eval()
            all_preds = []

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    outputs = model(X_batch)
                    all_preds.extend(outputs.cpu().numpy())

            all_preds = np.array(all_preds).flatten()
            val_aupr = average_precision_score(y_val, all_preds)

            print(f"    Epoch {epoch+1}/{max_epochs} - Loss: {train_loss/batch_count:.4f} - Val AUPR: {val_aupr:.4f}", end='')

            # Early stopping
            if val_aupr > best_aupr:
                best_aupr = val_aupr
                best_epoch = epoch + 1
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
                print(f" ✓")
            else:
                patience_counter += 1
                print()
                if patience_counter >= patience:
                    print(f"    Early stopping at epoch {epoch+1}")
                    break

        print(f"    最佳 Epoch: {best_epoch}, AUPR: {best_aupr:.4f}")

        # 恢复最佳状态
        model.load_state_dict(best_state)
        models.append(model)
        best_model_states.append(best_state)

    # 测试
    print(f"\n[4/5] 在独立测试集上评估 ({n_bags} 个CNN-DAMSR模型集成)...")

    for model in models:
        model.eval()

    all_preds = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)

            # 所有模型预测取平均
            outputs_sum = torch.zeros(X_batch.size(0), 1).to(device)
            for model in models:
                outputs_sum += model(X_batch)
            outputs = outputs_sum / len(models)

            all_preds.extend(outputs.cpu().numpy())

    y_pred_prob = np.array(all_preds).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)

    # 计算指标
    test_acc = (y_pred == y_test).mean()
    test_auc = roc_auc_score(y_test, y_pred_prob)
    test_aupr = average_precision_score(y_test, y_pred_prob)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    recall = recall_score(y_test, y_pred)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    print(f"\n独立测试集结果:")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Recall (Sn): {recall:.4f}")
    print(f"  Specificity (Sp): {specificity:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  MCC: {mcc:.4f}")
    print(f"  AUC: {test_auc:.4f}")
    print(f"  AUPR: {test_aupr:.4f} ⭐")

    print(f"\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['非位点', '位点']))

    print(f"\n混淆矩阵:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"              预测非位点  预测位点")
    print(f"实际非位点    {cm[0,0]:<10}  {cm[0,1]:<10}")
    print(f"实际位点      {cm[1,0]:<10}  {cm[1,1]:<10}")

    test_site_df = build_site_prediction_dataframe(
        test_samples,
        y_pred_prob,
        dataset_name='independent_test'
    )
    save_prediction_outputs(
        test_site_df,
        sequences,
        output_dir,           # <-- 修复: 使用参数而非全局变量
        'independent_test',
        '独立测试集'
    )

    # 保存结果
    test_results_dict = {
        'accuracy': float(test_acc),
        'recall': float(recall),
        'specificity': float(specificity),
        'f1_score': float(f1),
        'mcc': float(mcc),
        'auc': float(test_auc),
        'aupr': float(test_aupr),
        'n_models': n_bags
    }

    os.makedirs(output_dir, exist_ok=True)              # <-- 修复
    results_file = os.path.join(output_dir, 'test_results.json')  # <-- 修复
    with open(results_file, 'w') as f:
        json.dump(test_results_dict, f, indent=2)
    print(f"\n已保存测试结果: {results_file}")

    # 保存模型
    print(f"\n[5/5] 保存模型...")
    for i, state in enumerate(best_model_states):
        model_file = os.path.join(output_dir, f'model_{i+1}.pth')    # <-- 修复
        torch.save(state, model_file)
    print(f"已保存 {n_bags} 个模型到 {output_dir}/")

    return test_results_dict