"""按受试者划分的交叉验证（避免同一受试者的窗口同时出现在训练与测试）。

题目要求"在未用于拟合或选择模型的受试者上评估"，因此一切都以人为单位分组。
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def group_kfold(
    pids: Sequence[int], n_splits: int = 6, seed: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """把受试者分成 n_splits 折，返回 [(train_pids, test_pids), ...]。"""
    uniq = np.array(sorted(set(int(p) for p in pids)))
    if n_splits > len(uniq):
        raise ValueError(f"只有 {len(uniq)} 个受试者，无法做 {n_splits} 折")
    rng = np.random.default_rng(seed)
    order = rng.permutation(uniq)
    folds = np.array_split(order, n_splits)
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for i, test in enumerate(folds):
        train = np.concatenate([f for j, f in enumerate(folds) if j != i])
        out.append((np.sort(train), np.sort(test)))
    return out


def leave_one_subject_out(pids: Sequence[int]) -> List[Tuple[np.ndarray, np.ndarray]]:
    uniq = np.array(sorted(set(int(p) for p in pids)))
    return [(np.delete(uniq, i), np.array([p])) for i, p in enumerate(uniq)]


def inner_split(
    train_pids: Sequence[int], n_splits: int = 3, seed: int = 7
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """训练集内部的再划分，用于选超参 / 选阈值 / 选传感器布局。"""
    n = min(n_splits, len(set(train_pids)))
    if n < 2:
        uniq = np.array(sorted(set(int(p) for p in train_pids)))
        return [(uniq, uniq)]
    return group_kfold(train_pids, n, seed)


def subset_by_pids(pids: np.ndarray, keep: Sequence[int]) -> np.ndarray:
    return np.isin(pids, np.asarray(list(keep)))
