"""模型工厂与训练封装。

默认给一个"强但便宜"的基线：直方图梯度提升树（HGB）。
同时保留随机森林 / 极端随机树 / 逻辑回归做对照，便于在报告里做消融。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def _weight(class_weight: Optional[str]):
    return None if class_weight in (None, "none", "None") else class_weight


def build_model(
    name: str = "hgb",
    seed: int = 42,
    class_weight: Optional[str] = "balanced",
    n_jobs: int = -1,
    max_iter: int = 400,
    learning_rate: float = 0.08,
):
    cw = _weight(class_weight)
    name = name.lower()

    if name in ("hgb", "hist", "gbdt"):
        return HistGradientBoostingClassifier(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=25,
            random_state=seed,
            class_weight=cw,
        )
    if name in ("rf", "randomforest"):
        return RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=2,
            n_jobs=n_jobs,
            random_state=seed,
            class_weight="balanced_subsample" if cw else None,
        )
    if name in ("et", "extratrees"):
        return ExtraTreesClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=2,
            n_jobs=n_jobs,
            random_state=seed,
            class_weight="balanced_subsample" if cw else None,
        )
    if name in ("lr", "logreg"):
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        n_jobs=n_jobs,
                        random_state=seed,
                        class_weight=cw,
                    ),
                ),
            ]
        )
    if name in ("svm", "linsvc"):
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LinearSVC(C=1.0, random_state=seed, class_weight=cw)),
            ]
        )
    if name in ("vote", "ensemble"):
        hgb = build_model("hgb", seed, class_weight, n_jobs)
        rf = build_model("rf", seed, class_weight, n_jobs)
        lr = build_model("lr", seed, class_weight, n_jobs)
        return VotingClassifier(
            estimators=[("hgb", hgb), ("rf", rf), ("lr", lr)],
            voting="soft",
            n_jobs=n_jobs,
        )
    raise ValueError(f"未知模型 {name}，可选: hgb/rf/et/lr/svm/vote")


def supports_proba(est) -> bool:
    return hasattr(est, "predict_proba")


def fit_predict(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """返回 (预测标签, 类别概率或 None)。"""
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_test)
        except Exception:
            proba = None
    return np.asarray(pred), proba


def top_k_selector(
    X_train: np.ndarray,
    y_train: np.ndarray,
    k: int,
    seed: int = 42,
) -> np.ndarray:
    """用极端随机树的 impurity 重要性挑前 k 个特征（只在训练集上拟合）。"""
    if k <= 0 or k >= X_train.shape[1]:
        return np.arange(X_train.shape[1])
    et = ExtraTreesClassifier(
        n_estimators=120,
        max_features="sqrt",
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=seed,
    )
    et.fit(X_train, y_train)
    order = np.argsort(et.feature_importances_)[::-1]
    return np.sort(order[:k])
