"""评估：总体 / 逐类 / 跨受试者 / 混淆结构，以及论文用的图表。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)

from .sensors import ACTIVITY_NAMES, ACTIVITY_NAMES_CN

# matplotlib 在沙箱里没有可写的 HOME，指定缓存目录避免报错
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplcache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def label_names(labels: Sequence[int], cn: bool = False) -> List[str]:
    table = ACTIVITY_NAMES_CN if cn else ACTIVITY_NAMES
    return [table.get(int(l), str(l)) for l in labels]


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[int]
) -> Dict[str, object]:
    labels = [int(l) for l in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )

    rows = []
    for i, lab in enumerate(labels):
        key = str(lab)
        r = report.get(key, {})
        rows.append(
            {
                "label": lab,
                "activity": ACTIVITY_NAMES.get(lab, "?"),
                "activity_cn": ACTIVITY_NAMES_CN.get(lab, "?"),
                "support": int(cm[i].sum()),
                "precision": float(r.get("precision", 0.0)),
                "recall": float(r.get("recall", 0.0)),
                "f1": float(r.get("f1-score", 0.0)),
            }
        )
    per_class = pd.DataFrame(rows)

    # 主要混淆：真实类被错判成什么
    confusions = []
    for i, lab in enumerate(labels):
        support = cm[i].sum()
        if support == 0:
            continue
        for j, lab2 in enumerate(labels):
            if i == j or cm[i, j] == 0:
                continue
            confusions.append(
                {
                    "true": lab,
                    "true_name": ACTIVITY_NAMES.get(lab, "?"),
                    "pred": lab2,
                    "pred_name": ACTIVITY_NAMES.get(lab2, "?"),
                    "count": int(cm[i, j]),
                    "rate_of_true": float(cm[i, j] / support),
                }
            )
    conf_df = pd.DataFrame(confusions)
    if not conf_df.empty:
        conf_df = conf_df.sort_values("count", ascending=False).reset_index(drop=True)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "kappa": float(cohen_kappa_score(y_true, y_pred, labels=labels)),
        "n_samples": int(len(y_true)),
        "n_classes_present": int(len(np.unique(y_true))),
        "per_class": per_class,
        "confusion": cm,
        "confusions": conf_df,
        "labels": labels,
    }


def per_subject(
    y_true: np.ndarray, y_pred: np.ndarray, pids: np.ndarray, labels: Sequence[int]
) -> pd.DataFrame:
    rows = []
    for pid in sorted(np.unique(pids)):
        m = pids == pid
        if m.sum() == 0:
            continue
        # 只在"该受试者真实出现过的活动"上算宏平均，
        # 否则缺席的类别会被记 0 分，冤枉那些只做了部分活动的人。
        present = sorted(int(v) for v in np.unique(y_true[m]))
        rows.append(
            {
                "pid": int(pid),
                "n_windows": int(m.sum()),
                "accuracy": float(accuracy_score(y_true[m], y_pred[m])),
                "macro_f1": float(
                    f1_score(y_true[m], y_pred[m], labels=present,
                             average="macro", zero_division=0)
                ),
                "n_classes_present": len(present),
            }
        )
    return pd.DataFrame(rows).sort_values("pid").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 图表
# --------------------------------------------------------------------------- #
def plot_confusion(
    cm: np.ndarray, labels: Sequence[int], path: Path, normalize: bool = True
) -> None:
    labels = [int(l) for l in labels]
    M = cm.astype(float)
    if normalize:
        row = M.sum(axis=1, keepdims=True)
        M = np.divide(M, row, out=np.zeros_like(M), where=row > 0)
    names = label_names(labels)

    fig, ax = plt.subplots(figsize=(9.5, 8.2))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    ax.set_title(
        "Confusion matrix" + (" (row-normalized)" if normalize else " (counts)"),
        fontsize=12,
    )
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = M[i, j]
            if v > 0:
                ax.text(
                    j, i, f"{v:.2f}" if normalize else f"{int(v)}",
                    ha="center", va="center", fontsize=7,
                    color="white" if v > 0.5 else "black",
                )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_per_subject(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(df["pid"], df["accuracy"], color="#4C78A8", label="accuracy")
    ax.bar(df["pid"], df["macro_f1"], color="#F58518", alpha=0.6, label="macro F1")
    mean_acc = df["accuracy"].mean()
    ax.axhline(mean_acc, color="black", ls="--", lw=1,
               label=f"mean accuracy = {mean_acc:.3f}")
    ax.set_xlabel("Participant ID")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.02)
    ax.set_title("Per-participant performance (held-out subjects only)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_per_class(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(df))
    ax.bar(x - 0.27, df["precision"], width=0.27, label="precision")
    ax.bar(x, df["recall"], width=0.27, label="recall")
    ax.bar(x + 0.27, df["f1"], width=0.27, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(df["activity"], rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("Per-activity performance")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #
def df_to_markdown(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    try:
        return df.to_markdown(index=False, floatfmt=floatfmt)
    except Exception:
        cols = list(df.columns)
        lines = ["| " + " | ".join(map(str, cols)) + " |",
                 "| " + " | ".join(["---"] * len(cols)) + " |"]
        # 用 itertuples 逐列保留原始标量类型（iterrows 会把整行提升成 float）
        for row in df[cols].itertuples(index=False, name=None):
            vals = []
            for v in row:
                if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
                    vals.append(str(int(v)))
                elif isinstance(v, (float, np.floating)):
                    vals.append(f"{float(v):{floatfmt}}")
                else:
                    vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)


def save_json(obj: Dict[str, object], path: Path) -> None:
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=default),
                    encoding="utf-8")
