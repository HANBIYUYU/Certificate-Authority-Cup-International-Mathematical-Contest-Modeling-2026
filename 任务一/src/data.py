"""数据读取：受试者 CSV、试次边界、按试次切分。

关键约束（题目原文）：来自不同 trial 的记录不得拼接成一段连续运动。
因此所有代码都先按 trial 切分，再在 trial 内部切窗。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .sensors import ACTIVITY_NAMES, Schema, parse_columns


# --------------------------------------------------------------------------- #
# 文件发现
# --------------------------------------------------------------------------- #
PID_RE = re.compile(r"^P(\d{1,3})$", re.IGNORECASE)


def discover_participants(data_dir: str | Path) -> List[Tuple[int, Path]]:
    """在目录（含子目录）中找 P01.csv 这类文件，返回 [(1, path), ...]。"""
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"数据目录不存在: {root}")
    found: Dict[int, Path] = {}
    for p in root.rglob("*.csv"):
        m = PID_RE.match(p.stem)
        if not m:
            continue
        pid = int(m.group(1))
        if pid in found:
            # 优先选行数更多/层级更浅的那个
            if len(p.parts) < len(found[pid].parts):
                found[pid] = p
        else:
            found[pid] = p
    if not found:
        raise FileNotFoundError(f"在 {root} 下没有找到 Pxx.csv")
    return sorted(found.items())


def find_trial_file(data_dir: str | Path) -> Optional[Path]:
    root = Path(data_dir)
    for name in ("trial_information.csv", "trial information.csv", "trial_information.CSV"):
        for p in root.rglob(name):
            return p
    for p in root.rglob("*.csv"):
        if "trial" in p.name.lower():
            return p
    return None


# --------------------------------------------------------------------------- #
# 表头与 Schema
# --------------------------------------------------------------------------- #
def read_header(path: str | Path) -> List[str]:
    return list(pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns)


def build_schema(path: str | Path) -> Schema:
    return parse_columns(read_header(path))


# --------------------------------------------------------------------------- #
# 试次边界
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Trial:
    pid: int
    seq: int          # 该受试者内部第几个试次（0 起）
    start: int        # 0 基，含
    end: int          # 0 基，含

    @property
    def length(self) -> int:
        return self.end - self.start + 1


def load_trials(path: str | Path) -> Dict[int, List[Trial]]:
    df = pd.read_csv(path, encoding="utf-8-sig")
    cols = {c.lower().strip(): c for c in df.columns}

    def pick(*names: str) -> str:
        for n in names:
            if n in cols:
                return cols[n]
        raise KeyError(f"试次表缺少列 {names}，实际列: {list(df.columns)}")

    pid_col = pick("participant id", "participant", "pid", "subject", "subject id")
    s_col = pick("start index", "start", "start_index", "start sample")
    e_col = pick("end index", "end", "end_index", "end sample")

    out: Dict[int, List[Trial]] = {}
    for pid, grp in df.groupby(pid_col):
        rows = grp.sort_values(s_col)
        seq = 0
        for s, e in zip(rows[s_col].to_numpy(), rows[e_col].to_numpy()):
            s, e = int(s), int(e)
            if e < s:
                raise ValueError(f"P{int(pid):02d} 出现 end < start: {s}..{e}")
            out.setdefault(int(pid), []).append(Trial(int(pid), seq, s, e))
            seq += 1
    return out


def check_trials(trials: Sequence[Trial], n_rows: int) -> List[str]:
    warns: List[str] = []
    if not trials:
        return ["试次表为空"]
    if trials[-1].end >= n_rows:
        warns.append(f"试次末端 {trials[-1].end} 超出数据行数 {n_rows}，将被裁剪")
    for a, b in zip(trials[:-1], trials[1:]):
        if b.start <= a.end:
            warns.append(f"试次重叠: {a.start}..{a.end} 与 {b.start}..{b.end}")
        elif b.start > a.end + 1:
            warns.append(f"试次间有 {b.start - a.end - 1} 个样本的间隙")
    short = [t for t in trials if t.length < 60]
    if short:
        warns.append(f"有 {len(short)} 个试次短于 1 秒")
    return warns


# --------------------------------------------------------------------------- #
# 读取信号
# --------------------------------------------------------------------------- #
def read_activity(path: str | Path, schema: Schema) -> np.ndarray:
    s = pd.read_csv(path, usecols=[schema.label_column], encoding="utf-8-sig")
    y = pd.to_numeric(s[schema.label_column], errors="coerce").to_numpy()
    if np.isnan(y).any():
        raise ValueError(f"{path} 的活动标签存在非数值")
    return y.astype(np.int16)


def read_signals(path: str | Path, schema: Schema) -> Tuple[np.ndarray, np.ndarray]:
    """一次性读入一个受试者的信号矩阵 X (n, C) float32 与标签 y (n,) int16。"""
    cols = [schema.label_column] + schema.signal_columns
    dtypes = {c: np.float32 for c in schema.signal_columns}
    df = pd.read_csv(
        path,
        usecols=cols,
        encoding="utf-8-sig",
        dtype=dtypes,
        engine="c",
    )
    y = pd.to_numeric(df[schema.label_column], errors="coerce").to_numpy()
    if np.isnan(y).any():
        raise ValueError(f"{path} 的活动标签存在非数值")
    X = df[schema.signal_columns].to_numpy(dtype=np.float32, copy=True)
    return X, y.astype(np.int16)


def data_quality_report(X: np.ndarray, y: np.ndarray) -> Dict[str, object]:
    finite = np.isfinite(X)
    n_bad = int((~finite).sum())
    uniq, counts = np.unique(y, return_counts=True)
    return {
        "n_rows": int(X.shape[0]),
        "n_channels": int(X.shape[1]),
        "n_nonfinite": n_bad,
        "labels_present": uniq.astype(int).tolist(),
        "label_counts": {int(k): int(v) for k, v in zip(uniq, counts)},
        "unknown_labels": sorted(set(int(v) for v in uniq) - set(ACTIVITY_NAMES)),
    }
