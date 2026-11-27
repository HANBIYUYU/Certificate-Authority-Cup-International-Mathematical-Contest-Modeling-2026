"""按试次切段、在段内滑窗。

设计要点：
* 先在 trial 内部按"标签是否连续"切成 Segment，再在 Segment 内部开窗，
  因此窗口绝不会跨 trial，也绝不会跨活动切换点。
* 想保留"含切换段的窗口"做对照实验时，把 `purify=False` 并调小 `edge_trim_s`。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .data import Trial


@dataclass(frozen=True)
class Segment:
    pid: int
    trial: int
    activity: int
    start: int  # 该受试者内的行号，含
    end: int    # 含

    @property
    def length(self) -> int:
        return self.end - self.start + 1


@dataclass
class WindowPlan:
    starts: np.ndarray      # (n,) 窗口起点（受试者内行号）
    labels: np.ndarray      # (n,) 活动标签
    pids: np.ndarray        # (n,)
    trials: np.ndarray      # (n,)
    seg_ids: np.ndarray     # (n,) 对应 Segment 序号

    def __len__(self) -> int:
        return int(self.starts.shape[0])


def build_segments(
    pid: int,
    y: np.ndarray,
    trials: Sequence[Trial],
    min_segment_s: float,
    sr: float,
    drop_labels: Sequence[int] = (),
) -> List[Segment]:
    """把一个受试者的标签序列切成"单一活动 + 单一试次"的段落。"""
    min_len = max(1, int(round(min_segment_s * sr)))
    drop = set(int(v) for v in drop_labels)
    segs: List[Segment] = []

    for t in trials:
        s, e = t.start, min(t.end, len(y) - 1)
        if e < s:
            continue
        sub = y[s : e + 1]
        if sub.size == 0:
            continue
        change = np.flatnonzero(np.diff(sub) != 0) + 1
        bounds = np.concatenate(([0], change, [sub.size]))
        for a, b in zip(bounds[:-1], bounds[1:]):
            lab = int(sub[a])
            if lab in drop:
                continue
            if (b - a) < min_len:
                continue
            segs.append(Segment(pid, t.seq, lab, s + int(a), s + int(b) - 1))
    return segs


def make_windows(
    segments: Sequence[Segment],
    window: int,
    step: int,
    edge_trim: int = 0,
) -> WindowPlan:
    """在每个 Segment 内部生成窗口（不跨越段边界）。"""
    starts: List[int] = []
    labels: List[int] = []
    pids: List[int] = []
    trials: List[int] = []
    seg_ids: List[int] = []

    for si, seg in enumerate(segments):
        lo = seg.start + edge_trim
        hi = seg.end - edge_trim
        span = hi - lo + 1
        if span < window:
            continue
        n = 1 + (span - window) // step
        if n <= 0:
            continue
        s = lo + np.arange(n, dtype=np.int64) * step
        starts.append(s)
        labels.append(np.full(n, seg.activity, dtype=np.int16))
        pids.append(np.full(n, seg.pid, dtype=np.int32))
        trials.append(np.full(n, seg.trial, dtype=np.int32))
        seg_ids.append(np.full(n, si, dtype=np.int32))

    if not starts:
        empty = np.empty(0, dtype=np.int64)
        return WindowPlan(empty, empty.astype(np.int16), empty.astype(np.int32),
                          empty.astype(np.int32), empty.astype(np.int32))

    return WindowPlan(
        starts=np.concatenate(starts),
        labels=np.concatenate(labels),
        pids=np.concatenate(pids),
        trials=np.concatenate(trials),
        seg_ids=np.concatenate(seg_ids),
    )


def summarize_segments(segments: Sequence[Segment], sr: float) -> Dict[str, float]:
    if not segments:
        return {"n_segments": 0, "total_seconds": 0.0, "median_seconds": 0.0}
    lens = np.array([s.length for s in segments], dtype=np.float64) / sr
    return {
        "n_segments": float(len(segments)),
        "total_seconds": float(lens.sum()),
        "median_seconds": float(np.median(lens)),
        "shortest_seconds": float(lens.min()),
        "longest_seconds": float(lens.max()),
    }
