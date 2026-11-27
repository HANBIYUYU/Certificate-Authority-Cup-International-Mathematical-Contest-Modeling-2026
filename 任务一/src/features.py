"""向量化特征提取（时域 + 频域 + 单元内互相关）。

所有特征都归属于某一个传感器单元，因此任务 2 删单元时只需要按
`unit_of_col` 屏蔽整列，不会引入任何额外假设。

特征顺序（决定了列名与数组的对应关系）：
  1. 基础通道：按 schema 顺序的每个通道 × F 个特征
  2. 派生通道：|acc|、|gyro|、|q|  三类（类型优先，再按单元）
  3. 单元内互相关：acc(xy,xz,yz)、gyr(xy,xz,yz)
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.fft import rfft, rfftfreq

from .sensors import Schema, unit_names

EPS = 1e-8

TIME_FEATURES = (
    "mean", "std", "min", "max", "range", "rms", "sma", "iqr", "mad",
    "skew", "kurt", "zcr", "mcr", "energy", "jerk",
)
FREQ_FEATURES = (
    "spectral_centroid", "spectral_entropy", "dominant_frequency",
    "spectral_spread", "band_0_1", "band_1_3", "band_3_8", "band_8_15", "band_15_30",
)

FEATURE_SETS: Dict[str, Tuple[str, ...]] = {
    # 每个通道 10 个特征：够用且不容易过拟合
    "compact": (
        "mean", "std", "rms", "iqr", "skew", "kurt", "zcr",
        "spectral_centroid", "spectral_entropy", "dominant_frequency",
    ),
    "full": TIME_FEATURES + FREQ_FEATURES,
    "minimal": (
        "mean", "std", "rms", "spectral_entropy", "dominant_frequency",
    ),
}

BANDS = ((0.0, 1.0), (1.0, 3.0), (3.0, 8.0), (8.0, 15.0), (15.0, 30.0))
BAND_NAMES = tuple(f"band_{int(a)}_{int(b)}" for a, b in BANDS)
CORR_PAIRS = (("x", "y"), ("x", "z"), ("y", "z"))


# --------------------------------------------------------------------------- #
# 单块特征计算
# --------------------------------------------------------------------------- #
def _time_features(A: np.ndarray, want: Sequence[str]) -> Dict[str, np.ndarray]:
    """A: (k, W, c) -> 每个特征 (k, c)。"""
    W = A.shape[1]
    out: Dict[str, np.ndarray] = {}
    need = set(want)

    m = A.mean(axis=1, keepdims=True)
    sd = A.std(axis=1)

    if "mean" in need:
        out["mean"] = m[:, 0, :]
    if "std" in need:
        out["std"] = sd
    if "min" in need:
        out["min"] = A.min(axis=1)
    if "max" in need:
        out["max"] = A.max(axis=1)
    if "range" in need:
        out["range"] = A.max(axis=1) - A.min(axis=1)
    if "rms" in need:
        out["rms"] = np.sqrt((A * A).mean(axis=1))
    if "sma" in need:
        out["sma"] = np.abs(A).mean(axis=1)
    if "iqr" in need:
        q75, q25 = np.percentile(A, (75, 25), axis=1)
        out["iqr"] = q75 - q25
    if "mad" in need:
        out["mad"] = np.abs(A - m).mean(axis=1)
    if "skew" in need:
        out["skew"] = ((A - m) ** 3).mean(axis=1) / (sd ** 3 + EPS)
    if "kurt" in need:
        out["kurt"] = ((A - m) ** 4).mean(axis=1) / (sd ** 4 + EPS)
    if "zcr" in need:
        out["zcr"] = np.abs(np.diff(np.signbit(A), axis=1)).sum(axis=1) / max(W - 1, 1)
    if "mcr" in need:
        out["mcr"] = np.abs(np.diff(np.signbit(A - m), axis=1)).sum(axis=1) / max(W - 1, 1)
    if "energy" in need:
        out["energy"] = (A * A).sum(axis=1)
    if "jerk" in need:
        out["jerk"] = np.abs(np.diff(A, axis=1)).mean(axis=1)
    return out


def _freq_features(
    A: np.ndarray, want: Sequence[str], sr: float, freqs: np.ndarray
) -> Dict[str, np.ndarray]:
    need = set(want)
    K = freqs.shape[0]
    W = A.shape[1]

    Fh = rfft(A, axis=1)                       # complex64
    P = (np.abs(Fh) ** 2) / W
    if P.shape[1] > 1:
        P[:, 1:, :] *= 2.0
    total = P.sum(axis=1, keepdims=True) + EPS
    p = P / total

    out: Dict[str, np.ndarray] = {}
    f3 = freqs[None, :, None]

    if "spectral_centroid" in need or "spectral_spread" in need:
        centroid = (p * f3).sum(axis=1)
        if "spectral_centroid" in need:
            out["spectral_centroid"] = centroid
        if "spectral_spread" in need:
            out["spectral_spread"] = np.sqrt((p * (f3 - centroid[:, None, :]) ** 2).sum(axis=1))
    if "spectral_entropy" in need:
        out["spectral_entropy"] = -(p * np.log(p + EPS)).sum(axis=1) / np.log(max(K, 2))
    if "dominant_frequency" in need:
        out["dominant_frequency"] = freqs[np.argmax(P, axis=1)]
    for (lo, hi), name in zip(BANDS, BAND_NAMES):
        if name in need:
            mask = (freqs >= lo) & (freqs < hi)
            out[name] = p[:, mask, :].sum(axis=1)
    return out


def _block(A: np.ndarray, sr: float, feature_set: Sequence[str]) -> np.ndarray:
    """A: (k, W, c) -> (k, c*F)，列序为通道优先、特征其次。"""
    W = A.shape[1]
    freqs = rfftfreq(W, d=1.0 / sr)
    want_time = [f for f in feature_set if f not in FREQ_FEATURES]
    want_freq = [f for f in feature_set if f in FREQ_FEATURES]

    computed: Dict[str, np.ndarray] = {}
    if want_time:
        computed.update(_time_features(A, want_time))
    if want_freq:
        computed.update(_freq_features(A, want_freq, sr, freqs))

    cols = [computed[f].astype(np.float32, copy=False) for f in feature_set]
    stack = np.stack(cols, axis=-1)            # (k, c, F)
    k, c, f = stack.shape
    return stack.reshape(k, c * f)


# --------------------------------------------------------------------------- #
# 单元指针
# --------------------------------------------------------------------------- #
def _unit_pointers(schema: Schema) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (U,3) 的 acc / gyr / quat 通道下标，缺失用 -1 填充。"""
    U = schema.n_units
    acc = np.full((U, 3), -1, dtype=np.int64)
    gyr = np.full((U, 3), -1, dtype=np.int64)
    quat = np.full((U, 4), -1, dtype=np.int64)
    for ch in schema.channels:
        vec = {"acc": acc, "gyr": gyr, "quat": quat}[ch.kind]
        ax = {"x": 0, "y": 1, "z": 2, "w": 0}.get(ch.axis)
        if ch.kind == "quat":
            ax = {"w": 0, "x": 1, "y": 2, "z": 3}.get(ch.axis, 0)
        if ax is None:
            continue
        if vec[ch.unit, ax] == -1:
            vec[ch.unit, ax] = ch.index
    return acc, gyr, quat


# --------------------------------------------------------------------------- #
# 列名（必须与数组构造顺序严格一致）
# --------------------------------------------------------------------------- #
def feature_names(
    schema: Schema, feature_set: str
) -> Tuple[List[str], np.ndarray]:
    feats = FEATURE_SETS[feature_set]
    unames = unit_names(schema)
    names: List[str] = []
    unit_of: List[int] = []

    for ch in schema.channels:
        for f in feats:
            names.append(f"{unames[ch.unit]}_{ch.kind}_{ch.axis}_{f}")
            unit_of.append(ch.unit)

    for kind in ("accmag", "gyrmag", "quatnorm"):
        for u in range(schema.n_units):
            for f in feats:
                names.append(f"{unames[u]}_{kind}_{f}")
                unit_of.append(u)

    for kind in ("acc", "gyr"):
        for a, b in CORR_PAIRS:
            for u in range(schema.n_units):
                names.append(f"{unames[u]}_corr_{kind}_{a}{b}")
                unit_of.append(u)

    return names, np.asarray(unit_of, dtype=np.int16)


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def extract_features(
    X: np.ndarray,
    starts: np.ndarray,
    window: int,
    schema: Schema,
    sr: float,
    feature_set: str = "compact",
    chunk: int = 2048,
    verbose: bool = False,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """X: (n_rows, C) -> (n_win, P) 特征矩阵。"""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"未知特征集 {feature_set}，可选 {list(FEATURE_SETS)}")

    feats = FEATURE_SETS[feature_set]
    names, unit_of = feature_names(schema, feature_set)
    n = int(len(starts))
    P = len(names)
    out = np.empty((n, P), dtype=np.float32)

    sw = np.lib.stride_tricks.sliding_window_view(X, window, axis=0)
    acc_idx, gyr_idx, quat_idx = _unit_pointers(schema)
    U = schema.n_units

    has_acc = (acc_idx >= 0).all(axis=1)
    has_gyr = (gyr_idx >= 0).all(axis=1)
    has_quat = (quat_idx >= 0).all(axis=1)

    for c0 in range(0, n, chunk):
        c1 = min(n, c0 + chunk)
        idx = starts[c0:c1]
        A = np.ascontiguousarray(sw[idx].transpose(0, 2, 1))  # (k, W, C)

        base = _block(A, sr, feats)

        mag_acc = np.sqrt((A[:, :, acc_idx] ** 2).sum(axis=-1)) if has_acc.any() else None
        mag_gyr = np.sqrt((A[:, :, gyr_idx] ** 2).sum(axis=-1)) if has_gyr.any() else None
        qnorm = np.sqrt((A[:, :, quat_idx] ** 2).sum(axis=-1)) if has_quat.any() else None

        k = A.shape[0]
        D = np.zeros((k, A.shape[1], 3 * U), dtype=np.float32)
        for t, block in enumerate((mag_acc, mag_gyr, qnorm)):
            if block is None:
                continue
            D[:, :, t * U : (t + 1) * U] = block.astype(np.float32)
        der = _block(D, sr, feats)

        corr_cols: List[np.ndarray] = []
        for ptr in (acc_idx, gyr_idx):
            for a, b in CORR_PAIRS:
                ia, ib = {"x": 0, "y": 1, "z": 2}[a], {"x": 0, "y": 1, "z": 2}[b]
                ua, ub = ptr[:, ia], ptr[:, ib]
                valid = (ua >= 0) & (ub >= 0)
                va = np.where(valid, ua, 0)
                vb = np.where(valid, ub, 0)
                xa, xb = A[:, :, va], A[:, :, vb]
                ma, mb = xa.mean(1, keepdims=True), xb.mean(1, keepdims=True)
                cov = ((xa - ma) * (xb - mb)).mean(1)
                den = xa.std(1) * xb.std(1) + EPS
                c = (cov / den).astype(np.float32)
                c = np.where(valid[None, :], c, 0.0)
                corr_cols.append(c)

        corr = np.concatenate(corr_cols, axis=1)  # (k, 6U)
        out[c0:c1] = np.concatenate([base, der, corr], axis=1)

    return out, names, unit_of


def select_units(
    F: np.ndarray, unit_of_col: np.ndarray, keep_units: Sequence[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """按单元保留特征列（任务 2 复用）。"""
    mask = np.isin(unit_of_col, np.asarray(list(keep_units), dtype=np.int16))
    return F[:, mask], mask
