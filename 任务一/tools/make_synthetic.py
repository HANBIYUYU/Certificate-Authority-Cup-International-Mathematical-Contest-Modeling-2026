"""生成一个"结构完全仿真"的小数据集，用于在真实数据就绪前跑通全流程。

列顺序与真实数据完全一致：
    Activity, 然后每个单元  Acc_X..Z, Gyr_X..Z, Quat_X,Y,Z,W

用法：
    python tools/make_synthetic.py --out synth_data --participants 6 --activities 4
    python -m src.run_task1 --stage all --data-dir synth_data \
        --trial-file synth_data/trial_information.csv --out-dir outputs_synth --n-splits 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.sensors import BODY_PARTS  # noqa: E402

ACTIVITY_PARAMS = {
    # activity: (基频 Hz, 加速度幅值, 角速度幅值, 四元数摆角)
    2: (0.15, 0.10, 0.05, 0.05),
    3: (0.20, 0.12, 0.06, 0.06),
    4: (1.80, 1.20, 1.00, 0.35),
    5: (2.20, 2.00, 2.20, 0.55),
    10: (3.00, 5.00, 5.00, 0.90),
}


def make_trial(activity: int, n: int, sr: float, rng: np.random.Generator) -> np.ndarray:
    """返回 (n, 170) 的信号矩阵。"""
    freq, a_amp, g_amp, q_amp = ACTIVITY_PARAMS[activity]
    t = np.arange(n) / sr
    X = np.zeros((n, 170), dtype=np.float32)

    for u, _part in enumerate(BODY_PARTS):
        phase = rng.uniform(0, 2 * np.pi)
        gain = 0.5 + 1.5 * rng.random()
        base = u * 10

        for axis in range(3):
            w = 2 * np.pi * freq * (1 + 0.15 * (axis - 1))
            sig = a_amp * gain * np.sin(w * t + phase + 0.7 * axis)
            sig += 0.15 * a_amp * rng.standard_normal(n)
            X[:, base + axis] = (sig + (9.81 if axis == 2 else 0.0)).astype(np.float32)

        for axis in range(3):
            w = 2 * np.pi * freq * (1 + 0.1 * axis)
            sig = g_amp * gain * np.sin(w * t + phase + 1.1 * axis)
            sig += 0.1 * g_amp * rng.standard_normal(n)
            X[:, base + 3 + axis] = sig.astype(np.float32)

        ang = q_amp * gain * np.sin(2 * np.pi * freq * t + phase)
        half = ang / 2
        qx = np.sin(half) * 0.6
        qy = np.sin(half) * 0.3
        qz = np.sin(half) * 0.2
        qw = np.cos(half)
        norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
        X[:, base + 6] = (qx / norm).astype(np.float32)
        X[:, base + 7] = (qy / norm).astype(np.float32)
        X[:, base + 8] = (qz / norm).astype(np.float32)
        X[:, base + 9] = (qw / norm).astype(np.float32)

    return X


def columns() -> list[str]:
    cols = ["Activity"]
    for part in BODY_PARTS:
        cols += [f"Acc_{a}_{part}" for a in "XYZ"]
        cols += [f"Gyr_{a}_{part}" for a in "XYZ"]
        cols += [f"Quat_{a}_{part}" for a in "XYZW"]
    return cols


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="synth_data")
    ap.add_argument("--participants", type=int, default=6)
    ap.add_argument("--activities", type=int, default=4, help="使用前几个活动（从候选里取）")
    ap.add_argument("--trials-per-person", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--sr", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    sr = args.sr
    n = int(args.seconds * sr)
    cols = columns()
    cand = [4, 5, 10, 3, 2][: args.activities]

    rows = []
    for pid in range(1, args.participants + 1):
        frames, acts = [], []
        cursor = 0
        for k in range(args.trials_per_person):
            act = cand[(pid + k) % len(cand)]
            X = make_trial(act, n, sr, rng)
            block = np.zeros((n, 171), dtype=np.float32)
            block[:, 0] = act
            block[:, 1:] = X
            frames.append(block)
            acts.append((cursor, cursor + n - 1))
            cursor += n

        data = np.concatenate(frames, axis=0)
        df = pd.DataFrame(data, columns=cols)
        df["Activity"] = df["Activity"].astype(np.int16)
        df.to_csv(out / f"P{pid:02d}.csv", index=False, float_format="%.6f")

        for s, e in acts:
            rows.append({"Participant ID": pid, "Start Index": s, "End Index": e})
        used = [cand[(pid + k) % len(cand)] for k in range(args.trials_per_person)]
        print(f"P{pid:02d}: {len(df)} 行, {len(acts)} 试次, 活动 {used}")

    pd.DataFrame(rows).to_csv(out / "trial_information.csv", index=False)
    print(f"合成数据已写入 {out}（{len(cols)} 列）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
