"""数据集体检：列结构、试次边界、标签分布、四元数模长、缺失值。

输出一张受试者级别的表，便于写论文的"数据说明"一节。

用法：
    python tools/inspect_dataset.py --data-dir ../C_data/dataset \
        --trial-file ../C_data/trial_information.csv --out outputs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import data as D  # noqa: E402
from src.sensors import ACTIVITY_NAMES, check_schema  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="../C_data/dataset")
    ap.add_argument("--trial-file", default=None)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--max-participants", type=int, default=0)
    args = ap.parse_args()

    parts = D.discover_participants(args.data_dir)
    if args.max_participants:
        parts = parts[: args.max_participants]

    schema = D.build_schema(parts[0][1])
    print(schema.describe())
    for w in check_schema(schema):
        print(f"  [警告] {w}")

    tf = Path(args.trial_file) if args.trial_file else D.find_trial_file(args.data_dir)
    trials = D.load_trials(tf)
    print(f"试次表: {tf}")

    rows, label_totals = [], {}
    for pid, path in parts:
        X, y = D.read_signals(path, schema)
        qc = D.data_quality_report(X, y)
        for k, v in qc["label_counts"].items():
            label_totals[k] = label_totals.get(k, 0) + v

        # 四元数模长应接近 1；偏离说明该通道有问题
        qidx = [c.index for c in schema.channels if c.kind == "quat"]
        qn = np.linalg.norm(X[:, qidx], axis=1) if qidx else np.array([np.nan])
        tr = trials.get(pid, [])

        rows.append(
            {
                "pid": pid,
                "rows": len(X),
                "minutes": round(len(X) / 60 / 60, 1),
                "n_trials": len(tr),
                "trials_with_gap": sum(
                    1 for a, b in zip(tr[:-1], tr[1:]) if b.start > a.end + 1
                ),
                "labels": len(qc["labels_present"]),
                "nonfinite": qc["n_nonfinite"],
                "quat_norm_median": round(float(np.nanmedian(qn)), 4),
                "acc_absmax": round(float(np.abs(X[:, [c.index for c in schema.channels if c.kind == "acc"]]).max()), 2),
            }
        )
        print(f"P{pid:02d}: {len(X)} 行, {len(tr)} 试次, 标签 {qc['labels_present']}")

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "data_inventory.csv", index=False, encoding="utf-8-sig")

    lb = pd.DataFrame(
        {
            "label": sorted(label_totals),
            "activity": [ACTIVITY_NAMES.get(k, "?") for k in sorted(label_totals)],
            "samples": [label_totals[k] for k in sorted(label_totals)],
        }
    )
    lb["share"] = (lb["samples"] / lb["samples"].sum()).round(4)
    lb.to_csv(out / "label_balance.csv", index=False, encoding="utf-8-sig")

    print("\n=== 受试者级别 ===")
    print(df.to_string(index=False))
    print("\n=== 类别平衡 ===")
    print(lb.to_string(index=False))
    print(f"\n已写入 {out}/data_inventory.csv 与 {out}/label_balance.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
