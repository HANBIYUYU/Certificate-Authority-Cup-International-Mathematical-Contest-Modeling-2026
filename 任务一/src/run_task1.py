"""任务一主流程。

用法（在 任务一 目录下）：
    python -m src.run_task1 --stage prepare          # 切窗 + 提特征（只做一次）
    python -m src.run_task1 --stage train            # 按受试者交叉验证 + 出报告
    python -m src.run_task1 --stage all              # 全流程

设计原则：
  1. 任何窗口都不跨 trial、不跨活动切换点（题目硬性要求）。
  2. 训练/测试以"受试者"为单位划分，测试受试者从不参与拟合。
  3. 特征按传感器单元分组，任务 2 删单元时可直接按列屏蔽。
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):  # 允许直接 python src/run_task1.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

from . import compat
from . import data as D
from . import evaluation as E
from . import features as FE
from . import modeling as M
from . import splits as S
from . import windows as W
from .sensors import ACTIVITY_NAMES, ACTIVITY_NAMES_CN, check_schema

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# 项目根 = 任务一/，工作区根 = 它的上一级。默认路径都相对这两个位置解析，
# 这样无论从哪个目录启动脚本，路径都不会错位。
ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
DEFAULT_DATA = WORKSPACE / "C_data" / "dataset"
DEFAULT_TRIAL = WORKSPACE / "C_data" / "trial_information.csv"
DEFAULT_OUT = ROOT / "outputs"


def resolve(p: str | Path | None, fallback: Path | None = None) -> Path | None:
    if p in (None, ""):
        return fallback
    q = Path(p)
    if q.is_absolute():
        return q
    if q.exists():
        return q.resolve()
    cand = ROOT / q
    if cand.exists():
        return cand
    return (Path.cwd() / q).resolve()


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def secs(x: float) -> str:
    return f"{x:.1f}s" if x < 60 else f"{x / 60:.1f}min"


# --------------------------------------------------------------------------- #
# 参数
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="任务一：17 单元基线活动分类器",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--stage", choices=["prepare", "train", "report", "all"], default="all")
    p.add_argument("--data-dir", default=str(DEFAULT_DATA), help="含 Pxx.csv 的目录（会递归查找）")
    p.add_argument("--trial-file", default=str(DEFAULT_TRIAL), help="trial_information.csv")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))

    p.add_argument("--sr", type=float, default=60.0, help="采样率 Hz")
    p.add_argument("--window", type=float, default=2.0, help="时间窗长度（秒）")
    p.add_argument("--overlap", type=float, default=0.5, help="窗重叠比例")
    p.add_argument("--edge-trim", type=float, default=0.0, help="段两端各裁掉多少秒")
    p.add_argument("--min-segment", type=float, default=2.0, help="短于此秒数的活动段丢弃")

    p.add_argument("--feature-set", choices=list(FE.FEATURE_SETS), default="compact")
    p.add_argument("--chunk", type=int, default=2048, help="特征提取分块大小")

    p.add_argument(
        "--others-mode", choices=["include", "drop", "reject"], default="include",
        help="include=当第13类；drop=训练和评估都丢掉；reject=不训练但用置信度拒识",
    )
    p.add_argument("--n-splits", type=int, default=6, help="按受试者分几折")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", default="hgb", choices=["hgb", "rf", "et", "lr", "svm", "vote"])
    p.add_argument("--class-weight", default="balanced")
    p.add_argument("--max-iter", type=int, default=400, help="HGB 最大迭代轮数（越大越慢）")
    p.add_argument("--learning-rate", type=float, default=0.08)
    p.add_argument("--folds-limit", type=int, default=0, help="只跑前几折（调试/计时用）")
    p.add_argument("--top-k", type=int, default=0, help="0 表示不做特征选择")
    p.add_argument("--window-stride", type=int, default=1,
                   help="训练时对窗口等距抽样的步幅，2=只用一半窗口（省内存，等价把重叠降为 0）")

    p.add_argument("--limit", type=int, default=0, help="只跑前 N 个受试者（调试用）")
    p.add_argument("--resume", action="store_true", help="复用 metrics/partial 里已跑完的折")
    p.add_argument("--fold-only", type=int, default=0,
                   help="只跑第 N 折（1 起）。配合分进程运行，任何一折崩了只补那一折")
    p.add_argument("--no-cache", action="store_true", help="不缓存解析后的原始信号")
    p.add_argument("--force", action="store_true", help="重算已存在的特征文件")
    return p


def cfg_to_dict(a: argparse.Namespace) -> dict:
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}


# --------------------------------------------------------------------------- #
# 阶段 1：切窗 + 特征
# --------------------------------------------------------------------------- #
def cmd_prepare(cfg: argparse.Namespace) -> Path:
    out = Path(cfg.out_dir)
    feat_dir = out / "features"
    cache_dir = out / "cache"
    feat_dir.mkdir(parents=True, exist_ok=True)
    if not cfg.no_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)

    parts = D.discover_participants(cfg.data_dir)
    if cfg.limit:
        parts = parts[: cfg.limit]
    log(f"找到 {len(parts)} 个受试者文件")

    schema = D.build_schema(parts[0][1])
    log("列结构解析：\n" + schema.describe())
    for w in check_schema(schema):
        log(f"  [警告] {w}")

    trial_file = resolve(cfg.trial_file)
    if trial_file is None or not trial_file.exists():
        trial_file = D.find_trial_file(cfg.data_dir)
    if trial_file is None:
        raise FileNotFoundError("找不到 trial_information.csv，请用 --trial-file 指定")
    log(f"试次边界: {trial_file}")
    trials = D.load_trials(trial_file)

    win = int(round(cfg.window * cfg.sr))
    step = max(1, int(round(win * (1.0 - cfg.overlap))))
    trim = int(round(cfg.edge_trim * cfg.sr))
    drop_labels = (0,) if cfg.others_mode == "drop" else ()

    names, unit_of = FE.feature_names(schema, cfg.feature_set)
    log(f"窗口 {win} 采样（{cfg.window}s），步长 {step}，特征维数 {len(names)}")

    summary = []
    for i, (pid, path) in enumerate(parts, 1):
        t0 = time.time()
        cache = cache_dir / f"P{pid:02d}.npz"

        if cache.exists() and not cfg.force and not cfg.no_cache:
            with np.load(cache) as z:
                X, y = z["X"], z["y"]
            src = "cache"
        else:
            X, y = D.read_signals(path, schema)
            src = "csv"
            if not cfg.no_cache:
                np.savez(cache, X=X, y=y)

        qc = D.data_quality_report(X, y)
        if qc["n_nonfinite"]:
            log(f"  [警告] P{pid:02d} 有 {qc['n_nonfinite']} 个非有限值，已置零")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        pid_trials = trials.get(pid)
        if not pid_trials:
            raise ValueError(f"试次表里没有 P{pid:02d}")
        for w in D.check_trials(pid_trials, len(y)):
            log(f"  [警告] P{pid:02d}: {w}")

        segs = W.build_segments(pid, y, pid_trials, cfg.min_segment, cfg.sr, drop_labels)
        plan = W.make_windows(segs, win, step, trim)
        if len(plan) == 0:
            log(f"  [警告] P{pid:02d} 没有产生任何窗口，跳过")
            continue

        F, _, _ = FE.extract_features(
            X, plan.starts, win, schema, cfg.sr, cfg.feature_set, cfg.chunk
        )

        np.savez_compressed(
            feat_dir / f"P{pid:02d}.npz",
            F=F, y=plan.labels, pids=plan.pids, starts=plan.starts, trials=plan.trials,
        )

        seg_stat = W.summarize_segments(segs, cfg.sr)
        dt = time.time() - t0
        summary.append(
            {
                "pid": pid,
                "rows": int(X.shape[0]),
                "duration_s": round(X.shape[0] / cfg.sr, 1),
                "segments": len(segs),
                "windows": len(plan),
                "seconds_used": round(seg_stat["total_seconds"], 1),
                "source": src,
                "elapsed_s": round(dt, 1),
            }
        )
        log(
            f"({i}/{len(parts)}) P{pid:02d}: {X.shape[0]} 行 / {len(segs)} 段 / "
            f"{len(plan)} 窗 [{src}] {secs(dt)}"
        )
        del X, F

    index = {
        "feature_set": cfg.feature_set,
        "window_samples": win,
        "step_samples": step,
        "sampling_rate": cfg.sr,
        "n_features": len(names),
        "feature_names": names,
        "unit_of_col": unit_of.tolist(),
        "units": schema.units,
        "quat_order": schema.quat_order,
        "label_column": schema.label_column,
        "n_signal_channels": schema.n_channels,
    }
    (out / "feature_index.json").write_text(
        json.dumps(index, ensure_ascii=False), encoding="utf-8"
    )
    pd.DataFrame(summary).to_csv(out / "prepare_summary.csv", index=False, encoding="utf-8-sig")
    log(f"特征已保存到 {feat_dir}（维数 {len(names)}）")
    return out


# --------------------------------------------------------------------------- #
# 阶段 2：训练 + 评估
# --------------------------------------------------------------------------- #
def _feature_files(out: Path, limit: int = 0) -> list[Path]:
    files = sorted((out / "features").glob("P*.npz"))
    return files[:limit] if limit else files


def _pid_of(path: Path) -> int:
    return int(path.stem.lstrip("Pp"))


def load_meta(out: Path, limit: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """只读取标签与受试者编号（很小），用于划折。"""
    ys, ps = [], []
    for f in _feature_files(out, limit):
        with np.load(f) as z:
            ys.append(z["y"])
            ps.append(z["pids"])
    if not ys:
        raise FileNotFoundError("没有特征文件，请先运行 --stage prepare")
    return np.concatenate(ys), np.concatenate(ps)


def load_features(
    out: Path, limit: int = 0, pids_keep: set[int] | None = None, stride: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按需加载特征。

    内存吃紧时只加载某一折需要的受试者（`pids_keep`），避免全量矩阵常驻；
    `stride>1` 再对窗口等距抽样，进一步降低训练内存（等价于把重叠率降为 0）。
    先扫一遍行数再一次性分配，避免 vstack 的双倍峰值。
    """
    files = _feature_files(out, limit)
    if pids_keep is not None:
        files = [f for f in files if _pid_of(f) in pids_keep]
    if not files:
        raise FileNotFoundError("这一折没有对应的特征文件")
    stride = max(1, int(stride))

    total = 0
    for f in files:
        with np.load(f) as z:
            total += len(range(0, int(z["y"].shape[0]), stride))
    with np.load(files[0]) as z0:
        n_feat = int(z0["F"].shape[1])

    F = np.empty((total, n_feat), dtype=np.float32)
    y = np.empty(total, dtype=np.int16)
    pids = np.empty(total, dtype=np.int32)
    off = 0
    for f in files:
        with np.load(f) as z:
            Ff = z["F"][::stride]
            n = int(Ff.shape[0])
            F[off:off + n] = Ff
            y[off:off + n] = z["y"][::stride]
            pids[off:off + n] = z["pids"][::stride]
            off += n
    return F, y, pids


def calibrate_reject_threshold(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    pids_tr: np.ndarray,
    cfg: argparse.Namespace,
    labels: list[int],
) -> float:
    """在训练受试者内部再切一刀，用宏 F1 选一个"置信度低于 τ 就判 Others"的阈值。"""
    from sklearn.metrics import f1_score

    uniq = np.unique(pids_tr)
    if len(uniq) < 5:
        return 0.0
    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(uniq)
    n_val = max(1, len(uniq) // 4)
    val_pids, tr_pids = perm[:n_val], perm[n_val:]
    m_tr = np.isin(pids_tr, tr_pids)
    m_val = np.isin(pids_tr, val_pids)

    keep = ytr[m_tr] != 0
    if keep.sum() < 50 or len(np.unique(ytr[m_tr][keep])) < 2:
        return 0.0

    sub = M.build_model(
        cfg.model, cfg.seed, cfg.class_weight,
        max_iter=cfg.max_iter, learning_rate=cfg.learning_rate,
    )
    sub.fit(Xtr[m_tr][keep], ytr[m_tr][keep])
    proba = sub.predict_proba(Xtr[m_val])
    classes = np.asarray(sub.classes_)
    pred = classes[np.argmax(proba, axis=1)]
    conf = proba.max(axis=1)

    best_tau, best_f1 = 0.0, -1.0
    for tau in np.linspace(0.0, 1.0, 101):
        p = pred.copy()
        p[conf < tau] = 0
        f1 = f1_score(ytr[m_val], p, labels=labels, average="macro", zero_division=0)
        if f1 > best_f1:
            best_tau, best_f1 = float(tau), float(f1)
    return best_tau


def cmd_train(cfg: argparse.Namespace) -> None:
    out = Path(cfg.out_dir)
    met_dir = out / "metrics"
    fig_dir = out / "figures"
    met_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    y, pids = load_meta(out, cfg.limit)
    with np.load(_feature_files(out, cfg.limit)[0]) as z0:
        n_feat = int(z0["F"].shape[1])
    log(f"共 {len(y)} 个窗口 × {n_feat} 维；受试者 {len(np.unique(pids))} 人，"
        f"类别 {sorted(int(v) for v in np.unique(y))}")

    labels = sorted(int(v) for v in np.unique(y))
    folds = S.group_kfold(pids, cfg.n_splits, cfg.seed)
    if cfg.folds_limit:
        folds = folds[: cfg.folds_limit]
    if cfg.fold_only and not (1 <= cfg.fold_only <= len(folds)):
        raise ValueError(f"--fold-only 必须在 1..{len(folds)} 之间")
    log(f"按受试者 {cfg.n_splits} 折交叉验证；模型 {cfg.model}")
    log("按折加载特征（不常驻全量矩阵，避免内存峰值）")

    partial_dir = met_dir / "partial"
    partial_dir.mkdir(parents=True, exist_ok=True)

    # 配置指纹：只允许把"同一次实验"的分折结果合并到一起
    manifest = {
        "n_splits": len(folds), "seed": cfg.seed, "model": cfg.model,
        "max_iter": cfg.max_iter, "learning_rate": cfg.learning_rate,
        "window_stride": cfg.window_stride, "top_k": cfg.top_k,
        "others_mode": cfg.others_mode, "feature_dim": n_feat,
    }
    man_p = partial_dir / "manifest.json"
    if man_p.exists():
        try:
            old = json.loads(man_p.read_text(encoding="utf-8"))
        except Exception:
            old = None
        if old != manifest:
            log("[警告] partial/ 里是别的配置留下的结果，已清理，避免混用")
            for p in partial_dir.glob("fold_*"):
                p.unlink()
    man_p.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    # 收集已经跑完的折——这是"只跑剩下的"的关键
    collected: dict[int, tuple] = {}
    for fi in range(1, len(folds) + 1):
        npz_p = partial_dir / f"fold_{fi:02d}.npz"
        json_p = partial_dir / f"fold_{fi:02d}.json"
        if npz_p.exists() and json_p.exists():
            with np.load(npz_p) as z:
                collected[fi] = (
                    z["yte"], z["pred"], z["pte"],
                    json.loads(json_p.read_text(encoding="utf-8")),
                )

    selected = [cfg.fold_only] if cfg.fold_only else list(range(1, len(folds) + 1))
    todo = [fi for fi in selected if fi not in collected]
    log(f"{len(folds)} 折：已有 {sorted(collected) or '无'}，本次跑 {todo or '无'}")

    for fi in todo:
        train_pids, test_pids = folds[fi - 1]
        t0 = time.time()
        tag = f"fold_{fi:02d}"
        npz_p = partial_dir / f"{tag}.npz"
        json_p = partial_dir / f"{tag}.json"

        keep_tr = {int(p) for p in train_pids}
        keep_te = {int(p) for p in test_pids}
        Xtr, ytr, ptr = load_features(out, cfg.limit, keep_tr, cfg.window_stride)
        Xte, yte, pte = load_features(out, cfg.limit, keep_te, cfg.window_stride)

        col_idx = None
        if cfg.top_k > 0:
            if cfg.others_mode == "reject":
                keep_tk = ytr != 0
                col_idx = M.top_k_selector(Xtr[keep_tk], ytr[keep_tk], cfg.top_k, cfg.seed)
            else:
                col_idx = M.top_k_selector(Xtr, ytr, cfg.top_k, cfg.seed)
            Xtr, Xte = Xtr[:, col_idx], Xte[:, col_idx]

        tau = 0.0
        if cfg.others_mode == "reject":
            tau = calibrate_reject_threshold(Xtr, ytr, ptr, cfg, labels)

        model = M.build_model(
            cfg.model, cfg.seed, cfg.class_weight,
            max_iter=cfg.max_iter, learning_rate=cfg.learning_rate,
        )
        if cfg.others_mode == "reject":
            keep = ytr != 0
            model.fit(Xtr[keep], ytr[keep])   # 只有这条分支需要额外拷贝
        else:
            model.fit(Xtr, ytr)

        proba = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(Xte)
            except Exception:
                proba = None

        if proba is not None:
            classes = np.asarray(model.classes_)
            pred = classes[np.argmax(proba, axis=1)]
            if cfg.others_mode == "reject":
                conf = proba.max(axis=1)
                pred = pred.copy()
                pred[conf < tau] = 0
        else:
            pred = model.predict(Xte)

        from sklearn.metrics import accuracy_score, f1_score

        acc = accuracy_score(yte, pred)
        f1 = f1_score(yte, pred, labels=labels, average="macro", zero_division=0)
        row = {
            "fold": fi,
            "train_pids": " ".join(f"{p:02d}" for p in train_pids),
            "test_pids": " ".join(f"{p:02d}" for p in test_pids),
            "n_train": int(len(ytr)),
            "n_test": int(len(yte)),
            "accuracy": float(acc),
            "macro_f1": float(f1),
            "reject_tau": float(tau),
            "elapsed_s": round(time.time() - t0, 1),
        }
        collected[fi] = (yte, pred, pte, row)
        # 落盘：万一后面某折崩了，可以 --resume 或 --fold-only 接着跑
        np.savez(partial_dir / f"{tag}.npz", yte=yte, pred=pred, pte=pte)
        json_p.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
        log(
            f"fold {fi}/{len(folds)} 测试受试者 {list(map(int, test_pids))} "
            f"acc={acc:.4f} macroF1={f1:.4f} ({secs(time.time() - t0)})"
        )
        # 及时释放，避免多折累积把内存吃满
        del model, Xtr, Xte, ytr, yte, proba, pred
        gc.collect()

    if not collected:
        raise RuntimeError("没有任何一折可用（既没有历史结果，本次也没有成功跑出结果）")
    order = sorted(collected)
    y_true = np.concatenate([collected[i][0] for i in order])
    y_pred = np.concatenate([collected[i][1] for i in order])
    pids_oof = np.concatenate([collected[i][2] for i in order])
    fold_rows = [collected[i][3] for i in order]
    if len(order) < len(folds):
        log(f"[注意] 本次只合并了 {len(order)}/{len(folds)} 折，报告里会标注")

    metrics = E.compute_metrics(y_true, y_pred, labels)
    sub_df = E.per_subject(y_true, y_pred, pids_oof, labels)

    # ---- 保存数值结果 ----
    pd.DataFrame(fold_rows).to_csv(met_dir / "per_fold.csv", index=False, encoding="utf-8-sig")
    metrics["per_class"].to_csv(met_dir / "per_class.csv", index=False, encoding="utf-8-sig")
    sub_df.to_csv(met_dir / "per_subject.csv", index=False, encoding="utf-8-sig")
    if not metrics["confusions"].empty:
        metrics["confusions"].to_csv(met_dir / "confusions.csv", index=False, encoding="utf-8-sig")
    np.savetxt(met_dir / "confusion_matrix.csv", metrics["confusion"], fmt="%d", delimiter=",")

    head = {
        k: v for k, v in metrics.items()
        if k not in ("per_class", "confusion", "confusions", "labels")
    }
    head["labels"] = labels
    head["folds"] = fold_rows
    head["per_subject_accuracy_mean"] = float(sub_df["accuracy"].mean())
    head["per_subject_accuracy_std"] = float(sub_df["accuracy"].std(ddof=0))
    head["per_subject_accuracy_min"] = float(sub_df["accuracy"].min())
    head["per_subject_accuracy_max"] = float(sub_df["accuracy"].max())
    head["config"] = cfg_to_dict(cfg)
    head["feature_dim"] = int(n_feat)
    head["elapsed_s"] = round(time.time() - t_start, 1)
    E.save_json(head, met_dir / "results.json")

    # ---- 图表 ----
    E.plot_confusion(metrics["confusion"], labels, fig_dir / "confusion_matrix.png")
    E.plot_per_subject(sub_df, fig_dir / "per_subject.png")
    E.plot_per_class(metrics["per_class"], fig_dir / "per_class.png")

    # ---- 报告 ----
    summary_csv = out / "prepare_summary.csv"
    data_df = pd.read_csv(summary_csv) if summary_csv.exists() else None
    label_counts = pd.Series(y).value_counts().sort_index()
    write_report(out, cfg, metrics, sub_df, fold_rows, (int(len(y)), n_feat), data_df, label_counts)
    log(f"总体 acc={metrics['accuracy']:.4f}  macroF1={metrics['macro_f1']:.4f}  "
        f"balanced={metrics['balanced_accuracy']:.4f}")
    log(f"结果已写入 {out}")


# --------------------------------------------------------------------------- #
# 报告
# --------------------------------------------------------------------------- #
FAMILIES = {
    "步行强度": {"slow_walking", "moderate_walking", "brisk_walking"},
    "静态姿态": {"lying", "sitting", "standing"},
    "楼梯": {"ascending_stairs", "descending_stairs"},
    "器械": {"cycling", "rowing"},
}


def _family(name: str) -> str | None:
    for k, members in FAMILIES.items():
        if name in members:
            return k
    return None


def interpret_errors(metrics: dict) -> list[str]:
    """把混淆矩阵翻译成"测量信息"层面的结论——题目明确要求解释错误说明了什么。"""
    conf = metrics.get("confusions")
    per = metrics.get("per_class")
    if conf is None or len(conf) == 0:
        return []

    total_err = float(conf["count"].sum())
    if total_err <= 0:
        return []
    out: list[str] = []

    intra_mask = [
        _family(a) is not None and _family(a) == _family(b)
        for a, b in zip(conf["true_name"], conf["pred_name"])
    ]
    intra_share = conf.loc[intra_mask, "count"].sum() / total_err
    if intra_share > 0.35:
        out.append(
            f"约 {intra_share:.0%} 的错误发生在同一类活动族内部（步行强度 / 静态姿态 / 楼梯 / 器械）。"
            "这说明模型已经抓住了粗粒度信息，缺的是「强度」或「角度」这一层的细分辨力——"
            "属于**特征信息量**的限制，而不是分类器容量不足。"
        )

    walk = FAMILIES["步行强度"]
    w_err = conf[conf["true_name"].isin(walk) & conf["pred_name"].isin(walk)]["count"].sum()
    if total_err and w_err / total_err > 0.15:
        out.append(
            f"步行三档内部互错占全部错误的 {w_err / total_err:.0%}。慢走/中速走/快走是**同一动作的连续强度**，"
            "2 s 窗内的主频与谱质心差异很小，类边界本身带有主观性；这部分误差是**信息层面的**，"
            "换更强模型收益有限，要改善只能加长窗口、或引入步频/步长/对称性等专门特征。"
        )

    pos = FAMILIES["静态姿态"]
    p_err = conf[conf["true_name"].isin(pos) & conf["pred_name"].isin(pos)]["count"].sum()
    if total_err and p_err / total_err > 0.08:
        out.append(
            f"躺/坐/站之间的互错占 {p_err / total_err:.0%}。三者运动能量都接近零，"
            "唯一可用的区分信息是重力在各轴的投影（即躯干倾角）；而这个信息会被**佩戴朝向差异**污染，"
            "所以个体间的安装误差会直接变成标签误差——这解释了跨受试者性能波动的主要来源。"
        )

    if per is not None and len(per):
        strong = per[per["activity"].isin(
            ["ascending_stairs", "descending_stairs", "cycling", "rowing", "jumping"]
        )]
        if len(strong):
            out.append(
                f"上/下楼梯、骑行、划船、跳跃的 F1 均在 {strong['f1'].min():.2f} 以上："
                "这些活动有独特的角速度周期结构，信息高度冗余。"
                "也就是说在「信息充足」的活动上，17 单元布局是过剩的——这为任务二的裁剪留出了空间。"
            )

    if per is not None and (per["activity"] == "Others").any():
        row = per[per["activity"] == "Others"].iloc[0]
        others_err = conf[
            (conf["true_name"] == "Others") | (conf["pred_name"] == "Others")
        ]["count"].sum()
        out.append(
            f"Others 的 F1 仅 {row['f1']:.2f}，与 Others 相关的错误占 {others_err / total_err:.0%}。"
            "这不是模型弱，而是 **Others 本身是异质集合**（多种未命名动作被并进同一标签），"
            "类别定义的不确定性直接构成了性能上限。部署时应把它当「拒识类」而非第 13 个普通类别。"
        )

    return out


def _narrative(metrics: dict, sub_df: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    per = metrics["per_class"]
    lines.append(
        f"- **总体**：准确率 {metrics['accuracy']:.3f}，平衡准确率 "
        f"{metrics['balanced_accuracy']:.3f}，宏平均 F1 {metrics['macro_f1']:.3f}，"
        f"加权 F1 {metrics['weighted_f1']:.3f}，Cohen κ {metrics['kappa']:.3f}。"
    )

    worst = per.sort_values("f1").head(3)
    lines.append(
        "- **最弱的三类**："
        + "、".join(f"{r.activity_cn}（F1={r.f1:.3f}，支持={r.support}）"
                    for r in worst.itertuples())
        + "。"
    )
    best = per.sort_values("f1", ascending=False).head(3)
    lines.append(
        "- **最强的三类**："
        + "、".join(f"{r.activity_cn}（F1={r.f1:.3f}）" for r in best.itertuples())
        + "。"
    )

    conf = metrics["confusions"]
    if not conf.empty:
        top = conf.head(5)
        lines.append(
            "- **主要混淆**：" + "；".join(
                f"{r.true_name}→{r.pred_name}（{r.count} 次，占该类 {r.rate_of_true:.1%}）"
                for r in top.itertuples()
            ) + "。"
        )

    lines.append(
        f"- **跨受试者波动**：单人准确率均值 {sub_df['accuracy'].mean():.3f}，"
        f"标准差 {sub_df['accuracy'].std(ddof=0):.3f}，"
        f"范围 {sub_df['accuracy'].min():.3f}–{sub_df['accuracy'].max():.3f}。"
    )
    spread = sub_df["accuracy"].max() - sub_df["accuracy"].min()
    if spread > 0.2:
        lines.append(
            "- 个体差异显著：同一模型在不同人身上差出 "
            f"{spread:.1%}，说明**受试者层面的分布偏移**是主要不确定性来源，"
            "报告单一平均数会掩盖这一点。"
        )
    return lines


def write_report(
    out: Path,
    cfg: argparse.Namespace,
    metrics: dict,
    sub_df: pd.DataFrame,
    folds: list[dict],
    shape: tuple[int, int],
    data_df: pd.DataFrame | None = None,
    label_counts: pd.Series | None = None,
) -> None:
    per = metrics["per_class"].copy()
    per_out = per[["label", "activity_cn", "activity", "support", "precision", "recall", "f1"]]

    fold_df = pd.DataFrame(folds)[["fold", "test_pids", "n_train", "n_test", "accuracy", "macro_f1"]]
    conf_df = metrics["confusions"].head(12) if not metrics["confusions"].empty else pd.DataFrame()

    lines = [
        "# 任务一：姿态与活动分类（17 单元基线）",
        "",
        "> 本文件由 `src/run_task1.py` 自动生成。数值结果见 `metrics/`，图见 `figures/`。",
        "",
        "## 1. 数据概况",
        "",
    ]

    if data_df is not None and not data_df.empty:
        lines += [
            f"- 受试者：{len(data_df)} 人；总样本：{int(data_df['rows'].sum()):,} 行"
            f"（约 {data_df['rows'].sum() / cfg.sr / 3600:.2f} 小时 @ {cfg.sr:g} Hz）",
            f"- 单个受试者时长：{data_df['duration_s'].min() / 60:.1f} – "
            f"{data_df['duration_s'].max() / 60:.1f} 分钟",
            f"- 活动段（trial 内标签连续段）：{int(data_df['segments'].sum()):,} 个",
            "",
            E.df_to_markdown(data_df.head(8)),
            "",
            "（仅展示前 8 人，完整表见 `prepare_summary.csv`）",
            "",
        ]

    if label_counts is not None and len(label_counts):
        lc = pd.DataFrame(
            {
                "label": label_counts.index.astype(int),
                "activity_cn": [ACTIVITY_NAMES_CN.get(int(i), "?") for i in label_counts.index],
                "windows": label_counts.to_numpy(),
            }
        )
        lc["share"] = lc["windows"] / lc["windows"].sum()
        lines += [
            "**窗口层面的类别分布**（每个窗口只含一种活动；含切换的窗口已被剔除）：",
            "",
            E.df_to_markdown(lc),
            "",
        ]

    lines += [
        "## 2. 实验设置与方法",
        "",
        f"- 时间窗：{cfg.window} s（{int(round(cfg.window * cfg.sr))} 采样），重叠 {cfg.overlap:.0%}，"
        f"步长 {int(round(cfg.window * cfg.sr * (1 - cfg.overlap)))} 采样",
        f"- 特征：`{cfg.feature_set}` 特征集，共 {shape[1]} 维；每个特征都归属于某个传感器单元",
        f"- 划分：按受试者 {cfg.n_splits} 折交叉验证（GroupKFold），测试受试者完全不参与拟合",
        f"- 模型：{cfg.model}，类别权重 `{cfg.class_weight}`",
        f"- Others 处理：`{cfg.others_mode}`",
        f"- 窗口样本量：{shape[0]}",
        f"- 实际合并折数：{len(folds)} / {cfg.n_splits}",
        "  （如少于总折数，说明有折尚未跑完；指标只覆盖已完成的折）",
        "",
        "**如何处理时间维与活动切换**",
        "",
        "本题的硬约束是「不同 trial 的记录不得拼接」。实现方式是把每个受试者的序列先在"
        " trial 边界处切开，再在 trial 内部按标签是否变化切成「单一活动段」，窗口只在段内生成。"
        "因此：(a) 没有窗口跨越 trial；(b) 没有窗口跨越活动切换点，切换瞬间的过渡样本被自然剔除；"
        "(c) 短于一个时间窗（或短于 min_segment）的段直接丢弃。",
        "",
        "**Others 类别**",
        "",
        "- `include`：把 Others 当第 13 类训练与评估，模型必须显式区分「杂项」；",
        "- `drop`：训练与评估都剔除 Others（等价于作者基线脚本的做法），指标只反映 12 个具名活动；",
        "- `reject`：只用 12 个具名活动训练，再在训练受试者内部切一份验证集，"
        "用宏 F1 选一个置信度阈值 τ，测试时最大类别概率 < τ 即判为 Others。",
        "",
        "## 3. 结果总览",
        "",
        E.df_to_markdown(
            pd.DataFrame(
                [
                    {"指标": "准确率 accuracy", "数值": f"{metrics['accuracy']:.4f}"},
                    {"指标": "平衡准确率 balanced accuracy",
                     "数值": f"{metrics['balanced_accuracy']:.4f}"},
                    {"指标": "宏平均 F1 macro-F1", "数值": f"{metrics['macro_f1']:.4f}"},
                    {"指标": "加权 F1 weighted-F1", "数值": f"{metrics['weighted_f1']:.4f}"},
                    {"指标": "Cohen κ", "数值": f"{metrics['kappa']:.4f}"},
                    {"指标": "样本数", "数值": f"{metrics['n_samples']}"},
                ]
            )
        ),
        "",
        "## 4. 逐活动性能",
        "",
        E.df_to_markdown(per_out),
        "",
        "![逐类性能](figures/per_class.png)",
        "",
        "## 5. 混淆结构",
        "",
        "![混淆矩阵](figures/confusion_matrix.png)",
        "",
    ]

    if not conf_df.empty:
        lines += [
            "最主要的错误方向（真实 → 预测）：",
            "",
            E.df_to_markdown(
                conf_df[["true_name", "pred_name", "count", "rate_of_true"]]
            ),
            "",
        ]

    lines += [
        "## 6. 跨受试者稳定性",
        "",
        "![跨受试者](figures/per_subject.png)",
        "",
        E.df_to_markdown(
            sub_df[["pid", "n_windows", "accuracy", "macro_f1", "n_classes_present"]]
        ),
        "",
        "## 7. 各折结果",
        "",
        E.df_to_markdown(fold_df),
        "",
        "## 8. 自动结论",
        "",
    ]
    lines += _narrative(metrics, sub_df)
    interp = interpret_errors(metrics)
    if interp:
        lines += ["", "### 这些错误揭示了什么", ""]
        lines += [f"- {s}" for s in interp]
    lines += [
        "",
        "## 9. 复现",
        "",
        "```bash",
        f"python -m src.run_task1 --stage prepare --window {cfg.window} --overlap {cfg.overlap} "
        f"--feature-set {cfg.feature_set} --others-mode {cfg.others_mode}",
        f"python -m src.run_task1 --stage train --n-splits {cfg.n_splits} --model {cfg.model} "
        f"--others-mode {cfg.others_mode}",
        "```",
        "",
    ]

    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def load_saved_metrics(out: Path):
    """从 metrics/ 里把一次运行的结果读回来，用于"只重建报告"。"""
    met = out / "metrics"
    results = json.loads((met / "results.json").read_text(encoding="utf-8"))
    per_class = pd.read_csv(met / "per_class.csv")
    cm = np.loadtxt(met / "confusion_matrix.csv", delimiter=",", dtype=int)
    conf_path = met / "confusions.csv"
    conf = pd.read_csv(conf_path) if conf_path.exists() else pd.DataFrame()
    sub = pd.read_csv(met / "per_subject.csv")
    folds = pd.read_csv(met / "per_fold.csv").to_dict("records")
    metrics = {
        k: results[k]
        for k in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
                  "kappa", "n_samples", "labels")
    }
    metrics["per_class"] = per_class
    metrics["confusion"] = cm
    metrics["confusions"] = conf
    return metrics, sub, folds, results


def _load_partial_predictions(pdir: Path):
    """把各折的逐样本预测拼起来（按折号排序）。"""
    if not pdir.exists():
        return None
    files = sorted(pdir.glob("fold_*.npz"))
    if not files:
        return None
    ys, ps, ids = [], [], []
    for f in files:
        with np.load(f) as z:
            ys.append(z["yte"])
            ps.append(z["pred"])
            ids.append(z["pte"])
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(ids)


def cmd_report(cfg: argparse.Namespace) -> None:
    out = Path(cfg.out_dir)
    met = out / "metrics"
    results = json.loads((met / "results.json").read_text(encoding="utf-8"))

    # 用当初那次运行的配置覆盖命令行默认值，保证报告里的参数与结果一致
    for k, v in (results.get("config") or {}).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    labels = [int(x) for x in results["labels"]]
    preds = _load_partial_predictions(met / "partial")
    if preds is not None:
        y_true, y_pred, pids = preds
        metrics = E.compute_metrics(y_true, y_pred, labels)
        sub = E.per_subject(y_true, y_pred, pids, labels)
        # 顺手把逐类 / 逐人 / 混淆矩阵也按同一次口径重写一遍
        metrics["per_class"].to_csv(met / "per_class.csv", index=False, encoding="utf-8-sig")
        sub.to_csv(met / "per_subject.csv", index=False, encoding="utf-8-sig")
        np.savetxt(met / "confusion_matrix.csv", metrics["confusion"], fmt="%d", delimiter=",")
        if not metrics["confusions"].empty:
            metrics["confusions"].to_csv(met / "confusions.csv", index=False, encoding="utf-8-sig")
        log(f"由 {len(list((met / 'partial').glob('fold_*.npz')))} 折的逐样本预测重算了指标")
    else:
        metrics, sub, _, _ = load_saved_metrics(out)

    summary_csv = out / "prepare_summary.csv"
    data_df = pd.read_csv(summary_csv) if summary_csv.exists() else None
    per = metrics["per_class"]
    label_counts = pd.Series(per["support"].to_numpy(), index=per["label"].to_numpy())
    folds = pd.read_csv(met / "per_fold.csv").to_dict("records")
    shape = (int(results["n_samples"]), int(results["feature_dim"]))

    write_report(out, cfg, metrics, sub, folds, shape, data_df, label_counts)
    log(f"已重建 {out / 'REPORT.md'}")


# --------------------------------------------------------------------------- #
def main() -> int:
    compat.ensure_runnable()
    cfg = build_parser().parse_args()
    cfg.data_dir = str(resolve(cfg.data_dir, DEFAULT_DATA))
    cfg.out_dir = str(resolve(cfg.out_dir, DEFAULT_OUT))
    if cfg.trial_file:
        r = resolve(cfg.trial_file)
        cfg.trial_file = str(r) if r else None

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(
        json.dumps(cfg_to_dict(cfg), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if cfg.stage in ("prepare", "all"):
        cmd_prepare(cfg)
    if cfg.stage in ("train", "all"):
        cmd_train(cfg)
    if cfg.stage == "report":
        cmd_report(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
