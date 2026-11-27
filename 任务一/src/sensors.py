"""传感器布局定义与 CSV 列解析。

数据集每个受试者一个 CSV，列结构为：1 列活动标签 + 17 个单元 × 10 通道。
每个单元 10 通道 = 三轴加速度 + 三轴角速度 + 四元数 4 分量。

本模块只做"名字 -> 通道"的解析，不对数据做任何物理假设，
因此任务 2 删单元时可以直接复用这里的单元分组。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# 作者仓库 extract_features.py 中 BODY_PARTS 的顺序。
# 若列名解析失败而需要按位置兜底，就按这个顺序推断单元。
BODY_PARTS: Tuple[str, ...] = (
    "LowerBack",
    "RightThigh",
    "RightShank",
    "RightFoot",
    "LeftThigh",
    "LeftShank",
    "LeftFoot",
    "UpperBack",
    "Head",
    "RightShoulder",
    "RightUpperArm",
    "RightForeArm",
    "RightWrist",
    "LeftShoulder",
    "LeftUpperArm",
    "LeftForeArm",
    "LeftWrist",
)

BODY_PARTS_CN: Dict[str, str] = {
    "LowerBack": "下背L5",
    "UpperBack": "上背",
    "Head": "头部",
    "RightShoulder": "右肩",
    "LeftShoulder": "左肩",
    "RightUpperArm": "右上臂",
    "LeftUpperArm": "左上臂",
    "RightForeArm": "右前臂",
    "LeftForeArm": "左前臂",
    "RightWrist": "右腕",
    "LeftWrist": "左腕",
    "RightThigh": "右大腿",
    "LeftThigh": "左大腿",
    "RightShank": "右小腿",
    "LeftShank": "左小腿",
    "RightFoot": "右足",
    "LeftFoot": "左足",
}

# 题目给出的活动编码
ACTIVITY_NAMES: Dict[int, str] = {
    0: "Others",
    1: "lying",
    2: "sitting",
    3: "standing",
    4: "slow_walking",
    5: "moderate_walking",
    6: "brisk_walking",
    7: "ascending_stairs",
    8: "descending_stairs",
    9: "cycling",
    10: "running",
    11: "jumping",
    12: "rowing",
}

ACTIVITY_NAMES_CN: Dict[int, str] = {
    0: "其他",
    1: "躺卧",
    2: "坐",
    3: "站",
    4: "慢走",
    5: "中速走",
    6: "快走",
    7: "上楼梯",
    8: "下楼梯",
    9: "骑行",
    10: "跑步",
    11: "跳跃",
    12: "划船",
}

KIND_ALIASES: Dict[str, Tuple[str, ...]] = {
    "acc": ("acc", "acceleration", "accel", "a"),
    "gyr": ("gyr", "gyro", "gyroscope", "omega", "angularvelocity", "angvel", "g"),
    "quat": ("quat", "quaternion", "ori", "orientation", "rot", "q"),
}

ACC_AXES = ("x", "y", "z")
GYR_AXES = ("x", "y", "z")
QUAT_AXES = ("w", "x", "y", "z")


@dataclass(frozen=True)
class Channel:
    """扁平通道数组中的一个通道。"""

    index: int
    unit: int          # 单元序号（0..16）
    part: str          # 佩戴部位英文名
    kind: str          # acc / gyr / quat
    axis: str          # x / y / z / w
    column: str        # 原始 CSV 列名
    unit_column: str   # 在 10 通道块内的相对列号 0..9


@dataclass
class Schema:
    """一个受试者 CSV 的列结构。"""

    label_column: str
    channels: List[Channel]
    units: List[str]                     # 出现的部位，按 BODY_PARTS 顺序
    extra_columns: List[str] = field(default_factory=list)
    n_expected: Optional[int] = None     # 题目宣称 17 单元，用于自检
    quat_order: str = "xyzw"             # 原始表头里四元数分量的排列（任务三要用）

    # ------- 便捷视图 -------
    @property
    def signal_columns(self) -> List[str]:
        return [c.column for c in self.channels]

    @property
    def n_units(self) -> int:
        return len(self.units)

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    def unit_channel_indices(self, unit: int) -> List[int]:
        """某个单元在扁平通道数组中的位置（删除整单元时用）。"""
        return [c.index for c in self.channels if c.unit == unit]

    def describe(self) -> str:
        lines = [
            f"标签列        : {self.label_column}",
            f"单元数        : {self.n_units}",
            f"信号通道数    : {self.n_channels}",
            f"单元清单      : {', '.join(self.units)}",
        ]
        missing = [p for p in BODY_PARTS if p not in self.units]
        if missing:
            lines.append(f"缺失单元      : {', '.join(missing)}")
        if self.extra_columns:
            lines.append(f"未识别列({len(self.extra_columns)}) : {self.extra_columns[:8]}")
        by_kind: Dict[str, int] = {}
        for c in self.channels:
            by_kind[c.kind] = by_kind.get(c.kind, 0) + 1
        lines.append(f"通道类型统计  : {by_kind}")
        return "\n".join(lines)


def _norm(token: str) -> str:
    return re.sub(r"[^a-z0-9]", "", token.lower())


def _kind_of(token: str) -> Optional[str]:
    t = _norm(token)
    for kind, aliases in KIND_ALIASES.items():
        if t in {_norm(a) for a in aliases}:
            return kind
    # 兜底：以别名开头也算（例如 "AccX" 这种粘连写法）
    for kind, aliases in KIND_ALIASES.items():
        for a in aliases:
            if t.startswith(_norm(a)) and len(t) <= len(a) + 3:
                return kind
    return None


def _split_part(column: str) -> Tuple[Optional[str], str]:
    """把 'Acc_X_LowerBack' 拆成 ('LowerBack', 'Acc_X')。"""
    for part in sorted(BODY_PARTS, key=len, reverse=True):
        if column == part:
            return part, ""
        if column.endswith("_" + part):
            return part, column[: -(len(part) + 1)]
        if column.lower().endswith("_" + part.lower()):
            return part, column[: -(len(part) + 1)]
    return None, ""


def _sort_axes(pairs: List[Tuple[str, str]], preferred: Sequence[str]) -> List[str]:
    """按 preferred 顺序排列 (column, axis_token)。未知轴保留原顺序排在后面。"""

    def key(item: Tuple[str, str]) -> Tuple[int, int]:
        tok = _norm(item[1])
        if tok in preferred:
            return (preferred.index(tok), 0)
        return (len(preferred), 0)

    return [col for col, _ in sorted(pairs, key=key)]


def parse_columns(columns: Sequence[str], label_column: Optional[str] = None) -> Schema:
    """解析 CSV 列名，返回 Schema。

    支持 'Acc_X_LowerBack'、'Acc_LowerBack_X'、'LowerBack_Acc_X' 等常见写法，
    因为无法提前看到真实表头，这里做的是容错解析。
    """
    cols = list(columns)

    # 1) 找标签列
    if label_column is None:
        cand = [c for c in cols if _norm(c) in ("activity", "activitylabel", "label", "class")]
        if not cand:
            cand = [c for c in cols if "activity" in _norm(c)]
        if not cand:
            raise ValueError(f"找不到活动标签列，前 10 列是: {cols[:10]}")
        label_column = cand[0]

    # 2) 按部位分组
    per_part: Dict[str, List[Tuple[str, str, str]]] = {}
    extras: List[str] = []
    for c in cols:
        if c == label_column:
            continue
        part, prefix = _split_part(c)
        if part is None:
            extras.append(c)
            continue
        per_part.setdefault(part, []).append((c, prefix))

    # 3) 每个部位区分 acc / gyr / quat
    units = [p for p in BODY_PARTS if p in per_part]
    units += [p for p in per_part if p not in BODY_PARTS]  # 未知部位放后面

    if not units:
        raise ValueError("没有识别出任何传感器单元，表头格式与预期不符")

    channels: List[Channel] = []
    ch_index = 0
    quat_src_order = ""
    for ui, part in enumerate(units):
        entries = per_part[part]
        buckets: Dict[str, List[Tuple[str, str]]] = {"acc": [], "gyr": [], "quat": []}
        leftover: List[Tuple[str, str]] = []

        for col, prefix in entries:
            tokens = re.split(r"[_\-\. ]+", prefix) if prefix else [col]
            tokens = [t for t in tokens if t]
            kind = None
            axis_tok = ""
            for i, tok in enumerate(tokens):
                k = _kind_of(tok)
                if k is not None:
                    kind = k
                    if i + 1 < len(tokens):
                        axis_tok = tokens[i + 1]
                    elif i > 0:
                        axis_tok = tokens[i - 1]
                    break
            if kind is None:
                leftover.append((col, prefix))
                continue
            buckets[kind].append((col, axis_tok))

        # 同部位未识别出类型的列，按 10 通道块的位置兜底推断
        if leftover:
            for col, _ in leftover:
                if len(buckets["acc"]) < 3:
                    buckets["acc"].append((col, ACC_AXES[len(buckets["acc"])]))
                elif len(buckets["gyr"]) < 3:
                    buckets["gyr"].append((col, GYR_AXES[len(buckets["gyr"])]))
                elif len(buckets["quat"]) < 4:
                    buckets["quat"].append((col, QUAT_AXES[len(buckets["quat"])]))
                else:
                    extras.append(col)

        plan = (
            ("acc", ACC_AXES, 3),
            ("gyr", GYR_AXES, 3),
            ("quat", QUAT_AXES, 4),
        )
        if ui == 0 and buckets["quat"]:
            quat_src_order = "".join(_norm(t) for _, t in buckets["quat"])
        rel = 0
        for kind, order, need in plan:
            ordered = _sort_axes(buckets[kind], order)
            for axis_i, col in enumerate(ordered[:need]):
                axis = order[axis_i] if axis_i < len(order) else str(axis_i)
                channels.append(
                    Channel(
                        index=ch_index,
                        unit=ui,
                        part=part,
                        kind=kind,
                        axis=axis,
                        column=col,
                        unit_column=str(rel),
                    )
                )
                ch_index += 1
                rel += 1

    return Schema(
        label_column=label_column,
        channels=channels,
        units=units,
        extra_columns=extras,
        n_expected=17,
        quat_order=quat_src_order or "xyzw",
    )


def unit_names(schema: Schema) -> List[str]:
    """给每个单元一个短的稳定名字，用于特征列命名。"""
    return [f"U{i:02d}_{p}" for i, p in enumerate(schema.units)]


def check_schema(schema: Schema) -> List[str]:
    """返回自检警告列表（不抛异常，便于先跑通再修）。"""
    warns: List[str] = []
    if schema.n_units != 17:
        warns.append(f"单元数为 {schema.n_units}，题目基准是 17")
    if schema.n_channels != 170:
        warns.append(f"信号通道数为 {schema.n_channels}，题目基准是 170")
    for i, part in enumerate(schema.units):
        kinds = {c.kind for c in schema.channels if c.unit == i}
        counts = {k: sum(1 for c in schema.channels if c.unit == i and c.kind == k) for k in kinds}
        want = {"acc": 3, "gyr": 3, "quat": 4}
        for k, n in counts.items():
            if n != want[k]:
                warns.append(f"单元 {part} 的 {k} 通道数为 {n}，期望 {want[k]}")
        if "quat" not in kinds:
            warns.append(f"单元 {part} 没有四元数列")
    return warns
