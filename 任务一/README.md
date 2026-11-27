# 任务一：姿态与活动分类器（17 单元基线）

对应赛题 **Task 1 — Construct a posture and activity classifier**。

本目录实现一条**可复现、无数据泄漏**的活动识别流水线：用全部 17 个 IMU 单元建立基线，
在**从未参与训练或调参的受试者**上评估，并给出总体 / 逐类 / 跨受试者三个层面的结果与错误分析。

---

## 0. 结果摘要（已完成，6 折 × 30 人）

| 指标 | 数值 |
|---|---|
| 准确率 accuracy | **0.9344** |
| 平衡准确率 balanced accuracy | 0.9145 |
| 宏平均 F1 macro-F1 | **0.9165** |
| 加权 F1 weighted-F1 | 0.9329 |
| Cohen κ | 0.9285 |
| 评估窗口数 | 214,338（2.0 s 窗，50% 重叠） |

| 折 | 留出受试者 | 准确率 | macro-F1 |
|---|---|---|---|
| 1 | 08 17 25 26 30 | 0.923 | 0.893 |
| 2 | 06 11 22 23 27 | 0.940 | 0.926 |
| 3 | 07 10 19 20 21 | 0.934 | 0.920 |
| 4 | 01 04 16 18 24 | 0.949 | 0.927 |
| 5 | 03 12 13 15 29 | 0.919 | 0.902 |
| 6 | 02 05 09 14 28 | 0.941 | 0.923 |

单人准确率均值 0.934、标准差 0.037、范围 0.832–0.994。

**三个可直接写进论文的发现**

1. **最弱的不是某个精细动作，而是 Others（F1=0.541）**——它是异质集合，属于"类别定义"问题，
   不是模型能力问题；其余 12 类里只有中速走（0.838）明显偏低。
2. **57% 的错误发生在同一活动族内部**，其中步行三档互错占 40%。慢/中/快走是同一动作的连续强度，
   2 s 窗内的主频差异很小，这部分误差属于**信息层面**，换模型收益有限。
3. **信息充足的活动几乎无错**：跑步 0.993、划船 0.991、跳跃 0.990、骑行 0.986、上下楼梯 0.976/0.974。
   这说明 17 单元对这类活动是**过剩**的——正是任务二可以裁剪的空间。

完整报告见 [`outputs/REPORT.md`](outputs/REPORT.md)，图表见 [`outputs/figures/`](outputs/figures/)。

---

## 1. 题目要求 → 代码对应

| 赛题要求 | 本实现 |
|---|---|
| 用完整 17 单元布局建立基线 | `src/sensors.py` 解析 170 个信号通道，`src/features.py` 全部纳入 |
| 决定如何随时间使用测量 | 2.0 s 滑窗（120 采样），50% 重叠，步长 1 s（`--window/--overlap`） |
| 说明如何处理 Others | 三种策略可切换：`include` / `drop` / `reject`（见 §4） |
| 说明如何处理活动切换区间 | 先按 trial 切分，再按标签切"单一活动段"，**窗口不跨 trial、不跨切换点**（见 §3） |
| 在未参与拟合的受试者上评估 | GroupKFold（默认 6 折，按人分组），测试受试者完全不参与训练与选择 |
| 选择合适性能度量 | accuracy / balanced accuracy / macro-F1 / weighted-F1 / Cohen κ + 逐类 P/R/F1 |
| 报告总体与逐类性能 | `metrics/results.json`、`metrics/per_class.csv`、`figures/per_class.png` |
| 找出主要错误并解释 | `metrics/confusions.csv`（按次数排序的混淆方向）+ 报告 §5 + §8 自动结论 |
| 展示跨受试者波动 | `metrics/per_subject.csv`、`figures/per_subject.png`、报告 §6 |

---

## 2. 快速开始

```bash
# 在 任务一/ 目录下执行
python -m src.run_task1 --stage prepare     # 切窗 + 提特征（约 15 分钟，可复用缓存）
python -m src.run_task1 --stage train       # 按受试者交叉验证 + 出报告（约 20 分钟）
```

或者一步到位：

```bash
python -m src.run_task1 --stage all
```

只重建报告（不重新训练，几秒钟）：

```bash
python -m src.run_task1 --stage report
```

默认路径已经指向工作区里的真实数据（`../C_data/dataset`、`../C_data/trial_information.csv`），
不需要额外传参。换数据位置时：

```bash
python -m src.run_task1 --stage all \
  --data-dir D:/data/dataset \
  --trial-file D:/data/trial_information.csv \
  --out-dir outputs
```

Windows PowerShell 里如果中文输出乱码，先设 `$env:PYTHONIOENCODING="utf-8"`。

### 实测耗时（16 线程，30 人 / 21.4 万窗口 / 2312 维）

| 阶段 | 耗时 | 说明 |
|---|---|---|
| `prepare` | ≈ 11 min | 首次需读 30 个 CSV（共约 22 GB 文本）；有 `cache/` 时二次运行约 5 min |
| `train`（6 折，max_iter=80） | ≈ 35 min | 单折约 5–6 min；`--max-iter 40` 可减半，精度损失约 1 个点 |
| `report` | < 5 s | 纯读 CSV 重写报告 |

想先看个快结果：`--folds-limit 1 --max-iter 40`（约 3.5 min 出单折指标）。

**内存**：30 人 × 2312 维的全量特征矩阵约 2.0 GB，6 折训练峰值约 4.5–5 GB。
如果机器内存紧张（8 GB 以下），可加 `--top-k 1200`（先做一次特征选择再训练）或
`--n-splits 4`，都能显著降低峰值。

### 冒烟测试（不用真实数据，30 秒跑完）

```bash
python tools/make_synthetic.py --out synth_data --participants 6 --activities 4
python -m src.run_task1 --stage all --data-dir synth_data \
    --trial-file synth_data/trial_information.csv --out-dir outputs_synth --n-splits 3
```

---

## 3. 方法

### 3.1 数据结构

每个 `Pxx.csv` 有 **171 列** = `Activity` + 17 单元 × 10 通道。
每个单元：`Acc_X/Y/Z`、`Gyr_X/Y/Z`、`Quat_X/Y/Z/W`（实测表头里四元数是 **X,Y,Z,W** 顺序，
即标量在后，程序会自动识别并记录在 `feature_index.json:quat_order`）。

`trial_information.csv` 给出 `Participant ID, Start Index, End Index`（0 基、闭区间）。
P01 有 90 个试次，全表共 30 人。

### 3.2 切窗与"活动切换"的处理

赛题明令：**不同 trial 的记录不得拼接**。因此流程是

```
整个受试者序列
  └─ 按 trial 边界切开（trial_information.csv）
       └─ 在 trial 内按标签是否变化切成"单一活动段"（run-length 分段）
            └─ 只在段内滑窗（窗口必须在段内，不越界）
```

这样自然得到三个性质：

1. 没有窗口跨越 trial；
2. 没有窗口跨越活动切换点——切换瞬间的过渡态样本**不会被强行贴上某一类的标签**；
3. 长度不足一个时间窗的活动段被整段丢弃（`--min-segment`，默认 2 s）。

> 为什么这样处理：如果让窗口"跨"切换点再按多数投票赋标签，模型会学到"半个走 + 半个跑"
> 这种不存在的姿态，并在测试时把真正的过渡段判得很随机。宁可少样本，也不要坏标签。

### 3.3 特征（2312 维，**全部按单元分组**）

对每个通道算 10 个特征（`--feature-set compact`）：

- 时域：均值、标准差、RMS、四分位距、偏度、峰度、过零率
- 频域：谱质心、谱熵、主频

再加上两类只依赖单元自身的特征：

- **派生通道**：`|acc|`、`|gyro|`、`|q|`（四元数模长，用于自检）——各 10 个特征；
- **单元内互相关**：加速度/角速度三轴两两相关，各 3×17 个。

`--feature-set full` 会扩到每通道 24 个特征（增加 min/max/range/sma/mad/mcr/energy/jerk、
谱展宽、5 个频带能量），维数约 5000，通常能再涨一点但训练更慢。

**关键设计：每个特征列都唯一归属于一个传感器单元**（`feature_index.json:unit_of_col`）。
任务二删单元时就是"按列屏蔽"，不需要重算特征（见 §6）。

### 3.4 划分：按受试者，而不是按样本

```
6 折 GroupKFold（按 Participant ID 分组）
  每折：训练 25 人 / 测试 5 人；测试受试者的任何窗口都不出现在训练集
```

需要注意的两个"不许"：

- 不许把同一个人随机打乱后同时放进训练和测试（会造成 10 个点以上的虚高）；
- 不许用最终测试结果去选模型/选特征/选阈值。`--top-k` 的特征选择只在**当折训练集**上拟合。

### 3.5 模型

默认 `hgb`（`HistGradientBoostingClassifier`）：对 2000+ 维特征、20 万级样本足够快且强，
自带缺失值处理与早停。可选 `rf` / `et` / `lr` / `svm` / `vote`（hgb+rf+lr 软投票）做对照。

### 3.6 三种 Others 策略

`--others-mode`：

| 值 | 训练 | 评估 | 含义 |
|---|---|---|---|
| `include`（默认） | 13 类 | 13 类 | 模型必须显式区分"杂项"，最贴近实际部署 |
| `drop` | 12 类 | 12 类 | 等价于作者基线脚本（跳过 label 0），指标最高但不处理 Others |
| `reject` | 12 类 | 13 类 | 只用具名活动训练；在训练受试者内部再切一份验证集，按宏 F1 选置信度阈值 τ，测试时 `max_prob < τ → Others` |

`reject` 的阈值只在**训练受试者内部**校准，测试集不参与，因此不会泄漏。

---

## 4. 输出文件

```
outputs/
├── config.json            本次运行的完整参数（可复现）
├── feature_index.json     2312 个特征名、每个特征属于哪个单元、窗口参数
├── prepare_summary.csv    每人的行数、时长、活动段数、窗口数、耗时
├── features/Pxx.npz       F(窗口×2312) / y / pids / starts / trials
├── cache/Pxx.npz          解析后的原始信号（加速二次运行，可删）
├── metrics/
│   ├── results.json       总体指标 + 每折结果 + 跨人统计
│   ├── per_fold.csv       每折的测试受试者、样本量、acc、macro-F1
│   ├── per_class.csv      逐类 precision / recall / F1 / support
│   ├── per_subject.csv    每个测试受试者的 acc / macro-F1 / 窗口数
│   ├── confusions.csv     混淆方向（真实→预测，按次数排序）
│   └── confusion_matrix.csv
├── figures/
│   ├── confusion_matrix.png   行归一化混淆矩阵
│   ├── per_class.png          逐类 P/R/F1 柱状图
│   └── per_subject.png        跨受试者波动
└── REPORT.md              自动生成的完整报告（可直接摘进论文）
```

`REPORT.md` 里的 §8"自动结论"会把最弱的类、最强的类、主要混淆方向、跨人方差直接写成句子，
方便改写进论文正文。

---

## 5. 假设与已知限制

1. **标签对齐**：假设 `trial_information.csv` 的行号与 CSV 行号一致（0 基、闭区间）。
   代码会检查越界、重叠与间隙并打印警告。
2. **窗口不跨段**：代价是丢弃了切换瞬间的样本，因此"过渡态识别"不在本任务范围内；
   但题目只要求说明如何处理，而不是把过渡态判准。
3. **不假设坐标系与四元数约定**：本任务只把四元数当"信号"用（分量统计 + 模长），
   不做姿态解算，因此不依赖 wxyz/xyzw 约定。约定留到任务三处理。
4. **随机性**：HGB/RF 的 `random_state` 固定；GroupKFold 的分组用固定 seed 打乱，
   同参数重跑结果一致。
5. **没有真值位置**：本任务不需要，但也意味着分类准确率不能替代"运动重建精度"。
6. **运行环境兼容**：某些受限环境禁止创建 Windows 命名管道，会让 sklearn 的
   `joblib.Parallel(backend="threading")` 报 `WinError 5`。`src/compat.py` 会探测一次，
   若不可用就把 joblib 降级为串行，但保留 OpenMP 的直方图并行——只影响分箱一步的速度。
   如果你的机器没有这个限制，这层不会有任何作用。

---

## 6. 给任务二/任务三留的接口

`feature_index.json` 里的 `unit_of_col` 是一个长度 2312 的数组，取值 0–16，
表示每个特征列属于哪个单元。删单元就是：

```python
import json, numpy as np

idx = json.load(open("outputs/feature_index.json", encoding="utf-8"))
unit_of_col = np.array(idx["unit_of_col"])

keep_units = [0, 8, 16]                  # 例如只保留下背、头、左腕
mask = np.isin(unit_of_col, keep_units)
F_reduced = F[:, mask]                   # 直接用，无需重算特征
```

同一套 `src/` 下的切窗与特征代码，改 `--feature-set` 或窗口长度即可复用，
保证任务二与任务一用的是**完全相同的评估条件**（同一切分、同一模型、同一指标）。

---

## 7. 建议的消融实验（写论文用）

```bash
# 1) 窗口长度敏感性
for w in 1.0 1.5 2.0 3.0; do
  python -m src.run_task1 --stage prepare --window $w --out-dir outputs_w$w
  python -m src.run_task1 --stage train  --window $w --out-dir outputs_w$w
done

# 2) Others 的三种处理
python -m src.run_task1 --stage train --others-mode drop   --out-dir outputs_drop
python -m src.run_task1 --stage train --others-mode reject --out-dir outputs_reject

# 3) 模型对照
for m in hgb rf et; do
  python -m src.run_task1 --stage train --model $m --out-dir outputs_$m
done
```

注意：`--stage prepare` 与 `--stage train` 的窗口参数必须一致，否则 `feature_index.json` 与特征对不上。
