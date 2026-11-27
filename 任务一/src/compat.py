"""运行环境兼容层。

问题：sklearn 的部分代码用 ``joblib.Parallel(backend="threading")`` 做并行分箱，
它在 Windows 上通过 ``multiprocessing.Pipe`` 创建命名管道。某些受限环境
（沙箱 / 组策略）会拒绝创建命名管道，表现为::

    PermissionError: [WinError 5] 拒绝访问。

对策：启动时探测一次能否创建管道；若不能，把 joblib 的 Parallel 强制降级为
``n_jobs=1``（串行执行，不再建池），但 **保留 OpenMP 的直方图并行**，
因此 HGB 的训练速度基本不受影响，只有分箱一步变成串行。
"""

from __future__ import annotations

import sys
from typing import Any

_APPLIED = False


def pipes_available() -> bool:
    try:
        import multiprocessing as mp

        q = mp.SimpleQueue()
        q.close()
        return True
    except Exception:
        return False


def force_sequential_joblib() -> None:
    """把 joblib / sklearn 的 Parallel 全部改成 n_jobs=1。"""
    global _APPLIED
    if _APPLIED:
        return

    try:
        import joblib
        from joblib.parallel import Parallel as JoblibParallel
    except Exception:
        return

    try:
        from sklearn.utils.parallel import Parallel as SklearnParallel
    except Exception:
        SklearnParallel = None

    class _SeqJoblibParallel(JoblibParallel):
        def __init__(self, n_jobs: Any = None, *args: Any, **kwargs: Any) -> None:
            super().__init__(n_jobs=1, *args, **kwargs)

    replacement: Any = _SeqJoblibParallel

    if SklearnParallel is not None and SklearnParallel is not JoblibParallel:
        class _SeqSklearnParallel(SklearnParallel):  # type: ignore[misc,valid-type]
            def __init__(self, n_jobs: Any = None, *args: Any, **kwargs: Any) -> None:
                super().__init__(n_jobs=1, *args, **kwargs)

        replacement_cls = _SeqSklearnParallel
    else:
        replacement_cls = _SeqJoblibParallel

    # 把已经 import 过的模块里绑定的 Parallel 名字一起换掉
    swapped = 0
    for mod in list(sys.modules.values()):
        if mod is None:
            continue
        cur = getattr(mod, "Parallel", None)
        if cur is JoblibParallel:
            setattr(mod, "Parallel", _SeqJoblibParallel)
            swapped += 1
        elif SklearnParallel is not None and cur is SklearnParallel:
            setattr(mod, "Parallel", replacement_cls)
            swapped += 1

    joblib.Parallel = _SeqJoblibParallel
    joblib.parallel.Parallel = _SeqJoblibParallel
    try:
        import sklearn.utils.parallel as sup

        sup.Parallel = replacement_cls
    except Exception:
        pass

    _APPLIED = True
    print(f"[compat] 检测到命名管道不可用，joblib 已降级为串行（替换 {swapped} 处引用）")


def ensure_runnable() -> None:
    """脚本入口调用：不可用管道时自动降级，并给出提示。"""
    if not pipes_available():
        force_sequential_joblib()
