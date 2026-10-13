"""版本锁定清单的依赖探针：钉住两个会把「没用」说错的形状。

- 命名空间包：`langgraph-checkpoint-postgres` 的 top_level 是 `langgraph`，
  照名扫会把父包的 `from langgraph.graph import …` 算成它在用 ⇒ 零导入这条事实被工具抹掉。
- 框架隐式 import：`python-multipart` 本仓库不 import，但 FastAPI 在 `UploadFile = File(...)`
  时按需去 import ⇒ 机械扫会把它判成「没用到的依赖」，摘掉线上上传端点 500。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "m5_version_lock.py"
spec = importlib.util.spec_from_file_location("m5_version_lock", SCRIPT)
vl = importlib.util.module_from_spec(spec)
sys.modules["m5_version_lock"] = vl
spec.loader.exec_module(vl)


def test_namespace_pkg_does_not_inherit_parent_imports():
    prefixes, how = vl.import_prefixes("langgraph-checkpoint-postgres")
    assert prefixes == ["langgraph.checkpoint.postgres"]
    assert "人工映射" in how


def test_implicit_framework_dep_is_not_reported_as_unused():
    usage = vl.dep_usage()
    assert usage["python-multipart"]["判定"].startswith("隐式使用")
    assert usage["python-multipart"]["命中文件"] == []          # 确实扫不到，靠依据而不是靠扫
    assert usage["langgraph-checkpoint-postgres"]["判定"] == "声明了但零导入"


def test_every_declared_dep_gets_a_verdict():
    usage = vl.dep_usage()
    assert usage, "pyproject 依赖解析出来了才有判定"
    assert all(v["判定"] in {"在用", "声明了但零导入", "未判定"} or v["判定"].startswith("隐式使用")
               for v in usage.values())
    # 「未判定」不是失败：它必须带上为什么判定不了，否则读的人以为等同零导入
    for dep, v in usage.items():
        if v["判定"] == "未判定":
            assert v["前缀来源"].startswith("未判定"), dep


def test_prefixes_drop_directory_noise():
    """`numpy.libs` / `..` 这类首层目录不是 import 名，留在前缀里就是往正则里塞垃圾。"""
    usage = vl.dep_usage()
    for dep, v in usage.items():
        for p in v.get("前缀", []):
            assert "." not in p or p.startswith("langgraph."), f"{dep} 的前缀 {p} 不该有点分段的目录名"
            assert p not in {"..", "."}
