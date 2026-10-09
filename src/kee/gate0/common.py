"""Gate 0 公共件：证据快照、判定与报告落盘（所有验证项统一走这里，保证口径一致）。"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPORTS_DIR = PROJECT_ROOT / "reports" / "gate0"
VAR_DIR = PROJECT_ROOT / "var" / "gate0"

# 方案 3.2 的门禁口径，集中一处，报告与判定共用。
# gate_original = 方案 V1.0.1 原文；gate = 2026-10-09 经用户裁决修订后的口径（V1.0.2），
# 修订依据写在 docs/技术方案与MVP设计（V1.0）.md 的修订记录里。
GATE_CRITERIA = {
    "①": {
        "title": "8GB 显存下嵌入/重排与本地生成模型的分时/降级调度可行性",
        "blocking": True,
        "gate_original": "全流程零 OOM；嵌入、重排按需加载/释放与生成分时互斥；P95 端到端 < 3s（本地命中路径）",
        "gate": "零 OOM 且嵌入+重排可与 7B 共存驻留；P95 检索 < 800ms；P95 完整答案（256 token 预算）< 6s；"
        "流式首 token < 1.2s 待 M4 流式实现后补测",
    },
    "②": {
        "title": "Milvus v3.0.0 服务端 BM25 + 中文分析器可用性与中文检索有效性",
        "blocking": True,
        "gate_original": "jieba 生效且 BM25 相对默认分析器有明确正增益（非≈0）；术语/编号类 query 明显改善",
        "gate": "中文分析器生效且 BM25 相对默认分析器有明确正增益（非≈0）；术语/编号类 query 明显改善",
    },
    "③": {
        "title": "混合检索相对单一稠密的增益（消融对照）",
        "blocking": False,
        "gate_original": "增益可归因、可复现，且混合方案 Recall@5 ≥ 85% 具备达成路径",
        "gate": "增益可归因、可复现，且混合方案 Recall@5 ≥ 85% 具备达成路径",
    },
    "④": {
        "title": "多存储一致性（Outbox + 每日对账）",
        "blocking": False,
        "gate_original": "人为注入的不一致能被发现并修复；不出现静默缺数据",
        "gate": "人为注入的不一致能被发现并修复；不出现静默缺数据",
    },
    "⑤": {
        "title": "引用强制与置信度校准（温度缩放 + ECE）",
        "blocking": False,
        "gate_original": "ECE ≤ 0.10 具备达成路径；无来源必不输出结论；低置信度必触发降级/转人工",
        "gate": "按回退分支⑤口径：无来源必不输出结论（零引用结论=0）；无依据必拒答；敏感/低置信必转人工；"
        "结构化输出首次成功率 ≥ 90%。连续 ECE 降为报告值继续跟踪，不作阻塞判定",
    },
    "⑥": {
        "title": "单机 Docker Compose 全组件本地化与数据不出域",
        "blocking": True,
        "gate_original": "全组件一键启动可用；断网下核心链路可运行；确认无数据出域外呼",
        "gate": "全组件一键启动可用；断网下核心链路可运行；确认无数据出域外呼",
    },
}


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def env_snapshot() -> dict[str, Any]:
    import torch

    gpu = {"name": "", "total_mib": 0, "used_mib": 0}
    try:
        from kee.models.gpu import VramSampler

        sampler = VramSampler()
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        gpu = {
            "name": props.name,
            "total_mib": round(props.total_memory / 1048576),
            "used_mib": round(sampler._global_used_mib(), 1),  # noqa: SLF001
        }
    except Exception as e:  # noqa: BLE001
        gpu["error"] = str(e)

    try:
        import transformers

        tfv = transformers.__version__
    except Exception:  # noqa: BLE001
        tfv = ""
    try:
        import pymilvus

        pmv = pymilvus.__version__
    except Exception:  # noqa: BLE001
        pmv = ""

    return {
        "ts": now_iso(),
        "git_commit": _git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.release()}",
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu,
        "transformers": tfv,
        "pymilvus": pmv,
    }


def write_json(name: str, obj: Any) -> Path:
    VAR_DIR.mkdir(parents=True, exist_ok=True)
    p = VAR_DIR / name
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return p


def read_json(name: str) -> Any:
    return json.loads((VAR_DIR / name).read_text(encoding="utf-8"))


def write_md(name: str, lines: list[str]) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    p = REPORTS_DIR / name
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


ITEM_CODE = {"①": "01", "②": "02", "③": "03", "④": "04", "⑤": "05", "⑥": "06"}


def verdict(item: str, result: str, measured: str, note: str = "") -> str:
    """result: pass / partial / fail（partial = 具备达成路径但有保留条件）。

    同时把机器可读判定落盘，供 g00_summary 按方案 3.3 规则汇总放行结论。
    """
    mark = {"pass": "通过", "partial": "有条件通过", "fail": "不通过"}[result]
    crit = GATE_CRITERIA[item]
    line = f"- **判定**：{mark}｜门禁：{crit['gate']}｜实测：{measured}"
    if note:
        line += f"｜说明：{note}"
    VAR_DIR.mkdir(parents=True, exist_ok=True)
    (VAR_DIR / f"verdict_{ITEM_CODE[item]}.json").write_text(
        json.dumps(
            {
                "item": item,
                "title": crit["title"],
                "blocking": crit["blocking"],
                "result": result,
                "mark": mark,
                "gate": crit["gate"],
                "measured": measured,
                "note": note,
                "ts": now_iso(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return line


def md_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return out


def fmt_pct(x: float | None) -> str:
    return "-" if x is None or x != x else f"{x * 100:.1f}%"
