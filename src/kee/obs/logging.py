"""结构化日志：一行一个 JSON，字段稳定，`trace_id` 与响应体/审计/`session_turn` 同源。

为什么默认 JSON 而不是人读的那行 `INFO:xxx`：这台机器上的在线链路已经有一整套
「按 trace_id 串起来」的机制（响应头、audit_log、session_turn），日志是这条链上唯一
还在用自由文本的一环——`grep 超时` 查不出是哪个阶段，而 `--filter-json` 能。
要看日志的人：`tail -f logs/kee.log | jq -c '{t,level,msg,trace_id}'`。

重复配置防护：`configure_logging()` 会在 `create_app()` 里调，而验收脚本一晚上能调十几次，
所以按 handler 标记去重，不然每条日志会打三遍。
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any

_CONFIGURED_ATTR = "_kee_logging_configured"
_RESERVED = frozenset(
    {"args", "exc_info", "exc_text", "filename", "funcName", "levelname", "levelno", "lineno",
     "module", "msecs", "msg", "name", "pathname", "process", "processName", "relativeCreated",
     "stack_info", "taskName", "thread", "threadName"}
)


class JsonFormatter(logging.Formatter):
    """`extra` 里的自定义键一律平铺到顶层（`logger.info("msg", extra={"trace_id": ...})`）。"""

    def format(self, record: logging.LogRecord) -> str:
        row: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created))
            + f".{int(record.msecs):03d}",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, val in record.__dict__.items():
            if key in _RESERVED or key.startswith("_"):
                continue
            row[key] = val
        if record.exc_info:
            row["exc"] = self.formatException(record.exc_info)
        return json.dumps(row, ensure_ascii=False, default=str)


class TextFormatter(logging.Formatter):
    """`KEE_LOG_JSON=false` 时的本地读法，仍带 trace_id（关掉结构化不该把关联 id 一起关掉）。"""

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        trace = getattr(record, "trace_id", "")
        return f"{base} trace_id={trace}" if trace else base


def configure_logging(level: str | None = None, as_json: bool | None = None) -> logging.Handler:
    """幂等地装好根 logger。返回实际用的 handler，方便测试断言。"""
    root = logging.getLogger()
    lvl = getattr(logging, (level or os.environ.get("KEE_LOG_LEVEL", "INFO")).upper(), logging.INFO)
    if as_json is None:
        as_json = os.environ.get("KEE_LOG_JSON", "true").strip().lower() not in ("0", "false", "no", "off")
    fmt: logging.Formatter = JsonFormatter() if as_json else TextFormatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s"
    )

    handler = _existing_handler(root)
    if handler is not None:
        handler.setFormatter(fmt)
        root.setLevel(lvl)
        return handler

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(fmt)
    setattr(handler, _CONFIGURED_ATTR, True)
    root.addHandler(handler)
    root.setLevel(lvl)
    # 装了 root handler 之前，第三方库的 INFO 无处可去（`lastResort` 只放 WARNING 以上）；
    # 一装就全出来了：`verify_m1` 每次依赖调用一行 `httpx` INFO，验收输出被淹掉。
    # 所以这里显式把噪声源压到 WARNING——`pypdf` 的字体告警、Ollama 的 4xx 仍然出得来。
    for noisy in NOISY_LOGGERS:
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    return handler


NOISY_LOGGERS = (
    "httpx", "httpcore", "urllib3", "asyncio", "pymilvus", "huggingface_hub",
    "sentence_transformers", "transformers", "filelock",
)


def _existing_handler(root: logging.Logger) -> logging.Handler | None:
    for h in root.handlers:
        if getattr(h, _CONFIGURED_ATTR, False):
            return h
    return None


def log_kv(logger: logging.Logger, level: int, msg: str, **fields: Any) -> None:
    """带结构化字段的日志入口。走 `extra`，所以字段名不能与 `LogRecord` 的保留键同名。"""
    logger.log(level, msg, extra=dict(fields))
