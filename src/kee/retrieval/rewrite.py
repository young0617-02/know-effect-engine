"""查询改写：把口语化、省略式的提问补全成检索友好的查询（方案 4.1 在线链路第 2 环）。

两条硬约束决定了这里的取舍：
  1) **改写不得引入事实**。检索错了只是找不到，改写凭空造出项目编号/数值会让下游
     引用指向不存在的内容——那是引用造假。所以模型输出要过校验门：数字必须是原查询里
     出现过的、长度不得膨胀到 3 倍、必须能解析成 JSON，任一不过就退回原查询。
  2) **本地 7B 改写一次约 0.6~0.9s，而检索 P95 门禁是 800ms**。所以改写默认只在
     「像是要改写」的查询上触发（短查询、指代、省略），其余直接透传；命中缓存才算免费。
     真实代价由评测作业量出来，不靠假设收口。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from kee.config import get_settings
from kee.retrieval.cache import SearchCache

# 指代/省略/追问信号：中文口语里这些出现时，字面查询通常不足以定位段落
_ELLIPSIS = re.compile(r"(它|他|她|这|那|该|此|上面|刚才|之前|那个|这些|那些|呢|吗|怎么办|还有|另外)")
_DIGITS = re.compile(r"\d+(?:\.\d+)*")
# 明显过短的字面查询（如「线损」）几乎必然需要扩展
_MIN_AUTO_LEN = 8

# 输出只留 expanded：实测要求模型顺带解释理由时，reason 会吃到 num_predict 上限、
# 把整段 JSON 截断在半路，70 题里 21 题因此被判「无效输出」丢掉——理由不是这条链路的产物，
# 不值得用 30% 的失败率去换。
PROMPT = (
    "你是企业知识库检索的查询改写器。给定用户查询，只补全查询里已经隐含的信息：\n"
    "展开缩写、补出被指代或省略的主语、把口语同义说法换成规范术语。\n"
    "禁止：引入查询中没有的地名、人名、编号、数字、时间或结论。\n"
    '若查询本身已自足，expanded 就照抄原查询。只输出一行 JSON：{{"expanded": "..."}}\n'
    "查询：{query}\n"
)


@dataclass
class RewriteResult:
    original: str
    expanded: str
    changed: bool = False
    triggered: bool = False
    from_cache: bool = False
    skipped_reason: str = ""
    rejected_reason: str = ""
    ms: float = 0.0
    degraded: list[str] = field(default_factory=list)

    @property
    def effective(self) -> str:
        return self.expanded if self.changed else self.original


def needs_rewrite(query: str) -> bool:
    """启发式触发：只在字面查询明显不足以定位时才花一次 LLM 调用。"""
    q = query.strip()
    if not q:
        return False
    if _ELLIPSIS.search(q):
        return True
    if len(q) <= _MIN_AUTO_LEN:
        return True
    return False


def tidy(expanded: str) -> str:
    """去掉收尾句号：用户自己的问句可能以「。」结尾，模型照抄不该算违规。"""
    return (expanded or "").strip().rstrip("。；;")


def validate(original: str, expanded: str) -> str:
    """改写内容的准入校验；不过就返回拒绝原因。"""
    e = tidy(expanded)
    if not e:
        return "空改写"
    if len(e) > 3 * max(len(original), 8):
        return f"膨胀超限({len(e)}>{3 * max(len(original), 8)})"
    if any(d not in original for d in _DIGITS.findall(e)):
        return "引入了原查询没有的数字"
    for bad in ("。", "；", "\n"):
        if bad in e:
            return f"含句内多句结构({bad})"
    return ""


class QueryRewriter:
    def __init__(
        self,
        client=None,
        cache: SearchCache | None = None,
        budget_ms: float | None = None,
        always: bool = False,
    ) -> None:
        s = get_settings()
        self.settings = s
        self.cache = cache
        self.budget_ms = budget_ms if budget_ms is not None else s.rewrite_budget_ms
        self.always = always
        self._client = client
        self.stats: dict[str, int] = {
            "calls": 0, "changed": 0, "cached": 0, "skipped": 0, "rejected": 0, "failed": 0, "over_budget": 0,
        }

    @property
    def client(self):
        if self._client is None:
            from kee.llm.ollama import OllamaClient

            self._client = OllamaClient()
        return self._client

    def _cache_key(self, query: str) -> str:
        assert self.cache is not None
        return self.cache.make_key(query, "rewrite", "", 0)

    def rewrite(self, query: str) -> RewriteResult:
        q = (query or "").strip()
        res = RewriteResult(original=q, expanded=q)
        if not q:
            res.skipped_reason = "空查询"
            self.stats["skipped"] += 1
            return res
        if not (self.always or needs_rewrite(q)):
            res.skipped_reason = "字面查询已自足"
            self.stats["skipped"] += 1
            return res
        res.triggered = True

        if self.cache is not None:
            hit = self.cache.get(self._cache_key(q))
            if hit is not None:
                res.expanded = hit.get("expanded", q)
                res.changed = res.expanded != q
                res.from_cache = True
                self.stats["cached"] += 1
                return res

        t0 = time.perf_counter()
        self.stats["calls"] += 1
        try:
            r = self.client.chat(
                [{"role": "user", "content": PROMPT.format(query=q)}],
                json_mode=True,
                temperature=0.0,
                max_tokens=self.settings.rewrite_max_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - 改写故障绝不能拖垮检索
            res.ms = round((time.perf_counter() - t0) * 1000, 1)
            res.degraded.append(f"rewrite_error:{type(exc).__name__}")
            self.stats["failed"] += 1
            return res
        res.ms = round((time.perf_counter() - t0) * 1000, 1)
        if res.ms > self.budget_ms:
            res.degraded.append(f"rewrite_over_budget({res.ms:.0f}ms>{self.budget_ms:.0f}ms)")
            self.stats["over_budget"] += 1
        if not r.ok:
            res.degraded.append(f"rewrite_bad_output:{r.error or 'no_output'}")
            self.stats["failed"] += 1
            return res
        parsed = r.parsed if isinstance(r.parsed, dict) else {}
        salvaged = False
        if not str(parsed.get("expanded", "")).strip():
            parsed = parse_rewrite_json(r.text)
            salvaged = bool(parsed.get("salvaged"))
        cand = tidy(str(parsed.get("expanded", "")))
        if not cand:
            res.degraded.append("rewrite_bad_output:no_expanded_field")
            self.stats["failed"] += 1
            return res
        if salvaged:
            res.degraded.append("rewrite_json_salvaged")
        why = validate(q, cand)
        if why:
            res.rejected_reason = why
            self.stats["rejected"] += 1
            return res
        res.expanded = cand
        res.changed = cand != q
        if res.changed:
            self.stats["changed"] += 1
            if self.cache is not None:
                self.cache.set(self._cache_key(q), {"expanded": cand})
        return res

    def summary(self) -> dict[str, int]:
        return dict(self.stats)


_EXPANDED_FIELD = re.compile(r'"expanded"\s*:\s*"((?:[^"\\]|\\.)*)"')


def parse_rewrite_json(text: str) -> dict:
    """解析改写输出，并对**被截断的 JSON 做打捞**。

    7B 常在 num_predict 上限处把 JSON 截断，此时 expanded 已经完整、只是收尾括号没了，
    整段判废等于把一次正确改写扔掉。所以严格解析失败时只打捞「已闭合的 expanded 值」——
    正则要求引号成对，截断在未闭合字符串里的情况不会被误捞。
    """
    raw = text or ""
    m = re.search(r"\{.*\}", raw, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            if isinstance(d, dict):
                return d
        except json.JSONDecodeError:
            pass
    f = _EXPANDED_FIELD.search(raw)
    if not f:
        return {}
    try:
        val = json.loads('"' + f.group(1) + '"')
    except json.JSONDecodeError:
        return {}
    return {"expanded": val, "salvaged": True}
