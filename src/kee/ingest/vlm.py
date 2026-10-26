"""入库期 VLM：把图片/图表转成可检索的中文描述。

设计边界（与《技术方案》的「数据不出域」同源）：

1. **只在建索引时调用**。检索与回答阶段仍然只走文本嵌入 + 文本 LLM，VLM 不参与在线链路，
   所以 8GB 单卡不需要在服务时常驻一个视觉模型。
2. **默认关闭**。开启后，`vlm_base_url` 必须是回环/私网地址；公网地址要显式设
   `KEE_VLM_ALLOW_REMOTE=true` 才放行——那个开关就是一条「图片会离开本机」的书面确认。
3. **失败不阻断入库**。单张图超时/返回空/服务没起，都只丢这一张的描述，OCR 文字和原文照常入库；
   连续失败达到阈值后熔断，避免一篇 20 张图的文档被 20 次超时拖死。
4. **不把图片字节写日志**。日志只记 kind、耗时、状态与模型名。

实现协议是最小 OpenAI 兼容 `/chat/completions`（vLLM / Ollama / 兼容云厂商都可）。
刻意不引 SDK：与 Langfuse 同理，为一个入库期功能拖一整套客户端依赖不划算。
"""

from __future__ import annotations

import base64
import ipaddress
import logging
import re
import time
from io import BytesIO
from typing import Any
from urllib.parse import urlsplit

import httpx
import numpy as np

from kee.config import Settings, get_settings

log = logging.getLogger("kee.ingest.vlm")

ALLOWED_NAMES = frozenset({"localhost", "host.docker.internal", "gateway.docker.internal", "kubernetes.docker.internal"})

_KIND_PROMPTS = {
    "chart": (
        "这是一张文档中的图表。请用中文描述：图表类型、标题、坐标轴/图例含义、"
        "可见的关键数值或数据点、趋势或对比结论。只写图中可见的信息，不要补造数据。"
        "如果图中文字太少无法判断，就直说看不清。120 字以内。"
    ),
    "figure": (
        "这是一张文档中的插图或示意图。请用中文描述它表达的结构、流程、关系或关键信息。"
        "只写图中可见的信息，不要编造。120 字以内。"
    ),
    "image": (
        "这是一张文档中的图片。请用中文描述图片内容，并抄录其中与业务相关的关键文字或数据。"
        "如果它是图表，请描述图表类型、坐标轴/图例和可见趋势。只写图中可见的信息，不要编造。120 字以内。"
    ),
    "scanned_page": (
        "这是一页扫描件。请用中文概括页面类型（如表格、通知、规范页）与主要内容，"
        "不要逐字复述。120 字以内。"
    ),
    "formula": (
        "把图中的数学公式转写成 LaTeX。只输出 LaTeX 源码，不要解释、不要 Markdown 代码围栏。"
    ),
}

_SYSTEM_PROMPT = (
    "你是企业知识库入库阶段的视觉描述器。你的输出会作为图片的检索文本，"
    "所以只描述图中真实可见的内容；看不清就写看不清，禁止依据常识补全。"
)


def _in_domain(url: str) -> tuple[bool, str]:
    """回环/私网/链路本地或本机白名单名才算域内；公网一律默认拒绝。"""
    try:
        parts = urlsplit(url)
    except ValueError as exc:
        return False, f"URL 解析失败：{type(exc).__name__}"
    host = (parts.hostname or "").strip().lower()
    if not host:
        return False, "vlm_base_url 没有主机名"
    if parts.scheme not in ("http", "https"):
        return False, f"协议不支持：{parts.scheme or '(空)'}"
    if host in ALLOWED_NAMES:
        return True, f"名字白名单：{host}"
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False, f"非 IP 也非白名单名字：{host}"
    if ip.is_loopback or ip.is_private or ip.is_link_local:
        return True, f"私网/回环地址：{ip}"
    return False, f"公网地址 {ip}"


class VlmClient:
    """OpenAI 兼容的多模态描述客户端；禁用时所有方法安全返回空串。"""

    def __init__(self, settings: Settings | None = None, http: httpx.Client | None = None) -> None:
        self.s = settings or get_settings()
        self.stats: dict[str, Any] = {
            "calls": 0,
            "ok": 0,
            "failed": 0,
            "skipped": 0,
            "consecutive_failures": 0,
            "last_error": "",
            "last_ms": 0.0,
        }
        self.disabled_reason = ""
        self.enabled = False
        self._consecutive_failures = 0
        self._circuit_open = False
        self._http: httpx.Client | None = None
        self.endpoint = ""
        self._client_owned = http is None

        if not self.s.vlm_enabled:
            self.disabled_reason = "未启用（KEE_VLM_ENABLED=false，默认）"
            return
        if not (self.s.vlm_base_url or "").strip():
            self.disabled_reason = "未配置 KEE_VLM_BASE_URL"
            return
        if not (self.s.vlm_model or "").strip():
            self.disabled_reason = "未配置 KEE_VLM_MODEL"
            return

        base = self.s.vlm_base_url.rstrip("/")
        ok, why = _in_domain(base)
        remote = not ok
        if remote and not self.s.vlm_allow_remote:
            self.disabled_reason = (
                f"目标越域被拒：{why}；确需用云端 API 测试请显式设 KEE_VLM_ALLOW_REMOTE=true"
            )
            return
        if remote and not self.s.vlm_api_key:
            self.disabled_reason = "云端目标缺少 KEE_VLM_API_KEY"
            return
        if remote:
            log.warning("VLM 使用域外目标 %s：图片与提示词会离开本机（已由 KEE_VLM_ALLOW_REMOTE 显式放行）", base)

        self.endpoint = base if base.endswith("/chat/completions") else f"{base}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.s.vlm_api_key:
            headers["Authorization"] = f"Bearer {self.s.vlm_api_key}"
        self._headers = headers
        if http is None:
            http = httpx.Client(
                timeout=httpx.Timeout(self.s.vlm_timeout_sec, connect=5.0),
                headers=headers,
            )
        self._http = http
        self.enabled = True

    @classmethod
    def from_settings(cls, settings: Settings | None = None, http: httpx.Client | None = None) -> "VlmClient":
        return cls(settings=settings, http=http)

    def summary(self) -> dict[str, Any]:
        return {**self.stats, "enabled": self.enabled, "model": self.s.vlm_model if self.enabled else "",
                "disabled_reason": self.disabled_reason}

    def describe_image(
        self,
        image: np.ndarray | bytes,
        *,
        kind: str = "image",
        ocr_text: str = "",
        context: str = "",
    ) -> str:
        """返回中文描述/公式 LaTeX；禁用、熔断或失败时返回空串。"""
        if not self.enabled or self._http is None:
            self.stats["skipped"] += 1
            return ""
        if self._circuit_open:
            self.stats["skipped"] += 1
            return ""
        try:
            data_url = self._data_url(image)
        except Exception as exc:  # noqa: BLE001 - 一张图坏了不影响整篇
            self.stats["skipped"] += 1
            self.stats["last_error"] = f"encode:{type(exc).__name__}: {exc}"[:200]
            log.warning("VLM 图片编码失败，跳过：%s", type(exc).__name__)
            return ""

        prompt = _KIND_PROMPTS.get(kind, _KIND_PROMPTS["image"])
        if (ocr_text or "").strip():
            prompt += f"\nOCR 识别到的文字（可能不完整）：{(ocr_text or '').strip()[:1500]}"
        if (context or "").strip():
            prompt += f"\n文档/章节：{(context or '').strip()[:200]}"
        payload = {
            "model": self.s.vlm_model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            "temperature": 0.1,
            "max_tokens": self.s.vlm_max_tokens,
        }
        t0 = time.perf_counter()
        self.stats["calls"] += 1
        try:
            resp = self._http.post(self.endpoint, json=payload, headers=self._headers)
            resp.raise_for_status()
            text = self._content_of(resp.json()).strip()
            if not text:
                raise ValueError("empty content")
            if kind == "formula":
                text = re.sub(r"^```(?:latex|tex)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL)
                text = text.strip().strip("$").strip()
            text = " ".join(text.split())
            self.stats["ok"] += 1
            self.stats["consecutive_failures"] = 0
            self._consecutive_failures = 0
            self.stats["last_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            return text[:1200]
        except Exception as exc:  # noqa: BLE001 - 入库不能因 VLM 失败而失败
            self.stats["failed"] += 1
            self._consecutive_failures += 1
            self.stats["consecutive_failures"] = self._consecutive_failures
            self.stats["last_error"] = f"{type(exc).__name__}: {exc}"[:200]
            self.stats["last_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            log.warning("VLM 调用失败（%s/%s）：%s", self._consecutive_failures,
                        self.s.vlm_fail_fast_after, type(exc).__name__)
            if self._consecutive_failures >= max(1, self.s.vlm_fail_fast_after):
                self._circuit_open = True
                log.warning("VLM 连续失败达到阈值，本进程后续调用熔断；已入库文字不受影响")
            return ""

    def close(self) -> None:
        if self._http is not None and self._client_owned:
            self._http.close()

    # ---- 内部 ----
    def _data_url(self, image: np.ndarray | bytes) -> str:
        from PIL import Image

        if isinstance(image, (bytes, bytearray)):
            im = Image.open(BytesIO(bytes(image)))
        else:
            arr = np.asarray(image)
            if arr.ndim == 2:
                arr = np.repeat(arr[:, :, None], 3, axis=2)
            if arr.ndim != 3 or arr.shape[2] < 3:
                raise ValueError(f"不支持的图像形状：{arr.shape}")
            rgb = arr[:, :, :3][:, :, ::-1]
            im = Image.fromarray(rgb.astype("uint8"), mode="RGB")
        im = im.convert("RGB")
        if im.width * im.height < max(1, self.s.vlm_min_image_pixels):
            raise ValueError(f"图片过小：{im.width}x{im.height}")
        if max(im.size) > 1600:
            im.thumbnail((1600, 1600), Image.BILINEAR)
        buf = BytesIO()
        im.save(buf, format="JPEG", quality=max(40, min(95, self.s.vlm_jpeg_quality)), optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    @staticmethod
    def _content_of(data: dict[str, Any]) -> str:
        choices = data.get("choices") or []
        if not choices:
            raise ValueError("response 没有 choices")
        content = (choices[0].get("message") or {}).get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            return "".join(parts)
        return ""
