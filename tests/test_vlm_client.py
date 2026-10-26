import json

import httpx
import numpy as np

from kee.config import Settings
from kee.ingest.vlm import VlmClient


def _settings(**overrides):
    base = dict(
        vlm_enabled=True,
        vlm_base_url="http://127.0.0.1:8000/v1",
        vlm_model="Qwen3-VL-8B-Instruct",
        vlm_min_image_pixels=1,
        vlm_timeout_sec=5,
    )
    base.update(overrides)
    return Settings(_env_file=None, **base)


def _img() -> np.ndarray:
    return np.full((64, 64, 3), 255, dtype=np.uint8)


def test_disabled_by_default_and_no_http_call():
    seen = []

    def handler(req):
        seen.append(req)
        return httpx.Response(200, json={"choices": [{"message": {"content": "不该被调用"}}]})

    c = VlmClient(
        settings=Settings(_env_file=None, vlm_enabled=False),
        http=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert c.enabled is False
    assert c.describe_image(_img()) == ""
    assert seen == []
    assert "未启用" in c.disabled_reason


def test_public_target_needs_explicit_allow_remote():
    c = VlmClient(settings=_settings(vlm_base_url="https://api.example.com/v1", vlm_api_key="k"))
    assert c.enabled is False
    assert "ALLOW_REMOTE" in c.disabled_reason

    c2 = VlmClient(
        settings=_settings(
            vlm_base_url="https://api.example.com/v1",
            vlm_api_key="k",
            vlm_allow_remote=True,
        )
    )
    assert c2.enabled is True


def test_openai_compatible_payload_and_content_list():
    seen = {}

    def handler(req):
        seen["url"] = str(req.url)
        seen["body"] = json.loads(req.content)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": [{"type": "text", "text": "柱状图，Q1 到 Q4 上升。"}]}}]},
        )

    c = VlmClient(
        settings=_settings(),
        http=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    out = c.describe_image(_img(), kind="chart", ocr_text="Q1 10 Q4 40", context="某项目结项报告")
    assert out == "柱状图，Q1 到 Q4 上升。"
    assert seen["url"].endswith("/v1/chat/completions")
    body = seen["body"]
    assert body["model"] == "Qwen3-VL-8B-Instruct"
    assert body["messages"][1]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert "Q1 10 Q4 40" in body["messages"][1]["content"][0]["text"]
    assert c.stats["ok"] == 1


def test_fail_fast_stops_after_threshold():
    calls = []

    def handler(req):
        calls.append(1)
        raise httpx.ConnectError("local vlm down")

    c = VlmClient(
        settings=_settings(vlm_fail_fast_after=2),
        http=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert c.describe_image(_img()) == ""
    assert c.describe_image(_img()) == ""
    assert c.describe_image(_img()) == ""
    assert len(calls) == 2
    assert c.stats["skipped"] == 1
    assert "ConnectError" in c.stats["last_error"]


def test_missing_remote_key_is_disabled():
    c = VlmClient(
        settings=_settings(vlm_base_url="https://api.example.com/v1", vlm_api_key="", vlm_allow_remote=True)
    )
    assert c.enabled is False
    assert "API_KEY" in c.disabled_reason
