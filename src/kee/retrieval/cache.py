"""检索结果缓存（Redis，非权威、TTL 可驱逐；命中即短路整条检索链路）。"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import redis

from kee.config import get_settings


class SearchCache:
    def __init__(self, url: str | None = None, ttl_sec: int | None = None) -> None:
        s = get_settings()
        self._r = redis.Redis.from_url(url or s.redis_url, socket_connect_timeout=5, decode_responses=True)
        self.ttl_sec = ttl_sec if ttl_sec is not None else s.search_cache_ttl_sec
        self.hits = 0
        self.misses = 0
        self.errors = 0

    @staticmethod
    def make_key(query: str, mode: str, expr: str, top_k: int) -> str:
        payload = json.dumps([query.strip(), mode, expr, top_k], ensure_ascii=False)
        return "kee:search:" + hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Any | None:
        try:
            raw = self._r.get(key)
        except redis.exceptions.RedisError:
            self.errors += 1
            return None
        if raw is None:
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(raw)

    def set(self, key: str, value: Any) -> None:
        try:
            self._r.set(key, json.dumps(value, ensure_ascii=False), ex=self.ttl_sec)
        except redis.exceptions.RedisError:
            self.errors += 1

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return round(self.hits / total, 4) if total else 0.0

    def clear_namespace(self) -> int:
        n = 0
        for k in self._r.scan_iter("kee:search:*"):
            self._r.delete(k)
            n += 1
        return n

    def close(self) -> None:
        self._r.close()
