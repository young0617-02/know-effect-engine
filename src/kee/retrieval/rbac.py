"""RBAC：在检索层构造过滤表达式（「搜得到」的前提是有权限看）。

Milvus filter 是表达式字符串，直接把用户上下文拼进去等于开后门，
因此所有标识符先过白名单字符校验，再拼表达式。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# 允许中文、字母、数字、下划线、连字符、点、冒号、空格（项目/行业命名实测集合）
_SAFE = re.compile(r"^[0-9A-Za-z_\-\u4e00-\u9fff.: ]+$")


class UnsafeIdentifier(ValueError):
    pass


def _safe(value: str) -> str:
    if not isinstance(value, str) or not value or not _SAFE.match(value):
        raise UnsafeIdentifier(f"表达式标识符不合法：{value!r}")
    return value


# 角色 → 默认可见密级上限；行业/项目维度由上下文显式给出
ROLE_MAX_LEVEL = {
    "delivery_engineer": 2,
    "presales_architect": 2,
    "pm": 3,
    "ops_support": 2,
    "newcomer": 1,
    "knowledge_owner": 3,
    "system_admin": 3,
}


@dataclass
class AccessContext:
    role: str = "delivery_engineer"
    department: str = "交付中心"
    industries: list[str] = field(default_factory=list)
    projects: list[str] = field(default_factory=list)
    max_security_level: int | None = None

    @property
    def effective_max_level(self) -> int:
        if self.max_security_level is not None:
            return self.max_security_level
        return ROLE_MAX_LEVEL.get(self.role, 1)

    def is_unrestricted_scope(self) -> bool:
        return self.role in ("knowledge_owner", "system_admin")


def build_filter_expr(ctx: AccessContext) -> str:
    parts = [f"security_level <= {ctx.effective_max_level}"]
    if not ctx.is_unrestricted_scope():
        if ctx.industries:
            vals = ", ".join(f'"{_safe(i)}"' for i in ctx.industries)
            parts.append(f"industry in [{vals}]")
        if ctx.projects:
            vals = ", ".join(f'"{_safe(p)}"' for p in ctx.projects)
            parts.append(f"project in [{vals}]")
    return " and ".join(parts)
