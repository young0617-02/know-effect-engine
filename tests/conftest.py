"""测试通道的显式声明。

`allow_header_identity` 的**生产默认已是 False**（用户 2026-10-15 裁决），所以不带令牌、
只发 `X-Kee-Role` 的旧跑法在交付物里已经关掉。测试套件里有整批用例测的就是这条通道本身
（RBAC 过滤表达式、审计 actor、密级收窄），它们不是「忘了改」，是在**测这个功能**——
于是把开关在这里显式打开，而不是把默认值改回去。

真 HTTP 的验收脚本不吃这个环境变量：它们走 `scripts/_identity.py` 登录换 Bearer，
和界面用的是同一条入口。默认值本身由 `tests/test_auth.py::test_header_identity_defaults_off`
钉住，改回 True 会让那条测试变红。
"""

from __future__ import annotations

import os

os.environ.setdefault("KEE_ALLOW_HEADER_IDENTITY", "true")
