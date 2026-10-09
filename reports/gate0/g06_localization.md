# Gate 0 ⑥ 单机本地化部署与数据不出域验证报告

- 验证时间：2026-10-09T16:31:37+08:00
- 环境：Windows 11 / GPU NVIDIA GeForce RTX 4060 Laptop GPU 8188MiB / commit 5bb82f0

## 一键启动

- compose config：通过
- `docker compose up -d --wait` 用时：40.0s，返回码 0
- 全组件 healthy：是（期望 5 个服务）
- Milvus 就绪轮询：通过（等待 0.0s，服务端 3.0.0）

| 服务 | 状态 | 宿主端口 |
|---|---|---|
| kee-etcd | Up 39 seconds (healthy) | 2379-2380/tcp |
| kee-milvus | Up 6 seconds (healthy) | 127.0.0.1:39091->9091/tcp, 127.0.0.1:39530->19530/tcp |
| kee-minio | Up 39 seconds (healthy) | 9000/tcp |
| kee-postgres | Up 39 seconds (healthy) | 127.0.0.1:35432->5432/tcp |
| kee-redis | Up 39 seconds (healthy) | 127.0.0.1:36379->6379/tcp |

## 连通性与绑定

| 组件 | 结果 |
|---|---|
| milvus | {"ok": true, "server": "3.0.0", "collections": 0} |
| postgres | {"ok": true, "version": "PostgreSQL 18.6 (Debian 18.6-1"} |
| redis | {"ok": true, "version": "8.10.1"} |
| ollama | {"ok": true, "model": "qwen2.5:7b", "resident": true} |
- 仅绑定 127.0.0.1：是
- 云端兜底开关：`False`

## 核心链路（嵌入→混合检索→生成）与宿主出网审计

- dense 命中 2 条 / sparse 命中 2 条 / 生成 成功（JSON 合法 True）
- 本进程非回环连接：0 条；其中公网目标：0 条
- 观测到的连接（去重）：[('0.0.0.0', '0', 'Bound'), ('127.0.0.1', '11434', 'Established'), ('127.0.0.1', '39530', 'Established'), ('::', '0', 'Bound')]

## 静态出站端点扫描

- 命中 URL 总数：1；分类：连通性探测靶 1 处、容器内部名 0 处、**运行期数据出域嫌疑 0 处**
- 分类口径：探测靶（1.1.1.1 等，只用于验证出网被堵）与容器内部服务名不构成数据出域；只有真实外部目的地才算嫌疑。
| 文件 | 行 | URL |
|---|---|---|
| src\kee\gate0\g06_localization.py | 307 | http://1.1.1.1 |

## 容器侧连接审计（正常网络，核心链路运行期间）

| 容器 | ESTAB 连接数 | 可疑对外连接 | 对端 |
|---|---|---|---|
| kee-etcd | 0 | 0 | [] |
| kee-minio | 0 | 0 | [] |
| kee-milvus | 29 | 0 | [["00000000…", "34374"], ["00000000…", "34382"], ["00000000…", "36750"], ["00000000…", "36 |
| kee-postgres | 0 | 0 | [] |
| kee-redis | 0 | 0 | [] |
- 容器能否访问公网（正常模式下应为可，说明 NAT 存在）：True
- Milvus 容器内 healthz：200；etcd 健康应答：{"health":"true","reason":""}

## 断网（internal 网络）变体

- internal 网络下全组件 healthy：是
- 容器出网被封堵（公网探测全部失败）：确认
- 容器间互联仍可用（milvus→etcd 健康应答）：{"health":"true","reason":""}
- 容器侧可疑对外连接：False
- 宿主经发布端口访问 internal 网络：不可用（预期实测结论）
- ⚠ 结论：`internal: true` 会同时切断宿主↔容器的发布端口通路，因此「宿主跑应用 + 容器全内网」这一组合无法直接验证；完整断网链路验证需应用容器化后在同网段内跑（列入 M4），本轮以「容器侧无可疑对外连接 + 仅绑定 127.0.0.1 + 宿主进程连接审计」三条证据支撑无外呼判定。

- **判定**：通过｜门禁：全组件一键启动可用；断网下核心链路可运行；确认无数据出域外呼｜实测：一键启动 40.0s 全 healthy=True；非回环绑定 0；公网出网连接 0；静态扫描运行期出域嫌疑 0

## 已知边界

- 应用与 Nginx 容器本轮未纳入（按方案 3.4 回退分支⑥：先收敛到最小核心组件），列入 M4。
- Neo4j 图谱为可选增强，本轮未起。
- 宿主进程连接审计只覆盖本进程与 Ollama 侧可见连接；Ollama 自身出网需单独审计（本机 Ollama 绑定 127.0.0.1）。
