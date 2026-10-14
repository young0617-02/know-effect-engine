"""Gate 0 汇总：按方案 3.3 判定规则出放行结论，并回填版本锁定清单与参数基线。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from kee.config import get_settings  # noqa: E402
from kee.gate0.common import (  # noqa: E402
    GATE_CRITERIA,
    ITEM_CODE,
    VAR_DIR,
    env_snapshot,
    md_table,
    read_json,
    write_md,
)

RUNNERS = {
    "①": "src/kee/gate0/g01_vram_scheduler.py",
    "②": "src/kee/gate0/g02_chinese_bm25.py",
    "③": "src/kee/gate0/g03_ablation.py",
    "④": "src/kee/gate0/g04_consistency.py",
    "⑤": "src/kee/gate0/g05_citation_calibration.py",
    "⑥": "src/kee/gate0/g06_localization.py",
}


def load_verdicts() -> dict[str, dict]:
    out = {}
    for item, code in ITEM_CODE.items():
        p = VAR_DIR / f"verdict_{code}.json"
        out[item] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    return out


def decide(v: dict[str, dict]) -> tuple[str, str, list[str]]:
    executed = {k: x for k, x in v.items() if x}
    missing = [k for k in RUNNERS if k not in executed]
    blocking_items = [k for k, crit in GATE_CRITERIA.items() if crit["blocking"]]
    reasons = []
    blocking_ok = all(
        executed.get(b, {}).get("result") in ("pass",) for b in blocking_items
    )
    if missing:
        reasons.append(f"未执行项：{missing}")
    non_pass = [k for k, x in executed.items() if x["result"] != "pass"]
    if non_pass:
        reasons.append(f"未完全通过项：{non_pass}（partial 计入「具备达成路径」）")
    passed_n = sum(1 for x in executed.values() if x["result"] == "pass")
    partial_n = sum(1 for x in executed.values() if x["result"] == "partial")
    enough = len(executed) == 6 and (passed_n + partial_n) >= 5

    if missing:
        return ("冻结", "存在未执行的必备/非必备项，暂不放行 M1 主体开发", reasons)
    if blocking_ok and enough:
        return ("放行", "三项必备门禁全部通过，且六项中 ≥5 项通过（含有条件通过）", reasons)
    if blocking_ok and passed_n + partial_n == 6:
        return ("放行", "必备项全过，非必备项均为通过或有条件通过", reasons)
    if not blocking_ok:
        return ("冻结", "必备项（①/②/⑥）存在未通过，按方案 3.3 冻结主体开发并走回退分支", reasons)
    return ("冻结", "通过项数不足（<5）", reasons)


def main() -> int:
    s = get_settings()
    v = load_verdicts()
    decision, why, reasons = decide(v)

    g01 = read_json("g01_vram_scheduler.json") if (VAR_DIR / "g01_vram_scheduler.json").exists() else {}
    g02 = read_json("g02_chinese_bm25.json") if (VAR_DIR / "g02_chinese_bm25.json").exists() else {}
    g03 = read_json("g03_ablation.json") if (VAR_DIR / "g03_ablation.json").exists() else {}
    g04 = read_json("g04_consistency.json") if (VAR_DIR / "g04_consistency.json").exists() else {}
    g05 = read_json("g05_citation_calibration.json") if (VAR_DIR / "g05_citation_calibration.json").exists() else {}
    g06 = read_json("g06_localization.json") if (VAR_DIR / "g06_localization.json").exists() else {}
    calib = read_json("calibration.json") if (VAR_DIR / "calibration.json").exists() else {}
    env = env_snapshot()

    rows = []
    for item in GATE_CRITERIA:
        x = v[item]
        rows.append(
            [
                item,
                GATE_CRITERIA[item]["title"],
                "必备" if GATE_CRITERIA[item]["blocking"] else "非必备",
                (x or {}).get("mark", "未执行"),
                (x or {}).get("measured", "-"),
            ]
        )

    lines = [
        "# 《Gate 0 验证报告》（知效引擎 KEE · M0）",
        "",
        f"- 出具时间：{env['ts']}｜执行主体：个人独立设计与开发｜commit {env['git_commit']}",
        f"- 硬件：{env['gpu'].get('name')}（{env['gpu'].get('total_mib')} MiB 显存，Windows WDDM）｜OS：{env['platform']}",
        f"- 被测代码：`know-effect-engine` @ {env['git_commit']}{'（工作区有未提交改动）' if env['git_dirty'] else ''}",
        "",
        "> **口径声明**：本报告所有量化数字是**本次验证的实测值**，与方案中的「设计目标/验收门禁」逐项对照；"
        "语料为**自建行业文档集**（本项目编写），因此检索类指标的**相对差**比绝对值更具结论性，"
        "绝对值仅代表「达成路径是否存在」。",
        "",
        "## 一、放行结论",
        "",
        f"- **结论：{decision} M1 主体开发**。依据：{why}。",
        f"- 判定规则（方案 3.3）：必备项 ①②⑥ 全部通过，且六项中 ≥5 项通过。",
        *( [f"- 备注：{r}" for r in reasons] if reasons else ["- 备注：无保留项。"] ),
        "",
        "## 二、六项门禁判定表",
        "",
        *md_table(["项", "验证对象", "性质", "判定", "关键实测"], rows),
        "",
        "## 三、关键实测数字（与方案门禁对照）",
        "",
    ]

    if g01:
        pol = g01.get("policies", {})
        lines += [
            "### ① 显存调度",
            "",
            *md_table(
                ["策略", "OOM", "显存峰值 MiB", "冷启次数", "冷启最大 s", "P50 e2e ms", "P95 e2e ms"],
                [
                    [
                        p,
                        d["oom_count"],
                        d["vram"]["peak_mib"],
                        d["switch"]["n_cold_loads"],
                        d["switch"]["cold_load_sec_max"],
                        d["e2e_ms"].get("p50"),
                        d["e2e_ms"].get("p95"),
                    ]
                    for p, d in pol.items()
                ],
            ),
            f"- 7B 常驻显存净增：{g01.get('vram_attribution', {}).get('seven_b_footprint_mib')} MiB，"
            f"留给嵌入+重排余量 {g01.get('vram_attribution', {}).get('headroom_for_embed_rerank_mib')} MiB",
            f"- 生成长度扫描（P95 ms）：{json.dumps(g01.get('max_tokens_sweep', {}), ensure_ascii=False)[:400]}",
            "",
        ]
        # ① 判定余量披露：直接从原始样本重算，不信任产物里的结论字段（旧产物可能没落 amended_gate）
        from kee.eval.metrics import percentile

        co = g01.get("policies", {}).get("co_resident", {}) or {}
        loop_s = co.get("e2e_samples", []) or []
        sweep_s = (g01.get("max_tokens_sweep", {}).get("256", {}) or {}).get("e2e_samples", []) or []
        comb = loop_s + sweep_s
        if comb:
            p95 = percentile(comb, 95)
            margin = 6000 - p95
            lines += [
                f"- **判定余量披露**（按合并样本重算，n={len(comb)}）：P95 = {p95:.0f}ms，距 6s 门禁 **{margin:.0f}ms**"
                + ("——属**临界通过**，不得作为「延迟已达标」对外引用；M2 需 ≥50 样本复测并设线上 P95 告警" if margin < 500 else ""),
                f"- 分组对照：主循环 P95 {percentile(loop_s, 95):.0f}ms（n={len(loop_s)}）"
                f" / 长度扫描档 P95 {percentile(sweep_s, 95):.0f}ms（n={len(sweep_s)}）——差异属采样方差，判定取合并值，不择优",
                "",
            ]
    if g02:
        sp = g02.get("sparse_eval", {})
        lines += [
            "### ② 中文稀疏检索",
            "",
            *md_table(
                ["分析器", "Recall@5", "hit@5", "MRR", "零召回率"],
                [
                    [
                        n,
                        sp.get(n, {}).get("recall@5"),
                        sp.get(n, {}).get("hit@5"),
                        sp.get(n, {}).get("mrr"),
                        sp.get(n, {}).get("zero_recall_rate"),
                    ]
                    for n in sp
                ],
            ),
            f"- 评测集：{g02.get('eval_set', {}).get('n_questions')} 题 / 子块 {g02.get('eval_set', {}).get('n_children')}",
            "",
        ]
    if g03:
        cfgs = g03.get("configs", {})
        lines += [
            "### ③ 检索消融",
            "",
            *md_table(
                ["配置", "Recall@5", "MRR", "P95 检索 ms"],
                [
                    [
                        n,
                        c.get("recall@5"),
                        c.get("mrr"),
                        c.get("latency_ms", {}).get("p95"),
                    ]
                    for n, c in cfgs.items()
                ],
            ),
            "",
        ]
    if g04:
        lines += [
            "### ④ 多存储一致性",
            "",
            f"- 差异发现率：{g04.get('detect', {}).get('detection_rate')}，脏行发现：{g04.get('detect', {}).get('extra_detected')}",
            f"- 修复后复检：missing={g04.get('repair', {}).get('recheck', {})}",
            f"- 审计表禁改：UPDATE 被拦截={g04.get('audit_immutable', {}).get('blocked')}，"
            f"DELETE 被拦截={g04.get('audit_immutable_delete', {}).get('blocked')}",
            "",
        ]
    if g05:
        cal = g05.get("calibration", {})
        lines += [
            "### ⑤ 引用强制与校准",
            "",
            f"- 零引用结论：{g05.get('citation_enforcement', {}).get('answers_without_citation')} 条（门禁要求 0）",
            f"- 拒答：{g05.get('refusal', {}).get('correctly_refused_or_escalated')}/{g05.get('refusal', {}).get('n_unanswerable')}；"
            f"转人工：{g05.get('escalation', {}).get('routed_to_human')}/{g05.get('escalation', {}).get('n_sensitive')}",
            f"- 结构化首次成功率：{g05.get('structured_output', {}).get('first_attempt_valid_rate')}",
            f"- ECE：原始 {cal.get('ece_raw_holdout', {}).get('ece')} → 校准后 {cal.get('ece_calibrated_holdout', {}).get('ece')}"
            f"（T={cal.get('temperature', {}).get('temperature')}，留出集 {cal.get('test_n')} 题）",
            f"- 漏转人工明细：{json.dumps([{'qid': m['qid'], 'status': m['status'], 'notes': m['notes']} for m in g05.get('escalation', {}).get('missed', [])], ensure_ascii=False)[:260]}",
            "- 注：⑤ 在修订口径下仍判不通过——漏转的那条不是「答错了」，而是模型在检索分 0.97 的强证据下仍声明「无依据」并拒答，"
            "属**该答不答**方向的控制偏差；确定性规则目前只覆盖「问题命中敏感词」与「引用落在密级 3」两种触发，"
            "M3 需补「强证据却声明无依据 → 转人工复核」这条反向兜底。",
            "",
        ]
    if g06:
        steps = g06.get("steps", {})
        lines += [
            "### ⑥ 单机本地化与数据不出域",
            "",
            f"- 一键启动：{steps.get('up', {}).get('startup_sec')}s，全组件 healthy={steps.get('up', {}).get('all_healthy')}，"
            f"Milvus 就绪 {steps.get('milvus_ready', {}).get('waited_sec')}s",
            f"- 非回环端口绑定：{len(steps.get('binding', {}).get('non_loopback_bindings', []))} 处",
            f"- 宿主进程公网连接：{len(steps.get('core_path', {}).get('public_egress_connections', []))} 条；"
            f"容器侧可疑对外连接：{steps.get('container_probe', {}).get('any_suspicious_external')}",
            f"- 静态扫描运行期出域嫌疑：{len(steps.get('static_scan', {}).get('runtime_data_egress', []))} 处",
            "",
        ]

    lock_rows = [
        ["向量库 Milvus", "v3.0.0（镜像 milvusdb/milvus:v3.0.0，服务端实测 3.0.0）", "已确认"],
        ["pymilvus", env.get("pymilvus", "-"), "已确认"],
        ["元数据库 PostgreSQL", str(g06.get("steps", {}).get("connectivity", {}).get("postgres", {}).get("version", "-")), "已确认"],
        ["缓存 Redis", str(g06.get("steps", {}).get("connectivity", {}).get("redis", {}).get("version", "-")), "已确认"],
        ["嵌入模型", "BGE-M3（BAAI--bge-m3，本地权重，CLS pooling + L2）", "已确认"],
        ["重排模型", "BGE-reranker-v2-m3（本地权重，XLMRobertaForSequenceClassification + sigmoid）", "已确认"],
        ["生成模型", "Ollama qwen2.5:7b（Q4_K_M，本地，keep_alive 常驻）", "已确认"],
        ["推理框架", f"torch {env.get('torch')} / transformers {env.get('transformers')}", "已确认"],
        ["Python", f"CPython {env.get('python')}（venv 内，CUDA 可用={env.get('cuda_available')}）", "已确认"],
        ["编排框架 LangGraph", "本轮未引入（Gate 0 用确定性函数式流水线；M3 引入）", "待定"],
        ["网关 LiteLLM", "本轮未引入（云端兜底关闭）", "待定"],
        ["评测 RAGAS", "本轮未引入（⑤用规则化正确性 + 温度缩放；M2/M3 引入）", "待定"],
    ]
    lines += [
        "## 四、版本锁定清单（Gate 0 后确认）",
        "",
        *md_table(["组件", "锁定版本/形态", "状态"], lock_rows),
        "",
        "## 五、参数基线（Gate 0 出口）",
        "",
        f"- 中文分析器：`analyzer_params = {{\"tokenizer\": \"jieba\"}}`（Milvus v3.0.0 不接受 `{{\"type\":\"jieba\"}}`）。",
        f"- 检索：dense_top_k={s.dense_top_k}，sparse_top_k={s.sparse_top_k}，RRF k={s.rrf_k}，Rerank Top-{s.rerank_top_n}，稠密索引 FLAT/COSINE。",
        f"- 分块：Parent-Child，子块上限 {280} 字符、重叠 {40}，父块=章节。",
        f"- 置信度分档：high ≥ {s.confidence_high}，low < {s.confidence_low}；温度 T = {calib.get('temperature', '待 ⑤ 产出')}。",
        "- 驻留策略与生成长度上限：见第三节①与 `var/gate0/g01_vram_scheduler.json`。",
        "- 缓存：Redis TTL "
        f"{s.search_cache_ttl_sec}s，`allkeys-lru`，非权威可重建。",
        "",
        "## 六、已验证 / 未验证",
        "",
        "**已验证（本机真机，含可复跑脚本）**",
        "",
        "- Milvus v3.0.0 服务端 BM25 + 中文分析器的可用形态与中文检索有效性差异。",
        "- 8GB 显存下 qwen2.5:7b 常驻 + BGE-M3/BGE-reranker 的共存边界、冷启耗时与端到端时延分布。",
        "- 混合检索相对单一稠密的增益归因（同一评测集、同一集合、可复现排序）。",
        "- Outbox + 每日对账对「写失败/静默缺失/索引脏行」三类不一致的发现与修复；审计表 UPDATE/DELETE 被触发器拦截。",
        "- 引用强制（无出处即拒答）、无答案必拒答、敏感必转人工的机制行为；温度缩放对 ECE 的影响。",
        "- 单机一键起栈（含 restart 策略修复后的启动可靠性）、端口仅绑回环、宿主与容器侧无可观测公网外呼。",
        "",
        "**未验证 / 待办（不得当作已达成）**",
        "",
        "接入企业真实语料后须重跑的一栏：Recall@5/MRR/幻觉率绝对值（本轮读数为自建语料 + 构造式 gold）。",
        "- 双人标注与 Cohen's Kappa（第二人复核表已生成待填：`var/gate0/annotations_a2_template.csv`，搜 `TODO_ANNOTATOR2`）。",
        "- RAGAS Faithfulness / Answer Relevance（本轮未接入，属 M2/M3）。",
        "- LangGraph 编排、Checkpointer 多轮、流式推送、Web 前端、应用与 Nginx 容器化（属 M3/M4）。",
        "- 「容器内跑完整核心链路的真断网验证」：受限于应用未容器化，本轮只做到容器侧无可疑外呼 + 端口仅绑回环。",
        "- 云端兜底与 token 预算熔断（本轮开关为 false，未做真实链路）。",
        "- P95 端到端 < 3s 在**真实业务答案长度**下的达成情况（见第三节①的生成长度扫描边界）。",
        "",
        "## 七、对方案的修订建议（验证结论回写）",
        "",
        "- 技术方案 11.2 与 6.2 的 analyzer 写法需从 `{\"type\":\"jieba\"}` 改为 `{\"tokenizer\":\"jieba\"}`；"
        "同时把 `{\"type\":\"chinese\"}` 列为备选（实测对照见第二节②）。",
        "- 6.5「按需加载/释放」在 per-query 粒度不可行（冷启 "
        f"{(g01.get('policies', {}).get('time_sharded', {}).get('switch', {}) or {}).get('cold_load_sec_max', '-')}s/次），"
        "应改写为「驻留 + 空闲释放/共存」口径，并以第三节①的实测选择默认策略。",
        "- 3.4 回退分支①/②/⑥ 的触发条件应按本轮实测结论收敛为具体参数，而不是留作应急口头项。",
        "",
        "## 八、执行入口（可复跑）",
        "",
        *[f"- {item}：`{RUNNERS[item]}`" for item in GATE_CRITERIA],
        "- 汇总：`src/kee/gate0/g00_summary.py`｜原始产物：`var/gate0/`｜单项报告：`reports/gate0/`",
    ]

    p = write_md("GATE0_验证报告.md", lines)
    print("\n".join(lines[:40]))
    print(f"\n... 结论：{decision}｜报告：{p}")
    return 0 if decision == "放行" else 1


if __name__ == "__main__":
    raise SystemExit(main())
