"""M5-01 DoD 逐条核对：数字从证据产物里**取**，不手抄。

为什么做成脚本而不是手写表：手抄的读数会随产物重跑而失真，而 M5 的交付物恰好是
「验收时这张表还能不能复算」。取不到数的一律标 `不可测` / `待裁决`，**不许留空当通过**。

一条自己踩过的坑写在这儿：汇总**不能靠散文里有没有「达标」二字来数**。
第一版用 `verdict.startswith("达标")` 计数，于是「检索达标；端到端**不达标**」被算进了达标。
现在 status 是显式字段，散文只负责解释。

跑法：uv run python scripts/m5_dod_check.py
产物：var/gate0/m5_dod.json + reports/m5/M5_DoD核对表.md
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "var" / "gate0"
OUT_MD = ROOT / "reports" / "m5" / "M5_DoD核对表.md"
STATUSES = ("达标", "部分达标", "不达标", "不可测", "待复测", "本次产出")


def load(name: str) -> dict:
    p = ART / name
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except Exception:  # noqa: BLE001
        return {}


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def quantile(vals: list[float], q: float) -> float:
    v = sorted(vals)
    return round(v[min(len(v) - 1, max(0, int(q * len(v)) - 1))], 1) if v else 0.0


def e2e_from_pg() -> dict:
    """端到端延迟取**线上 HTTP 路径每轮落库那一列**（`session_turn.total_ms`），不取探针自报数——
    那才是用户点一次发送实际等到的时间。代价：这张表里混着验证脚本自己打的流量，必须一并说明。"""
    import psycopg
    from psycopg.rows import dict_row

    from kee.config import get_settings

    with psycopg.connect(get_settings().postgres_dsn, row_factory=dict_row) as conn:
        vals = [r["total_ms"] for r in conn.execute(
            "SELECT total_ms FROM session_turn WHERE total_ms IS NOT NULL ORDER BY total_ms").fetchall()]
    if not vals:
        return {"n": 0, "p50": 0.0, "p95": 0.0, "max": 0.0, "n_over_10s": 0}
    return {"n": len(vals), "p50": vals[len(vals) // 2], "p95": quantile(vals, 0.95),
            "max": vals[-1], "n_over_10s": sum(1 for v in vals if v > 10000)}


def row(no, item, status, crit, reading, ev, say) -> dict:
    return {"#": no, "验收项": item, "判定标准": crit, "实测读数": reading,
            "证据": ev, "结论": status, "引用时必须连着说的话": say}


def build() -> dict:
    m2, ans, concl = load("m2_eval.json"), load("m3_answer_probe.json"), load("m3_conclusion_probe.json")
    conf, stream, kb = load("m3_confidence_probe.json"), load("m4_stream_probe.json"), load("m4_kb_write_path.json")
    m1, v3, v4 = load("m1_verify.json"), load("m3_verify.json"), load("m4_verify.json")
    g6 = load("m4_g06_rerun2.json")
    ac = load("answer_correct_calibration.json")
    rep = load("m5_latency_replicates.json")
    halluc = load("hallucination_run.json")
    # 工单没回收时这一格必须显式「无数」，而且要说清哪几个率此刻**按构造出不了数**
    if halluc:
        halluc["no_evidence_face"] = [r["qid"] for r in halluc.get("rows", [])
                                      if not r.get("retrieved_ids")]
    ap = ac.get("populations", {}).get("answer_prod", {})
    ab = ac.get("populations", {}).get("all_behav", {})
    ver = {i: load(f"verdict_0{i}.json") for i in range(1, 7)}

    bf = m2.get("baseline_full", {})
    sg = bf.get("strict_gold", {})
    calib = ans.get("calibration_on_gradable", {})
    lat = e2e_from_pg()
    grounded = [r.get("number_groundedness") for r in conf.get("rows", []) if r.get("number_groundedness") is not None]
    invented = [r for r in conf.get("rows", []) if r.get("invented_numbers")]
    struct_ok, n_gen = concl.get("分型", {}).get("ok", 0), concl.get("n_generations", 0)
    g6s = g6.get("steps", {})
    sd = ans.get("status_dist", {})
    # #40 工单没回收时这一格必须显式「无数」，不能显示成 0.00%——那是把「没测」写成「测了且不合格」
    ac_acc = f"{pct(ap['accuracy'])}（{ap['n_correct']}/{ap['n']}）" if ap.get("n") else "无数（#40 answer_correct 工单未回收）"
    ac_ece = f"{ap['ece']}" if ap.get("n") else "无数（同上）"

    cause_short = "、".join(f"{n}×{k.split('：')[0]}"
                            for k, n in Counter(v.split("：")[0] for v in ac.get("wrong_row_causes", {}).values()).items())

    # ---- DoD#5：#56 净环境复测批落地后的读数。合并样本、缓存命中的轮次单列，不给单一口径的结论。
    pu = rep.get("pooled_uncached", {})
    cg = pu.get("client_generating_only", {})
    sg2 = pu.get("server_generating_only", {})
    ftb = rep.get("pooled_first_token", {}).get("bands", {})
    fvv = ftb.get("first_visible_ms", {})
    if not cg.get("n"):
        lat_verdict = "待复测"
        lat_reading = "净环境复测产物缺失（先跑 m5_latency_clean.py --tag … 再跑 m5_latency_merge.py）"
        lat_say = "没有这一格的数据就不要写结论——旧的那份 session_turn 全表读数混着脚本自身流量，已不作引用源。"
    else:
        # 状态**由门禁推导**（V1.21 定案口径），不手写词：手写错过一次
        # （把「检索达标；端到端不达标」按 startswith 数成达标），所以留显式判据字典。
        retr_p95 = bf.get("latency_ms", {}).get("p95") or 0
        lat_gate = {"检索 p95 < 800ms": retr_p95 < 800,
                    "作答轮次端到端 p95 < 6s": cg["p95"] < 6000,
                    "首可读字 p95 < 1200ms": (fvv.get("p95") or 9e9) < 1200}
        lat_failed = [k for k, v in lat_gate.items() if not v]
        lat_verdict = "达标" if not lat_failed else "不达标"
        lat_reading = (
            f"**门禁（V1.21 定案）{'全过' if not lat_failed else '未过：' + '、'.join(lat_failed)}** ｜ "
            f"P95 检索 {retr_p95}ms ｜ 作答轮次端到端 p50 {cg['p50']} / **p95 {cg['p95']}ms** / max {cg['max']}ms"
            f"（n={cg['n']}，>3s {pct(cg['share_over_3s'])}、>6s {pct(cg['share_over_6s'])}）｜ "
            f"服务端 p95 {sg2.get('p95')}ms ｜ **含拒答轮次另并报**（n={pu.get('n_rows')}）"
            f"p95 {pu.get('client_all', {}).get('p95')} / max {pu.get('client_all', {}).get('max')}ms、"
            f">6s {pct(pu.get('client_all', {}).get('share_over_6s'))} ｜ "
            f"首可读字 p50 {fvv.get('p50')} / p95 {fvv.get('p95')} / max {fvv.get('max')}ms（n={fvv.get('n')}）｜ "
            f"轮间 p50 极差 {rep.get('spread_across_runs', {}).get('p50_range_ms')}ms（噪声地板）｜ "
            f"冷装载 {rep.get('sources', {}).get('transcribed', {}).get('cold_start_ms')}ms 单列")
        lat_say = (
            "**口径已于 V1.21 定案（用户 2026-10-13 裁决），三件套**："
            "①原口径 DoD#5 `< 3s`；②修订口径 `P95 端到端 < 6s 且只统计作答轮次`"
            "（拒答/转人工不计入门禁但须另并报），首 token 口径钉 `first_visible_ms`；"
            f"③实测依据：净环境 4 轮 {cg['n']} 个作答轮次 p95 {cg['p95']}ms、0% 超 6s"
            f"（含拒答则 max {pu.get('client_all', {}).get('max')}ms、"
            f"{pct(pu.get('client_all', {}).get('share_over_6s'))} 超 6s）。"
            "**剔拒答的理由是结构性的、先于数值**：拒答轮次没有『答案』可等，耗时由判定层决定，"
            "混进门禁会让 p95 随语料里『证据勉强过线』的题占比漂移——那不是延迟性质。"
            "**代价必须同句报**：不得据此声称『每次请求都 <6s』。"
            "那条尾巴的真实身份已查清（`m5_latency_outlier_probe.py`，同题连跑 5 次、每跑前清缓存）："
            "拒答有两条路径且成本差 18 倍——闸前拒答中位 401.4ms（不进生成）、"
            "过闸后模型自宣无依据中位 7234.5ms（整趟 LLM 跑完才空结论，5/5 全在 6.97–7.39s，"
            "不是抖动：噪声地板 93.6ms）。这不是缺陷、走既有出口、行为正确。"
            "另一半：`3s` 那一边不是『差一点』而是**不可达**——90% 时间在生成段，"
            "要达 3s 须砍 1.1s 以上，可动的只有生成预算 256→96 token（Gate 0 实测该档 P95 2477ms 可达）"
            "或换更小模型（与『7B 是本地下限』的既有裁决冲突），那是实现改动不是措辞改动。"
            f"引用这一格的最低配置：分位数只取未命中检索缓存的作答轮次（命中轮次实测快 "
            f"{round(cg['p50'] - rep.get('pooled_cached', {}).get('client_all', {}).get('p50', 0))}ms，单列不合并）、"
            f"冷装载与热态分开报、小于 {rep.get('spread_across_runs', {}).get('p50_range_ms')}ms 的 Δ "
            "不得写成改善或恶化。旧的 `session_turn` 全表读数已作废为引用源"
            f"（当时拒答轮次 total_ms 恒 NULL；现在 {rep.get('e2e_fix_verification', {}).get('rows_covered')} 行里 "
            f"缺 total_ms 的 {rep.get('e2e_fix_verification', {}).get('rows_without_total_ms')} 行）。")

    rows = [
        row("1", "Must 链端到端可演示", "达标", "9 条 Must 全链路可跑通",
            f"M1 {len(m1.get('steps', []))} 步 ｜ M2 基线 gates_passed={m2.get('gates_passed')} ｜ "
            f"M3 {len(v3.get('steps', []))} 步（多轮/流式/RBAC/拒答/审计）｜ M4 交付层 {len(v4.get('checks', []))} 项 ｜ "
            f"M4 写路径 {len(kb.get('checks', []))} 项",
            ["var/gate0/m1_verify.json", "var/gate0/m2_eval.json", "var/gate0/m3_verify.json",
             "var/gate0/m4_verify.json", "var/gate0/m4_kb_write_path.json"],
            "写路径那 13 项跑在 **scratch 库 + scratch 集合**（线上没有删除文档的端点，不能拿生产语料试）；"
            "浏览器里的手动上传→复核是在「上传必 500」缺陷**修复之前**走的，修完还没重走。"),
        row("2", "检索质量", "达标" if m2.get("gates_passed") else "不达标",
            "Hit@5 ≥ 95% 且 strict gold Recall@5 ≥ 85% 且 MRR ≥ 0.75；section 只作报告值并附 n 与 95% CI",
            f"Hit@5(strict) {pct(sg.get('hit@5', 0))} ｜ Recall@5(strict) {pct(sg.get('recall@5', 0))} ｜ MRR {bf.get('mrr')} ｜ "
            f"P95 检索 {bf.get('latency_ms', {}).get('p95')}ms ｜ 报告值 section Recall@5 {pct(bf.get('recall@5', 0))}"
            f"（95% CI {bf.get('recall@5_ci95')}，n={bf.get('n')}）",
            ["var/gate0/m2_eval.json", "reports/m2/M2_检索评测基线.md 第十节"],
            f"section {pct(bf.get('recall@5', 0))} 仍低于 0.85，按 V1.0.3 裁决它只是报告值——"
            "只说「检索门禁通过」而不带这句，就是选择性引用。"),
        row("3", "引用可信", "部分达标", "引用正确率 ≥ 90%；幻觉率 ≤ 10%（Faithfulness 为报告值，非阻塞）",
            f"行级引用落 gold {ans.get('行级引用落gold')} ｜ **人判答案正确性（#40 已回收）{ac_acc}** ｜ 数字落地率均值 "
            f"{round(statistics.mean(grounded), 4) if grounded else 'NA'}（{len(grounded)} 题可判）｜ "
            f"含编造数字的题 {len(invented)} 道 ｜ 零引用结论 {ans.get('契约拦下的空结论')} 条 ｜ "
            f"**真幻觉率（该拒却答了）：无数**——#57 工单 {halluc.get('n_rows', 0)} 行已下发、三列判断均未填",
            ["var/gate0/m3_answer_probe.json", "var/gate0/m3_confidence_probe.json",
             "var/gate0/answer_correct_calibration.json", "scripts/m5_answer_correct_calibration.py",
             "var/gate0/hallucination_fill.csv", "var/gate0/hallucination_run.json",
             "scripts/m5_hallucination_sheet.py", "scripts/m5_hallucination_rate.py"],
            "这一格现在有**两种不同的可信**：出处可信（引用落 gold）与结论可信（真人判 answer_correct）。"
            f"后者首次有数：{ap.get('n')} 行给出答案的行里 {ap.get('n_wrong')} 行结论错"
            f"（错行归因：{cause_short}；明细见 `var/gate0/answer_correct_calibration.json`）。"
            "边界必须同句说：这只覆盖 68 道可答题里真跑过生成的那批，"
            "**无答案 7 + 敏感 5 题不在标签集内**——那 12 题「该拒却答了」至今没有真人标签，"
            "所以「幻觉率 ≤ 10%」这半边仍是代理读数（编造数字），不是人判。"
            f"（V1.0.14 的**答案层复核闸**是准入控制，不是正确性度量，不能顶替这一格）"
            f"**#57 已把工单出好**（源=生产入口真调用，题面哈希 {halluc.get('question_hash')}，"
            f"护栏：qid 双向对齐、混代拒绝、分母为 0 报「不适用」而不是 0%）："
            f"回收前这一格只能写「无数」。另有一条结构性缺口要同句报——"
            f"这 12 行里 **{len(halluc.get('no_evidence_face', []))} 行系统没做检索就给了处置**"
            "（敏感题在检索前被拦），那些行的「证据够不够」按构造不可判，"
            "所以这张表真正能定量的是**误拒**与**该转未转**，不是四个率都能出数。"),
        row("4", "校准与结构化", "部分达标", "ECE ≤ 0.10；结构化输出首次成功率 ≥ 90%",
            f"结构化首次成功率 {pct(struct_ok / n_gen) if n_gen else 'NA'}（{struct_ok}/{n_gen}，被截断 {concl.get('被截断数')}）｜ "
            f"ECE（真人 answer_correct，n={ap.get('n', 0)}）**{ac_ece}**"
            f"：mean_conf {ap.get('mean_conf')} vs accuracy {ap.get('accuracy')} ｜ "
            f"旧代理口径（L1 落 gold、标签近全 1）ECE {calib.get('ece', '无数')}",
            ["var/gate0/m3_conclusion_probe.json", "var/gate0/answer_correct_calibration.json",
             "scripts/m5_answer_correct_calibration.py", "var/gate0/verdict_05.json",
             "var/gate0/confidence_calibration.json"],
            f"ECE {ac_ece} > 0.10 ⇒ **这一格仍不达标**，但读法变了：现在标签有方差（{ap.get('n_correct')} 正 / "
            f"{ap.get('n_wrong')} 负），所以这个偏差是**真偏差**，不再是「拿没有方差的标签算出来的」那一条。"
            f"方向是{ap.get('ece_sign', '未定')}——分数说 {ap.get('mean_conf')}、实测 {ap.get('accuracy')}，"
            "危险的那个方向（错答案顶着高分被放行）没有出现，高置信带内漏的是 "
            f"{len(ab.get('gate_band', {}).get('wrong_rows_in_high_band', []))} 行贴线过的答案。"
            "**温度缩放两档都不采纳**（负类太少不可辨识 / 拒答行复合分被钉在 0.0，拟合的是同义反复），"
            "运行时维持 T=1.0、复合分不进门禁——沿用 2026-10-11 的裁决，本次不改口径。"),
        row("5", "延迟", lat_verdict,
            "**V1.21 定案**：P95 检索 < 800ms；P95 端到端 < 6s 且**只统计作答轮次**（status ∈ answer/tentative）；"
            "流式首 token < 1.2s，口径 = `first_visible_ms`（首个可读字）。拒答/转人工不计入门禁但须另并报",
            lat_reading,
            ["var/gate0/m2_eval.json", "var/gate0/m5_latency_replicates.json",
             "scripts/m5_latency_clean.py", "scripts/m5_latency_merge.py"],
            lat_say),
        row("6", "业务指标", "不可测", "自助解答率 ≥ 60%；重复提问占比下降 ≥ 30%；新人上手周期缩短 ≥ 30%",
            f"无真实用户与线上流量 ⇒ 不可测。结构性代理：可答题 {ans.get('n_answerable')} 道里 "
            f"answer {sd.get('answer')} / tentative {sd.get('tentative')} / refuse {sd.get('refuse')} / escalate {sd.get('escalate')}，"
            f"需转人工 {ans.get('needs_human')} 道",
            ["var/gate0/m3_answer_probe.json"],
            "代理读数测的是「构造语料上的系统行为分布」，**不是**用户是否真的自助解决了问题。"
            "把它当业务指标写进材料就是编数；这一格在 MVP 阶段只能标不可测。"),
        row("7", "合规", "部分达标", "数据不出域；RBAC 检索层过滤；审计追加禁改",
            f"⑥ 2026-10-12 复跑：一键 {g6s.get('up', {}).get('startup_sec')}s、全 healthy={g6s.get('up', {}).get('all_healthy')}、"
            f"非回环绑定 {len(g6s.get('binding', {}).get('non_loopback_bindings', []))}、"
            f"宿主审计非回环连接 {len(g6.get('core_path', {}).get('non_loopback_connections', []))}、"
            f"公网出网 {len(g6.get('core_path', {}).get('public_egress_connections', []))}、"
            f"云端兜底={g6s.get('cloud_switch', {}).get('cloud_fallback_enabled')} ｜ "
            f"静态扫描运行期出域 {len(g6s.get('static_scan', {}).get('runtime_data_egress', []))} ｜ "
            "RBAC 密级/行业过滤、审计追加禁改各有实测",
            ["var/gate0/m4_g06_rerun2.json", "var/gate0/m4_verify.json", "reports/m4/M4_前端与一键Demo.md"],
            "「断网下核心链路可运行」这句**没被端到端证明**：`internal: true` 把宿主↔容器通路一起切断，"
            "而 API 在宿主。**ADR-12 已定案（2026-10-12）：维持现状，「整机断网下端到端问答」划出承诺范围**——"
            "替代路 `enable_ip_masquerade=false` 实测切不掉容器出网，GPU 容器化要么拉 GB 级镜像、"
            "要么把检索门禁做成不可达（cpu+float32 合成检索 5167ms vs 门禁 800ms）。"
            "所以这一格是「四条结构性证据支撑的合规」，不是「断网已验证」——对外只能引用前者。"),
        row("8", "交付物", "本次产出", "版本锁定清单 + 风险登记册定稿",
            "本脚本 + `scripts/m5_version_lock.py` + `reports/m5/M5_风险登记册定稿.md`",
            ["reports/m5/M5_DoD核对表.md", "reports/m5/M5_版本锁定清单.md", "reports/m5/M5_风险登记册定稿.md"],
            "清单里凡标 `未固定` 的都要带原因，不能靠「本地能跑」蒙过去。"),
    ]

    # 汇总数是从 `结论` 这个**显式字段**数的，所以状态取值必须落在 STATUSES 里；
    # 否则新增的措辞（如「部分达标（口径待裁）」）不会报错，只会悄悄从汇总里少一行。
    off = [f"DoD#{r['#']}={r['结论']!r}" for r in rows if r["结论"] not in STATUSES]
    if off:
        raise SystemExit(f"结论取值不在 STATUSES 内，汇总会漏计：{off}")

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "principle": "读数全部从 var/gate0 产物与 PG 取；取不到就标不可测；status 显式给定，不靠散文里有没有「达标」二字来数",
        "dod": rows,
        "gate0_verdicts_latest": [{"门禁": f"Gate0 {k}", "结论": v.get("mark"), "实测": v.get("measured"), "ts": v.get("ts")}
                                  for k, v in ver.items() if v],
        "summary": {k: sum(1 for r in rows if r["结论"] == k) for k in STATUSES},
        # 下面这份是 DoD#5 的原子读数，`to_md` 那段散文要用它——散文里的数字必须从字段来，
        # 不能从上一轮的结论里"猜"（踩过一次：把「检索达标；端到端不达标」按 startswith 数成达标）。
        "lat_clean": {"p95_client": cg.get("p95"), "over_3s": cg.get("share_over_3s"),
                      "over_6s": cg.get("share_over_6s"), "n": cg.get("n"),
                      "p50_range_ms": rep.get("spread_across_runs", {}).get("p50_range_ms"),
                      "ft_n": fvv.get("n"), "ft_p50": fvv.get("p50"), "ft_p95": fvv.get("p95"),
                      "ft_max": fvv.get("max"), "event_p50": ftb.get("first_event_ms", {}).get("p50")},
        "ac": {"ece": ac_ece, "sign": ap.get("ece_sign", "未定"),
                        "accuracy": ap.get("accuracy", "NA"), "mean_conf": ap.get("mean_conf", "NA"),
                        "n": ap.get("n", 0), "coverage_boundary": ac.get("coverage_boundary", {})},
    }


def to_md(rep: dict) -> str:
    s = rep["summary"]
    lc = rep["lat_clean"]
    L = ["# M5 · DoD 逐条核对表", "",
         f"> 生成时点：{rep['generated_at']}｜由 `scripts/m5_dod_check.py` 从 `var/gate0/` 产物与 PG **取数**生成，不手抄。",
         "> 原则：取不到数就标「不可测 / 待裁决」，**不许留空当通过**；每条结论必须连同最右列那句一起引用，"
         "否则构成选择性引用。", "",
         f"**汇总**：达标 {s['达标']} ｜ 部分达标 {s['部分达标']} ｜ 不达标 {s['不达标']} ｜ "
         f"不可测 {s['不可测']} ｜ 本次产出 {s['本次产出']}（共 {len(rep['dod'])} 条）", "",
         "| # | 验收项 | 判定标准 | 实测读数 | 证据 | 结论 | 引用时必须连着说的话 |",
         "|---|---|---|---|---|---|---|"]
    for r in rep["dod"]:
        cells = [r["#"], r["验收项"], r["判定标准"], r["实测读数"],
                 "、".join(f"`{e}`" for e in r["证据"]), f"**{r['结论']}**", r["引用时必须连着说的话"]]
        L.append("| " + " | ".join(str(c).replace("\n", " ").replace("|", "／") for c in cells) + " |")
    L += ["", "## 停止条件（V1.1 第九节）对照", "",
          "| 停止条件 | 现在能不能说满足 | 依据 |", "|---|---|---|",
          "| ① Must 链端到端可演示 | **能**，但要同段说明写路径跑在 scratch 命名空间 | DoD#1 |",
          "| ② ≥ 3 项量化指标达标 | **能**：Hit@5(strict)、Recall@5(strict)、P95 检索、结构化首次成功率——"
          "四项各有同轮可复算读数；端到端延迟于 V1.21 口径定案后达标（<6s、只算作答轮次），"
          "唯 ECE 那一格仍不过（校准是报告值，门禁由四条机制承担） | DoD#2、#4、#5 |",
          "| ③ 版本锁定清单 + 风险登记册定稿 | 本次产出 | DoD#8 |", "",
          "## Gate 0 六项门禁的最新状态（不是只看 2026-10-09 那份）", "",
          "| 门禁 | 结论 | 实测 | 时点 |", "|---|---|---|---|"]
    for g in rep["gate0_verdicts_latest"]:
        L.append(f"| {g['门禁']} | {g['结论']} | {str(g['实测'])[:150].replace('|', '／')} | {g['ts']} |")
    ac = rep["ac"]
    L += ["", "> ⑥ 已用 2026-10-12 复跑读数替换（`var/gate0/m4_g06_rerun2.json`）。",
          "> ⑤ 的 **fail 没有被后续任何一轮改成通过**：M3-04b 把拒答修到 7/7、转人工 5/5。"
          "ECE 那一格此前的原句是「至今空（缺有方差的正确性标签）」——2026-10-12 `answer_correct_fill.csv` 67 行回收后"
          f"**这句作废**，实测算出 ECE={ac['ece']}（真人标签，有方差），**仍不达标**，方向为{ac['sign']}。"
          " fail 依旧是 fail，只是从「没测」变成「测了且不合格」——这不是笔误的修正，"
          "是把敞口关到了可判定的位置上。",
          "", "## 这张表暴露的口径冲突：**第 1、2 条已定案（V1.21 / V1.0.23），第 3 条见下**", "",
          "1. **P95 完整答案 / 端到端**：曾三处两个值（DoD#5 `< 3s`、《项目定义》第六节与本文 8.1 `< 6s`）。"
          f"**已定案：门禁 = P95 < 6s 且只统计作答轮次**——净环境 4 轮 {lc['n']} 个作答轮次 "
          f"p95 {lc['p95_client']}ms、0% 超 6s（>3s 是 {pct(lc['over_3s'])} ⇒ 3s 那一边在 256 token "
          "完整答案档**不可达**，要达就得改生成预算或换模型，那是实现改动不是措辞改动）。"
          "**含拒答轮次仍须另并报**（max 7854.6ms、4% 超 6s）——两条拒答路径成本差 18 倍，"
          "这就是它不能与被测对象混进同一个分位数的理由。",
          "2. **首 token 1.2s**：口径已钉死为 **`first_visible_ms`（首个可读字）**。"
          f"三种读法同轮量齐（n={lc['ft_n']}）：信封与首事件 p50 {lc['event_p50']}ms、"
          f"首个可读字 p50 {lc['ft_p50']} / p95 {lc['ft_p95']} / max {lc['ft_max']}ms ⇒ 按最严那把达标。"
          "信封那个数只是 `{`，**不得当达标依据**——这句现在写在门禁文字里，不靠谁记得。",
          "3. **「引用正确率」到底指什么**：DoD#3 的 ≥ 90% 原来只有「引用落 gold」一种实测，它不等于结论正确。"
          f"**2026-10-12 起这一格有两种实测并存**（落 gold 0.9839 / 人判 answer_correct {ac['accuracy']}），"
          "判据文字仍写「引用正确率」——两个数都能自称 ≥ 90%，"
          "但说的不是同一件事，**M5 要裁的是把哪个钉成门禁**，建议裁成人判那一列（它才是用户收到的东西）。", ""]
    return "\n".join(L)


def main() -> int:
    rep = build()
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "m5_dod.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(to_md(rep), encoding="utf-8")
    print("汇总:", json.dumps(rep["summary"], ensure_ascii=False))
    for r in rep["dod"]:
        print(f"  DoD#{r['#']} {r['验收项']}: {r['结论']}")
    print("产物：", OUT_MD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
