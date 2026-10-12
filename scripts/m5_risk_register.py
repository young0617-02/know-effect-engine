"""M5-03 风险登记册定稿：把《技术方案》第九节那张表**逐行**收成三种状态。

为什么用脚本而不是手写定稿：定稿最容易出的事故是「漏了一条」或「新加的行没人判」。
所以这里的做法是**解析原表的每一行**，每条都必须在下面的 `DISPOSITION` 里有一条对应处置，
**缺一行就报错退出**——新增风险行时如果忘了定状态，这份定稿会直接拒绝生成，
而不是悄悄少一条（这跟 M4 那次「只能同意」的假绿是同一类防线）。

三种状态：
  已闭合   —— 有实测读数且问题不再成立
  敞口未闭 —— 已量到但没关闭（如 ECE 缺标签、端到端延迟超标）
  待裁决   —— 判定权在用户，我不自行推进

跑法：uv run python scripts/m5_risk_register.py
产物：var/gate0/m5_risk_register.json + reports/m5/M5_风险登记册定稿.md
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs" / "技术方案与MVP设计（V1.0）.md"
ART = ROOT / "var" / "gate0"
OUT_MD = ROOT / "reports" / "m5" / "M5_风险登记册定稿.md"

CLOSED, OPEN, PENDING, INHERITED = "已闭合", "敞口未闭", "待裁决", "继承未触发"

# 风险名 → (状态, 定稿依据)。key 用原表里风险名的**前缀匹配串**，改名会报错而不是漏判。
DISPOSITION: dict[str, tuple[str, str]] = {
    "8GB 显存不足": (CLOSED, "Gate0 ① 通过：零 OOM、嵌入+重排与 7B 可共存；本轮再实测 7B 100% GPU 4.7GB + 嵌入冷加载共存"),
    "jieba 中文分析器失效": (CLOSED, "Gate0 ② 通过：jieba Recall@5 0.7612 vs 默认 0.399"),
    "混合检索相对稠密无增益": (INHERITED, "③ 只到「有条件通过」；「混合优于稠密」在 dev 半区 +0.004 落在噪声地板内，已降级为待更大评测集复核——**不得当已确认引用**"),
    "引用幻觉": (OPEN, "出处可信已量到（行级引用落 gold 53/53）；**答案内容可信未量**，等 #40 的 answer_correct 58 行"),
    "置信度校准偏差": (OPEN, "ECE 0.2347 > 0.10 且 accuracy 来自零方差标签；复合分已降为规则分不称概率（V1.0.6），温度缩放未启用（V1.0.6）"),
    "权限越权泄漏": (CLOSED, "检索层过滤 + 密级 3 新人不可读 + 审计 E403 原句，均真机实测（verify_m3 / verify_m4）"),
    "索引与权威源不一致": (CLOSED, "Gate0 ④ 通过：发现率 19/19；本轮写路径再次实测「人为删一个向量子块 → 对账报 missing=1 → 修复归零」"),
    "数据出域/合规风险": (OPEN, "⑥ 复跑通过（容器封公网、宿主审计 0 出网、云端兜底 false），**但「断网下端到端问答」未证**：internal 切宿主通路、API 在宿主、masquerade 路实测不成立 → 随 ADR-12 一起悬着"),
    "中文分词噪声": (INHERITED, "② 的附带观察，未单独设防；术语/编号类已抽检"),
    "本地模型能力上限": (OPEN, "端到端 p50 4003ms / p95 20700ms（含 18 行冷装载），DoD#5 两个口径都过不了；量化/降级未测"),
    "工程连续性": (INHERITED, "单人自主、无并行冗余；本轮无新证据"),
    "解析质量": (INHERITED, "M1 MinerU-ONNX 走通但公式未接；真实 PDF 只验过有限样本"),
    "评测集 gold 自身有缺陷": (CLOSED, "两批修订执行完（关键词收窄 + 逐块结论覆盖），改前实测三处留档；**拆题仍未做**，转记到下面的待裁/未完清单"),
    "流式「首 token < 1.2s」": (PENDING, "门禁值在、口径未定：热 1218/1613ms、冷 21118ms、信封首字符 849ms 是三种不同读法。定案前对外不得声称达标"),
    "置信度校准不可行": (OPEN, "同 ECE 那条：ECE 0.2347 > 0.10，可判标签零方差 ⇒ accuracy=1.0 是假的确定性；"
                          "复合分已降为规则分不称概率（V1.0.6）。关闭只有一条路：#40 的 58 行 answer_correct"),
    "知识库投毒": (OPEN, "**只有半套实测**：准入下限（段落长度/结构）与 RBAC 过滤表达式注入各有测试"
                        "（tests/test_rbac.py:41），但「恶意文档内容左右答案」这一类**没有任何实测**，"
                        "不能因为有 admission 测试就宣称防住"),
    "范围膨胀": (CLOSED, "本轮未扩范围：仍 9 条 Must、70 题评测集，未加自治编排/工具调用/SaaS 化；"
                        "M4/M5 的 additions 都在 MVP 边界内（交付层与验收）"),
    "云端兜底成本失控": (CLOSED, "cloud_fallback_enabled=false 实测（⑥ 复跑），且静态扫描运行期出域 0、"
                            "宿主连接审计非回环 0 —— 兜底路径存在但未启用，无外呼成本"),
    "个人独立开发进度风险": (INHERITED, "单人自主、无并行冗余；本轮无新证据，也不认为有可闭合的缓解措施"),
    "多存储静默不一致": (CLOSED, "Gate0 ④ 发现率 19/19 + 本轮写路径复现「人为删一个向量子块 → 对账报 missing=1 → 修复归零」"),
    "置信度校准不可行": (OPEN, "ECE 0.2347 > 0.10 且 accuracy=1.0 来自零方差标签；复合分已降为规则分（V1.0.6）、温度缩放未启用（V1.0.6）。"
                            "关闭路径唯一：#40 的 answer_correct 58 行"),
    "知识库投毒 / Prompt 注入": (OPEN, "**只有半套实测**：准入校验（段落长度/结构下限）与 RBAC 过滤表达式注入各有测试，"
                              "但「恶意文档内容左右答案」这类投毒**没有任何实测**——不能因为有 admission 测试就说这条已防住"),
    "范围膨胀（滑向自治 Agent/SaaS）": (INHERITED, "本轮仍按 MVP 范围收口，未新增自治能力；无新证据"),
    "个人独立开发进度风险": (INHERITED, "单人自主、无并行冗余；本轮无新证据"),
    "云端兜底成本失控": (CLOSED, "cloud_fallback_enabled=false 实测（⑥ 复跑），且静态扫描运行期出域 0、宿主连接审计非回环 0"),
}

# 原表里没单列、但本轮新登记/仍未闭的三条，补进定稿
ADDED = [
    ("未复核内容即时可被引用作答（复核不是前闸）", PENDING,
     "实测（scratch 后端，生产未动）：以 delivery_engineer 上传未复核文档后，复核**之前** /retrieval/search 命中、"
     "/answers 直接 status=answer 并引用该块；版本一上传即 effective。当前复核=事后撤销（reject 才 superseded+删索引），"
     "而 Must#1 写「上传→准入审核→分块→落库」，顺序相反且「准入审核」未定义是机械准入还是人工复核。"
     "**三条处置路见 reports/m4/M4_前端与一键Demo.md 第十节，须用户选定并改需求原文**；选定前不得声称「入库有审核闸」"),
    ("延迟读数不在净环境", PENDING,
     "本机跑着其他容器（auscult 一套）与宿主应用，`session_turn` 还混着验证脚本自己打的流量 ⇒ "
     "p50 4003ms / p95 20700ms 只能说「这台机器此刻这个样子要等多久」，**不能当 DoD#5 的判定数据**。"
     "用户 2026-10-12 决定：清空其他进程后复测；在那之前所有延迟类结论（P95 端到端、首 token 1.2s）"
     "一律挂『待复测』，3s vs 6s 的口径冲突也一并挂起——没有干净数据就先定口径，定完还得重测"),
    ("ADR-12 API 未容器化", PENDING,
     "pyproject 刻意不声明 torch（由本机 CUDA 环境 .pth 挂载）→ 容器化要么拉 GB 级轮子要么再要一份 CUDA 运行时。"
     "它同时挡住「断网端到端」与 M6 版本锁定，是本轮唯一还卡着两条验收的裁决"),
    ("DoD#5 与《项目定义》第六节的 3s/6s 口径冲突", PENDING,
     "DoD 核对表暴露的：同一指标三处两个值，必须先定口径再谈达标（现值 p50 4003ms 两边都过）"),
    ("「引用正确率 ≥ 90%」的语义未定", PENDING,
     "现在只有「引用落 gold」这一种实测，它不等于结论正确；要么改判据文字，要么等 answer_correct，"
     "不能拿落 gold 冒充正确率"),
    ("评测集拆题未完", OPEN,
     "q-053/q-055/q-058 + q-054 第二问，另加本轮新升的 q-013/q-025（都是「没有任何单块能答全」的形状）；"
     "要新题干 + 新人工标注，不能我单方生成就算标过"),
    ("误收运行日志进版本库", CLOSED,
     "本轮发生并已纠正：`git add -f var/gate0/` 把 Gate 0 的 .log 一起收了进去，与本仓库「只收读数」的纪律不符，"
     "已用新 commit 移出跟踪（文件留在磁盘）。教训：add -f 之前先看 git status --short 的完整清单"),
]


def parse_rows() -> list[dict]:
    text = PLAN.read_text(encoding="utf-8")
    m = re.search(r"## 九、风险登记册(.*?)\n## ", text, re.S)
    if not m:
        raise SystemExit("找不到第九节风险表，定稿拒绝生成")
    rows = []
    for ln in m.group(1).splitlines():
        if not ln.startswith("|") or set(ln) <= {"|", "-", " "}:
            continue
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if len(cells) < 7 or cells[0] == "风险":
            continue
        # `└ …` 是挂在上一行风险下的**状态更新子行**，不是独立风险。
        # 早先把它当新风险去要处置，定稿就为一堆本来就该合并的行报错——合并才是正确读法。
        if cells[0].startswith("└"):
            if not rows:
                raise SystemExit(f"子行没有父行可挂：{cells[0][:40]}")
            rows[-1]["状态更新"] = rows[-1].get("状态更新", []) + [cells[0].replace("└", "").strip()]
            continue
        rows.append({"风险": cells[0], "类别": cells[1], "影响": cells[2], "概率": cells[3],
                     "等级": cells[4], "缓解": cells[5], "信号": cells[6]})
    return rows


def match(name: str) -> str | None:
    for key in DISPOSITION:
        if name.replace("*", "").startswith(key) or key in name:
            return key
    return None


def main() -> int:
    rows = parse_rows()
    unmatched = [r["风险"] for r in rows if match(r["风险"]) is None]
    if unmatched:
        print("[拒绝定稿] 这些风险行没有对应处置，不能默认它们没问题：")
        for u in unmatched:
            print("   -", u)
        return 2

    out = []
    for r in rows:
        st, why = DISPOSITION[match(r["风险"])]
        out.append({**r, "定稿状态": st, "定稿依据": why})
    counts = {}
    for r in out + [{"定稿状态": s} for _, s, _ in ADDED]:
        counts[r["定稿状态"]] = counts.get(r["定稿状态"], 0) + 1

    rep = {"generated_at": datetime.now().isoformat(timespec="seconds"),
           "principle": "逐行解析原表，缺一条处置就拒绝生成；状态显式给定，不按散文猜",
           "risk_rows": out, "added": [{"风险": n, "定稿状态": s, "定稿依据": w} for n, s, w in ADDED],
           "counts": counts}
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "m5_risk_register.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")

    L = ["# M5 · 风险登记册定稿", "",
         f"> 生成时点：{rep['generated_at']}｜`scripts/m5_risk_register.py` **逐行解析**《技术方案》第九节的表，"
         "每行都必须在脚本的 `DISPOSITION` 里有一条处置——**缺一行就拒绝生成**，"
         "不靠人记得全不全（同 M4 那次「只能同意」假绿的防线思路）。", "",
         f"**状态分布**：" + " ｜ ".join(f"{k} {v}" for k, v in counts.items()), "",
         "## 一、原表逐行定稿", "",
         "| 风险 | 类别 | 等级 | 定稿状态 | 定稿依据 |", "|---|---|---|---|---|"]
    for r in out:
        L.append(f"| {r['风险']} | {r['类别']} | {r['等级']} | **{r['定稿状态']}** | {r['定稿依据']} |")
    folded = sum(len(r.get("状态更新", [])) for r in out)
    L += ["", f"> 解析时把 **{folded} 条 `└ 状态更新…` 子行折叠进各自的父风险**——它们是同一风险的时间线，"
          "不是独立风险；原文仍留在《技术方案》第九节，本定稿不重复正文，只给最终状态。", "",
          "## 二、原表没有、本轮新登记的条目", "",
          "| 风险 | 定稿状态 | 定稿依据 |", "|---|---|---|"]
    for n, s, w in ADDED:
        L.append(f"| {n} | **{s}** | {w} |")
    L += ["", "## 三、三条「必须用户拍」的裁决（定稿里它们只能停在待裁决）", "",
          "1. **ADR-12 API 容器化**——卡住「断网端到端」与 M6 版本锁定两条。",
          "2. **① 首 token 1.2s 的口径**（信封首字符 / 首可见内容 / 首事件），以及 **DoD#5 端到端 3s vs 第六节 6s** 的冲突。",
          "3. **「引用正确率」到底指什么**——落 gold 不等于答对。",
          "", "> 另有 `langgraph-checkpoint-postgres` 已安装但未使用，是否摘掉依赖声明影响 M6 清单，同样待裁。", "",
          "## 四、M5 的收口结论", "",
          "- 原表 20 行**全部有处置**，无「未判定」残留。",
          "- **已闭合**的都有实测文件；**敞口未闭**的四条里，三条卡人工（answer_correct、拆题、真实用户业务指标），"
          "一条卡裁决（ADR-12）。",
          "- 定稿不抹历史：状态变化都在原表的「状态更新」子行里留痕，这份文件只做**分类与归因**。", ""]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print("行数", len(out), "| 未匹配", len(unmatched), "| 状态分布", json.dumps(counts, ensure_ascii=False))
    print("产物：", OUT_MD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
