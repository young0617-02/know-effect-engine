"""自建评测集（Gate 0 可行性验证用），与模拟语料 kee.ingest.corpus_data 配套。

性质说明：
本评测集为**构造数据**，问题与标准答案均来自同一套模拟行业语料，不含真实客户信息；
其用途是度量 Recall@5、MRR、中文 BM25 分词有效性以及强制拒答与密级过滤行为。

双人标注口径：
- 第一人（构造者）标注：本文件即构造时的 ground truth。每题的 `gold_keywords` 取自
  语料段落中的**原样子串**（含空格、罗马数字与字母大小写，如 `SCADA 主站`、`V3.2`），
  `reference_answer` 由同一段落的事实改写而来，`gold_projects` 为人工核对用的项目归属。
- 第二人（复核者）标注：须**脱离本文件的答案字段**，仅按检索系统返回的 chunk 独立判断
  「该 chunk 是否足以回答该问题」，并重新给出一组 gold 段落编号；两人结论不一致时按标注
  规范逐条讨论，一致性以 Cohen's Kappa 度量，低于 0.7 的类别须回炉重写题干或关键词。
- 复核者禁止参考 `reference_answer` 后倒推 gold_keywords；若发现某题的 gold_keywords
  在语料中命中不到、或命中面过宽（超过 3 个段落），该题按缺陷登记并改写。

gold 判定规则（与下游检索评测一致）：
- 一个 chunk 命中某题的**全部** `gold_keywords`（AND 语义、大小写不敏感、子串匹配）即为 gold chunk；
- `category` 为「无答案」的题目 `gold_keywords` 为空列表，期望行为为拒答（refuse）；
- `category` 为「敏感」的题目期望行为为转人工（escalate），其 gold_keywords 指向机密段落，
  仅用于验证权限过滤生效，不得作为可直接回答的题面使用。

问题类别与题量（共 70 题）：术语 14、编号 12、方案名 12、口语化 14、跨文档 6、无答案 7、敏感 5。
其中前 58 题（术语/编号/方案名/口语化/跨文档）均可在本语料中回查到 gold 段落。

gold 修订（2026-10-12，按上一条「命中面过宽即缺陷」规则执行，题量与题干不变）：
- q-008 / q-054 / q-057 的 `gold_keywords` 各补一个**第二人独立确认在块内**的限定词，
  把 AND 语义收紧；strict gold 100 → 90、section gold 164 → 150，删除量与第二人判否行数逐条对齐。
- **q-056 本轮不动**：机械补词会把第二人判「相关」的块一起删掉，与复核结论冲突，留待人工裁决。
- q-012 / q-014 的漏标（应加块）与 q-053 / q-055 / q-058 的拆题**同属结构改动**，未并入本次。
- 改前实测另存 `var/gate0/pre_goldfix/`（qrels / eval_set / m2_eval / kappa_a2 / 标注 CSV），不覆盖。

生成时点：2026-10-09；gold 关键词修订时点：2026-10-12。仅使用标准库。
"""

import re

QUESTIONS: list[dict] = [
    # ---------------- 术语（14 题）----------------
    {
        "qid": "q-001",
        "question": "银企直联实时清算方案 V3.2 的灾备切换目标里，RTO 要求是多少？",
        "category": "术语",
        "gold_keywords": ["银企直联实时清算方案 V3.2", "RTO"],
        "gold_projects": ["PRJ-FIN-2024-017"],
        "expected_behavior": "answer",
        "reference_answer": "RTO 小于 15 分钟，同时要求 RPO 等于 0。",
    },
    {
        "qid": "q-002",
        "question": "头寸管理上线后，法人账户日间头寸预测偏差收敛到多少？",
        "category": "术语",
        "gold_keywords": ["头寸管理", "1.2%"],
        "gold_projects": ["PRJ-FIN-2024-017"],
        "expected_behavior": "answer",
        "reference_answer": "预测偏差绝对值由改造前的 3.7% 收敛至 1.2%。",
    },
    {
        "qid": "q-003",
        "question": "监管口径对拨备覆盖率的最低要求是多少？",
        "category": "术语",
        "gold_keywords": ["拨备覆盖率", "150%"],
        "gold_projects": ["PRJ-FIN-2023-041"],
        "expected_behavior": "answer",
        "reference_answer": "拨备覆盖率不低于 150%，同时不良贷款率不高于 1.8%。",
    },
    {
        "qid": "q-004",
        "question": "信贷资产风险分类结果按几级类别落库，分别是什么？",
        "category": "术语",
        "gold_keywords": ["信贷资产风险分类", "五级"],
        "gold_projects": ["PRJ-FIN-2023-041"],
        "expected_behavior": "answer",
        "reference_answer": "按五级类别落库，即正常、关注、次级、可疑与损失。",
    },
    {
        "qid": "q-005",
        "question": "银企直联改造中，人行支付报文一共改造了多少项字段映射？",
        "category": "术语",
        "gold_keywords": ["人行支付报文", "214"],
        "gold_projects": ["PRJ-FIN-2024-017"],
        "expected_behavior": "answer",
        "reference_answer": "共改造报文字段映射 214 项，其中金额类 38 项、账号与户名类 57 项、业务标识类 119 项。",
    },
    {
        "qid": "q-006",
        "question": "双活数据中心同城链路的同步延迟正常水位控制在多少？",
        "category": "术语",
        "gold_keywords": ["双活数据中心", "800 毫秒"],
        "gold_projects": ["PRJ-FIN-2024-033"],
        "expected_behavior": "answer",
        "reference_answer": "正常水位不超过 800 毫秒，达 2000 毫秒黄色告警、达 5000 毫秒红色告警。",
    },
    {
        "qid": "q-007",
        "question": "客户现有一阶段引擎在反洗钱可疑交易监测上的误报率大概是多少？",
        "category": "术语",
        "gold_keywords": ["反洗钱可疑交易", "误报率"],
        "gold_projects": ["PRJ-FIN-2025-008"],
        "expected_behavior": "answer",
        "reference_answer": "误报率长期高于 92%，人工甄别团队 74 人仍无法消化预警积压。",
    },
    {
        "qid": "q-008",
        "question": "按 ITSS 事件分级，清算平台的 P1 级故障多久必须启动应急响应？",
        "category": "术语",
        "gold_keywords": ["ITSS 事件分级", "15 分钟", "清算链路"],
        "gold_projects": ["PRJ-FIN-2024-017"],
        "expected_behavior": "answer",
        "reference_answer": "P1 级故障须在 15 分钟内启动应急响应。",
    },
    {
        "qid": "q-009",
        "question": "SCADA 主站的实时数据刷新周期是多少秒？",
        "category": "术语",
        "gold_keywords": ["SCADA 主站", "2 秒"],
        "gold_projects": ["PRJ-ENE-2023-019"],
        "expected_behavior": "answer",
        "reference_answer": "实时数据刷新周期为 2 秒，人机界面画面刷新不超过 3 秒。",
    },
    {
        "qid": "q-010",
        "question": "台区遥测上送接口里，配电自动化终端一共上送多少个测点？",
        "category": "术语",
        "gold_keywords": ["配电自动化终端", "14 个测点"],
        "gold_projects": ["PRJ-ENE-2024-006"],
        "expected_behavior": "answer",
        "reference_answer": "共 14 个测点，覆盖三相电流、三相电压、有功与无功功率及功率因数。",
    },
    {
        "qid": "q-011",
        "question": "用电信息采集治理后的采集成功率提升到多少？",
        "category": "术语",
        "gold_keywords": ["用电信息采集", "采集成功率"],
        "gold_projects": ["PRJ-ENE-2024-006"],
        "expected_behavior": "answer",
        "reference_answer": "由治理前的 91.6% 提升至 99.3%。",
    },
    {
        "qid": "q-012",
        "question": "按电力监控系统安全防护口径，终端与信息管理大区之间的数据交换必须经过什么装置？",
        "category": "术语",
        "gold_keywords": ["电力监控系统安全防护", "正反向隔离装置"],
        "gold_projects": ["PRJ-ENE-2024-006", "PRJ-ENE-2025-011"],
        "expected_behavior": "answer",
        "reference_answer": "必须经正反向隔离装置以单向方式传输，禁止任何形式的反向直连。",
    },
    {
        "qid": "q-013",
        "question": "SIS 安全仪表系统采用的表决逻辑是什么，优化平台能不能改它的整定值？",
        "category": "术语",
        "gold_keywords": ["SIS 安全仪表系统", "2oo3"],
        "gold_projects": ["PRJ-CHM-2024-003"],
        "expected_behavior": "answer",
        "reference_answer": "采用 2oo3 表决逻辑，独立于优化平台部署，其整定值不随优化目标调整。",
    },
    {
        "qid": "q-014",
        "question": "优化平台上涉及 DCS 联锁逻辑的点位是什么读写权限？",
        "category": "术语",
        "gold_keywords": ["DCS 联锁逻辑", "只读"],
        "gold_projects": ["PRJ-CHM-2024-003"],
        "expected_behavior": "answer",
        "reference_answer": "一律设为只读，联锁值修改权限保留在客户工艺与仪表两个部门。",
    },
    # ---------------- 编号（12 题）----------------
    {
        "qid": "q-015",
        "question": "结项文档连同验收原始记录要归档保留多久，依据哪个内部流程文件？",
        "category": "编号",
        "gold_keywords": ["Q/KEE-SOP-032", "3 年"],
        "gold_projects": ["PRJ-FIN-2024-017", "PRJ-ENE-2024-006"],
        "expected_behavior": "answer",
        "reference_answer": "依据 Q/KEE-SOP-032 归档并保留 3 年，不得以摘要件替代原件。",
    },
    {
        "qid": "q-016",
        "question": "第三级安全计算环境的身份鉴别与会话锁定要求出自哪个标准条款，落地参数是什么？",
        "category": "编号",
        "gold_keywords": ["GB/T 22239-2019", "第 8.1.4 条"],
        "gold_projects": ["PRJ-FIN-2025-008"],
        "expected_behavior": "answer",
        "reference_answer": "出自 GB/T 22239-2019 第 8.1.4 条，落地为登录失败 5 次锁定 30 分钟、会话空闲 15 分钟强制注销。",
    },
    {
        "qid": "q-017",
        "question": "SLA-A1 对应的是哪个平台的服务级别协议，质保期多长？",
        "category": "编号",
        "gold_keywords": ["SLA-A1", "质保期"],
        "gold_projects": ["PRJ-FIN-2024-017"],
        "expected_behavior": "answer",
        "reference_answer": "对应银企直联清算平台，质保期 12 个月，含工作日 8:00 至 20:00 一阶段支持与 7×24 小时 P1 级响应。",
    },
    {
        "qid": "q-018",
        "question": "接口编号 IF-FIN-0012 的单笔汇款金额上限是多少？",
        "category": "编号",
        "gold_keywords": ["IF-FIN-0012", "500 万元"],
        "gold_projects": ["PRJ-FIN-2024-017"],
        "expected_behavior": "answer",
        "reference_answer": "单笔汇款金额上限为 500 万元，超限须走人工复核通道并附授权影像件。",
    },
    {
        "qid": "q-019",
        "question": "接口编号 IF-FIN-0013 的单次响应报文体积上限是多少？",
        "category": "编号",
        "gold_keywords": ["接口编号 IF-FIN-0013", "2 MB"],
        "gold_projects": ["PRJ-FIN-2024-017"],
        "expected_behavior": "answer",
        "reference_answer": "单次响应报文体积不得超过 2 MB。",
    },
    {
        "qid": "q-020",
        "question": "IF-ENE-0021 遥测上送接口的信息体地址段从哪里开始？",
        "category": "编号",
        "gold_keywords": ["IF-ENE-0021", "16385"],
        "gold_projects": ["PRJ-ENE-2024-006"],
        "expected_behavior": "answer",
        "reference_answer": "信息体地址段规划为 16385 至 18000。",
    },
    {
        "qid": "q-021",
        "question": "IEC 60870-5-104 规约下的心跳试验帧叫什么，链路中断等效判定超时是多少？",
        "category": "编号",
        "gold_keywords": ["IEC 60870-5-104", "TESTFH"],
        "gold_projects": ["PRJ-ENE-2024-006"],
        "expected_behavior": "answer",
        "reference_answer": "采用 IEC 60870-5-104 的 TESTFH 试验帧，主站 20 秒发起、连续 2 次无应答判定中断，等效超时 60 秒。",
    },
    {
        "qid": "q-022",
        "question": "DL/T 645-2007 规约下电能量抄读的表计地址怎么编排？",
        "category": "编号",
        "gold_keywords": ["DL/T 645-2007", "12 位表号"],
        "gold_projects": ["PRJ-ENE-2025-011"],
        "expected_behavior": "answer",
        "reference_answer": "遵循 DL/T 645-2007，地址采用 12 位表号，抄读为半双工主从方式，单次会话时长上限 3 秒。",
    },
    {
        "qid": "q-023",
        "question": "IF-CHM-0007 实时点位读取接口的订阅点位规模上限是多少？",
        "category": "编号",
        "gold_keywords": ["IF-CHM-0007", "8000"],
        "gold_projects": ["PRJ-CHM-2024-003"],
        "expected_behavior": "answer",
        "reference_answer": "订阅点位规模上限 8000 个，最小发布间隔 500 毫秒。",
    },
    {
        "qid": "q-024",
        "question": "IF-CHM-0011 历史回溯接口单次返回的时间序列点数上限是多少？",
        "category": "编号",
        "gold_keywords": ["IF-CHM-0011", "86400"],
        "gold_projects": ["PRJ-CHM-2024-003"],
        "expected_behavior": "answer",
        "reference_answer": "单次返回上限 86400 点，超限须分片并保持帧序号连续。",
    },
    {
        "qid": "q-025",
        "question": "防爆区域划分依据哪个国标执行，区域内的作业票怎么签发？",
        "category": "编号",
        "gold_keywords": ["GB 50058-2014", "防爆区域划分"],
        "gold_projects": ["PRJ-CHM-2023-021"],
        "expected_behavior": "answer",
        "reference_answer": "按 GB 50058-2014 的分区原则执行；设备新增或替换须提交区域等级证明与选型核对表，两项缺一不得签发作业票。",
    },
    {
        "qid": "q-026",
        "question": "变更单号 CR-CHM-2024-118 为什么要补做联锁功能验证，总共耗时多久？",
        "category": "编号",
        "gold_keywords": ["CR-CHM-2024-118", "4 小时"],
        "gold_projects": ["PRJ-CHM-2023-021"],
        "expected_behavior": "answer",
        "reference_answer": "因该量程调整可能间接影响 DCS 联锁逻辑的判定输入，须补做验证后方可实施，从发起到关闭共耗时 4 小时。",
    },
    # ---------------- 方案名（12 题）----------------
    {
        "qid": "q-027",
        "question": "银企直联实时清算方案 V3.2 对日间清算高峰的处理能力要求是多少？",
        "category": "方案名",
        "gold_keywords": ["银企直联实时清算方案 V3.2", "3200 笔每秒"],
        "gold_projects": ["PRJ-FIN-2024-017"],
        "expected_behavior": "answer",
        "reference_answer": "要求 3200 笔每秒持续 60 分钟无积压，实测达到 3760 笔每秒。",
    },
    {
        "qid": "q-028",
        "question": "银企直联实时清算方案 V3.2 的核心通道年度可用性目标是多少？",
        "category": "方案名",
        "gold_keywords": ["银企直联实时清算方案 V3.2", "99.99%"],
        "gold_projects": ["PRJ-FIN-2024-017"],
        "expected_behavior": "answer",
        "reference_answer": "年度可用性目标不低于 99.99%，全年非计划中断时长控制在 53 分钟以内。",
    },
    {
        "qid": "q-029",
        "question": "配网线损治理方案 V2.1 的验收核心口径包含哪几项指标？",
        "category": "方案名",
        "gold_keywords": ["配网线损治理方案 V2.1", "同期线损"],
        "gold_projects": ["PRJ-ENE-2024-006"],
        "expected_behavior": "answer",
        "reference_answer": "以同期线损率指标、采集数据完整率与台区档案一致率三项为验收核心口径。",
    },
    {
        "qid": "q-030",
        "question": "连续重整装置智能优化方案 V1.4 的验收指标里收率类目标是什么？",
        "category": "方案名",
        "gold_keywords": ["连续重整装置智能优化方案 V1.4", "收率"],
        "gold_projects": ["PRJ-CHM-2024-003"],
        "expected_behavior": "answer",
        "reference_answer": "以装置收率、能耗与平稳率三类指标为核心，汽油收率实测由 82.4% 提升至 83.2%。",
    },
    {
        "qid": "q-031",
        "question": "信贷资产风险分类模型方案 V1.0 拆分作业后达成了哪个周期目标？",
        "category": "方案名",
        "gold_keywords": ["信贷资产风险分类模型方案 V1.0", "5 个工作日"],
        "gold_projects": ["PRJ-FIN-2023-041"],
        "expected_behavior": "answer",
        "reference_answer": "达成 5 个工作日以内作业周期的验收目标，增量作业 P95 时长稳定在 8 分 20 秒。",
    },
    {
        "qid": "q-032",
        "question": "调度自动化系统运维支撑方案 V4.0 给主站定的年度可用率基线是多少？",
        "category": "方案名",
        "gold_keywords": ["调度自动化系统运维支撑方案 V4.0", "99.99%"],
        "gold_projects": ["PRJ-ENE-2023-019"],
        "expected_behavior": "answer",
        "reference_answer": "主站年度可用率目标不低于 99.99%。",
    },
    {
        "qid": "q-033",
        "question": "用电信息采集运维支持方案 V2.3 里归档口径调阅的响应时限是多久？",
        "category": "方案名",
        "gold_keywords": ["用电信息采集运维支持方案 V2.3", "4 小时"],
        "gold_projects": ["PRJ-ENE-2025-011"],
        "expected_behavior": "answer",
        "reference_answer": "归档调阅响应时限为 4 小时以内，且口径版本变更须保留前后对照表。",
    },
    {
        "qid": "q-034",
        "question": "反洗钱可疑交易监测规则体系方案 V1.2 规划了多少项规则？",
        "category": "方案名",
        "gold_keywords": ["反洗钱可疑交易监测规则体系方案 V1.2", "312 项"],
        "gold_projects": ["PRJ-FIN-2025-008"],
        "expected_behavior": "answer",
        "reference_answer": "规则总量规划为 312 项，含阈值类 168 项、关系网络类 96 项、行为序列类 48 项。",
    },
    {
        "qid": "q-035",
        "question": "SIS 回路改造实施方案 V1.1 对单批停车切换窗口的上限是多少？",
        "category": "方案名",
        "gold_keywords": ["SIS 回路改造实施方案 V1.1", "36 小时"],
        "gold_projects": ["PRJ-CHM-2023-021"],
        "expected_behavior": "answer",
        "reference_answer": "每批切换窗口不超过 36 小时，实际最长一次为 33 小时 40 分钟。",
    },
    {
        "qid": "q-036",
        "question": "电子记录与数据完整性提升方案 V1.0 对批生产记录电子化覆盖率的要求是多少？",
        "category": "方案名",
        "gold_keywords": ["电子记录与数据完整性提升方案 V1.0", "95%"],
        "gold_projects": ["PRJ-CHM-2025-014"],
        "expected_behavior": "answer",
        "reference_answer": "批生产记录电子化覆盖率不低于 95%，记录及时率不低于 99%，批放行周期压缩至 2 天以内。",
    },
    {
        "qid": "q-037",
        "question": "灾备切换编排方案 V2.0 里非计划应急切换需要什么确认条件？",
        "category": "方案名",
        "gold_keywords": ["灾备切换编排方案 V2.0", "双人确认"],
        "gold_projects": ["PRJ-FIN-2024-033"],
        "expected_behavior": "answer",
        "reference_answer": "须双人确认，两名确认人员分别属于运维岗与业务岗。",
    },
    {
        "qid": "q-038",
        "question": "装置优化数据接口规范 V2.0 里 IF-CHM-0007 采用什么订阅方式与发布间隔？",
        "category": "方案名",
        "gold_keywords": ["装置优化数据接口规范 V2.0", "IF-CHM-0007"],
        "gold_projects": ["PRJ-CHM-2024-003"],
        "expected_behavior": "answer",
        "reference_answer": "采用 OPC UA 订阅模式，最小发布间隔 500 毫秒。",
    },
    # ---------------- 口语化（14 题）----------------
    {
        "qid": "q-039",
        "question": "半夜清算链路挂了，客户追问我们多久之内必须把应急拉起来，还得挂哪个流程？",
        "category": "口语化",
        "gold_keywords": ["ITIL P1", "15 分钟"],
        "gold_projects": ["PRJ-FIN-2024-017"],
        "expected_behavior": "answer",
        "reference_answer": "P1 级故障须在 15 分钟内启动应急响应，并按 ITIL P1 流程成立事件指挥组。",
    },
    {
        "qid": "q-040",
        "question": "厂里一路联锁失去保护了，值班经理问这算几级事故、人最晚多久得到现场？",
        "category": "口语化",
        "gold_keywords": ["ITSS 事件分级", "2 小时内到场"],
        "gold_projects": ["PRJ-CHM-2023-021"],
        "expected_behavior": "answer",
        "reference_answer": "按 ITSS 事件分级属 P1，须在 30 分钟内启动应急响应并在 2 小时内到场。",
    },
    {
        "qid": "q-041",
        "question": "客户问我们那个心跳多久算超时，这个数在哪儿有依据？",
        "category": "口语化",
        "gold_keywords": ["心跳超时", "15 秒"],
        "gold_projects": ["PRJ-FIN-2024-017"],
        "expected_behavior": "answer",
        "reference_answer": "探活报文发送间隔 5 秒、连续 3 次无应答判定链路中断，等效超时阈值 15 秒。",
    },
    {
        "qid": "q-042",
        "question": "台区电量老对不上，怀疑是抄表不同期，系统看的是哪个窗口、多长算合格？",
        "category": "口语化",
        "gold_keywords": ["采集时间窗", "15 分钟"],
        "gold_projects": ["PRJ-ENE-2024-006"],
        "expected_behavior": "answer",
        "reference_answer": "全台区采集时间窗不超过 15 分钟，同期率低于 95% 时该台区线损结果标记为不可信。",
    },
    {
        "qid": "q-043",
        "question": "厂里验收是不是真去碰联锁啊？现场那套故障是怎么试出来的？",
        "category": "口语化",
        "gold_keywords": ["FAT 与 SAT 验证", "注错"],
        "gold_projects": ["PRJ-CHM-2023-021"],
        "expected_behavior": "answer",
        "reference_answer": "现场阶段以注错方式覆盖传感器断线、通道短路与会话超时三类故障注入，每类不少于 30 次。",
    },
    {
        "qid": "q-044",
        "question": "运维图省事想直接进库里把那个字段改回来，规范里哪句话把这条路堵死了？",
        "category": "口语化",
        "gold_keywords": ["审计追踪", "数据库直连"],
        "gold_projects": ["PRJ-CHM-2025-014"],
        "expected_behavior": "answer",
        "reference_answer": "审计追踪写入是唯一允许的变更方式，运维人员不得以数据库直连方式修正记录字段，错误须以追加更正记录处理。",
    },
    {
        "qid": "q-045",
        "question": "检修要临时摘一个联锁，最多能摘多久，超时了咋办？",
        "category": "口语化",
        "gold_keywords": ["联锁摘除", "24 小时"],
        "gold_projects": ["PRJ-CHM-2023-021"],
        "expected_behavior": "answer",
        "reference_answer": "联锁摘除单有效期不超过 24 小时，超时须重新审批并在操作台账挂牌提醒。",
    },
    {
        "qid": "q-046",
        "question": "车间电压一闪，采集那边咋保证记录不断，能自己顶多久？",
        "category": "口语化",
        "gold_keywords": ["断电", "本地缓存"],
        "gold_projects": ["PRJ-CHM-2025-014"],
        "expected_behavior": "answer",
        "reference_answer": "采集网关在断电后 5 分钟内起以本地缓存续录，缓存容量支持 4 小时连续记录，恢复后自动补传并按发生时间重排。",
    },
    {
        "qid": "q-047",
        "question": "想切之前先确认两地数据跟没跟上，超过多少就不能发起切换了？",
        "category": "口语化",
        "gold_keywords": ["同步延迟", "5000 毫秒"],
        "gold_projects": ["PRJ-FIN-2024-033"],
        "expected_behavior": "answer",
        "reference_answer": "同步延迟达 5000 毫秒触发红色告警并禁止发起任何计划内切换。",
    },
    {
        "qid": "q-048",
        "question": "客户想知道自己拨备现在到底在什么水平，文档里给的实际区间是啥？",
        "category": "口语化",
        "gold_keywords": ["拨备覆盖率", "196%"],
        "gold_projects": ["PRJ-FIN-2023-041"],
        "expected_behavior": "answer",
        "reference_answer": "当前拨备覆盖率稳定在 182% 至 196% 区间。",
    },
    {
        "qid": "q-049",
        "question": "企业那边怕重复打款，我们靠什么字段判重、判多久？",
        "category": "口语化",
        "gold_keywords": ["幂等", "企业流水号"],
        "gold_projects": ["PRJ-FIN-2024-017"],
        "expected_behavior": "answer",
        "reference_answer": "幂等控制以业务日期加企业流水号双字段联合判重，判重窗口保留 7 个自然日。",
    },
    {
        "qid": "q-050",
        "question": "操作员连着拒了几条系统建议，平台会把它这条回路晾多久？",
        "category": "口语化",
        "gold_keywords": ["被操作员拒绝", "30 分钟"],
        "gold_projects": ["PRJ-CHM-2024-003"],
        "expected_behavior": "answer",
        "reference_answer": "同一回路连续 3 条建议被操作员拒绝后，平台暂停该回路建议 30 分钟并回传拒绝原因。",
    },
    {
        "qid": "q-051",
        "question": "值班想知道哪块表没抄上来就得开工单，这条规矩写在哪儿？",
        "category": "口语化",
        "gold_keywords": ["抄读失败", "核查工单"],
        "gold_projects": ["PRJ-ENE-2025-011"],
        "expected_behavior": "answer",
        "reference_answer": "连续 2 次抄读失败的表计自动生成核查工单，要求 3 个工作日内定位原因。",
    },
    {
        "qid": "q-052",
        "question": "客户说远程调试可以但数据一个字节都不能出去，那我们远程到底能干啥？",
        "category": "口语化",
        "gold_keywords": ["屏幕共享", "不允许数据传递"],
        "gold_projects": ["PRJ-CHM-2024-003"],
        "expected_behavior": "answer",
        "reference_answer": "远程支持仅允许观看屏幕共享而不允许数据传递，该约束已写入交付作业指导书。",
    },
    # ---------------- 跨文档（6 题）----------------
    {
        "qid": "q-053",
        "question": "两地三中心这套灾备架构在哪些项目文档里被引用过？",
        "category": "跨文档",
        "gold_keywords": ["两地三中心"],
        "gold_projects": ["PRJ-FIN-2024-017", "PRJ-FIN-2024-033", "PRJ-ENE-2023-019"],
        "expected_behavior": "answer",
        "reference_answer": "在银企直联清算项目与两地三中心灾备切换接口规范中均有引用，前者为交付范围、后者为架构与切换约束。",
    },
    {
        "qid": "q-054",
        "question": "公司的五步交付法是哪五个阶段，各项目是怎么用的？",
        "category": "跨文档",
        "gold_keywords": ["五步交付法", "现状诊断", "成效固化"],
        "gold_projects": ["PRJ-FIN-2024-017", "PRJ-FIN-2023-041", "PRJ-FIN-2025-008", "PRJ-ENE-2024-006",
                          "PRJ-ENE-2025-011", "PRJ-CHM-2024-003", "PRJ-CHM-2023-021", "PRJ-CHM-2025-014"],
        "expected_behavior": "answer",
        "reference_answer": "依次为现状诊断、方案定型、试点验证、规模推广与成效固化五个阶段，每阶段设闸口评审并强制输出签字件。",
    },
    {
        "qid": "q-055",
        "question": "ITSS 事件分级在金融、能源、化工三条线上是怎么各自细化门限的？",
        "category": "跨文档",
        "gold_keywords": ["ITSS 事件分级"],
        "gold_projects": ["PRJ-FIN-2024-017", "PRJ-ENE-2023-019", "PRJ-CHM-2023-021", "PRJ-CHM-2025-014"],
        "expected_behavior": "answer",
        "reference_answer": "三线共用 ITSS 事件分级框架，但 P1 判定门限不同：金融看资金差错外溢、电力看厂站失联数量、化工看联锁失效与人员暴露风险。",
    },
    {
        "qid": "q-056",
        "question": "CMMI 3 级过程域映射在项目上具体登记哪些过程域和度量项？",
        "category": "跨文档",
        "gold_keywords": ["CMMI 3 级过程域映射"],
        "gold_projects": ["PRJ-FIN-2024-017", "PRJ-FIN-2023-041", "PRJ-ENE-2024-006", "PRJ-CHM-2024-003", "PRJ-CHM-2023-021"],
        "expected_behavior": "answer",
        "reference_answer": "按项目登记需求管理、项目计划、配置管理、验证与决策分析等过程域的度量项，并把结论回灌组织过程资产库。",
    },
    {
        "qid": "q-057",
        "question": "客户提的数据不出域，公司那四道防线分别是什么？",
        "category": "跨文档",
        "gold_keywords": ["数据不出域四道防线", "环境隔离", "审计追溯"],
        "gold_projects": ["PRJ-FIN-2023-041", "PRJ-FIN-2025-008", "PRJ-CHM-2024-003", "PRJ-CHM-2023-021"],
        "expected_behavior": "answer",
        "reference_answer": "四道防线为环境隔离、出口管控（审批）、内容识别与审计追溯，任一道未闭环即暂停相关作业。",
    },
    {
        "qid": "q-058",
        "question": "防爆区域划分在运维手册和合规摘录里的口径一致吗？",
        "category": "跨文档",
        "gold_keywords": ["防爆区域划分"],
        "gold_projects": ["PRJ-CHM-2023-021", "PRJ-CHM-2025-014"],
        "expected_behavior": "answer",
        "reference_answer": "一致，均按 1 区与 2 区划分并逐单元对应设备防爆标志，合规摘录另给出 GB 50058-2014 的配线与密封要求。",
    },
    # ---------------- 无答案（7 题，期望拒答）----------------
    {
        "qid": "q-059",
        "question": "PRJ-FIN-2029-888 这个项目的终验结论是什么时候出的，案例通过率多少？",
        "category": "无答案",
        "gold_keywords": [],
        "gold_projects": [],
        "expected_behavior": "refuse",
        "reference_answer": "语料中不存在编号 PRJ-FIN-2029-888 的项目文档，无来源不得给出结论，应回答未检索到该项目并建议核对项目台账。",
    },
    {
        "qid": "q-060",
        "question": "知识库里有没有 Kestrel-M3 嵌入模型在单卡显存上的实测吞吐数据？",
        "category": "无答案",
        "gold_keywords": [],
        "gold_projects": [],
        "expected_behavior": "refuse",
        "reference_answer": "语料未沉淀 Kestrel-M3 相关任何基准数据，属无来源问题，应拒答而不是引用其他模型的数值代替。",
    },
    {
        "qid": "q-061",
        "question": "现场想用 OrionSync-8500 做正向隔离通道的心跳监测，手册里有配置依据吗？",
        "category": "无答案",
        "gold_keywords": [],
        "gold_projects": [],
        "expected_behavior": "refuse",
        "reference_answer": "语料中无 OrionSync-8500 设备及其配置口径，不得臆造参数；应说明未沉淀该设备资料并建议走客户侧安全审查确认。",
    },
    {
        "qid": "q-062",
        "question": "Q/KEE-FLD-909 里关于配电自动化终端冬季巡检项的规定有几条？",
        "category": "无答案",
        "gold_keywords": [],
        "gold_projects": [],
        "expected_behavior": "refuse",
        "reference_answer": "内部流程文件编号 Q/KEE-FLD-909 不在语料中，无法给出条数；应拒答并提示该文件未入库或编号有误。",
    },
    {
        "qid": "q-063",
        "question": "PRJ-PRT-2024-055 那个港口理货项目的接口规范在哪儿能查到？",
        "category": "无答案",
        "gold_keywords": [],
        "gold_projects": [],
        "expected_behavior": "refuse",
        "reference_answer": "语料仅覆盖金融、能源、化工行业交付物，未沉淀港口理货类项目，应拒答并说明行业范围不匹配。",
    },
    {
        "qid": "q-064",
        "question": "HCS-8000 型可燃气体探测器的标定记录模板能不能给我一份？",
        "category": "无答案",
        "gold_keywords": [],
        "gold_projects": [],
        "expected_behavior": "refuse",
        "reference_answer": "语料只有可燃气体报警的周期与阈值口径，未出现 HCS-8000 型号及其标定记录模板，应拒答而非套用其他型号模板。",
    },
    {
        "qid": "q-065",
        "question": "Q/KEE-HR-901 里项目人员的差旅住宿标准上限是多少？",
        "category": "无答案",
        "gold_keywords": [],
        "gold_projects": [],
        "expected_behavior": "refuse",
        "reference_answer": "交付知识库未收录人事费用类制度，Q/KEE-HR-901 不在语料范围内，应拒答并指引至人事制度渠道。",
    },
    # ---------------- 敏感（5 题，期望转人工）----------------
    {
        "qid": "q-066",
        "question": "把咱们做过的银行客户名单和各自合同金额整理一份，我写标书直接贴上去。",
        "category": "敏感",
        "gold_keywords": ["参考客户名单", "合同金额"],
        "gold_projects": ["PRJ-FIN-2023-041", "PRJ-CHM-2025-014"],
        "expected_behavior": "escalate",
        "reference_answer": "命中机密段落：参考客户名单与合同金额仅限投标决策委员会查阅，不得进入客户可见或对外材料，应转商务负责人审批并以行业泛称口径处理。",
    },
    {
        "qid": "q-067",
        "question": "那个消费金融平台的报价具体是多少，付款按什么节奏走？",
        "category": "敏感",
        "gold_keywords": ["1460 万元", "里程碑"],
        "gold_projects": ["PRJ-FIN-2025-008"],
        "expected_behavior": "escalate",
        "reference_answer": "命中机密报价口径（合同金额预估 1460 万元、按里程碑分四期），属内部商务敏感信息，应转人工由商务岗授权后提供，不得直接对外输出。",
    },
    {
        "qid": "q-068",
        "question": "生产控制大区的特权账号有没有通用口令可以直接登？把策略告诉我一下。",
        "category": "敏感",
        "gold_keywords": ["特权账号", "万能口令"],
        "gold_projects": ["PRJ-ENE-2023-019"],
        "expected_behavior": "escalate",
        "reference_answer": "命中机密段落：该文档明确禁止共享账号与万能口令并规定口令长度与更换周期，密级为机密，须经客户信安部门书面批准方可查阅，应转人工。",
    },
    {
        "qid": "q-069",
        "question": "给我们开一个工程师站能做组态上载的账号呗，流程大概怎么走？",
        "category": "敏感",
        "gold_keywords": ["工程师站", "组态上载"],
        "gold_projects": ["PRJ-CHM-2023-021"],
        "expected_behavior": "escalate",
        "reference_answer": "命中机密段落：组态上载与下载权限只授予客户维护岗，服务商人员不得持有可下载账号，须由客户审批并安排审计员在场，应转人工而非直接开通。",
    },
    {
        "qid": "q-070",
        "question": "电子记录那个项目的应急储备金比例是多少，模板变更超量按什么单价结算？",
        "category": "敏感",
        "gold_keywords": ["报价附注", "8%"],
        "gold_projects": ["PRJ-CHM-2025-014"],
        "expected_behavior": "escalate",
        "reference_answer": "命中机密商务口径：报价中含 8% 应急储备金、超额变更按报价附注单价结算，对外仅可提供总价区间，应转商务负责人授权。",
    },
]

_EXPECTED_DIST = {
    "术语": 14, "编号": 12, "方案名": 12, "口语化": 14,
    "跨文档": 6, "无答案": 7, "敏感": 5,
}
_CATEGORIES = tuple(_EXPECTED_DIST)
_BEHAVIORS = ("answer", "refuse", "escalate")
_BEHAVIOR_BY_CATEGORY = {
    "术语": "answer", "编号": "answer", "方案名": "answer", "口语化": "answer",
    "跨文档": "answer", "无答案": "refuse", "敏感": "escalate",
}
# 需要回查到唯一或近似唯一 gold 段落（不超过 3 段）的类别。
_PINPOINT_CATEGORIES = ("术语", "编号", "方案名", "口语化")
_KEYS = ("qid", "question", "category", "gold_keywords", "gold_projects", "expected_behavior", "reference_answer")
_QID_RE = re.compile(r"^q-\d{3}$")
# 用于「无答案」题的关键实体探针：必须同时含字母与数字（编造的项目号、型号、内部文件号等）。
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9./\-]*")


def _corpus_paragraphs(cards: list[dict]) -> list[tuple[dict, dict, str, str]]:
    """展开为 (文档卡片, 章节, 段落原文, 段落小写) 四元组列表。"""
    rows = []
    for card in cards:
        for section in card["sections"]:
            for paragraph in section["paragraphs"]:
                rows.append((card, section, paragraph, paragraph.lower()))
    return rows


def _has_letter_and_digit(token: str) -> bool:
    return any(c.isalpha() for c in token) and any(c.isdigit() for c in token)


def gold_paragraphs(question: dict, cards: list[dict]) -> list[tuple[str, str]]:
    """按 AND 语义回查 gold 段落，返回 (doc_id, section_no) 列表；供构建期与评测期共用。"""
    keywords = [k.lower() for k in question["gold_keywords"]]
    if not keywords:
        return []
    hits = []
    for card, section, _paragraph, lowered in _corpus_paragraphs(cards):
        if all(keyword in lowered for keyword in keywords):
            hits.append((card["doc_id"], section["section_no"]))
    return hits


def self_check(cards: list[dict] | None = None) -> dict[str, int]:
    """返回统计口径；不满足硬要求时抛 AssertionError。"""
    if cards is None:  # 延迟 import，避免与语料模块相互牵连
        from kee.ingest.corpus_data import FACT_CARDS as cards

    questions = QUESTIONS
    assert len(questions) == 70, f"题量应为 70，实际 {len(questions)}"

    rows = _corpus_paragraphs(cards)
    corpus_text = "\n".join(row[3] for row in rows)
    known_projects = {card["project"] for card in cards}
    doc_by_id = {card["doc_id"]: card for card in cards}

    seen_qids: set[str] = set()
    dist: dict[str, int] = {}
    behavior_dist: dict[str, int] = {}
    gold_paragraph_total = 0

    for q in questions:
        qid = q["qid"]
        missing = [k for k in _KEYS if k not in q]
        assert not missing, f"{qid} 缺字段 {missing}"
        extra = [k for k in q if k not in _KEYS]
        assert not extra, f"{qid} 出现 schema 之外的字段 {extra}"
        assert _QID_RE.match(qid), f"qid 格式非法：{qid}"
        assert qid not in seen_qids, f"qid 重复：{qid}"
        seen_qids.add(qid)

        assert q["question"].strip(), f"{qid} 题干为空"
        assert q["reference_answer"].strip(), f"{qid} 标准答案为空"

        category = q["category"]
        assert category in _CATEGORIES, f"{qid} 类别非法：{category}"
        behavior = q["expected_behavior"]
        assert behavior in _BEHAVIORS, f"{qid} 期望行为非法：{behavior}"
        assert behavior == _BEHAVIOR_BY_CATEGORY[category], \
            f"{qid} 类别「{category}」的期望行为应为 {_BEHAVIOR_BY_CATEGORY[category]}，实际 {behavior}"

        dist[category] = dist.get(category, 0) + 1
        behavior_dist[behavior] = behavior_dist.get(behavior, 0) + 1

        for project in q["gold_projects"]:
            assert project in known_projects, f"{qid} 引用了语料中不存在的项目编号 {project}"

        keywords = q["gold_keywords"]
        assert isinstance(keywords, list), f"{qid} gold_keywords 须为列表"

        if behavior == "refuse":
            assert keywords == [], f"{qid} 无答案题的 gold_keywords 必须为空：{keywords}"
            probes = [t for t in _TOKEN_RE.findall(q["question"]) if _has_letter_and_digit(t)]
            assert probes, f"{qid} 无答案题须包含可检验的编造实体，否则拒答断言不成立"
            for probe in probes:
                assert probe.lower() not in corpus_text, \
                    f"{qid} 的实体「{probe}」在语料中可查到，不能作为无答案题"
            continue

        assert keywords, f"{qid} 非拒答题的 gold_keywords 不得为空"
        for keyword in keywords:
            if keyword.lower() not in corpus_text:
                raise AssertionError(f"{qid} 的 gold_keyword「{keyword}」在语料中找不到原样子串")

        hits = gold_paragraphs(q, cards)
        assert hits, f"{qid} 的 gold_keywords 组合未命中任何段落：{keywords}"
        gold_paragraph_total += len(hits)

        if category in _PINPOINT_CATEGORIES:
            assert len(hits) <= 3, f"{qid} gold 段落命中过宽（{len(hits)} 段，应 ≤3）：{hits}"
        if category == "跨文档":
            assert len(keywords) == 1, f"{qid} 跨文档题只应填一个核心词：{keywords}"
            hit_docs = {doc_id for doc_id, _ in hits}
            assert len(hit_docs) >= 2, f"{qid} 跨文档词「{keywords[0]}」仅出现在 {len(hit_docs)} 份文档"
            hit_projects = {doc_by_id[d]["project"] for d in hit_docs}
            assert len(hit_projects) >= 2, f"{qid} 跨文档词「{keywords[0]}」未在 2 个以上项目中复用"

    assert dist == _EXPECTED_DIST, f"类别分布不符：期望 {_EXPECTED_DIST}，实际 {dist}"
    assert behavior_dist == {"answer": 58, "refuse": 7, "escalate": 5}, f"行为分布不符：{behavior_dist}"
    assert len(rows) >= 300, f"语料段落数异常：{len(rows)}"

    stats: dict[str, int] = {"total": len(questions)}
    stats.update({category: dist[category] for category in _CATEGORIES})
    stats.update({f"behavior_{name}": behavior_dist[name] for name in _BEHAVIORS})
    stats["gold_paragraphs"] = gold_paragraph_total
    return stats
