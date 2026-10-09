# Gate 0 ④ 多存储一致性验证报告（Outbox + 每日对账）

- 验证时间：2026-10-09T16:23:33+08:00｜集合 kb_chunks（jieba）｜commit 5bb82f0
- 基线对账：权威子块 193，索引 193，差异 missing=0 extra=0（0.28s）

## 注入与发现

| 注入类型 | 注入量 | 对账发现 | 发现率 |
|---|---|---|---|
| A 权威源已写、向量写失败（升版 v2） | 14 | 14 | 1.0 |
| B 索引子块被删（静默缺数据） | 5 | 5 | 1.0 |
| A∪B 去重后合计 | 19 | 19 | 1.0 |
| C 索引脏行（权威源无登记） | 1 | 1 | 1.0 |
- 误报（发现但不属于注入集）：无
- 对账耗时：0.25s；Outbox 状态分布：{"dispatched": 43}
- 缺失样本：['chm-closeout-prj-chm-2023-021@chm-closeout-prj-chm-2023-021@v1#1.1#c1', 'chm-closeout-prj-chm-2023-021@chm-closeout-prj-chm-2023-021@v1#2.1#c1', 'chm-closeout-prj-chm-2023-021@chm-closeout-prj-chm-2023-021@v1#2.1#c2', 'chm-closeout-prj-chm-2023-021@chm-closeout-prj-chm-2023-021@v1#3.1#c1', 'chm-closeout-prj-chm-2023-021@chm-closeout-prj-chm-2023-021@v1#4.1#c1']

## 修复与复检

- 补偿事件：4 条，重投成功 19 条，脏行清理 1 条（92.63s）
- 复检：missing=0，extra=0，权威子块 220，索引 220

## 版本回滚演练（本次验证的收尾）

- 撤销验证用的 v2 版本：['chm-closeout-prj-chm-2023-021@v2', 'chm-closeout-prj-chm-2024-003@v2', 'chm-compliance-prj-chm-2023-021@v2']
- 回滚后对账：missing=0，extra=0，权威子块 193，索引 193

## 审计不可篡改

- UPDATE audit_log：被触发器拒绝｜audit_log 为追加表，禁止 UPDATE 操作
CONTEXT:  PL/pgSQL function kee_forbid_audit_mutation() line 3 at RAISE
- DELETE audit_log：被触发器拒绝
- 审计行数：24

- **判定**：通过｜门禁：人为注入的不一致能被发现并修复；不出现静默缺数据｜实测：发现率 1.0（19/19）+ 脏行 1；修复后 missing/extra 归零=True｜说明：Outbox 注入失败为受控故障注入，非生产行为

## 一致性机制参数（④出口）

- Outbox 事件粒度：一份文档一个版本按「批次」拆分（本次验证用两半批，以暴露部分失败）。
- 重试：指数退避，base 0.2s，最多 5 轮；retry_count≥5 落入死信（需人工介入）。
- 对账：以权威源子块集合为基准做双向差集；缺失走 Outbox 重投，多余走索引删除。
- 修复不依赖 GPU：向量物料仓（派生物）+ 权威源即可完成索引重建。
