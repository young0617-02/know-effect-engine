⑤ 探针重跑前留档（#56 净环境复测批，2026-10-12）
为什么必须先留：answer_correct_fill.csv 那 67 行人判标签是按 m3_confidence_probe.json 的 conclusion 逐条判的；
探针一重跑就会覆盖那份源，标签与结论的对应关系会断——这与本项目犯过两次的『同名覆盖证据』是同一类事故。
同时留 session_turn 快照（若存在）与两份探针产物，便于对比冷热。
