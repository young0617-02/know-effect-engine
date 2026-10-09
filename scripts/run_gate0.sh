#!/usr/bin/env bash
# Gate 0 全量复跑：按依赖顺序执行六个验证项并出汇总判定。
# 任一项失败不中断后续（除 ② 建库失败外），最终汇总按方案 3.3 规则判定。
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python.exe}
G=src/kee/gate0
LOG=${LOG:-var/gate0/run.log}
mkdir -p var/gate0

echo "== Gate 0 复跑开始 $(date -Iseconds) ==" | tee "$LOG"

# 存储栈必须在跑
docker compose -f infra/docker-compose.gate0.yml up -d --wait --wait-timeout 240 || {
  echo "!! 存储栈未就绪，终止"; exit 2;
}

run() {
  local name=$1; shift
  echo "-- $name" | tee -a "$LOG"
  "$PY" "$@" >>"$LOG" 2>&1
  local rc=$?
  echo "   rc=$rc" | tee -a "$LOG"
  return $rc
}

# ② 含首次建库，失败则后面都无意义
run "② 中文 BM25（含建库）" "$G/g02_chinese_bm25.py" || { echo "!! 建库失败，终止"; exit 1; }
run "③ 检索消融" "$G/g03_ablation.py" || true
run "① 显存调度" "$G/g01_vram_scheduler.py" --n 8 || true
run "④ 一致性" "$G/g04_consistency.py" || true
run "⑤ 引用强制与校准" "$G/g05_citation_calibration.py" || true
run "⑥ 单机本地化" "$G/g06_localization.py" --isolation || true
run "汇总判定" "$G/g00_summary.py" || true

echo "== 报告目录 reports/gate0/ ==" | tee -a "$LOG"
ls -1 reports/gate0 | tee -a "$LOG"
