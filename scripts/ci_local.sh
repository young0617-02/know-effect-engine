#!/usr/bin/env bash
# 本地一条命令复验：把「我手动按顺序跑八支脚本」这件事写成机制。
#
# 这不是 CI 服务。这个仓库**没有 remote**（用户定过不推），而且被测系统绑着本机
# D 盘的 CUDA 权重、Ollama、Milvus 与宿主回环端口——搬到第三方 runner 上既跑不动，
# 也会把带业务原文的测试语料送出域外。所以这里做的是**本机**的强制点：
# 一条命令、顺序固定、失败即停、产物留档、**服务状态复原**。
#
# 用法：
#   scripts/ci_local.sh                      # 全量（约 12~18 分钟，含检索回归复跑与两轮真机问答）
#   scripts/ci_local.sh --quick              # 只跑不依赖 GPU 的四段（约 1 分钟）
#   scripts/ci_local.sh --only static,backend
#   scripts/ci_local.sh --only eval          # 只检检索回归（会先停宿主 API，跑完按原状起回）
#   scripts/ci_local.sh --skip obs-probe     # 观测栈没起的时候
#   scripts/ci_local.sh --keep-going         # 失败不中断，最后汇总
#   scripts/ci_local.sh --regen-reports      # 允许重写 M5 三件套（默认不重写，见下）
#   scripts/ci_local.sh --with-test-user     # 允许跑 m5_test_user_run（会刷新带漂移的业务读数）
#
# 四条设计约束，都是这个项目踩过的坑：
#   1. **GPU 互斥**：8GB 单卡零余量。`verify_m1` / `verify_m3` 在**自己进程内**加载嵌入与重排，
#      宿主 API 若同时占着模型，症状不是报错而是**静默退化**（曾把检索量纲换成 BM25 的几十，
#      报出 36.9408 这种"过闸"读数）。所以这两段之前必须先停 API、跑完再按原状起回来。
#   2. **证据产物不许被悄悄重写**：默认不跑那三个"生成报告"的脚本；即使用 --regen-reports，
#      也会先记 md5、跑完对比并打印 diff 清单——"报告又变好看了"这种事必须当场看得见。
#   3. **人工标注工单只读**：脚本清单里刻意不含任何会写 `var/gate0/*_fill.csv` 的作业。
#   4. **重启之后必须预热**：正因为 1 要把 API 停起来再起回，HTTP 那三段拿到的进程是冷的。
#      8GB 卡上 Ollama 常驻 4.7GB，第一个问答还要装嵌入+重排，实测 >180s 无首字节——
#      `verify_m4` 与观测探针会把它读成失败（2026-10-16 全量第一次就是这么三连红的）。
#      所以 `scripts/api_warm.py` 先付这笔账；它没过就**跳过**后面三段，而不是让它们带病出数。
set -uo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-.venv/Scripts/python.exe}
API_PORT=${KEE_API_PORT:-28000}
# 必须 export：`infra/docker-compose.m4.yml` 的 nginx 模板取的是 `${KEE_API_PORT:-38000}`，
# 而 serve_api / obs_targets 的默认也是 38000。这里不导出，冷启动那一次 compose up 会把
# 反代渲染成 38000，而宿主 API 在 28000 —— 症状是 38080 整条同源通路 502（不是配置错，是两处默认值不同步）。
export KEE_API_PORT=$API_PORT
WEB_PORT=${KEE_WEB_PORT:-38080}
PROM_URL=${KEE_PROM_URL:-http://127.0.0.1:39090}
RUN_ID=$(date +%Y%m%d-%H%M%S)
OUT=var/ci/$RUN_ID
KEEP_GOING=0
REGEN=0
WITH_TU=0
STAGES=""
SKIP=""

ALL_STAGES="preflight static backend frontend verify-m1 verify-m3 eval verify-m4 obs-probe test-user reports"

while [ $# -gt 0 ]; do
  case "$1" in
    --quick) STAGES="preflight,static,backend,frontend" ;;
    --only) shift; STAGES="$1" ;;
    --skip) shift; SKIP="$1" ;;
    --keep-going) KEEP_GOING=1 ;;
    --regen-reports) REGEN=1 ;;
    --with-test-user) WITH_TU=1 ;;
    --list) echo "$ALL_STAGES" | tr ' ' '\n'; exit 0 ;;
    *) echo "未知参数：$1（--list 看阶段）"; exit 2 ;;
  esac
  shift
done

mkdir -p "$OUT"
SUMMARY="$OUT/summary.tsv"
: > "$SUMMARY"
FAILED=0
declare -a FAILED_NAMES=()
RAN=0

in_list() { case ",$1," in *",$2,"*) return 0 ;; *) return 1 ;; esac; }

# 阶段名拼错不许静默变成「一个都没跑然后报通过」——这个函数第一版就是这么假绿的：
# `case " $1 " in *" $2 "*` 要求两边各有一个空格，而逗号分隔的列表里没有 ⇒ 永远不匹配
# ⇒ `--quick` 跑了 0 个阶段还打印「全部通过」。下面既有正确的匹配，也有 RAN 计数兜底。
wanted() {
  local name=$1
  if [ -n "$SKIP" ] && in_list "$SKIP" "$name"; then return 1; fi
  if [ -z "$STAGES" ]; then return 0; fi
  in_list "$STAGES" "$name"
}

validate_names() {
  local list=$1 bad=0 n
  [ -z "$list" ] && return 0
  for n in ${list//,/ }; do
    if ! in_list "${ALL_STAGES// /,}" "$n"; then echo "未知阶段名：$n（--list 看可选）" >&2; bad=1; fi
  done
  return $bad
}

validate_names "$STAGES" || exit 2
validate_names "$SKIP" || exit 2

# 只跑一次、跑完就退出的阶段用 stage；需要"先决条件不满足就跳过"的用 skip_stage
LAST_RC=0
stage() {
  local name=$1; shift
  if ! wanted "$name"; then return 0; fi
  local log="$OUT/$name.log" t0 t1 rc
  printf '%-42s' "$name"
  t0=$(date +%s)
  "$@" >"$log" 2>&1
  rc=$?
  LAST_RC=$rc
  t1=$(date +%s)
  printf '%-14s %4ss  %s\n' "$([ $rc -eq 0 ] && echo PASS || echo "FAIL(rc=$rc)")" "$((t1 - t0))" "$log"
  printf '%s\t%s\t%s\t%s\n' "$name" "$rc" "$((t1 - t0))" "$log" >> "$SUMMARY"
  RAN=$((RAN + 1))
  if [ $rc -ne 0 ]; then
    FAILED=1; FAILED_NAMES+=("$name")
    tail -25 "$log" | sed 's/^/      │ /'
    [ $KEEP_GOING -eq 1 ] || { echo; echo "!! $name 失败即停（--keep-going 可跑完再汇总）；日志在 $OUT/"; exit 1; }
  fi
}

say() { printf '\n\033[1m%s\033[0m\n' "$1"; }

# ---------- 服务与端口 ----------
port_pid() { netstat -ano 2>/dev/null | grep -E ":$1\s" | grep LISTENING | awk '{print $5}' | head -1; }

kill_tree() { # venv 的 python.exe 是 trampoline：只杀 netstat 那个 PID 会留下活着的服务
  local pid=$1
  [ -z "$pid" ] && return 0
  MSYS_NO_PATHCONV=1 taskkill /PID "$pid" /T /F >/dev/null 2>&1
}

API_WAS_UP=0
api_up() {
  local pid; pid=$(port_pid "$API_PORT")
  if [ -n "$pid" ]; then return 0; fi
  KEE_API_PORT=$API_PORT nohup "$PY" -u scripts/serve_api.py > "$OUT/api.log" 2>&1 &
  for _ in $(seq 1 40); do
    sleep 1
    curl -sf "http://127.0.0.1:$API_PORT/healthz" >/dev/null 2>&1 && return 0
  done
  echo "宿主 API 起不来（$OUT/api.log）" >&2; return 1
}

api_down() { kill_tree "$(port_pid "$API_PORT")"; sleep 2; }

# 预热不是可选阶段，所以**不走 stage()/wanted()**：`--only verify-m4` 时它也必须跑，
# 否则第一个真 HTTP 读数又打在冷进程上（2026-10-16 那次三连败就是这么来的）。
WARM_OK=0
api_warm() {
  local log="$OUT/warm.log" t0 t1 rc
  printf '%-42s' "warm（冷装载预热）"
  t0=$(date +%s)
  "$PY" -u scripts/api_warm.py --api "http://127.0.0.1:$API_PORT" >"$log" 2>&1
  rc=$?
  t1=$(date +%s)
  printf '%-14s %4ss  %s\n' "$([ $rc -eq 0 ] && echo PASS || echo "FAIL(rc=$rc)")" "$((t1 - t0))" "$log"
  printf '%s\t%s\t%s\t%s\n' "warm" "$rc" "$((t1 - t0))" "$log" >> "$SUMMARY"
  RAN=$((RAN + 1))
  if [ $rc -ne 0 ]; then
    tail -6 "$log" | sed 's/^/      │ /'
    echo "      ↑ 预热没成：后面的真 HTTP 验收会被跳过，因为冷进程上的读数不是失败、是没测"
    WARM_OK=0; FAILED=1; FAILED_NAMES+=("warm")
    [ $KEEP_GOING -eq 1 ] || { echo; echo "!! warm 失败即停；日志在 $OUT/"; exit 1; }
  else
    WARM_OK=1
    grep -E "预热完成|热态复打" "$log" | sed 's/^/      · /'
  fi
}

cleanup() {
  local rc=$?
  if [ "$API_WAS_UP" -eq 0 ]; then
    kill_tree "$(port_pid "$API_PORT")" >/dev/null 2>&1
    echo "  复原：宿主 API 起跑前不在运行，已停掉本次起的实例（端口 $API_PORT）"
  else
    api_up >/dev/null 2>&1 && echo "  复原：宿主 API 已按原状起回 $API_PORT"
  fi
  exit $rc
}
trap cleanup EXIT

# ---------- 会被重写的证据产物：跑前记 md5，跑后对比 ----------
EVIDENCE="var/gate0/m1_verify.json var/gate0/m3_verify.json var/gate0/m4_verify.json
var/gate0/m5_dod.json var/gate0/m5_versions.json var/gate0/m5_risk_register.json
var/gate0/m5_observability.json var/gate0/m2_eval.json"
MD5_BEFORE="$OUT/md5_before.txt"
md5_snapshot() {
  : > "$MD5_BEFORE"
  for f in $EVIDENCE; do [ -f "$f" ] && md5sum "$f" >> "$MD5_BEFORE"; done
}

say "本地复验 $RUN_ID   阶段：${STAGES:-全部}${SKIP:+   跳过：$SKIP}"
echo "  产物目录 $OUT/   python=$PY   API 端口=$API_PORT"
API_WAS_UP=0; [ -n "$(port_pid "$API_PORT")" ] && API_WAS_UP=1
echo "  宿主 API 起跑前：$([ $API_WAS_UP -eq 1 ] && echo 在跑 || echo 未运行)（结束时会复原）"
md5_snapshot

# ---------- 1. 前置事实核查 ----------
if wanted preflight; then
  say "1 前置：torch/CUDA、权重目录、Milvus/PG/Redis、Ollama 模型、容器健康"
  stage preflight "$PY" -u scripts/check_stack.py
fi

# ---------- 2. 静态 ----------
if wanted static; then
  say "2 静态：能编译、能解析、配置合法（今天两次事故都是这一层能拦住、但当时没有这一层的形状）"
  stage static "$PY" -u scripts/ci_static.py
fi

# ---------- 3. 后端单测 ----------
if wanted backend; then
  say "3 后端单测（进程内 TestClient，需要存储栈；不占 GPU，可与 API 并存）"
  stage backend "$PY" -m pytest -q
fi

# ---------- 4. 前端 ----------
if wanted frontend; then
  say "4 前端：单测 + 构建（构建产物是 kee-web 挂载的那份 dist，不 build 就等于没改）"
  stage frontend bash -c 'cd frontend && npm run test && npm run build'
fi

# ---------- 5/6/7. 需要独占 GPU 的三支 ----------
if wanted verify-m1 || wanted verify-m3 || wanted eval; then
  say "5 在线链路验收 + 检索回归（进程内加载模型 ⇒ 先停宿主 API，8GB 零余量）"
  api_down
  stage verify-m1 "$PY" -u scripts/verify_m1.py
  stage verify-m3 "$PY" -u scripts/verify_m3.py
  # 检索侧的**版本回归**：只跑基线（--skip-grid），与 m2_eval_prev.json 同尺子对照。
  # 退出码 0=门禁过且无回归／1=门禁没过／2=同尺子 strict Recall@5 掉超 0.02（回归）。
  # 读数写到本次运行目录：默认路径那份是带 dev 网格的历史证据，`--skip-grid` 写回去等于抹掉 grid_dev。
  if wanted eval; then
    stage eval "$PY" -u scripts/m2_eval.py --skip-grid --out "$OUT/m2_eval.json"
    if [ "${LAST_RC:-0}" -eq 2 ]; then
      echo "      ↑ rc=2 是**回归**（同尺子比出来的跌幅），不是门禁没过；两者的修法不一样"
    elif [ "${LAST_RC:-0}" -eq 1 ]; then
      echo "      ↑ rc=1 = 四条检索门禁里有条没过了（判定行见日志「判定:」）"
    fi
  fi
fi

# ---------- 7. 真 HTTP 通路 ----------
if wanted verify-m4 || wanted obs-probe || wanted test-user; then
  say "6 走生产入口（真 HTTP + Bearer 令牌，与界面同一条路）"
  api_up || exit 1
  (cd frontend && npm run build) >/dev/null 2>&1 || true
  api_warm
  [ $WARM_OK -eq 0 ] && echo "  下面三段全部跳过：预热未成，冷进程上的问答读数只会伪装成失败"
  if [ $WARM_OK -eq 1 ] && wanted verify-m4; then
    docker compose -f infra/docker-compose.m4.yml up -d >/dev/null 2>&1 || true
    KEE_BASE="http://127.0.0.1:$API_PORT" stage verify-m4-direct "$PY" -u scripts/verify_m4.py
    if curl -sf "http://127.0.0.1:$WEB_PORT/" >/dev/null 2>&1; then
      KEE_BASE="http://127.0.0.1:$WEB_PORT" stage verify-m4-web "$PY" -u scripts/verify_m4.py
    else
      echo "  跳过 verify-m4-web：$WEB_PORT 没起（经 Nginx 那 13 项没验到，不等于通过）"
    fi
  fi
  if [ $WARM_OK -eq 1 ] && wanted obs-probe; then
    if curl -sf "$PROM_URL/-/healthy" >/dev/null 2>&1; then
      stage obs-probe "$PY" -u scripts/m5_observability_probe.py --api "http://127.0.0.1:$API_PORT" --prom "$PROM_URL"
    else
      echo "  跳过 obs-probe：Prometheus 不在 $PROM_URL（起法见 infra/docker-compose.observability.yml）"
    fi
  fi
  if [ $WARM_OK -eq 1 ] && wanted test-user; then
    if [ $WITH_TU -eq 1 ]; then
      KEE_TU_BASE="http://127.0.0.1:$API_PORT" stage test-user "$PY" -u scripts/m5_test_user_run.py
    else
      echo "  跳过 test-user：它会刷新那份带 LLM 漂移的业务读数产物（要跑加 --with-test-user）"
    fi
  fi
fi

# ---------- 8. M5 三件套（默认不动） ----------
if wanted reports; then
  say "7 M5 三件套（重写 tracked 报告，默认跳过）"
  if [ $REGEN -eq 1 ]; then
    stage reports "$PY" -u scripts/m5_dod_check.py
    "$PY" -u scripts/m5_version_lock.py >> "$OUT/reports.log" 2>&1 && echo "   version_lock ok" >> "$OUT/reports.log"
    "$PY" -u scripts/m5_risk_register.py >> "$OUT/reports.log" 2>&1 && echo "   risk_register ok" >> "$OUT/reports.log"
  else
    echo "  跳过（加 --regen-reports 才会重生成；它们会改 reports/m5/*.md 与 var/gate0/*.json）"
  fi
fi

# ---------- 汇总 ----------
say "汇总"
if [ "$RAN" -eq 0 ]; then
  echo "!! 一个阶段都没跑（阶段名拼错，或全被 --skip 掉）——这不是通过，是没测。" >&2
  exit 1
fi
echo "  实际执行阶段数：$RAN"
printf '%-22s %-10s %6s\n' 阶段 结果 秒
while IFS=$'\t' read -r name rc sec log; do
  printf '%-22s %-10s %6s\n' "$name" "$([ "$rc" = 0 ] && echo PASS || echo "FAIL($rc)")" "$sec"
done < "$SUMMARY"

CHANGED=0
for f in $EVIDENCE; do
  [ -f "$f" ] || continue
  # Windows/Git Bash 的 md5sum 输出是「<hash> *<路径>」（二进制模式带星号），
  # 早先用 grep -F " $f" 找的是「空格+路径」⇒ 永远匹配不到 ⇒ old 恒空 ⇒ **无论有没有变都报「无变化」**。
  # 这条守卫的全部意义就是「报告又变好看了必须当场看得见」，它自己假绿比没有这条守卫更糟。
  old=$(awk -v p="$f" '{f=$2; sub(/^\*/,"",f); if (f==p) {print $1; exit}}' "$MD5_BEFORE" 2>/dev/null)
  new=$(md5sum "$f" | awk '{print $1}')
  if [ -z "$old" ]; then
    echo "  证据产物不在起跑快照里（本次新增）：$f"; CHANGED=1
  elif [ "$old" != "$new" ]; then
    echo "  证据产物已变化：$f"; CHANGED=1
  fi
done
[ $CHANGED -eq 0 ] && echo "  证据产物 md5：无变化"
# 工作区噪声只报数量：本次到底改没改证据，上面那段 md5 已经回答了，
# 列 20 行既有改动只会把关键那一行挤出屏幕。
echo "  工作区当前有 $(git status --short | wc -l | tr -d ' ') 个未提交改动（含本次之前的；本次改了哪些看上面的 md5）"

if [ $FAILED -eq 0 ]; then
  echo; echo "== 全部通过。日志与清单在 $OUT/ =="; exit 0
fi
echo; echo "== 失败阶段：${FAILED_NAMES[*]}   详见 $OUT/ =="; exit 1
