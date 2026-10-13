/** 类名 token：组件与测试都从这里取，**不在两边各写一遍字符串**。
 *
 * 这么做的理由很具体：`provisional` / `confirmed` / `retracted` 三个态是流式渲染契约的
 * 可验证面（M4-07 要用 DOM 断言钉住「未完成态与最终答案样式不同」「撤回不得静默删除」）。
 * 如果类名散落在 JSX 里，重命名时测试会跟着瞎掉——那正是「断言看起来绿、其实什么都没钉住」。
 */
export const styles = {
  shell: "kee-shell",
  brand: "kee-brand",
  tabs: "kee-tabs",
  tab: "kee-tab",
  tabOn: "kee-tab--on",
  panel: "kee-panel",

  identity: "kee-identity",
  field: "kee-field",
  chips: "kee-chips",
  chip: "kee-chip",
  chipOn: "kee-chip--on",
  link: "kee-link",
  scopeNote: "kee-scope-note",

  authWrap: "kee-auth",
  authCard: "kee-auth__card",
  authTabs: "kee-auth__tabs",
  authForm: "kee-auth__form",
  authHint: "kee-auth__hint",
  authError: "kee-auth__error",

  row: "kee-row",
  muted: "kee-muted",
  formMsg: "kee-form-msg",

  health: "kee-health",
  healthOk: "kee-health--ok",
  healthBad: "kee-health--bad",

  ask: "kee-ask",
  askDock: "kee-ask-dock",
  askForm: "kee-ask-form",
  turns: "kee-turns",
  turn: "kee-turn",
  turnQuestion: "kee-turn__question",
  turnAnswer: "kee-turn__answer",

  /** 三态：未完成（provisional 增量）/ 已确认（最终答案）/ 已撤回 */
  provisional: "kee-state--provisional",
  confirmed: "kee-state--confirmed",
  retracted: "kee-state--retracted",
  retractedMark: "kee-retraction",

  conclusion: "kee-conclusion",
  points: "kee-points",
  point: "kee-point",
  cites: "kee-cites",
  cite: "kee-cite",
  badge: "kee-badge",
  badgeWarn: "kee-badge--warn",
  badgeRefuse: "kee-badge--refuse",
  conf: "kee-conf",
  meta: "kee-meta",
  error: "kee-error",

  table: "kee-table",
  toolbar: "kee-toolbar",
  btn: "kee-btn",
  btnPrimary: "kee-btn--primary",
  btnDanger: "kee-btn--danger",
  empty: "kee-empty",
  /** 引用高亮：点/悬停某个 evidence_id 时，引用它的要点一起亮——这是「引用可核对」的界面兑现 */
  hit: "kee-hit",
} as const;

/** tab 上「选中态」用同一个 token 拼，避免各处写法不一。 */
export const tabClass = (active: boolean) => `${styles.tab}${active ? ` ${styles.tabOn}` : ""}`;
