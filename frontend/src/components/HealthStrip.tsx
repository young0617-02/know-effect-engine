import { useEffect, useState } from "react";
import { getHealth, type Health } from "../api/client";
import { styles as s } from "../styles";

/** 依赖状态条：把「服务活着」和「依赖挂了」分开显示。
 *
 * 后端刻意让 /healthz 在依赖挂掉时仍返回 200 + degraded 结构（启动期不硬连依赖），
 * 所以这里不能只看 HTTP 码，要看 deps.*.ok——否则 Milvus 宕机时界面会是一片绿。
 */
export function HealthStrip() {
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let alive = true;
    getHealth()
      .then((h) => alive && setHealth(h))
      .catch((e: unknown) => alive && setError(e instanceof Error ? e.message : String(e)));
    return () => {
      alive = false;
    };
  }, []);

  if (error) {
    return (
      <p className={`${s.health} ${s.healthBad}`}>
        无法读取依赖状态：{error}
      </p>
    );
  }
  if (!health) return <p className={s.health}>正在读取依赖状态…</p>;

  const deps = Object.entries(health.deps ?? {});
  return (
    <p className={`${s.health} ${health.status === "ok" ? s.healthOk : s.healthBad}`}>
      <span>依赖：{health.status === "ok" ? "全部就绪" : "有降级"}</span>
      {deps.map(([name, d]) => (
        <span key={name} className={s.badge}>
          {name} {d.ok ? "OK" : `挂（${d.error ?? "?"}）`}
        </span>
      ))}
    </p>
  );
}
