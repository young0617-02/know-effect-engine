/** 角色表：**只用于界面呈现，不是权限实现**。
 *
 * 强制点在检索层（`kee/retrieval/rbac.py:build_filter_expr` 把 security_level/industry/project
 * 拼进 Milvus 过滤表达式），服务端另有写入/复核的角色门槛。这份镜像若与后端漂移，
 * 后果只是「按钮该不该显示」，不会有越权——所以这里刻意不做任何判断逻辑，
 * 所有越权都由服务端返回 403、前端把理由原样显示出来。
 *
 * 来源：`kee/retrieval/rbac.py` ROLE_MAX_LEVEL、`kee/api/app.py` REVIEW_ROLES 与 upload 的角色白名单。
 */

export interface RoleSpec {
  role: string;
  label: string;
  maxLevel: number;
  canWrite: boolean;
  canReview: boolean;
}

export const ROLES: RoleSpec[] = [
  { role: "delivery_engineer", label: "交付工程师", maxLevel: 2, canWrite: true, canReview: false },
  { role: "presales_architect", label: "售前架构师", maxLevel: 2, canWrite: true, canReview: false },
  { role: "pm", label: "项目经理", maxLevel: 3, canWrite: true, canReview: false },
  { role: "ops_support", label: "运维支持", maxLevel: 2, canWrite: false, canReview: false },
  { role: "newcomer", label: "新人", maxLevel: 1, canWrite: false, canReview: false },
  { role: "knowledge_owner", label: "知识 Owner", maxLevel: 3, canWrite: true, canReview: true },
  { role: "system_admin", label: "系统管理员", maxLevel: 3, canWrite: true, canReview: true },
];

export const roleSpec = (role: string): RoleSpec | undefined => ROLES.find((r) => r.role === role);

export const INDUSTRIES = ["金融", "能源", "化工"];

export const SECURITY_LEVEL_LABEL: Record<number, string> = {
  1: "内部",
  2: "机密",
  3: "绝密（强制转人工）",
};
