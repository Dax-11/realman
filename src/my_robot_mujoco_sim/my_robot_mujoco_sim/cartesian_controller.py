from __future__ import annotations

import numpy as np
import mujoco


class CartesianIKController:
    def __init__(self, model: mujoco.MjModel, site_name: str = "gripper_tcp"):
        # 保存 MuJoCo 模型引用，后续 Jacobian、关节范围、位姿计算都依赖它。
        self.model = model
        self.site_name = site_name

        # 查找末端 site 的 ID。IK 的所有几何计算都以这个 site 为目标点。
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self.site_id < 0:
            raise ValueError(f"Site '{site_name}' not found in MuJoCo model.")

        # 找到机械臂第一个关节 `joint1` 在 qpos / qvel 中的起始地址。
        # 这样可以把整个 7 轴关节块切出来，而不必依赖硬编码偏移。
        joint1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint1")
        self.qpos_start = model.jnt_qposadr[joint1_id] if joint1_id >= 0 else 0
        self.dof_start = model.jnt_dofadr[joint1_id] if joint1_id >= 0 else 0

        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))

    def solve_ik(
        self,
        data: mujoco.MjData,
        target_pos: np.ndarray,
        target_quat: np.ndarray | None = None,
        q_current: np.ndarray | None = None,
        damping: float = 0.05,
        max_steps: int = 150,
        tol: float = 1e-4,
    ) -> np.ndarray:
        # 先保存完整仿真状态，避免 IK 计算污染外部调用方的 MuJoCo 数据。
        original_qpos = data.qpos.copy()

        # 如果调用方给了初始关节猜测，就先把它写入 MuJoCo 状态，作为 IK 初值。
        if q_current is not None:
            data.qpos[self.qpos_start : self.qpos_start + 7] = q_current
            mujoco.mj_forward(self.model, data)

        # 当前迭代关节向量（只取 7 轴机械臂部分）。
        q = data.qpos[self.qpos_start : self.qpos_start + 7].copy()

        # 记录“当前最好”的候选解。
        # 即便没在 max_steps 内达到 tol，也能返回误差最小的那一个。
        best_q = q.copy()
        best_err_metric = float("inf")

        for step in range(max_steps):
            # 计算当前位形下末端的几何 Jacobian。
            # mj_jacSite 会把 site 对全部自由度的雅可比写入 jacp / jacr。
            mujoco.mj_jacSite(self.model, data, self.jacp, self.jacr, self.site_id)

            # 只保留机器人手臂相关的 7 个自由度。
            # 对于本项目来说，后面的夹爪自由度通常不参与末端 IK 主体求解。
            Jp = self.jacp[:, self.dof_start : self.dof_start + 7]
            Jr = self.jacr[:, self.dof_start : self.dof_start + 7]

            # 当前末端位置与目标位置的差值。
            curr_pos = data.site_xpos[self.site_id]
            err_pos = target_pos - curr_pos

            if target_quat is not None:
                # 同时考虑位置和姿态时，把位置 Jacobian 和旋转 Jacobian 纵向拼接。
                J = np.vstack((Jp, Jr))

                # 取当前末端姿态矩阵并转成四元数。
                # 这里使用 MuJoCo 的矩阵转四元数接口，以保证格式和内部约定一致。
                curr_quat = np.zeros(4)
                mujoco.mju_mat2Quat(curr_quat, data.site_xmat[self.site_id])

                # 计算相对旋转：q_rel = q_target * inverse(q_current)
                # 这样得到的是从当前姿态旋到目标姿态的“误差旋转”。
                curr_quat_inv = np.array([curr_quat[0], -curr_quat[1], -curr_quat[2], -curr_quat[3]])
                q_rel = np.zeros(4)
                mujoco.mju_mulQuat(q_rel, target_quat, curr_quat_inv)

                # 四元数的符号存在双覆盖问题（q 和 -q 表示同一旋转）。
                # 这里取标量部分的符号，保证走“更短”的旋转路径。
                sign = 1.0 if q_rel[0] >= 0 else -1.0
                err_rot = 2.0 * sign * q_rel[1:4]

                # 姿态误差在这里被缩小了 0.2 倍，避免在混合误差中压过位置误差。
                # 这样更适合末端抓取 / 对位类任务。
                err = np.concatenate((err_pos, err_rot * 0.2))

                # 用位置误差 + 缩放后的姿态误差作为本轮质量指标。
                err_metric = np.linalg.norm(err_pos) + np.linalg.norm(err_rot) * 0.2
                if err_metric < best_err_metric:
                    best_err_metric = err_metric
                    best_q = q.copy()

                # 同时满足位置和姿态的阈值，则提前结束迭代。
                if np.linalg.norm(err_pos) < tol and np.linalg.norm(err_rot) < tol:
                    break
            else:
                # 只求位置 IK 时，目标就是把末端点拉到指定坐标。
                J = Jp
                err = err_pos

                err_metric = np.linalg.norm(err_pos)
                if err_metric < best_err_metric:
                    best_err_metric = err_metric
                    best_q = q.copy()

                if np.linalg.norm(err_pos) < tol:
                    break

            # 阻尼最小二乘更新公式：
            # dq = J^T (J J^T + λ^2 I)^-1 e
            # 它相当于“带正则项的伪逆”，能在接近奇异位形时保持稳定。
            m = J.shape[0]
            reg = damping**2 * np.eye(m)
            dq = J.T @ np.linalg.solve(J @ J.T + reg, err)

            # 限制单步更新幅度，防止一次迭代跳得太大。
            # 这对于真实机器人或高刚度仿真都很重要。
            max_step_size = 0.08
            norm_dq = np.linalg.norm(dq)
            if norm_dq > max_step_size:
                dq = (dq / norm_dq) * max_step_size

            # 把本轮增量加到关节向量上。
            q += dq

            # 关节限位裁剪：防止 IK 解跑出硬件/模型的可达范围。
            # 这里从 joint1 开始截取 7 个关节的范围进行裁剪。
            joint1_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "joint1")
            jnt_idx = joint1_id if joint1_id >= 0 else 0
            q = np.clip(
                q,
                self.model.jnt_range[jnt_idx : jnt_idx + 7, 0],
                self.model.jnt_range[jnt_idx : jnt_idx + 7, 1],
            )

            # 将新关节值写回 MuJoCo，并前向更新，以便下一轮重新计算 Jacobian。
            data.qpos[self.qpos_start : self.qpos_start + 7] = q
            mujoco.mj_forward(self.model, data)

        # 计算完毕后，恢复原始仿真状态。
        # 这使得 solve_ik 变成“纯计算函数”，不会改变外部状态。
        data.qpos[:] = original_qpos
        mujoco.mj_forward(self.model, data)

        return best_q


def minimum_jerk_interpolate(
    start_pos: np.ndarray,
    end_pos: np.ndarray,
    current_time: float,
    total_duration: float,
) -> np.ndarray:

    if current_time >= total_duration:
        return end_pos.copy()
    if current_time <= 0.0:
        return start_pos.copy()

    # 将时间归一化到 [0, 1] 区间。
    t = current_time / total_duration

    # 五次多项式的最小 jerk 轨迹：
    # s(t) = 10t^3 - 15t^4 + 6t^5
    # 它满足：
    # - s(0)=0, s(1)=1
    # - 起点/终点速度为 0
    # - 起点/终点加速度为 0
    s = 10.0 * (t ** 3) - 15.0 * (t ** 4) + 6.0 * (t ** 5)

    return start_pos + s * (end_pos - start_pos)
