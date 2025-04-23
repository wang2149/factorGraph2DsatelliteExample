#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2-D 卫星轨迹模拟器（方位角 θ   +   线速度 v_x、v_y 可选）
------------------------------------------------------------------
生成四个数据文件
  1) slide_example.npz                  仅 θ   测量
  2) orbit_run_restricted.npz           仅 θ   测量（不含 truth，给图优化用）
  3) slide_example_velocity.npz         θ + v  测量
  4) orbit_run_restricted_velocity.npz  θ + v  测量（不含 truth）
每个文件字段见下方 np.savez 调用处的注释。

⚠️ 关键修正
-----------
• *过程噪声 Q 只加到 truth 的副本*，**绝不**回写 curr_x！
  这样下一步积分仍然沿“干净轨迹”进行，与 EKF / 图优化假设匹配。
"""
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from math import pi, atan2

# ---------------- 全局参数 ----------------
G_E       = 3.986e14                  # 地心引力常数   m³/s²
dt        = 15.0                      # 观测间隔 (s)
num_steps = 1_800                     # 共计 7.5 h

# 测量噪声
R_th = (0.2 * pi / 180.0)**2          # θ 方差   (rad²) ≈ 0.2°
σ_v  = 10.0                           # 速度噪声标准差  m/s
R_v  = np.diag([σ_v**2, σ_v**2])      # v_x,v_y 方差矩阵

# 过程噪声（随机游走模型）
Q = np.diag([0.01, 0.01, 0.0025, 0.0025])   # x,y,vx,vy

# 初始状态 & 协方差
x0 = np.array([0.0, 20_000e3, 4_500.0, 0.0])  # 20 000 km 圆轨道附近
P0 = np.diag([1e6**2, 3e6**2, 800.0**2, 800.0**2])

# ---------------- 测量函数 ----------------
def meas_theta(x: np.ndarray) -> float:
    return atan2(x[1], x[0]) + np.sqrt(R_th) * np.random.randn()

def meas_velocity(x: np.ndarray) -> np.ndarray:
    return x[2:] + la.cholesky(R_v, lower=True) @ np.random.randn(2)

# ---------------- 动力学积分准备 ----------------
dt_div   = 150                        # 把 15 s 切成 0.1 s 小步
sub_dt   = dt / dt_div
F_small  = np.eye(4)
F_small[:2, 2:] = np.eye(2) * sub_dt  # x,y ← + vx,vy * sub_dt
Q_small  = Q * sub_dt                 # 对应子步过程噪声

# ---------------- 预分配数组 ----------------
truth       = np.zeros((num_steps + 1, 4))
meas_th_arr = np.zeros(num_steps + 1)
meas_v_arr  = np.zeros((num_steps + 1, 2))

# ---------------- 初始化 ----------------
rng     = np.random.default_rng()     # 可改 seed
curr_x  = x0 + la.cholesky(P0, lower=True) @ rng.standard_normal(4)

truth[0]       = curr_x
meas_th_arr[0] = meas_theta(curr_x)
meas_v_arr[0]  = meas_velocity(curr_x)

# ---------------- 主循环 ----------------
for k in range(num_steps):
    comp_Q = np.zeros((4, 4))
    for _ in range(dt_div):
        r     = np.linalg.norm(curr_x[:2])
        accel = -G_E * curr_x[:2] / r**3

        # x_k+1 = F_small x_k + [0.5 a dt², a dt]ᵀ
        curr_x = F_small @ curr_x + np.hstack((0.5 * sub_dt**2 * accel,
                                               sub_dt * accel))
        comp_Q = F_small @ comp_Q @ F_small.T + Q_small

    # -------- 给 “真值” 加一次过程噪声，但 **不** 回写 curr_x --------
    truth_k1 = curr_x + rng.multivariate_normal(np.zeros(4), comp_Q)
    truth[k+1] = truth_k1

    # -------- 生成测量 --------
    meas_th_arr[k+1] = meas_theta(truth_k1)
    meas_v_arr[k+1]  = meas_velocity(truth_k1)

# ---------------- 保存文件 ----------------
# 1) 仅 θ
np.savez('slide_example.npz',
         R=R_th, Q=Q, dt=dt,
         meas=meas_th_arr,
         x0=x0, P0=P0,
         truth=truth)

np.savez('orbit_run_restricted.npz',
         R=R_th, Q=Q, dt=dt,
         meas=meas_th_arr,
         x0=x0, P0=P0)

# 2) θ + v
np.savez('slide_example_velocity.npz',
         R_th=R_th, R_v=R_v, Q=Q, dt=dt,
         meas_th=meas_th_arr,
         meas_vel=meas_v_arr,
         x0=x0, P0=P0,
         truth=truth)

np.savez('orbit_run_restricted_velocity.npz',
         R_th=R_th, R_v=R_v, Q=Q, dt=dt,
         meas_th=meas_th_arr,
         meas_vel=meas_v_arr,
         x0=x0, P0=P0)

print("✅  四个 npz 文件已生成！")

# ---------------- 可选：绘图快速检查 ----------------
plt.figure(figsize=(5, 5))
plt.plot(truth[:, 0], truth[:, 1])
plt.gca().set_aspect('equal'); plt.grid(True)
plt.title('True Orbit  (x vs y)')

plt.figure()
plt.plot(truth[:, :2]); plt.legend(['x', 'y'])
plt.title('True Position vs Time')

plt.figure()
plt.plot(truth[:, 2:]); plt.legend(['v_x', 'v_y'])
plt.title('True Velocity vs Time')

plt.tight_layout(); plt.show()
