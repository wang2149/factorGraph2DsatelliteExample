import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def compute_position_rmse(est, truth):
    """RMSE of √(Δx²+Δy²).  est, truth shape (N,4)"""
    err = est[:,:2] - truth[:,:2]
    return np.sqrt(np.mean(np.sum(err**2, axis=1)))

# ---------- 1. 读 FG 结果 ----------
base     = np.load('fg_slide_example_res.npz')
truth    = base['truth']
fg_base  = base['fg_res']          # 没有别名，就直接 fg_res

vel      = np.load('fg_slide_example_velocity_res.npz')
fg_vel   = vel['fg_res']

# ---------- 2. 计算 RMSE ----------
rmse_base = compute_position_rmse(fg_base, truth)
rmse_vel  = compute_position_rmse(fg_vel,  truth)

print(f"FG  baseline position RMSE  = {rmse_base:.1f} m")
print(f"FG  +velocity position RMSE = {rmse_vel:.1f} m")

# ---------- 3. 画两张子图 ----------
millions = FuncFormatter(lambda x, pos: f"{x/1e6:.0f}")

fig, axes = plt.subplots(1,2, figsize=(12,6))

# --- 3.1 Baseline FG ---
ax = axes[0]
ax.plot(truth[:,0], truth[:,1], 'b',  label='Truth')
ax.plot(fg_base[:,0], fg_base[:,1], 'r',
        label=f'Baseline FG\nRMSE={rmse_base:.0f} m')
ax.set_aspect('equal')
ax.grid(linestyle='-.', color='#CCC')
ax.xaxis.set_major_formatter(millions)
ax.yaxis.set_major_formatter(millions)
ax.set_xlabel('x (×10⁶ m)'); ax.set_ylabel('y (×10⁶ m)')
ax.set_title('FG Baseline vs Truth'); ax.legend()

# --- 3.2 FG +Velocity ---
ax = axes[1]
ax.plot(truth[:,0], truth[:,1], 'b',  label='Truth')
ax.plot(fg_vel[:,0], fg_vel[:,1], 'r',
        label=f'FG +Velocity\nRMSE={rmse_vel:.0f} m')
ax.set_aspect('equal')
ax.grid(linestyle='-.', color='#CCC')
ax.xaxis.set_major_formatter(millions)
ax.yaxis.set_major_formatter(millions)
ax.set_xlabel('x (×10⁶ m)'); ax.set_ylabel('y (×10⁶ m)')
ax.set_title('FG +Velocity vs Truth'); ax.legend()

plt.tight_layout()
plt.savefig('comparison_FG_combined.pdf', bbox_inches='tight')
plt.show()
