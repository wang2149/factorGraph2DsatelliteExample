import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def compute_position_rmse(est, truth):
    """
    est, truth: shape (N,4) with columns [x, y, vx, vy]
    Returns scalar RMSE over Euclidean position error.
    """
    err = est[:, :2] - truth[:, :2]
    return np.sqrt(np.mean(np.sum(err**2, axis=1)))

# formatter for axes (millions of meters)
millions = FuncFormatter(lambda x, pos: f"{x/1e6:.0f}")

# --- Load data ---
base     = np.load('ekf_slide_example_res.npz')
truth    = base['truth']
ekf_base = base.get('ekf_res', base.get('ekf_ref'))

vel      = np.load('ekf_slide_example_velocity_res.npz')
ekf_vel  = vel['ekf_res']

# --- Compute RMSEs ---
rmse_base = compute_position_rmse(ekf_base, truth)
rmse_vel  = compute_position_rmse(ekf_vel,  truth)

print(f"EKF baseline position RMSE       = {rmse_base:.1f} m")
print(f"EKF +velocity position RMSE      = {rmse_vel:.1f} m")

# --- Combined figure with two subplots ---
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Subplot 1: Baseline EKF vs Truth
ax = axes[0]
ax.plot(truth[:,0], truth[:,1], c='blue', label='Truth')
ax.plot(ekf_base[:,0], ekf_base[:,1], c='red',
        label=f'Baseline EKF\nRMSE={rmse_base:.0f} m')
ax.set_aspect('equal')
ax.legend(loc='upper right')
ax.grid(which='major', linestyle='-.', color='#CCCCCC')
ax.xaxis.set_major_formatter(millions)
ax.yaxis.set_major_formatter(millions)
ax.set_xlabel('x (×10⁶ m)')
ax.set_ylabel('y (×10⁶ m)')
ax.set_title('EKF Baseline vs Truth')

# Subplot 2: EKF +Velocity vs Truth
ax = axes[1]
ax.plot(truth[:,0], truth[:,1], c='blue', label='Truth')
ax.plot(ekf_vel[:,0], ekf_vel[:,1], c='red',
        label=f'EKF +Velocity\nRMSE={rmse_vel:.0f} m')
ax.set_aspect('equal')
ax.legend(loc='upper right')
ax.grid(which='major', linestyle='-.', color='#CCCCCC')
ax.xaxis.set_major_formatter(millions)
ax.yaxis.set_major_formatter(millions)
ax.set_xlabel('x (×10⁶ m)')
ax.set_ylabel('y (×10⁶ m)')
ax.set_title('EKF +Velocity vs Truth')

plt.tight_layout()
plt.savefig('comparison_EKF_combined.pdf', bbox_inches='tight')
plt.show()
