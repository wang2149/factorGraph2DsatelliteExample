import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def compute_position_rmse(est, truth):
    """
    est, truth: shape (N,4), columns [x, y, vx, vy]
    Returns the position RMSE (scalar) based on Euclidean distance error.
    """
    # compute error in x,y
    pos_err = est[:, :2] - truth[:, :2]        # (N,2)
    dist2   = np.sum(pos_err**2, axis=1)        # (N,)
    return np.sqrt(np.mean(dist2))             # scalar RMSE

# --- Load data ---
prefix    = 'slide_example'
ekf_data  = np.load(f'ekf_{prefix}_res.npz')
fg_data   = np.load(f'fg_{prefix}_res.npz')

truth     = ekf_data['truth']
# fix key if needed: replace 'ekf_ref' with 'ekf_res' if that's the correct one
ekf_res   = ekf_data.get('ekf_res', ekf_data.get('ekf_ref'))
fg_res    = fg_data['fg_res']

# --- Compute position RMSE ---
ekf_pos_rmse = compute_position_rmse(ekf_res, truth)
fg_pos_rmse  = compute_position_rmse(fg_res,  truth)

print(f"EKF position RMSE = {ekf_pos_rmse:.1f} m")
print(f"FG  position RMSE = {fg_pos_rmse:.1f} m")

# formatter for axes (in millions of meters)
millions = FuncFormatter(lambda x, pos: f"{x/1e6:.0f}")

# --- Figure 1: EKF vs Truth ---
plt.figure()
plt.plot(ekf_res[:,0], ekf_res[:,1], c='b',  label='EKF estimate')
plt.plot(truth[:,0],  truth[:,1],  'r--', label='truth')
plt.legend(loc='center left', bbox_to_anchor=(0.5,0.5))
plt.grid(which='major', linestyle='-.', color='#CCCCCC')

ax = plt.gca()
ax.set_aspect('equal')
# move spines
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
# draw gray box
x_min, x_max = ax.get_xlim()
y_min, y_max = ax.get_ylim()
ax.axhline(y_min, color='gray', linewidth=1.5)
ax.axhline(y_max, color='gray', linewidth=1)
ax.axvline(x_min, color='gray', linewidth=1)
ax.axvline(x_max, color='gray', linewidth=1.5)
# format ticks
ax.xaxis.set_major_formatter(millions)
ax.yaxis.set_major_formatter(millions)
# labels & title with RMSE
ax.set_xlabel(r'x (m$\times10^6$)', labelpad=65)
ax.set_ylabel(r'y (m$\times10^6$)', labelpad=85)
ax.set_title(f"EKF vs Truth — Position RMSE: {ekf_pos_rmse:.0f} m")

plt.savefig(f'ekf_{prefix}.pdf', bbox_inches='tight')

# --- Figure 2: FG vs Truth ---
plt.figure()
plt.plot(fg_res[:,0], fg_res[:,1], c='b',  label='FG estimate')
plt.plot(truth[:,0],  truth[:,1],  'r--', label='truth')
plt.legend(loc='center left', bbox_to_anchor=(0.5,0.5))
plt.grid(which='major', linestyle='-.', color='#CCCCCC')

ax = plt.gca()
ax.set_aspect('equal')
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
x_min, x_max = ax.get_xlim()
y_min, y_max = ax.get_ylim()
ax.axhline(y_min, color='gray', linewidth=1.5)
ax.axhline(y_max, color='gray', linewidth=1)
ax.axvline(x_min, color='gray', linewidth=1)
ax.axvline(x_max, color='gray', linewidth=1.5)
ax.xaxis.set_major_formatter(millions)
ax.yaxis.set_major_formatter(millions)
ax.set_xlabel(r'x (m$\times10^6$)', labelpad=65)
ax.set_ylabel(r'y (m$\times10^6$)', labelpad=85)
ax.set_title(f"FG vs Truth  — Position RMSE: {fg_pos_rmse:.0f} m")

plt.savefig(f'fg_{prefix}.pdf', bbox_inches='tight')
plt.show()
