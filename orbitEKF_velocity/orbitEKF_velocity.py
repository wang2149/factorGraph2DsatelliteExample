import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from math import pi, tan, atan2, floor

def plot_1d_est_with_cov(est, covs, truth=None, xs=None, sigmas=3):
    '''
    This function takes in (n,) numpy arrays and plots the estimate with
    "error bars" around the estimate.  Can also plot the truth.  The estimate
    will be red, the 
    
    Args:
        est:  The estimate values over time. An (n,) array
        covs:  The covariance over time.  An (n,) array
        truth:  (optional)  The true values over time.  An (n,) array
        xs: (optional) If nothing passed in, will plot 1->n on the x axis.
            Otherwise, will put xs along the x axis
        sigmas:  (default=3) How many sigmas out to put the "error bars"
            when plotting

    Returns:  nothing. Just plots stuff using matplotlib
    '''
    assert len(est)==len(covs), 'est and covs must be the same length'
    if truth is not None:
        assert len(est)==len(truth), 'est and truth must be the same length'
    if xs is not None:
        assert len(est)==len(xs), 'est and xs must be the same length'    
    else:
        xs = np.arange(len(est))
    plt.plot(xs,est,c='r', label='estimate')
    top_vals = est + np.sqrt(covs)*sigmas
    low_vals = est - np.sqrt(covs)*sigmas
    plt.fill_between(xs,low_vals,top_vals, facecolor='b',alpha=.5)
    if truth is not None:
        plt.plot(xs,truth,c='k',linestyle='--',label='truth')
        plt.legend()
    
def plot_2d_est_with_cov(est,covs,truth=None, sigmas=3, sigma_decimate=10):
    '''
    This function takes in (n,2) numpy arrays and plots the estimate with
    "error bars" around the estimate.  Can also plot the truth.  The estimate
    will be red, the truth black, and ellipses blue
    
    Args:
        est:  The estimate values over time. An (n,2) array
        covs:  The covariance over time.  An (n,2,2) array
        truth:  (optional)  The true values over time.  An (n,2) array
        sigmas:  (default=3) How many sigmas out to put the "error bars"
            when plotting
        sigma_decimate: (default=10) How many of the values to plot covariances
            around (plus stars on the corresponding locations).  If 1, will plot
            an ellipse around every point

    Returns:  nothing. Just plots stuff using matplotlib
    '''
    assert len(est)==len(covs), 'est and covs must be the same length'
    if truth is not None:
        assert len(est)==len(truth), 'est and truth must be the same length'
    plt.plot(est[:,0],est[:,1],c='b', label='estimate')
    plt.plot(est[0::sigma_decimate,0], est[0::sigma_decimate,1], 'b+')
    
    #Create a circle for plotting
    angs = np.arange(0,2*pi+.1,.1)
    circ_pts =np.zeros((2,len(angs)))
    circ_pts[0]=np.sin(angs)
    circ_pts[1]=np.cos(angs)
    circ_pts *= sigmas
    for i in range(len(est)):
        if i%sigma_decimate == 0:
            S_cov = la.cholesky(covs[i,:2,:2],lower=True)
            ellipse = S_cov.dot(circ_pts) + est[i,:2].reshape((2,1)) #reshape enables broadcast
            plt.plot(ellipse[0],ellipse[1],'b')
    if truth is not None:
        plt.plot(truth[:,0],truth[:,1],c='r',label='truth')
        plt.plot(truth[0::sigma_decimate,0], truth[0::sigma_decimate,1], 'r+')
        plt.legend()
    ax=plt.gca()
    ax.set_aspect('equal')

prefix='slide_example_velocity'

data     = np.load('slide_example_velocity.npz')

# 方位角观测 θ_k, shape = (N,)
meas_th  = data['meas_th']

# 线速度观测 [v_x, v_y], shape = (N,2)
meas_vel = data['meas_vel']

# 步数
num_steps = len(meas_th) - 1

# 观测噪声
R_th = data['R_th']    # 标量方位角噪声方差
R_v  = data['R_v']     # 2×2 速度噪声协方差

# 合成 3×3 观测噪声协方差 R
# （将 R_th 放在第一行列，后面两维直接用 R_v）
R_mat = np.block([
    [R_th      , 0    , 0    ],
    [0         , R_v[0,0], R_v[0,1]],
    [0         , R_v[1,0], R_v[1,1]]
])

# 过程噪声、时间步长、初始状态等
Q      = data['Q']       # 4×4
dt     = data['dt'].item()
curr_x = data['x0'].copy()
curr_P = data['P0'].copy()
truth  = data['truth']   # (N+1)×4


est_state = np.zeros((num_steps+1,4))
est_cov = np.zeros((num_steps+1,4,4))

est_state[0] = curr_x
est_cov[0] = curr_P
#See if this fixes things.
G_E = 3.986E14

def f(x,dt):
    accel = -G_E*x[:2]/la.norm(x[:2])**3
    F = np.eye(4)
    F[:2,2:] = np.eye(2) * dt
    accel_add = np.concatenate((dt**2/2 * accel, accel*dt))
    return F.dot(x) + accel_add

def f2(x,dt):
    dt_divider=50
    my_dt = dt/dt_divider
    F = np.eye(4)
    F[:2,2:] = np.eye(2) * my_dt
    for _ in range(dt_divider):
        accel = -G_E *x[:2]/la.norm(x[:2])**3
        move_accel = np.concatenate((accel * 0.5*my_dt**2, my_dt*accel))
        x = F.dot(x)+move_accel
    return x

def h(x):
    return atan2(x[1],x[0])

for i in range(num_steps):
    # —— 2. 预测步 ——
    # 2.1 线性化雅可比 F
    dist = la.norm(curr_x[:2])
    x,y = curr_x[0], curr_x[1]
    dt5 = dt*G_E/dist**5
    M = np.array([
        [ dt*(2*x*x - y*y)/2, 3*dt*x*y/2, 0,0],
        [3*dt*x*y/2, dt*(2*y*y - x*x)/2,0,0],
        [2*x*x - y*y, 3*x*y,           0,0],
        [3*x*y,       2*y*y - x*x,     0,0]
    ])
    T = np.array([1,0,dt,0, 0,1,0,dt, 0,0,1,0, 0,0,0,1]).reshape(4,4)
    F = T + dt5*M

    curr_x = f(curr_x, dt)
    curr_P = F.dot(curr_P).dot(F.T) + Q*dt

    # —— 3. 更新步 ——
    # 3.1 构造 3×4 观测雅可比 H
    dist2 = x*x + y*y
    H = np.array([
        [-y/dist2,  x/dist2, 0, 0],
        [       0,        0, 1, 0],
        [       0,        0, 0, 1]
    ])

    # 3.2 计算卡尔曼增益 K = P Hᵀ (H P Hᵀ + R)⁻¹
    S = H.dot(curr_P).dot(H.T) + R_mat   # 3×3
    K = curr_P.dot(H.T).dot(la.inv(S))   # 4×3

    # 3.3 残差向量 ỹ
    #     第一分量做 [-π,π] 归一化
    innov = np.zeros(3)
    pred_th = atan2(curr_x[1], curr_x[0])
    innov[0] = meas_th[i+1] - pred_th
    if innov[0] >  pi: innov[0] -= 2*pi
    if innov[0] < -pi: innov[0] += 2*pi
    innov[1:] = meas_vel[i+1] - curr_x[2:]  # [v_x err, v_y err]

    # 3.4 状态和协方差更新
    curr_x = curr_x + K.dot(innov)            # 4×1
    curr_P = (np.eye(4) - K.dot(H)).dot(curr_P)

    # 存储
    est_state[i+1] = curr_x
    est_cov[i+1]   = curr_P

decimate = int(floor(num_steps/20))
plot_2d_est_with_cov(est_state, est_cov, truth[:(num_steps+1)], sigma_decimate=decimate)
plt.savefig(prefix+'_loc.png')
plt.figure()
plot_1d_est_with_cov(est_state[:,2],est_cov[:,2,2],truth[:(num_steps+1),2])
plt.title('X Velocity')
plt.savefig(prefix+'_x_vel.png')
plt.figure()
plot_1d_est_with_cov(est_state[:,3],est_cov[:,3,3],truth[:(num_steps+1),3])
plt.title('Y Velocity')
plt.savefig(prefix+'_y_vel.png')

plt.figure()
plt.plot(est_state-truth)
plt.legend (['x','y','vx','vy'])
plt.title('errors')
plt.savefig(f'{prefix}_EKF_errors.png')

plt.figure()
plt.plot(est_state[:,0],est_state[:,1],c='b', label='estimate')
plt.plot(truth[:,0],truth[:,1],'r--',label='truth')
plt.legend()
ax=plt.gca()
ax.set_aspect('equal')
plt.savefig('EKF_'+prefix+'.png')
plt.show()

np.savez('ekf_'+prefix+'_res',ekf_res=est_state, truth=truth)