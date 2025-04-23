import scipy.sparse as sp
import scipy.sparse.linalg as spla
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from math import pi, sqrt, ceil, atan2

def dense_2_sp_lists(M: np.array, tl_row: int, tl_col: int, row_vec=True):
    """
    Helper to linearize a dense matrix M into data, row, col arrays for sparse.
    """
    data = M.flatten()
    if M.ndim == 2:
        rows, cols = M.shape
    elif M.ndim == 1:
        if row_vec:
            rows, cols = 1, M.size
        else:
            rows, cols = M.size, 1
    else:
        raise ValueError("M must be 1d or 2d")
    row_list = np.repeat(np.arange(rows)[:,None], cols, axis=1).flatten() + tl_row
    col_list = np.repeat(np.arange(cols)[None,:], rows, axis=0).flatten() + tl_col
    return data, row_list, col_list

class satelliteModelBatch:
    """
    Batch factor-graph optimizer for 2D satellite orbit with
    bearing+velocity measurements.
    """
    MEAS_DIM = 3  # theta, vx, vy

    def __init__(self, meas_th: np.ndarray, meas_vel: np.ndarray,
                 R_th: float, R_v: np.ndarray,
                 Q: np.ndarray, dt: float = 5.0,
                 x0: np.ndarray = np.array([0,2e7,4500.,0])):
        self.N = len(meas_th)
        self.dt = dt
        if dt > 1:
            self.prop_dt = dt / ceil(dt)
            self.n_timesteps = int(ceil(dt))
        else:
            self.prop_dt = dt
            self.n_timesteps = 1
        self.T = np.array([1,0,self.prop_dt,0,
                           0,1,0,self.prop_dt,
                           0,0,1,0,
                           0,0,0,1]).reshape(4,4)
        self.GE = 3.986e14

        # stack measurements into (N,3) array
        self.meas = np.hstack([
            meas_th.reshape(-1,1),
            meas_vel
        ])  # shape (N,3)

        # process noise
        self.S_Q_inv = la.inv(la.cholesky(Q, lower=True))

        # measurement noise
        R_big = np.diag([R_th, R_v[0,0], R_v[1,1]])
        self.S_R_inv = la.inv(la.cholesky(R_big, lower=True))

        # states initialization
        self.states = np.zeros((self.N,4))
        self.states[0] = x0.copy()
        self.create_init_state()

    def create_init_state(self):
        for i in range(1, self.N):
            self.states[i] = self.prop_one_timestep(self.states[i-1])

    def prop_one_timestep(self, state: np.ndarray) -> np.ndarray:
        x = state.copy()
        for _ in range(self.n_timesteps):
            r = la.norm(x[:2])
            add = self.GE*self.prop_dt/(r**3) * np.array([
                0.5*self.prop_dt*x[0],
                0.5*self.prop_dt*x[1],
                x[0],
                x[1]
            ])
            x = self.T.dot(x) - add
        return x

    def state_idx(self, i: int) -> int:
        return 4*i

    def dyn_idx(self, i: int) -> int:
        return 4*(i-1)

    def meas_idx(self, i: int) -> int:
        return 4*(self.N-1) + self.MEAS_DIM * i

    def H_mat(self, state: np.ndarray) -> np.ndarray:
        x, y = state[0], state[1]
        r2 = x*x + y*y
        row1 = np.array([-y/r2, x/r2, 0, 0])
        row2 = np.array([0,0,1,0])
        row3 = np.array([0,0,0,1])
        return np.vstack([row1, row2, row3])  # (3,4)

    def F_mat(self, state: np.ndarray) -> np.ndarray:
        F = np.eye(4)
        curr = state.copy()
        for _ in range(self.n_timesteps):
            x,y = curr[0], curr[1]
            r = la.norm(curr[:2])
            k = self.prop_dt*self.GE/(r**5)
            t_mat = np.array([
                [-self.prop_dt*(y**2-2*x**2)/2, 3*x*y*self.prop_dt/2, 0, 0],
                [ 3*x*y*self.prop_dt/2, -self.prop_dt*(x**2-2*y**2)/2, 0,0],
                [ 2*x**2-y**2,            3*x*y,                      0,0],
                [ 3*x*y,                  2*y**2-x**2,               0,0]
            ])
            F = (self.T + k*t_mat).dot(F)
            add = self.GE*self.prop_dt/(r**3) * np.array([
                0.5*self.prop_dt*curr[0],
                0.5*self.prop_dt*curr[1],
                curr[0],
                curr[1]
            ])
            curr = self.T.dot(curr) - add
        return F

    def create_L(self) -> sp.csr_matrix:
        H_size = 4 * self.MEAS_DIM  # each measurement block has 12 entries
        F_size = 16
        nnz = 2*F_size*(self.N-1) + H_size*self.N
        data = np.zeros(nnz)
        row = np.zeros(nnz, dtype=int)
        col = np.zeros(nnz, dtype=int)
        t = 0
        # dynamics blocks
        for i in range(1, self.N):
            Fk = self.S_Q_inv.dot(self.F_mat(self.states[i-1]))
            d,r,c = dense_2_sp_lists(Fk, self.dyn_idx(i), self.state_idx(i-1))
            data[t:t+F_size], row[t:t+F_size], col[t:t+F_size] = d,r,c
            t += F_size
            Mk = -self.S_Q_inv
            d,r,c = dense_2_sp_lists(Mk, self.dyn_idx(i), self.state_idx(i))
            data[t:t+F_size], row[t:t+F_size], col[t:t+F_size] = d,r,c
            t += F_size
        # measurement blocks
        for i in range(self.N):
            Hi = self.S_R_inv.dot(self.H_mat(self.states[i]))  # (3,4)
            d,r,c = dense_2_sp_lists(Hi, self.meas_idx(i), self.state_idx(i))
            data[t:t+H_size], row[t:t+H_size], col[t:t+H_size] = d,r,c
            t += H_size
        return sp.csr_matrix((data,(row,col)))

    def create_y(self, state_vec: np.ndarray=None) -> np.ndarray:
        if state_vec is None:
            states = self.states
        else:
            states = self.vec_to_data(state_vec)
        y = np.zeros(4*(self.N-1) + self.MEAS_DIM*self.N)
        # dynamics residuals
        for i in range(1,self.N):
            pred = self.prop_one_timestep(states[i-1]) - states[i]
            y[self.dyn_idx(i):self.dyn_idx(i+1)] = self.S_Q_inv.dot(-pred)
        # measurement residuals
        for i in range(self.N):
            x,y_,vx,vy = states[i]
            z_pred = np.array([atan2(y_,x), vx, vy])
            innov = self.meas[i] - z_pred
            if innov[0] >  pi: innov[0] -= 2*pi
            if innov[0] < -pi: innov[0] += 2*pi
            y[self.meas_idx(i):self.meas_idx(i)+self.MEAS_DIM] = \
                self.S_R_inv.dot(innov)
        return y

    def vec_to_data(self, vec: np.ndarray) -> np.ndarray:
        return vec.reshape(self.N, 4)

    def add_delta(self, delta: np.ndarray=None) -> np.ndarray:
        if delta is None:
            delta = np.zeros(self.N*4)
        return (self.states.flatten() + delta)

    def update_state(self, delta: np.ndarray):
        self.states += delta.reshape(self.N,4)

    def opt(self):
        finished = False
        while not finished:
            L = self.create_L()
            y = self.create_y()
            M = L.T.dot(L)
            Lty = L.T.dot(y)
            delta = spla.spsolve(M, Lty)
            scale = 1.0
            # damping
            if la.norm(delta) >= 10:
                while True:
                    y_new = self.create_y(self.add_delta(delta*scale))
                    pred = y - L.dot(delta*scale)
                    num = y.dot(y) - y_new.dot(y_new)
                    den = y.dot(y) - pred.dot(pred)
                    ratio = num/den if den!=0 else 0
                    if 0.25 < ratio < 4.0 or scale < 1e-3:
                        break
                    scale /= 2
            self.update_state(delta*scale)
            if la.norm(delta) < 12:
                finished = True

# ---------------- main ----------------------------------------------------- #
if __name__ == '__main__':
    import sys, pathlib
    # ------- 1. 读数据 ----------------------------------------------------- #
    prefix = sys.argv[1] if len(sys.argv) > 1 else 'slide_example_velocity'
    data   = np.load(f'{prefix}.npz')

    meas_th  = data['meas_th'] if 'meas_th' in data.files else data['meas']
    meas_vel = data['meas_vel'] if 'meas_vel' in data.files else None
    R_th     = data['R_th']     if 'R_th'     in data.files else data['R']
    R_v      = data['R_v']      if 'R_v'      in data.files else None
    Q, dt    = data['Q'], data['dt'].item()
    truth, x0 = data['truth'], data['x0']

    # ------- 2. 初始化 FG -------------------------------------------------- #
    model = satelliteModelBatch(meas_th, meas_vel, R_th, R_v, Q, dt, x0)

    # ------- 3. 显式 Gauss-Newton 迭代，收集 Δx --------------------------- #
    history = []
    MAX_IT, TOL = 100, 1e-2
    print(f"run FG on «{prefix}.npz»  (N = {len(meas_th)})")
    for it in range(1, MAX_IT+1):
        L = model.create_L()
        y = model.create_y()
        delta = spla.spsolve(L.T @ L, L.T @ y)
        norm  = la.norm(delta)
        history.append(norm)
        print(f"[iter {it:2d}]  ||Δx|| = {norm:8.3f}")
        model.update_state(delta)
        if norm < TOL:
            break

    # ------- 4. 图 1：真轨迹 / 初值 / 优化后 ------------------------------ #
    plt.figure()
    plt.plot(truth[:,0], truth[:,1], 'b',  label='truth')
    plt.plot(model.states[:,0], model.states[:,1], 'r',  label='opt')
    plt.plot(model.states[0,0], model.states[0,1], 'go', label='init')
    plt.gca().set_aspect('equal'); plt.legend()
    plt.title('FG optimisation result')
    plt.savefig(f'FG_{prefix}_opt.png', dpi=200)

    # ------- 5. 图 2：opt vs truth（对比用） ------------------------------ #
    plt.figure()
    plt.plot(truth[:,0], truth[:,1], 'b',label='truth')
    plt.plot(model.states[:,0], model.states[:,1],'r',label='opt')
    plt.gca().set_aspect('equal'); plt.legend()
    plt.title('Optimised vs Truth')
    plt.savefig(f'FG_{prefix}_compare.png', dpi=200)

    # ------- 6. 图 3：误差 + 收敛曲线 ------------------------------------ #
    fig, ax = plt.subplots()
    ax.plot(model.states-truth)
    ax.legend(['x','y','vx','vy']); ax.set_title('state errors')
    ax2 = fig.add_axes([0.62,0.55,0.3,0.3])
    ax2.plot(history,'k'); ax2.set_yscale('log'); ax2.set_title('‖Δx‖')
    plt.savefig(f'FG_{prefix}_errors.png', dpi=200)
    plt.show()

    # ------- 7. 保存结果 --------------------------------------------------- #
    np.savez(f'fg_{prefix}_res', fg_res=model.states, truth=truth)
    print("done – results & figures saved.\n")
