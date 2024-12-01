########################################################## FUNCTIONS
import numpy as np
import numba as nb


rng = np.random.default_rng(1)

@nb.jit(nopython=True, parallel=True)
def get_kernel_mat_nb(t1list, t2list):
    mat = np.zeros((t1list.shape[0], t2list.shape[0]))
    for i in nb.prange(t1list.shape[0]):
        for j in nb.prange(t2list.shape[0]):
            mat[i][j] = np.exp(-0.13 * (t1list[i] - t2list[j]) ** 2) * 2.0
    return mat.T


def mvn_sample_normal(_num_samples, _tsteps, _Lmat):
    init_samples = rng.standard_normal(size=(_tsteps, _num_samples))
    new_samples = _Lmat @ init_samples
    return new_samples.T


def get_min_dist(x_trajs, y_trajs):
    dx_trajs = x_trajs[:, np.newaxis, :] - x_trajs
    dy_trajs = y_trajs[:, np.newaxis, :] - y_trajs
    dist_trajs = np.sqrt(dx_trajs ** 2 + dy_trajs ** 2)

    min_dist_trajs = np.min(dist_trajs, axis=2)
    min_dist_trajs += np.eye(min_dist_trajs.shape[0]) * 1e6
    min_dist = np.min(min_dist_trajs)

    return min_dist


@nb.njit  # ('float64[:, :](float64[:, :])')
def cholesky_numba(A):
    n = A.shape[0]
    L = np.zeros_like(A)
    for i in range(n):
        for j in range(i + 1):
            s = 0
            for k in range(j):
                s += L[i][k] * L[j][k]

            if (i == j):
                L[i][j] = (A[i][i] - s) ** 0.5
            else:
                L[i][j] = (1.0 / L[j][j] * (A[i][j] - s))
    return L


@nb.jit(nopython=True)
def get_Lmat_nb(train_ts, test_ts, train_noise):
    covmat_11 = get_kernel_mat_nb(train_ts, train_ts)
    covmat_11 += np.diag(train_noise)
    covmat_12 = get_kernel_mat_nb(test_ts, train_ts).T
    covmat_22 = get_kernel_mat_nb(test_ts, test_ts)
    cov_mat = covmat_22 - covmat_12 @ np.linalg.inv(covmat_11) @ covmat_12.T
    cov_mat += np.eye(test_ts.shape[0]) * 1e-06
    return cholesky_numba(cov_mat), cov_mat


@nb.jit(nopython=True, parallel=True)
def costs_nb(trajs_x, trajs_y, num_agents, num_pts, tsteps):
    vals = np.zeros((num_agents * num_pts, num_agents * num_pts))
    for i in nb.prange(num_pts * num_agents):
        traj_xi = trajs_x[i]
        traj_yi = trajs_y[i]
        for j in nb.prange(num_pts * num_agents):
            traj_xj = trajs_x[j]
            traj_yj = trajs_y[j]
            traj_costs = np.zeros(tsteps)
            for t in nb.prange(tsteps):
                dist = (traj_xi[t] - traj_xj[t]) ** 2 + (traj_yi[t] - traj_yj[t]) ** 2
                traj_costs[t] = 2 - 2 / (1.0 + np.exp(-10.0 * dist))
            vals[i, j] = np.max(traj_costs) * 100.0
    return vals


@nb.jit(nopython=True, parallel=True)
def weights_update_nb(all_costs, old_weights, index_table, all_pt_index, num_agents, num_pts):
    weights = old_weights.copy()
    for i in range(num_agents):
        row = index_table[i]
        # other_pt_index = all_pt_index[row[1:], :].ravel()

        for j1 in nb.prange(num_pts):
            cost1 = 0.0
            idx1 = all_pt_index[row[0], j1]
            for i2 in nb.prange(num_agents - 1):
                for j2 in range(num_pts):
                    idx2 = all_pt_index[row[i2 + 1], j2]
                    cost1 += all_costs[idx1, idx2] * weights[row[i2 + 1], j2]
            cost1 /= (num_agents - 1) * num_pts
            weights[i, j1] = np.exp(-1.0 * cost1)
        weights[i] /= np.mean(weights[i])
    return weights


@nb.jit(nopython=True, parallel=True)
def get_index_table(num_agents):
    index_table = np.zeros((num_agents, num_agents))
    for i in nb.prange(num_agents):
        index_table[i][0] = i
        idx = 1
        for j in range(num_agents):
            if i == j:
                continue
            index_table[i][idx] = j
            idx += 1
    return index_table


def brne_nav(xmean_list, ymean_list, x_pts, y_pts, num_agents, tsteps, num_pts):
    index_table = get_index_table(num_agents).astype(int)
    all_pt_index = np.arange(num_agents * num_pts).reshape(num_agents, num_pts)

    x_opt_trajs = np.array(xmean_list)
    y_opt_trajs = np.array(ymean_list)

    weights = np.ones((num_agents, num_pts))
    all_traj_pts_x = np.zeros((num_agents * num_pts, tsteps))
    all_traj_pts_y = np.zeros((num_agents * num_pts, tsteps))
    
    i = 0
    all_traj_pts_x[i * num_pts:i * num_pts + num_pts] = xmean_list[i] + x_pts[i * num_pts:i * num_pts + num_pts]
    all_traj_pts_y[i * num_pts:i * num_pts + num_pts] = ymean_list[i] + y_pts[i * num_pts:i * num_pts + num_pts]
    
    for i in range(1, num_agents):
        all_traj_pts_x[i * num_pts:i * num_pts + num_pts] = xmean_list[i] + x_pts[i * num_pts:i * num_pts + num_pts]
        all_traj_pts_y[i * num_pts:i * num_pts + num_pts] = ymean_list[i] + y_pts[i * num_pts:i * num_pts + num_pts]
    all_costs = costs_nb(all_traj_pts_x, all_traj_pts_y, num_agents, num_pts, tsteps)

    for iter_num in range(10):
        weights = weights_update_nb(all_costs, weights, index_table, all_pt_index, num_agents, num_pts)
    for i in range(num_agents):
        x_opt_trajs[i] = xmean_list[i] + np.mean(
            x_pts[i * num_pts: i * num_pts + num_pts] * weights[i][:, np.newaxis], axis=0)
        y_opt_trajs[i] = ymean_list[i] + np.mean(
            y_pts[i * num_pts: i * num_pts + num_pts] * weights[i][:, np.newaxis], axis=0)

    return x_opt_trajs, y_opt_trajs, weights

# @nb.jit(nopython=True)
def get_min_dist(x_trajs, y_trajs):
    dx_trajs = x_trajs[:,np.newaxis,:] - x_trajs
    dy_trajs = y_trajs[:,np.newaxis,:] - y_trajs
    dist_trajs = np.sqrt(dx_trajs**2 + dy_trajs**2)

    min_dist_trajs = np.min(dist_trajs, axis=2)
    min_dist_trajs += np.eye(min_dist_trajs.shape[0]) * 1e6
    min_dist = np.min(min_dist_trajs[0])
    # print('min dist verify: ', np.min(min_dist_trajs[1]))

    return min_dist
