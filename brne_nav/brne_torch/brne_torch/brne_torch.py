########################################################## FUNCTIONS
import numpy as np
rng = np.random.default_rng(1)
import time

import torch 
gpu_index = 0
device = torch.device(f'cuda:{gpu_index}')
print(f'device: {device}')


# new NumPy-only version
def get_kernel_mat(t1list, t2list, kernel_a1, kernel_a2):
    dt_mat = t1list - t2list[:,None]
    k_mat = np.exp(-kernel_a1 * dt_mat**2) * kernel_a2 
    return k_mat.T


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


# @nb.njit  # ('float64[:, :](float64[:, :])')
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


# new NumPy-only version
def get_Lmat(train_ts, test_ts, train_noise, kernel_a1, kernel_a2):
    covmat_11 = get_kernel_mat(train_ts, train_ts, kernel_a1, kernel_a2)
    covmat_11 += np.diag(train_noise)
    covmat_12 = get_kernel_mat(test_ts, train_ts, kernel_a1, kernel_a2)
    covmat_22 = get_kernel_mat(test_ts, test_ts, kernel_a1, kernel_a2)
    cov_mat = covmat_22 - covmat_12 @ np.linalg.inv(covmat_11) @ covmat_12.T
    cov_mat += np.eye(test_ts.shape[0]) * 1e-06
    return np.linalg.cholesky(cov_mat), cov_mat


# @nb.jit(nopython=True, parallel=True)
# def costs_nb(trajs_x, trajs_y, num_agents, num_pts, tsteps, cost_a1, cost_a2, cost_a3):
#     vals = np.zeros((num_agents * num_pts, num_agents * num_pts))
#     for i in nb.prange(num_pts * num_agents):
#         traj_xi = trajs_x[i]
#         traj_yi = trajs_y[i]
#         for j in nb.prange(num_pts * num_agents):
#             traj_xj = trajs_x[j]
#             traj_yj = trajs_y[j]
#             traj_costs = np.zeros(tsteps)
#             for t in nb.prange(tsteps):
#                 dist = (traj_xi[t] - traj_xj[t]) ** 2 + (traj_yi[t] - traj_yj[t]) ** 2
#                 traj_costs[t] = 2 - 2 / (1.0 + np.exp(-cost_a1 * np.power(dist, cost_a2)))
#             vals[i, j] = np.max(traj_costs) * cost_a3
#     return vals



########################################################################################################
########################################################################################################
########################################################################################################

def risk(dist2, a1, a2, a3):
    val = a3 * (2.0 - 2.0 / (1.0 + torch.exp(-a1 * torch.pow(dist2, a2))))
    return val


def risk_traj(traj1, traj2, a1, a2, a3):
    dist2_traj = torch.sum(torch.square(traj1 - traj2), dim=1)
    min_dist2 = torch.min(dist2_traj)
    return risk(min_dist2, a1, a2, a3)

# @torch.script.jit
def costs_torch(trajs_x, trajs_y, cost_a1, cost_a2, cost_a3):
    trajs_all = np.stack([trajs_x, trajs_y], axis=2)
    trajs_all = torch.tensor(trajs_all, device=device, dtype=torch.float32)
    vals = torch.vmap(torch.vmap(risk_traj, in_dims=(None,0)), in_dims=(0,None))(
        trajs_all, trajs_all, 
        a1=cost_a1, a2=cost_a2, a3=cost_a3
    )
    return vals.cpu().detach().numpy()

########################################################################################################
########################################################################################################
########################################################################################################



# # original Numba version
# def weights_update_nb(all_costs, old_weights, index_table, all_pt_index, num_agents, num_pts):
#     weights = old_weights.copy()
#     for i in range(num_agents):
#         row = index_table[i]
#         # other_pt_index = all_pt_index[row[1:], :].ravel()
#         for j in nb.prange(num_pts):
#             cost1 = 0.0
#             idx1 = all_pt_index[row[0], j]
#             for k in nb.prange(num_agents - 1):
#                 for l in range(num_pts):
#                     idx2 = all_pt_index[row[k + 1], l]
#                     cost1 += all_costs[idx1, idx2] * weights[row[k + 1], l]
#             cost1 /= (num_agents - 1) * num_pts
#             weights[i, j] = np.exp(-1.0 * cost1)
#         weights[i] /= np.mean(weights[i])
#     return weights


# new NumPy-only version
def weights_update(all_costs, old_weights, index_table, all_pt_index, num_agents, num_pts):
    weights = old_weights.copy()
    for i in range(num_agents):
        row = index_table[i].astype(int)
        cost1 = np.zeros(num_pts)
        for j in row[1:]:
            cost1 += np.sum(all_costs[i*num_pts:(i+1)*num_pts, j*num_pts:(j+1)*num_pts] * weights[j], axis=1)
        cost1 /= (num_agents - 1) * num_pts
        weights[i] = np.exp(-1.0 * cost1)
        weights[i] /= np.mean(weights[i])
    return weights

# new NumPy-only version (faster)
def weights_update_fast(all_costs, old_weights, index_table, all_pt_index, num_agents, num_pts):
    weights = old_weights.copy()
    for i in range(num_agents):
        row = index_table[i].astype(int)
        # cost1 = np.zeros(num_pts)
        # for j in row[1:]:
        #     cost1 += np.sum(all_costs[i*num_pts:(i+1)*num_pts, j*num_pts:(j+1)*num_pts] * weights[j], axis=1)
        idx_list = (np.arange(num_pts) + row[1:][:,None] * num_pts).ravel()
        sub_costs = all_costs[i*num_pts:(i+1)*num_pts, idx_list] * weights[row[1:], :].ravel()
        cost1 = np.sum(sub_costs, axis=1)
        cost1 /= (num_agents - 1) * num_pts
        weights[i] = np.exp(-1.0 * cost1)
        weights[i] /= np.mean(weights[i])
    return weights

def get_index_table(num_agents):
    index_table = np.zeros((num_agents, num_agents))
    for i in range(num_agents):
        index_table[i][0] = i
        idx = 1
        for j in range(num_agents):
            if i == j:
                continue
            index_table[i][idx] = j
            idx += 1
    return index_table

def coll_beck(trajs_y, y_min, y_max):
    lower_mask = trajs_y > y_min
    upper_mask = trajs_y < y_max
    return lower_mask * upper_mask

def brne_nav(xtraj_samples, ytraj_samples, num_agents, tsteps, num_pts, cost_a1, cost_a2, cost_a3, ped_sample_scale, y_min, y_max):
    index_table = get_index_table(num_agents).astype(int)
    all_pt_index = np.arange(num_agents * num_pts).reshape(num_agents, num_pts)
    weights = np.ones((num_agents, num_pts))
    coll_mask = coll_beck(ytraj_samples, y_min, y_max).all(axis=1).astype(float).reshape(num_agents, num_pts)

    # all_costs = costs_nb(xtraj_samples, ytraj_samples, num_agents, num_pts, tsteps, cost_a1, cost_a2, cost_a3)
    # tic = time.time()
    all_costs = costs_torch(xtraj_samples, ytraj_samples, cost_a1, cost_a2, cost_a3)
    # toc = time.time()
    

    # if we are going out of bounds, coll_mask[0] is all 0s and we will encounter divide by 0 error.
    if not np.any(coll_mask[0]):
        return None

    for iter_num in range(10):
        # weights = weights_update(all_costs, weights, index_table, all_pt_index, num_agents, num_pts)
        weights = weights_update_fast(all_costs, weights, index_table, all_pt_index, num_agents, num_pts)

    # print(f"\nFinal Weights\n{weights}")

    # for i in range(num_agents):
    #     agent_weights = weights[i] * coll_mask[i]
    #     agent_weights /= np.mean(agent_weights)
    #     weights[i] = agent_weights.copy()
    agent_weights = weights[0] * coll_mask[0]
    agent_weights /= np.mean(agent_weights)
    weights[0] = agent_weights.copy()

    # print(f"\nWeights after masking\n{weights}")
    return weights



##########################################################################################################
def dyn(st, ut):
    # print(f"st\n{st.T}")
    # print(f"ut\n{ut.T}")
    sdot = np.array([
        ut[0] * np.cos(st[2]),
        ut[0] * np.sin(st[2]),
        ut[1]
    ])
    # print(f"Sdot\n{sdot.T}")
    return sdot

def dyn_step(st, ut, dt):
    k1 = dt * dyn(st, ut)
    k2 = dt * dyn(st + k1/2.0, ut)
    k3 = dt * dyn(st + k2/2.0, ut)
    k4 = dt * dyn(st + k3, ut)

    return st + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

def traj_sim(st, ulist, dt):
    tsteps = len(ulist)
    traj = np.zeros((tsteps, 3))
    for t in range(tsteps):
        st = dyn_step(st, ulist[t])
        traj[t] = st.copy()
    return traj

def traj_sim_essemble(st, ulist, dt):
    """
    st.shape = (3, num_samples)
    ulist = (tsteps, num_samples, 2)
    """
    tsteps = ulist.shape[0]
    num_samples = ulist.shape[1]
    print('st.shape: ', st.shape)
    print('ulist.shape: ', ulist.shape)
    assert st.shape[1] == ulist.shape[1]

    traj = np.zeros((tsteps, 3, num_samples))

    for t in range(tsteps):
        st = dyn_step(st, ulist[t].T, dt)
        # print(f"Updated state\n{st.T}")
        traj[t] = st.copy()

    return traj

def get_ulist_essemble(ulist, u1_max, u2_max, num_samples):
    # square root should be an even number. this is for meshgrid
    num_essembles_per_dim = int(np.sqrt(num_samples))
    # empirically, it is more important to have more resolution in linear vs angular
    num_essembles_per_dim_u1 = int(num_essembles_per_dim * 2)
    num_essembles_per_dim_u2 = int(num_essembles_per_dim / 2)

    # print(f"N per u1 {num_essembles_per_dim_u1} per u2 {num_essembles_per_dim_u2}")

    u1_offset = np.minimum(
        ulist[:,0].min(),
        u1_max-ulist[:,0].max()
    )

    # print(f"Ulist :,0 {ulist[:,0]} :,1 {ulist[:,1]}")
    
    u2_offset = np.minimum(
        u2_max+ulist[:,1].min(),
        u2_max-ulist[:,1].max()
    )

    # print(f"U1 offset {u1_offset}")
    # print(f"U2 offset {u2_offset}")

    # ### grid approach
    u1_perturbs, u2_perturbs = np.meshgrid(
       np.linspace(-u1_offset, u1_offset, num_essembles_per_dim_u1),
       np.linspace(-u2_offset, u2_offset, num_essembles_per_dim_u2)
    )

    # print(f"Linspace u1\n{np.linspace(-u1_offset, u1_offset, num_essembles_per_dim_u1)}")
    # print(f"Linspace u2\n{np.linspace(-u2_offset, u2_offset, num_essembles_per_dim_u2)}")

    # print(f"U1 perturbs\n{u1_perturbs}")
    # print(f"U2 perturbs\n{u2_perturbs}")

    ### gaussian sampling
    # u1_perturbs = np.random.normal(size=(num_samples,))
    # u1_perturbs /= -u1_perturbs.min() / (ulist[0][0] - 0.0)
    # u2_perturbs = np.random.normal(size=(num_samples,))
    # u2_perturbs /= u2_perturbs.max()

    u_perturbs = np.array([
        u1_perturbs.ravel(), u2_perturbs.ravel()
    ]).T

    # print(f"U perturbs\n{u_perturbs}")

    # print(f"Ulist newaxis {ulist[:,np.newaxis]}")

    # print(f"Ulist\n{ulist}")

    ulist_essemeble = ulist[:,np.newaxis] + u_perturbs
    # ulist_essemeble[:,:,0] = np.clip(ulist_essemeble[:,:,0], 0.0, u1_max)
    # ulist_essemeble[:,:,1] = np.clip(ulist_essemeble[:,:,1], -u2_max, u2_max)

    assert ulist_essemeble.shape[0] == ulist.shape[0]
    assert ulist_essemeble.shape[1] == u_perturbs.shape[0]
    assert ulist_essemeble.shape[2] == ulist.shape[1]

    return ulist_essemeble
