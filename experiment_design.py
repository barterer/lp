import numpy as np
from scipy.linalg import qr
from scipy.optimize import minimize

# greedy

def pivot_columns(a, rank=None, columns_to_avoid=None):
    """Computes the QR decomposition of a matrix with column pivoting, i.e. solves the equation AP=QR such that Q is
    orthonormal, R is upper triangular, and P is a permutation matrix.
    
    When setting arguments, note that one and only one of threshold and rank should be specified.
    
    Arguments:
        a (np.ndarray):    Matrix for which to compute QR decomposition.
        rank (int):        The approximate rank.
        columns_to_avoid (list): The columns to be avoided.
                
    Returns:
        list: all the columns (except the ones in columns_to_avoid) in the pivot order.

    """
    if columns_to_avoid is not None:
        set_of_columns_to_avoid = set(columns_to_avoid)
    else:
        set_of_columns_to_avoid = set()
    
    qr_columns = qr(a, pivoting=True)[2]
    
    r = []
    i = 0
    while(len(r) < rank):
        if qr_columns[i] not in set_of_columns_to_avoid:
            r.append(qr_columns[i])
        i += 1
    return r


def greedy_stepwise_selection(Y, initialization, n_greedy, columns_to_avoid=None, verbose=False):
    """
    The greedy algorithm for D-optimal experiment design.
    
    Arguments:
    Y (np.ndarray):                The d-dimensional design vectors are aligned by columns.
    initialization (list):         The list of initializations, with cardinality at least d.
    n_greedy (int):                Number of design vectors to select.
    columns_to_avoid (list):         The list of design vector indices to avoid selecting from.
    verbose (Boolean):             Whether to show intermediate information.
    
    Returns:
    selected (list):               The list of selected design vector indices.
    """
    n_cols = Y.shape[1]
    
    X = Y[:, initialization] @ Y[:, initialization].T
    selected = list(initialization)
    
    if columns_to_avoid is not None:
        set_of_indices_to_avoid = set(columns_to_avoid)
    else:
        set_of_indices_to_avoid = set()
    
    for iteration in range(n_greedy):
        if verbose and not iteration % 10:
            print("Iteration {}".format(iteration))
        to_select = list(set(range(n_cols)).difference(set(selected)).difference(set_of_indices_to_avoid))
        try:
            X_inv = np.linalg.inv(X)
        except:
            break
        obj_all = np.array([Y[:, i] @ X_inv @ Y[:, i] for i in to_select])
        
        if not all(np.isnan(obj_all)):
            to_add = to_select[np.nanargmax(obj_all)]
        else:
            break        

        X = X + np.outer(Y[:, to_add], Y[:, to_add])
        selected.append(to_add)
    
    if verbose:
        print("condition number of final design matrix: {}".format(np.linalg.cond(X)))
    return selected

# convexification

def number_of_entries_solve(N, Y, scalarization='D'):
    n = Y.shape[1]
    # It is observed the scipy.optimize solver in this problem usually converges within 50 iterations. Thus a maximum of 50 step is set as limit.
    if scalarization == 'D':
        def objective(v):
            sign, log_det = np.linalg.slogdet(Y @ np.diag(v) @ Y.T)
            return -1 * sign * log_det
    elif scalarization == 'A':
        def objective(v):
            return np.trace(np.linalg.pinv(Y @ np.diag(v) @ Y.T))
    elif scalarization == 'E':
        def objective(v):
            return np.linalg.norm(np.linalg.pinv(Y @ np.diag(v) @ Y.T), ord=2)
    def constraint(v):
        return N - np.sum(v)
    v0 = np.full((n, ), 0.5)
    constraints = {'type': 'ineq', 'fun': constraint}
    v_opt = minimize(objective, v0, method='SLSQP', bounds=[(0, 1)] * n, options={'maxiter': 50},
                     constraints=constraints)
    return v_opt.x


def convexification_solve(Y, n_cvx, scalarization='D', pick_largest_v_opt=True, columns_to_avoid=None, get_cvx_solution=False):
    if columns_to_avoid is not None:
        set_of_indices_to_avoid = set(columns_to_avoid)
    else:
        set_of_indices_to_avoid = set()
    
    valid = np.array(list(set(np.arange(Y.shape[1])) - set_of_indices_to_avoid))
    
    v_opt = np.zeros(Y.shape[1])
    cvx_solution = number_of_entries_solve(n_cvx, Y[:, valid], scalarization)
    v_opt[valid] = cvx_solution
    
    if pick_largest_v_opt:
        to_sample = np.argsort(-v_opt)[:n_cvx]
    else:
        to_sample = np.where(v_opt > 0.9)[0]
    
    if get_cvx_solution:
        return to_sample, cvx_solution
    else:
        return to_sample
