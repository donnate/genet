import numpy as np
import cvxpy as cp


def _ensure_edge_by_node(D, beta):
    # beta is a cvxpy Variable with shape (p,) or (p,1)
    p = int(beta.shape[0])
    # D might be a numpy array or a scipy sparse matrix; both have .shape
    if D.shape[1] == p:
        return D                     # (m, p) already good
    if D.shape[0] == p:
        return D.T                   # transpose node-by-edge -> edge-by-node
    raise ValueError(f"Incompatible D shape {D.shape} for beta of length {p}")



def lasso_penalty(beta, l1):
    return l1 * cp.norm1(beta)


def smoothlasso_penalty(beta, l1, l2, D):
    return l1 * cp.norm1(beta) + l2 * cp.norm2(D @ beta)**2


def elasticnet_penalty(beta, l1, l2):
    return l1 * cp.norm1(beta) + l2 * cp.norm2(beta)**2


def fusedlasso_penalty(beta, l1, l2, D):
    D = _ensure_edge_by_node(D, beta)
    return l1 * cp.norm1(beta) + l2 * cp.norm1(D @ beta)


def ee_penalty(beta, l1, l2, D):
    D = _ensure_edge_by_node(D, beta)
    return l1 * cp.norm1(D @ beta) + l2*cp.norm2(D @ beta)**2


def gtv_penalty(beta, l1, l2, l3, D):
    D = _ensure_edge_by_node(D, beta)
    return l1 * cp.norm1(D @ beta) + l2*cp.norm2(D @ beta)**2 + l3 * cp.norm1(beta)
