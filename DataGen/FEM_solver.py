import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _p1_triangle_matrices(p0, p1, p2):
    """
    Local stiffness and load for P1 on a triangle with vertices p0,p1,p2.
    Assumes coefficient a is constant on the triangle and f is constant (handled outside).
    Returns:
      Kref: 3x3 stiffness matrix for a=1, i.e. integral( grad phi_i.grad phi_j )dx
      m:    3-vector with int(φ_i) dx  (so load for constant f is f*m)
      area: triangle area
    """
    x0, y0 = p0
    x1, y1 = p1
    x2, y2 = p2

    # Twice area (signed)
    detJ2 = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    area = 0.5 * abs(detJ2)
    if area == 0.0:
        raise ValueError("Degenerate triangle encountered.")

    # Gradients of barycentric basis functions:
    # grad phi_i = [b_i, c_i] / (2*area)
    b = np.array([y1 - y2, y2 - y0, y0 - y1], dtype=np.float64)
    c = np.array([x2 - x1, x0 - x2, x1 - x0], dtype=np.float64)
    grads = np.vstack([b, c]).T / (2.0 * area)  # shape (3,2)

    Kref = area * (grads @ grads.T)  # integral( grad phi_i.grad phi_j) dx

    # For P1, int_T phi_i dx = area/3
    m = np.full(3, area / 3.0, dtype=np.float64)

    return Kref, m, area


def solve_darcy_fem_p1_dirichlet(a_nodes, f_value=1.0, L=1.0, cell_coef="mean",
                                tol=1e-9, maxiter=2000):
    """
    Solve -div(a grad u) = f_value on [0,L]^2 with homogeneous Dirichlet BC,
    using P1 FEM on a structured grid with (N x N) nodal samples of a(x,y).

    Parameters
    ----------
    a_nodes : (N,N) array
        Coefficient sampled at grid points including the boundary.
    f_value : float
        Constant RHS value (default 1.0).
    L : float
        Domain size; coordinates are [0,L]x[0,L].
    cell_coef : {"mean","majority"}
        How to map nodal a to cellwise constant a_K per square cell.
    tol, maxiter : CG parameters.

    Returns
    -------
    u : (N,N) array
        Nodal solution including boundary (boundary nodes are zero).
    info : dict
        Convergence info.
    """
    a_nodes = np.asarray(a_nodes, dtype=np.float64)
    if a_nodes.ndim != 2 or a_nodes.shape[0] != a_nodes.shape[1]:
        raise ValueError("a_nodes must be a square (N,N) array.")

    N = a_nodes.shape[0]
    if N < 3:
        raise ValueError("Need at least N>=3 nodal points to have interior unknowns.")

    h = L / (N - 1)  # include both endpoints
    n_total = N * N

    def node_id(i, j):
        return i + N * j

    # Identify interior degrees of freedom (Dirichlet boundary eliminated)
    dof = -np.ones(n_total, dtype=np.int32)
    idx = 0
    for j in range(1, N - 1):
        for i in range(1, N - 1):
            dof[node_id(i, j)] = idx
            idx += 1
    n_dof = idx

    # Precompute local matrices for the two triangles in a reference cell
    # Cell corners: (0,0), (h,0), (0,h), (h,h)
    p00 = (0.0, 0.0)
    p10 = (h, 0.0)
    p01 = (0.0, h)
    p11 = (h, h)

    # Two triangles: (00,10,11) and (00,11,01)
    K1_ref, m1, _ = _p1_triangle_matrices(p00, p10, p11)
    K2_ref, m2, _ = _p1_triangle_matrices(p00, p11, p01)

    # Sparse assembly in COO triplets
    rows = []
    cols = []
    data = []
    b = np.zeros(n_dof, dtype=np.float64)

    for j in range(N - 1):
        for i in range(N - 1):
            # Nodal indices of the square cell corners
            n00 = node_id(i, j)
            n10 = node_id(i + 1, j)
            n01 = node_id(i, j + 1)
            n11 = node_id(i + 1, j + 1)

            # Map nodal a to a cellwise constant coefficient
            a_cell = 0.25 * (a_nodes[i, j] + a_nodes[i + 1, j] + a_nodes[i, j + 1] + a_nodes[i + 1, j + 1])
            if cell_coef == "majority":
                a_cell = 12.0 if a_cell >= 7.5 else 3.0
            elif cell_coef != "mean":
                raise ValueError("cell_coef must be 'mean' or 'majority'.")

            # Triangle 1: (00,10,11)
            tri1 = [n00, n10, n11]
            Kloc = a_cell * K1_ref
            floc = f_value * m1

            for a in range(3):
                I = dof[tri1[a]]
                if I >= 0:
                    b[I] += floc[a]
                    for bb in range(3):
                        J = dof[tri1[bb]]
                        if J >= 0:
                            rows.append(I)
                            cols.append(J)
                            data.append(Kloc[a, bb])

            # Triangle 2: (00,11,01)
            tri2 = [n00, n11, n01]
            Kloc = a_cell * K2_ref
            floc = f_value * m2

            for a in range(3):
                I = dof[tri2[a]]
                if I >= 0:
                    b[I] += floc[a]
                    for bb in range(3):
                        J = dof[tri2[bb]]
                        if J >= 0:
                            rows.append(I)
                            cols.append(J)
                            data.append(Kloc[a, bb])

    A = sp.coo_matrix((data, (rows, cols)), shape=(n_dof, n_dof)).tocsr()
    A.sum_duplicates()

    # Jacobi diagonal preconditioner
    diag = A.diagonal()
    if np.any(diag <= 0):
        raise RuntimeError("Non-positive diagonal encountered; check coefficient positivity and assembly.")

    M_inv = spla.LinearOperator((n_dof, n_dof), matvec=lambda x: x / diag)

    # Solve with preconditions conjugate gradients
    u_int, cg_info = spla.cg(A, b, M=M_inv, rtol=tol, maxiter=maxiter)

    info = {
        "cg_info": cg_info,   # 0 means converged
        "n_dof": n_dof,
    }

    # Scatter interior solution back to full grid with boundary zero
    u = np.zeros((N, N), dtype=np.float64)
    for j in range(1, N - 1):
        for i in range(1, N - 1):
            I = dof[node_id(i, j)]
            u[i, j] = u_int[I]

    return u, info
