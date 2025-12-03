import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
import time

def assemble_divAgrad_dirichlet(
    N,
    a,                 # either callable a(x,y) or ndarray (N,N) of nodal values on interior points
    f,                 # either callable f(x,y) or ndarray (N,N) on interior points
    g=None             # Dirichlet BC. None -> 0. Either callable g(x,y) or dict/tuple of arrays for sides
):
    """
    Assemble the linear system A u = b for -div(a grad u) = f on (0,1)^2 with Dirichlet BC on all sides.
    Grid: N x N interior points, uniform spacing h = 1/(N+1). Unknowns ordered lexicographically:
      k = i + N*j with i=0..N-1 (x-index), j=0..N-1 (y-index).
    Returns:
      A (csr_matrix of shape (N*N, N*N)), b (np.ndarray of shape (N*N,)),
      x (N,), y (N,), and a helper lambda vec_to_grid(u) -> (N,N) array in the same ordering.
    Notes:
      * If 'a' is an ndarray (N,N), face values are built by harmonic means for interior faces and by copying the
        adjacent interior value on boundary faces.
      * 'g' can be:
          - None: homogeneous (g=0);
          - callable g(x,y);
          - dict with keys {'west','east','south','north'} each an (N,) array, or a 4-tuple (gW,gE,gS,gN).
    """
    h = 1.0 / (N + 1)
    x = h * (np.arange(N) + 1)  # interior x
    y = h * (np.arange(N) + 1)  # interior y

    # Helpers to evaluate data
    def eval_on_nodes(func_or_arr):
        if callable(func_or_arr):
            X, Y = np.meshgrid(x, y, indexing='ij')
            return func_or_arr(X, Y)
        arr = np.asarray(func_or_arr)
        if arr.shape != (N, N):
            raise ValueError(f"Expected array of shape {(N,N)} for nodal data.")
        return arr

    # Build face-centered a on vertical (x) and horizontal (y) faces
    def build_face_a(a_in):
        if callable(a_in):
            # vertical faces: (N+1, N), at (x_{i+1/2}, y_j)
            xv = h*(np.arange(N+1) + 0.5)[:, None]
            yv = y[None, :]
            ax = a_in(xv, yv)

            # horizontal faces: (N, N+1), at (x_i, y_{j+1/2})
            xh = x[:, None]
            yh = h*(np.arange(N+1) + 0.5)[None, :]
            ay = a_in(xh, yh)
        else:
            an = eval_on_nodes(a_in)  # (N,N) at interior nodes
            # Harmonic mean for interior faces; copy adjacent interior value on domain faces
            ax = np.empty((N+1, N), dtype=float)
            ay = np.empty((N, N+1), dtype=float)

            # vertical faces between i-1 and i for i=1..N-1
            for i in range(1, N):
                ai = an[i, :]     # at node i
                aim = an[i-1, :]  # at node i-1
                ax[i, :] = 2.0 * ai * aim / (ai + aim)
            ax[0, :]  = an[0, :]      # west boundary face ≈ adjacent interior
            ax[N, :]  = an[N-1, :]    # east boundary face ≈ adjacent interior

            # horizontal faces between j-1 and j for j=1..N-1
            for j in range(1, N):
                aj  = an[:, j]
                ajm = an[:, j-1]
                ay[:, j] = 2.0 * aj * ajm / (aj + ajm)
            ay[:, 0]  = an[:, 0]      # south boundary face
            ay[:, N]  = an[:, N-1]    # north boundary face
        return ax, ay

    ax, ay = build_face_a(a)

    fn = eval_on_nodes(f).astype(float)

    # Prepare Dirichlet boundary values
    if g is None:
        gW = np.zeros(N)
        gE = np.zeros(N)
        gS = np.zeros(N)
        gN = np.zeros(N)
    elif callable(g):
        gW = g(np.zeros(N), y)
        gE = g(np.ones(N),  y)
        gS = g(x, np.zeros(N))
        gN = g(x, np.ones(N))
    else:
        if isinstance(g, dict):
            gW = np.asarray(g['west'])
            gE = np.asarray(g['east'])
            gS = np.asarray(g['south'])
            gN = np.asarray(g['north'])
        else:  # assume 4-tuple/list
            gW, gE, gS, gN = map(np.asarray, g)
        for arr in (gW, gE, gS, gN):
            if arr.shape != (N,):
                raise ValueError("Boundary arrays must have shape (N,)")

    # Triplet assembly
    rows, cols, data = [], [], []
    b = fn.reshape(N*N, order='F').copy()  # Fortran order so k=i+N*j matches reshape(..., order='F')

    def idx(i, j):  # i in [0,N-1], j in [0,N-1]
        return i + N * j

    invh2 = 1.0 / (h*h)

    for j in range(N):
        for i in range(N):
            k = idx(i, j)

            # Face coefficients at this node
            aL = ax[i,   j]     # face between (i-1,j) and (i,j)   (west)
            aR = ax[i+1, j]     # face between (i,  j) and (i+1,j) (east)
            aS = ay[i,   j]     # face between (i,  j-1) and (i,  j) (south)
            aN = ay[i,   j+1]   # face between (i,  j)   and (i,  j+1) (north)

            diag = (aL + aR + aS + aN) * invh2
            rows.append(k); cols.append(k); data.append(diag)

            # West neighbor / boundary
            if i > 0:
                rows.append(k); cols.append(idx(i-1, j)); data.append(-aL * invh2)
            else:
                b[k] += aL * invh2 * gW[j]

            # East
            if i < N-1:
                rows.append(k); cols.append(idx(i+1, j)); data.append(-aR * invh2)
            else:
                b[k] += aR * invh2 * gE[j]

            # South
            if j > 0:
                rows.append(k); cols.append(idx(i, j-1)); data.append(-aS * invh2)
            else:
                b[k] += aS * invh2 * gS[i]

            # North
            if j < N-1:
                rows.append(k); cols.append(idx(i, j+1)); data.append(-aN * invh2)
            else:
                b[k] += aN * invh2 * gN[i]

    A = coo_matrix((np.array(data), (np.array(rows), np.array(cols))),
                   shape=(N*N, N*N)).tocsr()

    vec_to_grid = lambda u: np.asarray(u).reshape((N, N), order='F')
    return A, b, x, y, vec_to_grid

# ---------------------------
if __name__ == "__main__":
    tic = time.perf_counter()
    N = 1024
    a = lambda X, Y: 1.5*np.exp(-X**2/0.2-Y**2/0.2) # constant coefficient
    f = lambda X, Y: np.sin(-np.pi*X)*np.sin(1.5*np.pi*Y)  # unit source
    x = np.linspace(0,1,N)

    gW = np.sin(2*np.pi*x)
    gE = np.zeros(N)
    gS = np.zeros(N)
    gN = np.zeros(N)
    g = (gW,gE,gN,gN)

    A, b, x, y, to_grid = assemble_divAgrad_dirichlet(N, a, f, g)

    # Solve
    from scipy.sparse.linalg import spsolve
    u_vec = spsolve(A, b)
    U = to_grid(u_vec)  # shape (N,N)
    toc = time.perf_counter()
    print(f"Time taken: {toc-tic:.3f}s")
    import matplotlib.pyplot as plt
    plt.imshow(
        U.T,                 # transpose to make columns = x
        origin='lower',      # y=0 at the bottom
        extent=[x[0], x[-1], y[0], y[-1]],
        aspect='equal'
    )
    plt.xlabel('x'); plt.ylabel('y'); plt.colorbar(); plt.title('u(x,y)')
    plt.show()