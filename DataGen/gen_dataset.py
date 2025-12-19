import numpy as np
import h5py
from pathlib import Path
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# FEM precompute
def _p1_triangle_matrices(p0, p1, p2):
    x0, y0 = p0
    x1, y1 = p1
    x2, y2 = p2

    detJ2 = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    area = 0.5 * abs(detJ2)
    if area == 0.0:
        raise ValueError("Degenerate triangle encountered.")

    b = np.array([y1 - y2, y2 - y0, y0 - y1], dtype=np.float64)
    c = np.array([x2 - x1, x0 - x2, x1 - x0], dtype=np.float64)
    grads = np.vstack([b, c]).T / (2.0 * area)  # (3,2)

    Kref = area * (grads @ grads.T)  # 3x3
    m = np.full(3, area / 3.0, dtype=np.float64)
    return Kref, m, area


class DarcyFEMP1DirichletPrecomputed:

    def __init__(self, N_nodes, L=1.0, f_value=1.0, cell_coef="mean"):
        self.N = int(N_nodes)
        self.L = float(L)
        self.f_value = float(f_value)
        self.cell_coef = cell_coef

        if self.N < 3:
            raise ValueError("Need N_nodes >= 3.")

        self.h = self.L / (self.N - 1)
        self.n_total = self.N * self.N

        def node_id(i, j):
            return i * self.N + j
        self.node_id = node_id

        # DOF map: interior nodes only
        dof = -np.ones(self.n_total, dtype=np.int32)
        idx = 0
        for i in range(1, self.N - 1):
            for j in range(1, self.N - 1):
                dof[node_id(i, j)] = idx
                idx += 1
        self.dof = dof
        self.n_dof = int(idx)

        # Reference element matrices for the two triangles in each cell
        h = self.h
        p00 = (0.0, 0.0)
        p10 = (h, 0.0)
        p01 = (0.0, h)
        p11 = (h, h)

        K1_ref, m1, _ = _p1_triangle_matrices(p00, p10, p11)  # (00,10,11)
        K2_ref, m2, _ = _p1_triangle_matrices(p00, p11, p01)  # (00,11,01)

        rows = []
        cols = []
        wts  = []
        cids = []

        b = np.zeros(self.n_dof, dtype=np.float64)

        Nc = self.N - 1
        cell_id = 0

        for i in range(Nc):
            for j in range(Nc):
                n00 = node_id(i, j)
                n10 = node_id(i + 1, j)
                n01 = node_id(i, j + 1)
                n11 = node_id(i + 1, j + 1)

                # Triangle 1: (00,10,11)
                tri1 = [n00, n10, n11]
                floc1 = self.f_value * m1
                for a_loc in range(3):
                    I = dof[tri1[a_loc]]
                    if I >= 0:
                        b[I] += floc1[a_loc]
                        for b_loc in range(3):
                            J = dof[tri1[b_loc]]
                            if J >= 0:
                                rows.append(I); cols.append(J)
                                wts.append(K1_ref[a_loc, b_loc])
                                cids.append(cell_id)

                # Triangle 2: (00,11,01)
                tri2 = [n00, n11, n01]
                floc2 = self.f_value * m2
                for a_loc in range(3):
                    I = dof[tri2[a_loc]]
                    if I >= 0:
                        b[I] += floc2[a_loc]
                        for b_loc in range(3):
                            J = dof[tri2[b_loc]]
                            if J >= 0:
                                rows.append(I); cols.append(J)
                                wts.append(K2_ref[a_loc, b_loc])
                                cids.append(cell_id)

                cell_id += 1

        self.b = b
        rows = np.asarray(rows, dtype=np.int32)
        cols = np.asarray(cols, dtype=np.int32)
        wts  = np.asarray(wts,  dtype=np.float64)
        cids = np.asarray(cids, dtype=np.int32)

        # Compress duplicates once 
        key = rows.astype(np.int64) * self.n_dof + cols.astype(np.int64)
        uniq_key, inv = np.unique(key, return_inverse=True)  # sorted by (row,col)
        self.inv = inv.astype(np.int32)
        self.wts = wts
        self.cids = cids

        rows_u = (uniq_key // self.n_dof).astype(np.int32)
        cols_u = (uniq_key - rows_u.astype(np.int64) * self.n_dof).astype(np.int32)

        counts = np.bincount(rows_u, minlength=self.n_dof).astype(np.int32)
        indptr = np.empty(self.n_dof + 1, dtype=np.int32)
        indptr[0] = 0
        np.cumsum(counts, out=indptr[1:])

        indices = cols_u.astype(np.int32)
        data = np.zeros_like(indices, dtype=np.float64)
        A = sp.csr_matrix((data, indices, indptr), shape=(self.n_dof, self.n_dof))

        # Diagonal positions for fast Jacobi
        diag_pos = np.full(self.n_dof, -1, dtype=np.int64)
        for r in range(self.n_dof):
            start, end = indptr[r], indptr[r + 1]
            cols_row = indices[start:end]
            hit = np.where(cols_row == r)[0]
            if hit.size:
                diag_pos[r] = start + hit[0]
        if np.any(diag_pos < 0):
            raise RuntimeError("Diagonal entry missing in CSR structure")

        self.A = A
        self.diag_pos = diag_pos

    def _cellwise_a(self, a_nodes):
        a = np.asarray(a_nodes, dtype=np.float64)
        if a.shape != (self.N, self.N):
            raise ValueError(f"a_nodes must have shape ({self.N},{self.N}). Got {a.shape}.")

        a_cell = 0.25 * (a[:-1, :-1] + a[1:, :-1] + a[:-1, 1:] + a[1:, 1:])
        if self.cell_coef == "majority":
            a_cell = np.where(a_cell >= 7.5, 12.0, 3.0)
        elif self.cell_coef != "mean":
            raise ValueError("cell_coef must be 'mean' or 'majority'.")

        return a_cell.ravel(order="C")
    # Spare conjugate gradients
    def solve(self, a_nodes, tol=1e-9, maxiter=2000):
        a_cell_flat = self._cellwise_a(a_nodes)

        contrib = self.wts * a_cell_flat[self.cids]
        data_u = np.bincount(self.inv, weights=contrib, minlength=self.A.data.size).astype(np.float64)
        self.A.data[:] = data_u

        diag = self.A.data[self.diag_pos]
        M_inv = spla.LinearOperator((self.n_dof, self.n_dof), matvec=lambda x: x / diag)

        u_int, cg_info = spla.cg(self.A, self.b, M=M_inv, rtol=tol, maxiter=maxiter)

        u = np.zeros((self.N, self.N), dtype=np.float64)
        u[1:-1, 1:-1] = u_int.reshape((self.N - 2, self.N - 2), order="C")
        return u, {"cg_info": cg_info, "n_dof": self.n_dof}


# HDF5 writer
class PDEH5Appender:
    def __init__(self, path, C_a=1, C_u=1, H=257, W=257,
                 compression=None, chunk_N=64):
        self.path = Path(path)
        self.Ca, self.Cu, self.H, self.W = C_a, C_u, H, W

        self.f = h5py.File(self.path, "a")
        g = self.f.require_group("data")

        self.a = g.require_dataset(
            "a",
            shape=(0, C_a, H, W),
            maxshape=(None, C_a, H, W),
            chunks=(chunk_N, C_a, H, W),
            dtype="f4",
            compression=compression,
            shuffle=False,
        )
        self.u = g.require_dataset(
            "u",
            shape=(0, C_u, H, W),
            maxshape=(None, C_u, H, W),
            chunks=(chunk_N, C_u, H, W),
            dtype="f4",
            compression=compression,
            shuffle=False,
        )

    @property
    def N(self):
        return self.a.shape[0]

    def append_batch(self, a_batch, u_batch):
        a = np.asarray(a_batch, dtype=np.float32)
        u = np.asarray(u_batch, dtype=np.float32)
        assert a.shape[1:] == (self.Ca, self.H, self.W)
        assert u.shape[1:] == (self.Cu, self.H, self.W)
        B = a.shape[0]

        n0, n1 = self.N, self.N + B
        self.a.resize((n1, self.Ca, self.H, self.W))
        self.u.resize((n1, self.Cu, self.H, self.W))
        self.a[n0:n1] = a
        self.u[n0:n1] = u

    def close(self):
        if self.f:
            self.f.flush()
            self.f.close()
            self.f = None


# Dataset generation
if __name__ == "__main__":
    from sample_rf import RFsample
    n_datapoints = 2500
    N_cells = 256
    N_nodes = N_cells + 1
    BATCH = 64
    file_name = f"lognormal_dataset_test_nodes{N_nodes}_{n_datapoints}.h5"

    # Precompute FEM structures once
    solver = DarcyFEMP1DirichletPrecomputed(
        N_nodes=N_nodes,
        L=1.0,
        f_value=1.0,
        cell_coef="mean",
    )

    app = PDEH5Appender(
        file_name,
        C_a=1, C_u=1,
        H=N_nodes, W=N_nodes,
        chunk_N=BATCH,
    )

    # Preallocate batch buffers
    a_batch = np.empty((BATCH, 1, N_nodes, N_nodes), dtype=np.float32)
    u_batch = np.empty((BATCH, 1, N_nodes, N_nodes), dtype=np.float32)
    bcount = 0

    try:
        for i in range(n_datapoints):
            # Avoid failed sample, if it so rarely happens that cg does not conv...
            while True:
                # Sample on 256x256 periodic grid, threshold, then wrap-pad to 257x257 nodal grid
                a_256 = RFsample(L=1.0, N=N_cells)
                #a_256 = np.where(a_256 >= 0, 12.0, 3.0) # threshold field
                a_256 = np.exp(a_256)   # log-normal field
                a_nodes = np.pad(a_256, ((0, 1), (0, 1)), mode="wrap")  # (257,257)

                u, info = solver.solve(a_nodes, tol=1e-9, maxiter=2000)

                if info["cg_info"] == 0:
                    break

            a_batch[bcount, 0] = a_nodes.astype(np.float32, copy=False)
            u_batch[bcount, 0] = u.astype(np.float32, copy=False)
            bcount += 1

            if bcount == BATCH:
                app.append_batch(a_batch, u_batch)
                bcount = 0

            if (i + 1) % 100 == 0:
                print(f"{i + 1} / {n_datapoints} samples generated")

        if bcount > 0:
            app.append_batch(a_batch[:bcount], u_batch[:bcount])

    finally:
        app.close()
