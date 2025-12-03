import numpy as np
from Darcy2DAssemble import assemble_divAgrad_dirichlet
from sampleRF import RFsample
from scipy.sparse.linalg import spsolve
import h5py
from pathlib import Path

class PDEH5Appender:
    """
    Single-file append-only writer for (a, u) pairs of shape [C, H, W].
    For your case C=1, H=W=128, but the class allows general C/H/W.
    """
    def __init__(self, path, C_a=1, C_u=1, H=128, W=128,
                 compression="gzip", chunk_N=256):
        self.path = Path(path)
        self.Ca, self.Cu, self.H, self.W = C_a, C_u, H, W
        mode = "a"  # create if missing, append if exists
        self.f = h5py.File(self.path, mode)

        g = self.f.require_group("data")
        # Create resizable datasets if they don't exist
        self.a = g.require_dataset(
            "a",
            shape=(0, C_a, H, W),
            maxshape=(None, C_a, H, W),
            chunks=(chunk_N, C_a, H, W),
            dtype="f4",
            compression=compression,
            shuffle=True,
        )
        self.u = g.require_dataset(
            "u",
            shape=(0, C_u, H, W),
            maxshape=(None, C_u, H, W),
            chunks=(chunk_N, C_u, H, W),
            dtype="f4",
            compression=compression,
            shuffle=True,
        )

        # Optional metadata (set once; update as needed)
        if "attrs_written" not in self.f.attrs:
            self.f.attrs.update(dict(
                grid="tensor-product",
                H=H, W=W, C_a=C_a, C_u=C_u,
                dtype="float32",
            ))
            self.f.attrs["attrs_written"] = True

    @property
    def N(self):
        return self.a.shape[0]

    def append(self, a_sample: np.ndarray, u_sample: np.ndarray):
        """
        Append ONE sample with shapes [C_a, H, W] and [C_u, H, W].
        """
        # Validate and cast
        a = np.asarray(a_sample, dtype=np.float32)
        u = np.asarray(u_sample, dtype=np.float32)
        assert a.shape == (self.Ca, self.H, self.W)
        assert u.shape == (self.Cu, self.H, self.W)

        n0, n1 = self.N, self.N + 1
        self.a.resize((n1, self.Ca, self.H, self.W))
        self.u.resize((n1, self.Cu, self.H, self.W))
        self.a[n0] = a
        self.u[n0] = u
        self.f.flush()  # make sure it’s on disk

    def append_batch(self, a_batch: np.ndarray, u_batch: np.ndarray):
        """
        Append a BATCH with shapes [B, C_a, H, W] and [B, C_u, H, W].
        """
        a = np.asarray(a_batch, dtype=np.float32)
        u = np.asarray(u_batch, dtype=np.float32)
        assert a.ndim == 4 and u.ndim == 4
        assert a.shape[1:] == (self.Ca, self.H, self.W)
        assert u.shape[1:] == (self.Cu, self.H, self.W)
        B = a.shape[0]

        n0, n1 = self.N, self.N + B
        self.a.resize((n1, self.Ca, self.H, self.W))
        self.u.resize((n1, self.Cu, self.H, self.W))
        self.a[n0:n1] = a
        self.u[n0:n1] = u
        self.f.flush()

    def close(self):
        if self.f:
            self.f.close()
            self.f = None


if __name__ == "__main__":

    buf_a, buf_u = [], []
    n_datapoints = 1000
    grid_size = 256
    field_type = "threshold"

    if field_type == "threshold":
        file_name = f"threshold_dataset_{grid_size}_{n_datapoints}.h5"
    elif field_type == "log_normal":
        file_name = f"log_normal_dataset_{grid_size}_{n_datapoints}.h5"

    app = PDEH5Appender(file_name, C_a=1, C_u=1, H=grid_size, W=grid_size, compression="lzf", chunk_N=256)
    for i in range(n_datapoints):
        # Get sample from RF with periodic BC using FFT
        a = RFsample(L=1, N = grid_size)
        # Threshold field 
        if field_type == "threshold":
            a = np.where(a>=0, 12 , 3)
        elif field_type == "log_normal":
            a = np.exp(a)
        else:
            raise Exception("Invalid Field Type")
        # Unit source function
        f = lambda X, Y: np.ones_like(X)
        # Assemble system for Darcy Problem
        A, b, x, y, to_grid  = assemble_divAgrad_dirichlet(N=grid_size, a=a, f=f)
        u_sol = spsolve(A, b)
        # Reshape to (n,n)
        u = to_grid(u_sol)
        # append to buffers
        buf_a.append(np.expand_dims(a, 0)); buf_u.append(np.expand_dims(u,0))
        if len(buf_a) == 64:
            app.append_batch(np.stack(buf_a,0), np.stack(buf_u,0))
            buf_a.clear(); buf_u.clear()
        if (i+1) % 500 == 0:
            print(f"{i} of {n_datapoints} done..")

    # flush remainder
    if buf_a:
        app.append_batch(np.stack(buf_a,0), np.stack(buf_u,0))
    app.close()
