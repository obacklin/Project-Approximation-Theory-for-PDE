import numpy as np
import matplotlib.pyplot as plt
from sample_rf import RFsample
from FEM_solver import solve_darcy_fem_p1_dirichlet

def plot_darcy_solution(u, a, L=1.0, title=None):
    """
    Visualize coefficient a(x,y) and FEM solution u(x,y) on [0,L]^2.

    u, a are (N,N) arrays with indexing u[i,j] where i->x and j->y.
    We transpose for imshow so axes match (x horizontal, y vertical).
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    im_a = axs[0].imshow(
        a.T, origin="lower", extent=(0, L, 0, L), aspect="equal"
    )
    axs[0].set_title("Coefficient a(x,y)")
    axs[0].set_xlabel("x"); axs[0].set_ylabel("y")
    fig.colorbar(im_a, ax=axs[0])

    im_u = axs[1].imshow(
        u.T, origin="lower", extent=(0, L, 0, L), aspect="equal"
    )
    axs[1].set_title("Solution u(x,y)")
    axs[1].set_xlabel("x"); axs[1].set_ylabel("y")
    fig.colorbar(im_u, ax=axs[1])

    if title is not None:
        fig.suptitle(title)

    plt.show()

def plot_solution_surface(u, L=1.0, stride=2, elev=35, azim=-60, title="u(x,y) surface"):
    """
    3D surface plot for u on [0,L]^2.

    u is assumed indexed as u[i,j] with i->x and j->y (as in the FEM code).
    """
    u = np.asarray(u)
    N = u.shape[0]
    x = np.linspace(0.0, L, N)
    X, Y = np.meshgrid(x, x, indexing="ij")

    # Optional downsample for speed/clarity
    Xp = X[::stride, ::stride]
    Yp = Y[::stride, ::stride]
    Up = u[::stride, ::stride]

    fig = plt.figure(figsize=(10, 7), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(Xp, Yp, Up, linewidth=0, antialiased=True)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u")
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title)

    fig.colorbar(surf, ax=ax, shrink=0.7, pad=0.08)
    plt.show()
def plot_solution_surface_colormap(u, L=1.0, stride=2, elev=35, azim=-60,
                                   title="u(x,y) surface (colormap)",
                                   robust=True, robust_q=(1.0, 99.0)):
    """
    3D surface with colormap + colorbar. Two key points:
      - u is indexed as u[i,j] with i->x, j->y (matches indexing='ij')
      - robust=True uses percentiles for vmin/vmax to enhance contrast
    """
    u = np.asarray(u, dtype=np.float64)
    N = u.shape[0]
    x = np.linspace(0.0, L, N)
    X, Y = np.meshgrid(x, x, indexing="ij")

    Xp = X[::stride, ::stride]
    Yp = Y[::stride, ::stride]
    Up = u[::stride, ::stride]

    if robust:
        vmin, vmax = np.percentile(Up, robust_q)
    else:
        vmin, vmax = float(np.min(Up)), float(np.max(Up))

    fig = plt.figure(figsize=(10, 7), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        Xp, Yp, Up,
        cmap="viridis",   # explicit colormap, as requested
        vmin=vmin, vmax=vmax,
        linewidth=0,
        antialiased=True
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u")
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title)

    fig.colorbar(surf, ax=ax, shrink=0.7, pad=0.08)
    plt.show()


def plot_solution_surface_wireframe(u, L=1.0, stride=2, elev=35, azim=-60,
                                    title="u(x,y) surface (wireframe)"):
    """
    3D wireframe plot.
    """
    u = np.asarray(u, dtype=np.float64)
    N = u.shape[0]
    x = np.linspace(0.0, L, N)
    X, Y = np.meshgrid(x, x, indexing="ij")

    Xp = X[::stride, ::stride]
    Yp = Y[::stride, ::stride]
    Up = u[::stride, ::stride]

    fig = plt.figure(figsize=(10, 7), constrained_layout=False)
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_wireframe(Xp, Yp, Up)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u")
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title)

    plt.show()


if __name__ == "__main__":
    N = 256
    a_256 = RFsample(L=1.0, N=N)      # your sampler
    a_256 = np.where(a_256 >= 0, 12.0, 3.0)
    # Make nodal coefficient on 257x257 nodes by periodic wrap (adds last row/col)
    a_nodes = np.pad(a_256, ((0, 1), (0, 1)), mode="wrap")  # shape (257,257)
    u, info = solve_darcy_fem_p1_dirichlet(a_nodes, f_value=1.0, L=1.0, cell_coef="mean")
    print(info)
    plot_solution_surface_wireframe(u, L=1.0, stride=2)
