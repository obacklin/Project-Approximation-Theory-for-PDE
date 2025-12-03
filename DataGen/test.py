from Darcy2DAssemble import assemble_divAgrad_dirichlet
from sampleRF import RFsample
import numpy as np
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import time

# Time test 100 samples:

acoeff = RFsample(L = 1.0, N=128)
acoeff = np.where(acoeff >= 0, 12, 3) # To satify coercivity assumption
f = lambda X, Y: np.ones_like(X)  # unit source
A, b, x, y, to_grid  = assemble_divAgrad_dirichlet(N=128, a = acoeff, f=f )
u_vec = spsolve(A, b)
U = to_grid(u_vec)  # shape (N,N)

print(type(U))
print(type(acoeff))
print(U.shape)
print(acoeff.shape)
print(acoeff.dtype)
print(U.dtype)
print(np.expand_dims(acoeff, 0).shape)
