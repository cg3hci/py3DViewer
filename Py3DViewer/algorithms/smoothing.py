import numpy as np
from scipy.sparse import lil_matrix as sp_matrix
from scipy.sparse import eye as identity
from scipy.sparse.linalg import spsolve

def laplacian_smoothing(mesh, lambda_=1.0, iterations=1):
    
    n = mesh.num_vertices
    e = mesh.edges
    A = sp_matrix((n, n))
    A[e[:,0], e[:,1]] = 1
    D = sp_matrix(A.shape)
    D.setdiag(np.sum(A, axis=1))
    L = D - A
    I = identity(n)
    
    for i in range(iterations):
        mesh.vertices[:] += spsolve( I + lambda_ * L, mesh.vertices) - mesh.vertices[:]
        

