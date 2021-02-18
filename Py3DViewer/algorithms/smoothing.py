import numpy as np
from numba import jit
from scipy.sparse import lil_matrix as sp_matrix
from scipy.sparse import eye as identity
from scipy.sparse.linalg import spsolve

def laplacian_smoothing(mesh, lambda_=1.0, iterations=1):
    
    n = mesh.num_vertices
    e = mesh.edges
    A = sp_matrix((n, n))
    A[e[:,0], e[:,1]] = 1
    A[e[:,1], e[:,0]] = 1
    D = sp_matrix(A.shape)
    D.setdiag(np.sum(A, axis=1))
    L = D - A
    I = identity(n)
    
    for i in range(iterations):
        mesh.vertices[:] += spsolve( I + lambda_ * L, mesh.vertices) - mesh.vertices[:]
        
@jit(nopython=True, parallel=True, cache=True)
def __taubin_smoothing_internal(n, e, vertices, lambda_, mu, iterations):
    A = np.zeros((n, n))
    for i in range(e.shape[0]):
        A[e[i][0], e[i][1]] = 1
        A[e[i][1], e[i][0]] = 1
    for i in range(A.shape[0]):
        A[i]/=np.sum(A[i])
    I = np.eye(n)
    K = I - A
    for i in range(iterations):
        vertices = (I - lambda_*K)@vertices
        vertices = (I - mu*K)@vertices
    return vertices

def taubin_smoothing(mesh, lambda_ = 0.89, mu = -0.9, iterations = 1):
    n = mesh.num_vertices
    e = mesh.edges
    mesh.vertices[:] = __taubin_smoothing_internal(n, e, mesh.vertices, lambda_, mu, iterations)
