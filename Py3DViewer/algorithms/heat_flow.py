from ..utils import matrices
from scipy.sparse.linalg import spsolve
import numpy as np

def solve_heat_flow(mesh, heat_sources, time):
    
    heat_sources = np.array(heat_sources)
    #hs = heat_sources.shape[0]
    L = matrices.laplacian_matrix(mesh)
    MM = matrices.mass_matrix(mesh)
    rhs = np.zeros(mesh.num_vertices)
    rhs[heat_sources] = 1
    heat = spsolve(MM-time*L,rhs)
    return heat