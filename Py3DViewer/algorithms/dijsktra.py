from numba import njit, float64, int64
from numba.types import Tuple, ListType as LT
import numpy as np

@njit((float64[:])(float64[:,::1], LT(LT(int64)), int64), cache=True)
def __dijsktra(vertices, adj_vtx2vtx, start):
    
    dists = np.full(vertices.shape[0], np.inf)    
    dists[start] = 0
    
    queue = []
    queue.append((0.0, start))
    
    while(len(queue) > 0):
       
        queue = sorted(queue)
        d, v = queue.pop(0)
       
        for adj in adj_vtx2vtx[v]:
            
            nd = np.round(dists[v] + np.linalg.norm(vertices[v]-vertices[adj]),9)
            if dists[adj] > nd:
                
                if dists[adj] < np.inf:
                    queue.remove((dists[adj], adj))
                    
                dists[adj] = nd
                queue.append((nd, adj))
    
    return dists


def dijsktra(mesh, start):
     """
        Returns the shortest distance between a vertex and all the other vertices of the mesh by using
        the dijsktra algorithm.

        Parameters:

            mesh : a mesh of any type
            start (int) : the starting vertex
        """
    return __dijsktra(mesh.vertices, mesh.adj_vtx2vtx, start)