import numpy as np
from numba import njit, float64, int64
from numba.types import Tuple

@njit(Tuple((float64[:,::1], int64[:,::1]))(float64[:,::1],int64[:,::1]), cache=True)
def remove_duplicated_vertices(vertices, faces):
    
    vtx_dictionary = dict()
    support_set = set()
    vtx_dictionary[(-1.,-1.,-1.)] = -1.
    support_set.add((-1.,-1.,-1.))
    new_vertices = np.zeros(vertices.shape, dtype=np.float64)
    
    j=0
    for i in range(vertices.shape[0]):
        
        v = (vertices[i][0], vertices[i][1], vertices[i][2])
        
        if v not in support_set:
            
            vtx_dictionary[v] = i
            support_set.add(v)
            new_vertices[j] = vertices[i]
            j+=1
        
        else:
            idx = vtx_dictionary[v]
            r = np.where(faces==i)
            for k in zip(r[0], r[1]):
                faces[k[0]][k[1]] = idx
    
    
    return new_vertices[:j], faces