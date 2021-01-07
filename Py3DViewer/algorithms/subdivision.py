import numpy as np
from numba import njit, float64, int64
from numba.types import Tuple


@njit(Tuple((float64[:,::1], int64[:,::1]))(float64[:,::1],int64[:,::1], int64[:,::1]), cache=True)
def _mid_point_subdivision_trimesh(vertices, faces, edges):
    
    vtx_dictionary = dict()
    support_set = set()
    vtx_dictionary[(-1.,-1.,-1.)] = -1
    support_set.add((-1.,-1.,-1.))
    n_vertices = vertices.shape[0]
    
    new_vertices = np.empty((edges.shape[0], 3), dtype=np.float64)
    new_faces    = np.empty((faces.shape[0]*4, 3), dtype=np.int64)
    
    j = 0
    for i in range(faces.shape[0]):
        
        new_vtx1 = (vertices[edges[i*3][0]] + vertices[edges[i*3][1]]) / 2
        new_vtx2 = (vertices[edges[i*3+1][0]] + vertices[edges[i*3+1][1]]) / 2
        new_vtx3 = (vertices[edges[i*3+2][0]] + vertices[edges[i*3+2][1]]) / 2
        
        index0 = 0
        index1 = 0
        index2 = 0
        
        v1 = (new_vtx1[0], new_vtx1[1], new_vtx1[2])
        v2 = (new_vtx2[0], new_vtx2[1], new_vtx2[2])
        v3 = (new_vtx3[0], new_vtx3[1], new_vtx3[2])
        
        if v1 not in support_set:
            
            new_vertices[j] = new_vtx1
            vtx_dictionary[v1] = j+n_vertices
            index0 = j+n_vertices
            j+=1
            
        else:
            index0 = vtx_dictionary[v1]
        
        if v2 not in support_set:
            
            new_vertices[j] = new_vtx2
            vtx_dictionary[v2] = j+n_vertices
            index1 = j+n_vertices
            j+=1
            
        else:
            index1 = vtx_dictionary[v2]
            
        if v3 not in support_set:
            
            new_vertices[j] = new_vtx3
            vtx_dictionary[v3] = j+n_vertices
            index2 = j+n_vertices
            j+=1
            
        else:
            index2 = vtx_dictionary[v3]
            
        
        new_faces[i*4] = np.array([faces[i][0], index0, index2])
        new_faces[i*4+1] = np.array([index0, index1, index2])
        new_faces[i*4+2] = np.array([index0, faces[i][1], index1])
        new_faces[i*4+3] = np.array([index2, index1, faces[i][2]])
        
    return np.concatenate((vertices, new_vertices[:j]), axis=0), new_faces



def mid_point_subdivision(mesh):

    if 'Trimesh' in type(mesh):
        v, f = _mid_point_subdivision_trimesh(mesh.vertices, mesh.faces, mesh.edges)
    else:
        print("Implemented only for Trimesh")
        return
    mesh.__init__(vertices=v, faces=f)


def hex_to_tet_subdivision(hexes, subdivision_rule=3):

    if subdivision_rule == 0:
        split_rules = np.array([[0,1,2,5], [0,2,7,5], [0,2,3,7], [0,5,7,4], [2,7,5,6]], dtype=np.int)
    elif subdivision_rule == 1:
        split_rules = np.array([[0,5,7,4], [0,1,7,5], [1,6,7,5], [0,7,2,3], [0,7,1,2], [1,7,6,2]], dtype=np.int)
    elif subdivision_rule == 2:
        split_rules = np.array([[0,4,5,6], [0,3,7,6], [0,7,4,6], [0,1,2,5], [0,3,6,2], [0,6,5,2]], dtype=np.int)
    elif subdivision_rule == 3:
        split_rules = np.array([[0,2,3,6], [0,3,7,6], [0,7,4,6], [0,5,6,4], [1,5,6,0], [1,6,2,0]], dtype=np.int)
    else:
        raise ValueError("subdivision_rule must be an integer between 0 and 3")

    tetrahedra = np.ascontiguousarray(hexes[:,split_rules])
    tetrahedra.shape = (-1,4)
    return tetrahedra