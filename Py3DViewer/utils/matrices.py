import numpy as np
from scipy.sparse import lil_matrix as sp_matrix
from .metrics import *

def laplacian_matrix(mesh):
    n = mesh.num_vertices
    #e =  np.c_[mesh.faces[:,:2], mesh.faces[:,1:], mesh.faces[:,2], mesh.faces[:,0]]
    e = mesh.edges
    A = sp_matrix((n, n))
    A[e[:,0], e[:,1]] = -1
    D = sp_matrix(A.shape)
    D.setdiag(np.sum(A, axis=1))
    L = D-A
    return L


def mass_matrix(mesh):
    
    nv = mesh.num_vertices
    mass = np.ones((nv))

    for i in range(nv):
        volumes = []
        if mesh.mesh_is_volumetric:
            if mesh.polys.shape[1] == 4:
                 _ , volumes = tet_volume(mesh.vertices, mesh.polys[mesh.adj_vtx2poly[i]])
            else:
                _ , volumes = hex_volume(mesh.vertices, mesh.polys[mesh.adj_vtx2poly[i]])
           
        elif mesh.mesh_is_surface:
            if(mesh.polys.shape[1] == 3):
                _ , volumes = triangle_area(mesh.vertices, mesh.polys[mesh.adj_vtx2poly[i]])
            else:
                _ , volumes = quad_area(mesh.vertices, mesh.polys[mesh.adj_vtx2poly[i]])
        
        mass[i] *= (np.sum(volumes) / volumes.shape[0])
    
    MM = sp_matrix((nv,nv))
    MM.setdiag(mass)
    
    return MM