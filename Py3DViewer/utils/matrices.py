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
        if(hasattr(mesh, 'hexes')):
            v2p = mesh.vtx2hex[i]
            _ , volumes = hex_volume(mesh.vertices, mesh.hexes[v2p])
        elif(hasattr(mesh, 'tets')):
            v2p = mesh.vtx2tet[i]
            _ , volumes = tet_volume(mesh.vertices, mesh.tets[v2p])
        elif(hasattr(mesh, 'faces')):
            v2p = mesh.vtx2face[i]
            if(mesh.faces.shape[1] == 3):
                _ , volumes = triangle_area(mesh.vertices, mesh.faces[v2p])
            else:
                _ , volumes = quad_area(mesh.vertices, mesh.faces[v2p])
        
        mass[i] *= (np.sum(volumes) / volumes.shape[0])
    
    MM = sp_matrix((nv,nv))
    MM.setdiag(mass)
    
    return MM