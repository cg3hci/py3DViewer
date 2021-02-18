import numpy as np
from scipy.sparse import lil_matrix as sp_matrix
from .metrics import *

def laplacian_matrix(mesh):
    n = mesh.num_vertices
    #e =  np.c_[mesh.faces[:,:2], mesh.faces[:,1:], mesh.faces[:,2], mesh.faces[:,0]]
    e = mesh.edges
    A = sp_matrix((n, n))
    A[e[:,0], e[:,1]] = 1
    A[e[:,1], e[:,0]] = 1
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


def rotation_matrix(alpha, c):

    sin = np.sin(np.radians(alpha))
    if alpha > 0:
        cos = np.cos(np.radians(alpha))
    else:
        cos = -np.cos(np.radians(np.abs(alpha)))

    if type(c) is str or type(c) is int:
        if c == 'x' or c == 0:
            matrix = np.identity(4)
            matrix[1:3, 1:3] = [[cos, -sin], [sin, cos]]
        elif c =='y' or c == 1:
            matrix = np.identity(4)
            matrix[:3, :3] = [[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]]
        elif c == 'z' or c == 2:
            matrix = np.identity(4)
            matrix[:2, :2] = [[cos, -sin], [sin, cos]]
        else:
            raise Exception('Not a valid axis')
        return matrix
    else:
        raise Exception('Not a str')