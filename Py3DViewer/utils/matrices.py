import numpy as np
from scipy.sparse import lil_matrix as sp_matrix
from .metrics import *

def adjacency_matrix(mesh, type='std'):
    assert(type=='std' or type=='cot')
    n = mesh.num_vertices
    e = mesh.edges
    A = sp_matrix((n, n), dtype=np.float64)
    if type=='std':
        A[e[:,0], e[:,1]] = 1
        A[e[:,1], e[:,0]] = 1
    else:
        raise NotImplementedError("not implemented yet")

    return A

def degree_matrix(A):
    D = sp_matrix(A.shape, dtype=np.float64)
    D.setdiag(np.sum(A, axis=1))
    return D



def laplacian_matrix(mesh, type='std'):
    assert(type=='std' or type=='cot')
    A = adjacency_matrix(mesh, type)
    D = degree_matrix(A)
    L = D-A
    return L

def symmetric_normalized_laplacian_matrix(mesh, type='std'):
    assert(type=='std' or type=='cot')
    A = adjacency_matrix(mesh, type)
    D = degree_matrix(A)
    L = D-A
    D=D.power(-0.5) 
    return D*L*D

def random_walk_normalized_laplacian(mesh, type='std'):
    assert(type=='std' or type=='cot')
    A = adjacency_matrix(mesh, type)
    D = degree_matrix(A)
    L = D-A
    D=D.power(-1) 
    return D*L
    



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


#def rotation_matrix(alpha, c):
#
#    sin = np.sin(np.radians(alpha))
#    if alpha > 0:
#        cos = np.cos(np.radians(alpha))
#    else:
#        cos = -np.cos(np.radians(np.abs(alpha)))
#
#    if type(c) is str or type(c) is int:
#        if c == 'x' or c == 0:
#            matrix = np.identity(4)
#            matrix[1:3, 1:3] = [[cos, -sin], [sin, cos]]
#        elif c =='y' or c == 1:
#            matrix = np.identity(4)
#            matrix[:3, :3] = [[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]]
#        elif c == 'z' or c == 2:
#            matrix = np.identity(4)
#            matrix[:2, :2] = [[cos, -sin], [sin, cos]]
#        else:
#            raise Exception('Not a valid axis')
#        return matrix
#    else:
#        raise Exception('Not a str')


def rotation_matrix(angle, axis):
    
    angle = np.array(angle, dtype=np.float)
    assert(angle.size == 1 or angle.size == 3)
    assert(axis.size == 3)
    
    if(angle.size == 1):
        angle = np.repeat(angle, 3)
    
    angle[axis == 0] = 0 
        
    tx = np.radians(angle[0]) 
    ty = np.radians(angle[1])
    tz = np.radians(angle[2])

    sinx = np.sin(tx)
    siny = np.sin(ty)
    sinz = np.sin(tz)


    cosx = np.cos(tx) if tx >= 0 else -np.cos(tx)
    cosy = np.cos(ty) if ty >= 0 else -np.cos(ty)
    cosz = np.cos(tz) if tz >= 0 else -np.cos(tz)

    matrix = np.identity(4)
    matrix[:3, :3] = [[cosy*cosz, -cosy*sinz, siny],\
                       [sinx*siny*cosz + cosx*sinz, -sinx*siny*sinz + cosx*cosz, -sinx*cosy],\
                       [-cosx*siny*cosz + sinx*sinz, cosx*siny*sinz + sinx*cosz, cosx*cosy]]
    
    return matrix