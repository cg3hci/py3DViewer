from numba import njit, int64, float64
from numba.typed import List as L
from numba.types import Tuple, ListType as LT
import numpy as np


@njit(Tuple((int64[:,::1],LT(LT(int64)),LT(LT(int64))))(int64[:,::1], int64, int64))
def compute_surface_mesh_adjs(edges, num_vertices, edges_per_face):
        
    num_faces = edges.shape[0]//edges_per_face
    adjs =  np.zeros((num_faces, edges_per_face), dtype=np.int64)-1
    vtx2vtx = L()
    vtx2face = L()
        
    for k in range(num_vertices):
        tmp1 = L()
        tmp2 = L()
        tmp1.append(-1)
        tmp2.append(-1)
        vtx2vtx.append(tmp1)
        vtx2face.append(tmp2)
        
    tmp = np.arange(num_faces)
    faces_idx = np.repeat(tmp, edges_per_face)
        
    map_ = dict()
    support_set = set()
    map_[(-1,-1)] = -1
    support_set.add((-1,-1))
        
    for i in range(edges.shape[0]):
            
        e = (edges[i][0], edges[i][1])
        f = faces_idx[i]
            
        if vtx2vtx[e[0]][0] == -1:
            vtx2vtx[e[0]][0] = e[1]
        else:
            vtx2vtx[e[0]].append(e[1])
                
        if vtx2face[e[0]][0] == -1:
            vtx2face[e[0]][0] = f
        else:
            vtx2face[e[0]].append(f)
            
        if e not in support_set:
            map_[(edges[i][1], edges[i][0])] = f
            support_set.add((edges[i][1], edges[i][0]))
        else:
            idx_to_append1 = np.where(adjs[f] == -1)[0][0]
            idx_to_append2 = np.where(adjs[map_[e]] == -1)[0][0]
            adjs[f][idx_to_append1] = map_[e]
            adjs[map_[e]][idx_to_append2] = f
                
       
    return adjs, vtx2vtx, vtx2face   


def compute_face_normals(vertices, faces, quad=False):
        
    e1_v = vertices[faces][:,1] - vertices[faces][:,0]
    e2_v = vertices[faces][:,2] - vertices[faces][:,1]
    
    if quad:
        e2_v = vertices[faces][:,2] - vertices[faces][:,0]
        
    face_normals = np.cross(e1_v, e2_v)
    norm = np.linalg.norm(face_normals, axis=1)
    norm.shape = (-1,1)
    return face_normals / norm



@njit(float64[:,::1](float64[:,::1], LT(LT(int64))))
def compute_vertex_normals(face_normals, vtx2face):

    mean = np.zeros((1, 3), dtype=np.float64)
    vtx_normals = np.zeros((len(vtx2face),3), dtype=np.float64)
    idx = 0
    for v2f in vtx2face:
        for v in v2f:
            mean = mean+face_normals[v]
        mean/=len(v2f)

        vtx_normals[idx] = mean
        mean-=mean
        idx+=1

    norm = np.sqrt(np.sum(vtx_normals**2, axis=1))
    norm=np.reshape(norm, (-1,1))
    return vtx_normals/norm
