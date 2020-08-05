from numba import njit, int64, float64
from numba.typed import List as L
from numba.types import Tuple, ListType as LT
import numpy as np


@njit(Tuple((int64[:,::1],LT(LT(int64)),LT(LT(int64))))(int64[:,::1], int64, int64), cache=True)
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


@njit(Tuple((int64[:,::1],LT(LT(int64)),LT(LT(int64)),LT(LT(int64))))(int64[:,::1], int64), cache=True)
def compute_tet_mesh_adjs(faces, num_vertices):
        
    num_poly = faces.shape[0]//4
    adjs =  np.zeros((num_poly, 4), dtype=np.int64)-1
    vtx2vtx = L()
    vtx2poly = L()
    vtx2face = L()
        
    for k in range(num_vertices):
        tmp1 = L()
        tmp2 = L()
        tmp3 = L()
        tmp1.append(-1)
        tmp2.append(-1)
        tmp3.append(-1)
        vtx2vtx.append(tmp1)
        vtx2poly.append(tmp2)
        vtx2face.append(tmp3)
        
    tmp = np.arange(num_poly)
    poly_idx = np.repeat(tmp, 4)
        
    map_ = dict()
    support_set = set()
    map_[(-1,-1,-1)] = -1
    support_set.add((-1,-1,-1))
        
    for i in range(faces.shape[0]):
        
        f1 = (faces[i][0], faces[i][1], faces[i][2])
        f2 = (faces[i][2], faces[i][1], faces[i][0])
        f3 = (faces[i][0], faces[i][2], faces[i][1])
        f4 = (faces[i][1], faces[i][0], faces[i][2])
        
        t = poly_idx[i]
            
        if vtx2vtx[faces[i][0]][0] == -1:
            vtx2vtx[faces[i][0]][0] = faces[i][1]
        else:
            if faces[i][1] not in vtx2vtx[faces[i][0]]:
                vtx2vtx[faces[i][0]].append(faces[i][1])
            
        if vtx2vtx[faces[i][1]][0] == -1:
            vtx2vtx[faces[i][1]][0] = faces[i][2]
        else:
            if faces[i][2] not in vtx2vtx[faces[i][1]]:
                vtx2vtx[faces[i][1]].append(faces[i][2])
        
        if vtx2vtx[faces[i][2]][0] == -1:
            vtx2vtx[faces[i][2]][0] = faces[i][0]
        else:
            if faces[i][0] not in vtx2vtx[faces[i][2]]:
                vtx2vtx[faces[i][2]].append(faces[i][0])
        
        for j in range(3):
            if vtx2face[faces[i][j]][0] == -1:
                vtx2face[faces[i][j]][0] = i
            else:
                if faces[i][j] not in vtx2face[faces[i][j]]:
                    vtx2face[faces[i][j]].append(i)
                
        if vtx2poly[faces[i][0]][0] == -1:
            vtx2poly[faces[i][0]][0] = t
        else:
            if t not in vtx2poly[faces[i][0]]:
                vtx2poly[faces[i][0]].append(t)
        
        if vtx2poly[faces[i][1]][0] == -1:
            vtx2poly[faces[i][1]][0] = t
        else:
            if t not in vtx2poly[faces[i][1]]:
                vtx2poly[faces[i][1]].append(t)
        
        if vtx2poly[faces[i][2]][0] == -1:
            vtx2poly[faces[i][2]][0] = t
        else:
            if t not in vtx2poly[faces[i][2]]:
                vtx2poly[faces[i][2]].append(t)
            
            
            
        if f1 not in support_set:
            map_[f2] = t
            map_[f3] = t
            map_[f4] = t
            support_set.add(f2)
            support_set.add(f3)
            support_set.add(f4)
        else:
            idx_to_append1 = np.where(adjs[t] == -1)[0][0]
            idx_to_append2 = np.where(adjs[map_[f1]] == -1)[0][0]
            adjs[t][idx_to_append1] = map_[f1]
            adjs[map_[f1]][idx_to_append2] = t
                
       
    return adjs, vtx2vtx, vtx2poly, vtx2face   

@njit(Tuple((int64[:,::1],LT(LT(int64)),LT(LT(int64)),LT(LT(int64))))(int64[:,::1], int64), cache=True)
def compute_hex_mesh_adjs(faces, num_vertices):
        
    num_poly = faces.shape[0]//6
    adjs =  np.zeros((num_poly, 6), dtype=np.int64)-1
    vtx2vtx = L()
    vtx2poly = L()
    vtx2face = L()
        
    for k in range(num_vertices):
        tmp1 = L()
        tmp2 = L()
        tmp3 = L()
        tmp1.append(-1)
        tmp2.append(-1)
        tmp3.append(-1)
        vtx2vtx.append(tmp1)
        vtx2poly.append(tmp2)
        vtx2face.append(tmp3)
        
    tmp = np.arange(num_poly)
    poly_idx = np.repeat(tmp, 6)
        
    map_ = dict()
    support_set = set()
    map_[(-1,-1,-1,-1)] = -1
    support_set.add((-1,-1,-1,-1))
        
    for i in range(faces.shape[0]):
        
        f1 = (faces[i][0], faces[i][1], faces[i][2], faces[i][3])
        f2 = (faces[i][3], faces[i][2], faces[i][1], faces[i][0])
        f3 = (faces[i][2], faces[i][1], faces[i][0], faces[i][3])
        f4 = (faces[i][1], faces[i][0], faces[i][3], faces[i][2])
        f5 = (faces[i][0], faces[i][3], faces[i][2], faces[i][1])
        
        t = poly_idx[i]
            
        if vtx2vtx[faces[i][0]][0] == -1:
            vtx2vtx[faces[i][0]][0] = faces[i][1]
        else:
            if faces[i][1] not in vtx2vtx[faces[i][0]]:
                vtx2vtx[faces[i][0]].append(faces[i][1])
            
        if vtx2vtx[faces[i][1]][0] == -1:
            vtx2vtx[faces[i][1]][0] = faces[i][2]
        else:
            if faces[i][2] not in vtx2vtx[faces[i][1]]:
                vtx2vtx[faces[i][1]].append(faces[i][2])
        
        if vtx2vtx[faces[i][2]][0] == -1:
            vtx2vtx[faces[i][2]][0] = faces[i][3]
        else:
            if faces[i][3] not in vtx2vtx[faces[i][2]]:
                vtx2vtx[faces[i][2]].append(faces[i][3])
        
        if vtx2vtx[faces[i][3]][0] == -1:
            vtx2vtx[faces[i][3]][0] = faces[i][0]
        else:
            if faces[i][0] not in vtx2vtx[faces[i][3]]:
                vtx2vtx[faces[i][3]].append(faces[i][0])
            
        for j in range(4):
            if vtx2face[faces[i][j]][0] == -1:
                vtx2face[faces[i][j]][0] = i
            else:
                if faces[i][j] not in vtx2face[faces[i][j]]:
                    vtx2face[faces[i][j]].append(i)
                
        if vtx2poly[faces[i][0]][0] == -1:
            vtx2poly[faces[i][0]][0] = t
        else:
            if t not in vtx2poly[faces[i][0]]:
                vtx2poly[faces[i][0]].append(t)
        
        if vtx2poly[faces[i][1]][0] == -1:
            vtx2poly[faces[i][1]][0] = t
        else:
            if t not in vtx2poly[faces[i][1]]:
                vtx2poly[faces[i][1]].append(t)
        
        if vtx2poly[faces[i][2]][0] == -1:
            vtx2poly[faces[i][2]][0] = t
        else:
            if t not in vtx2poly[faces[i][2]]:
                vtx2poly[faces[i][2]].append(t)
                
        if vtx2poly[faces[i][3]][0] == -1:
            vtx2poly[faces[i][3]][0] = t
        else:
            if t not in vtx2poly[faces[i][3]]:
                vtx2poly[faces[i][3]].append(t)
            
            
            
        if f1 not in support_set:
            map_[f2] = t
            map_[f3] = t
            map_[f4] = t
            map_[f5] = t
            support_set.add(f2)
            support_set.add(f3)
            support_set.add(f4)
            support_set.add(f5)
        else:
            idx_to_append1 = np.where(adjs[t] == -1)[0][0]
            idx_to_append2 = np.where(adjs[map_[f1]] == -1)[0][0]
            adjs[t][idx_to_append1] = map_[f1]
            adjs[map_[f1]][idx_to_append2] = t
                
       
    return adjs, vtx2vtx, vtx2poly, vtx2face
   

@njit(int64[:,::1](int64[:,::1]),cache=True)
def compute_adj_f2f_volume(faces):
    
    adjs =  np.zeros((faces.shape[0], 1), dtype=np.int64)-1
    map_ = dict()
    map_[(-1,-1,-1,-1)] = -1
    for idx in range(faces.shape[0]):
        
        f = np.copy(faces[idx])
        f.sort()
        support = (f[0],f[1],f[2],-1) if faces.shape[1] == 3 else (f[0],f[1],f[2],f[3])
        if(support in map_):
            idx_to_append1 = np.where(adjs[map_[support]] == -1)[0][0]
            idx_to_append2 = np.where(adjs[idx] == -1)[0][0]
            adjs[map_[support]][idx_to_append1] = idx
            adjs[idx][idx_to_append2] = map_[support]
        else:
            map_[support] = idx
    return adjs


def compute_face_normals(vertices, faces, quad=False):
        
    e1_v = vertices[faces][:,1] - vertices[faces][:,0]
    e2_v = vertices[faces][:,2] - vertices[faces][:,1]
    
    if quad:
        e2_v = vertices[faces][:,2] - vertices[faces][:,0]
        
    face_normals = np.cross(e1_v, e2_v)
    norm = np.linalg.norm(face_normals, axis=1)
    norm.shape = (-1,1)
    return face_normals / norm



@njit(float64[:,::1](float64[:,::1], LT(LT(int64))), cache=True)
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



def _compute_three_vertex_normals(tri_soup):
    
    tmp = tri_soup[0::3]
    a = tri_soup[1::3] - tmp
    b = tri_soup[2::3] - tmp
    cross = np.cross(a,b)
    face_normals = cross / np.linalg.norm(cross, axis=1, keepdims=True)
    vtx_normals = np.repeat(face_normals, 3, axis=0)
    
    return vtx_normals

