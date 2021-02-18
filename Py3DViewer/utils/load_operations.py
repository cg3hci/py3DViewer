from numba import njit, int64, float64
from numba.typed import List as L
from numba.types import Tuple, List, ListType as LT
import numpy as np

#edges, vtx2vtx, vtx2edge, vtx2poly, edge2vtx, edge2edge, edge2poly, poly2vtx, poly2edge, poly2poly   , LT(LT(int64)),LT(LT(int64)), LT(LT(int64)), int64[:,::1], LT(LT(int64)), LT(LT(int64)), int64[:,::1], int64[:,::1], int64[:,::1] 
@njit(Tuple((int64[:,::1], LT(LT(int64)), LT(LT(int64)), LT(LT(int64)), int64[:,::1], LT(LT(int64)), LT(LT(int64)), int64[:,::1],int64[:,::1],int64[:,::1] ))(int64, int64[:,::1]), cache=True)
def get_connectivity_info_surface(num_vertices, polys):

    vtx2vtx   = L()
    vtx2edge  = L()
    vtx2poly  = L()
    edge2edge = L()
    edge2poly = L()

    for i in range(num_vertices):
        tmp1 = L()
        tmp2 = L()
        tmp3 = L()
        tmp1.append(-1)
        tmp2.append(-1)
        tmp3.append(-1)
        vtx2vtx.append(tmp1) 
        vtx2edge.append(tmp2) 
        vtx2poly.append(tmp3)
    
    for i in range(polys.shape[0]*4):
        tmp_1 = L()
        tmp_2 = L()
        tmp_1.append(-1)
        tmp_2.append(-1)
        edge2edge.append(tmp_1) 
        edge2poly.append(tmp_2)

    poly2edge = np.zeros((polys.shape[0], polys.shape[1]), dtype=np.int64)-1 
    poly2poly = np.zeros((polys.shape[0], polys.shape[1]), dtype=np.int64)-1

    edges_list = [(0,0)]
    edges_map = dict()
    edge_poly_map = dict()

    for pid in range(polys.shape[0]):
        edges_tmp = [[0,0]]
        if(polys.shape[1] == 3):
            edges_tmp = [[polys[pid][0], polys[pid][1]],  [polys[pid][1], polys[pid][2]], [polys[pid][2], polys[pid][0]]]
        else:
            edges_tmp = [[polys[pid][0], polys[pid][1]],  [polys[pid][1], polys[pid][2]], [polys[pid][2], polys[pid][3]], [polys[pid][3], polys[pid][0]]]
        
        for e_idx in range(len(edges_tmp)):

            edges_tmp[e_idx].sort()
            e = (edges_tmp[e_idx][0], edges_tmp[e_idx][1])
            eid = 0
            not_checked = False
            if e not in edges_map:
                eid = len(edges_list)-1
                edges_list.append(e)
                edges_map[e] = eid
                not_checked = True
            else:
                eid = edges_map[e]
            
            adj_pid = -1
            if eid in edge_poly_map:
                adj_pid = edge_poly_map[eid]
            else:
                edge_poly_map[eid] = pid

            if not_checked:
                #vtx2vtx
                if(vtx2vtx[e[0]][0] == -1):
                    vtx2vtx[e[0]][0] = e[1]
                else:
                    vtx2vtx[e[0]].append(e[1]) 
                
                if(vtx2vtx[e[1]][0] == -1):
                    vtx2vtx[e[1]][0] = e[0]
                else:
                    vtx2vtx[e[1]].append(e[0])

                #vtx2edge
                if(vtx2edge[e[0]][0] == -1):
                    vtx2edge[e[0]][0] = eid
                else:
                    vtx2edge[e[0]].append(eid) 
                
                if(vtx2edge[e[1]][0] == -1):
                    vtx2edge[e[1]][0] = eid
                else:
                    vtx2edge[e[1]].append(eid)

            
            
            #edge2poly
            if(edge2poly[eid][0] == -1):
                edge2poly[eid][0] = pid
            else:
                edge2poly[eid].append(pid) 

            #poly2edge
            idx_to_append = np.where(poly2edge[pid] == -1)[0][0]
            poly2edge[pid][idx_to_append] = eid 

            #poly2poly
            if adj_pid != -1:
                idx_to_append1 = np.where(poly2poly[pid] == -1)[0][0]
                idx_to_append2 = np.where(poly2poly[adj_pid] == -1)[0][0]
                poly2poly[pid][idx_to_append1] = adj_pid 
                poly2poly[adj_pid][idx_to_append2] = pid 
                
        
        for vid in polys[pid]:
            #vtx2poly
            if(vtx2poly[vid][0] == -1):
                vtx2poly[vid][0] = pid
            else:
               vtx2poly[vid].append(pid) 
            

    for eid, e in enumerate(edges_list[1:]):
        #edge2edge
        a = vtx2edge[e[0]].copy()
        b = vtx2edge[e[1]].copy() 
        a.remove(eid)
        b.remove(eid)
        for el in b:
            a.append(el)
        edge2edge[eid] = a 

    edges = np.array(edges_list[1:])
    edge2poly = edge2poly[:edges.shape[0]]
    edge2edge = edge2edge[:edges.shape[0]]


    return edges, vtx2vtx, vtx2edge, vtx2poly, edges, edge2edge, edge2poly, polys, poly2edge, poly2poly 


@njit(Tuple((int64[:,::1], LT(LT(int64)), LT(LT(int64)), LT(LT(int64)), int64[:,::1], LT(LT(int64)), LT(LT(int64)), int64[:,::1],int64[:,::1], LT(LT(int64))))(int64, int64[:,::1]), cache=True)
def get_connectivity_info_volume_faces(num_vertices, polys):

    vtx2vtx   = L()
    vtx2edge  = L()
    vtx2poly  = L()
    edge2edge = L()
    edge2poly = L()
    poly2poly = L()

    for i in range(num_vertices):
        tmp1 = L()
        tmp2 = L()
        tmp3 = L()
        tmp1.append(-1)
        tmp2.append(-1)
        tmp3.append(-1)
        vtx2vtx.append(tmp1) 
        vtx2edge.append(tmp2) 
        vtx2poly.append(tmp3)
    
    for i in range(polys.shape[0]*4):
        tmp_1 = L()
        tmp_2 = L()
        tmp_3 = L()
        tmp_1.append(-1)
        tmp_2.append(-1)
        tmp_3.append(-1)
        edge2edge.append(tmp_1) 
        edge2poly.append(tmp_2)
        if(i < polys.shape[0]):
            poly2poly.append(tmp3)

    poly2edge = np.zeros((polys.shape[0], polys.shape[1]), dtype=np.int64)-1 

    edges_list = [(0,0)]
    edges_map = dict()
    edge_poly_map = dict()

    for pid in range(polys.shape[0]):
        edges_tmp = [[0,0]]
        if(polys.shape[1] == 3):
            edges_tmp = [[polys[pid][0], polys[pid][1]],  [polys[pid][1], polys[pid][2]], [polys[pid][2], polys[pid][0]]]
        else:
            edges_tmp = [[polys[pid][0], polys[pid][1]],  [polys[pid][1], polys[pid][2]], [polys[pid][2], polys[pid][3]], [polys[pid][3], polys[pid][0]]]
        
        for e_idx in range(len(edges_tmp)):

            edges_tmp[e_idx].sort()
            e = (edges_tmp[e_idx][0], edges_tmp[e_idx][1])
            eid = 0
            not_checked = False
            if e not in edges_map:
                eid = len(edges_list)-1
                edges_list.append(e)
                edges_map[e] = eid
                not_checked = True
            else:
                eid = edges_map[e]
            
            adj_pid = -1
            if eid in edge_poly_map:
                adj_pid = edge_poly_map[eid]
            else:
                edge_poly_map[eid] = pid

            if not_checked:
                #vtx2vtx
                if(vtx2vtx[e[0]][0] == -1):
                    vtx2vtx[e[0]][0] = e[1]
                else:
                    vtx2vtx[e[0]].append(e[1]) 
                
                if(vtx2vtx[e[1]][0] == -1):
                    vtx2vtx[e[1]][0] = e[0]
                else:
                    vtx2vtx[e[1]].append(e[0])

                #vtx2edge
                if(vtx2edge[e[0]][0] == -1):
                    vtx2edge[e[0]][0] = eid
                else:
                    vtx2edge[e[0]].append(eid) 
                
                if(vtx2edge[e[1]][0] == -1):
                    vtx2edge[e[1]][0] = eid
                else:
                    vtx2edge[e[1]].append(eid)

            
            
            #edge2poly
            if(edge2poly[eid][0] == -1):
                edge2poly[eid][0] = pid
            else:
                edge2poly[eid].append(pid) 

            #poly2edge
            idx_to_append = np.where(poly2edge[pid] == -1)[0][0]
            poly2edge[pid][idx_to_append] = eid 

        
        for vid in polys[pid]:
            #vtx2poly
            if(vtx2poly[vid][0] == -1):
                vtx2poly[vid][0] = pid
            else:
               vtx2poly[vid].append(pid) 
            

    for eid, e in enumerate(edges_list[1:]):
        #edge2edge
        a = vtx2edge[e[0]].copy()
        b = vtx2edge[e[1]].copy() 
        a.remove(eid)
        b.remove(eid)
        for el in b:
            a.append(el)
        edge2edge[eid] = a 
    
    for pid in range(polys.shape[0]):
        adjs = L()
        
        for eid in poly2edge[pid]:
            for adj_pid in edge2poly[eid]:
                if adj_pid != pid:
                    adjs.append(adj_pid)
        poly2poly[pid] = adjs[1:]

    edges = np.array(edges_list[1:])
    edge2poly = edge2poly[:edges.shape[0]]
    edge2edge = edge2edge[:edges.shape[0]]


    return edges, vtx2vtx, vtx2edge, vtx2poly, edges, edge2edge, edge2poly, polys, poly2edge, poly2poly 

#faces, edges, vtx2vtx, vtx2edge, vtx2face, vtx2poly, edges, edge2edge, edge2face, edge2poly, faces, face2edge, face2face, face2poly, polys, poly2edge, poly2face, poly2poly
@njit(Tuple((int64[:,::1], int64[:,::1], LT(LT(int64)), LT(LT(int64)), LT(LT(int64)), LT(LT(int64)), int64[:,::1], LT(LT(int64)), LT(LT(int64)), LT(LT(int64)), int64[:,::1], int64[:,::1], LT(LT(int64)), int64[:,::1], int64[:,::1], int64[:,::1], int64[:,::1], int64[:,::1]))(int64, int64[:,::1]), cache=True)
def get_connectivity_info_volume_tet(num_vertices, polys):

    vtx2poly  = L()
    edge2poly = L()


    for i in range(num_vertices):
        tmp1 = L()
        tmp1.append(-1)
        vtx2poly.append(tmp1)
    

   

    face2poly = np.zeros((polys.shape[0]*4, 2), dtype=np.int64)-1  
    poly2face = np.zeros((polys.shape[0], 4), dtype=np.int64)-1 
    poly2poly = np.zeros((polys.shape[0], 4), dtype=np.int64)-1 
    poly2edge = np.zeros((polys.shape[0], 6), dtype=np.int64)-1 

    faces_list = [(0,0,0)]
    faces_map = dict()
    face_poly_map = dict() 

    for pid in range(polys.shape[0]):
        faces_tmp = [[polys[pid][0], polys[pid][2], polys[pid][1]], [polys[pid][0], polys[pid][1], polys[pid][3]], [polys[pid][1], polys[pid][2], polys[pid][3]], [polys[pid][0], polys[pid][3], polys[pid][2]] ]

        
        for f_idx in range(len(faces_tmp)):

            faces_tmp[f_idx].sort()
            f = (faces_tmp[f_idx][0], faces_tmp[f_idx][1], faces_tmp[f_idx][2]) 
            fid = 0
            if f not in faces_map:
                fid = len(faces_list)-1
                faces_list.append(f)
                faces_map[f] = fid
            else:
                fid = faces_map[f]

            adj_pid = -1
            if fid in face_poly_map:
                adj_pid = face_poly_map[fid]
            else:
                face_poly_map[fid] = pid

            #face2poly
            idx_to_append = np.where(face2poly[fid] == -1)[0][0]
            face2poly[fid][idx_to_append] = pid 

            #poly2face
            idx_to_append = np.where(poly2face[pid] == -1)[0][0]
            poly2face[pid][idx_to_append] = fid 

            if adj_pid != -1:
                idx_to_append1 = np.where(poly2poly[pid] == -1)[0][0]
                idx_to_append2 = np.where(poly2poly[adj_pid] == -1)[0][0]
                poly2poly[pid][idx_to_append1] = adj_pid 
                poly2poly[adj_pid][idx_to_append2] = pid 
                
        for vid in polys[pid]:
            #vtx2poly
            if(vtx2poly[vid][0] == -1):
                vtx2poly[vid][0] = pid
            else:
               vtx2poly[vid].append(pid) 

    faces = np.array(faces_list[1:])
    face2poly = face2poly[:faces.shape[0]]

    edges, vtx2vtx, vtx2edge, vtx2face, edges, edge2edge, edge2face, faces, face2edge, face2face = get_connectivity_info_volume_faces(num_vertices, faces)

    for i in range(edges.shape[0]):
        tmp_2 = L()
        tmp_2.append(-1)
        edge2poly.append(tmp_2)

    for pid in range(polys.shape[0]):
        adj_edges = []
        for fid in poly2face[pid]:
            for eid in face2edge[fid]:
                adj_edges.append(eid)
        
        unique = np.unique(np.array(adj_edges))
        poly2edge[pid] = unique
        for eid in unique:
            if(edge2poly[eid][0] == -1):
                edge2poly[eid][0] = pid
            else:
               edge2poly[eid].append(pid) 


    return faces, edges, vtx2vtx, vtx2edge, vtx2face, vtx2poly, edges, edge2edge, edge2face, edge2poly, faces, face2edge, face2face, face2poly, polys, poly2edge, poly2face, poly2poly 


@njit(Tuple((int64[:,::1], int64[:,::1], LT(LT(int64)), LT(LT(int64)), LT(LT(int64)), LT(LT(int64)), int64[:,::1], LT(LT(int64)), LT(LT(int64)), LT(LT(int64)), int64[:,::1], int64[:,::1], LT(LT(int64)), int64[:,::1], int64[:,::1], int64[:,::1], int64[:,::1], int64[:,::1]))(int64, int64[:,::1]), cache=True)
def get_connectivity_info_volume_hex(num_vertices, polys):

    vtx2poly  = L()
    edge2poly = L()


    for i in range(num_vertices):
        tmp1 = L()
        tmp1.append(-1)
        vtx2poly.append(tmp1)
    


    face2poly = np.zeros((polys.shape[0]*6, 2), dtype=np.int64)-1  
    poly2face = np.zeros((polys.shape[0], 6), dtype=np.int64)-1 
    poly2poly = np.zeros((polys.shape[0], 6), dtype=np.int64)-1 
    poly2edge = np.zeros((polys.shape[0], 12), dtype=np.int64)-1 

    faces_list = [(0,0,0,0)]
    faces_map = dict()
    face_poly_map = dict() 

    for pid in range(polys.shape[0]):

        faces_tmp = [[polys[pid][0], polys[pid][3], polys[pid][2], polys[pid][1]], 
                        [polys[pid][1], polys[pid][2], polys[pid][6], polys[pid][5]],
                        [polys[pid][4], polys[pid][5], polys[pid][6], polys[pid][7]],
                        [polys[pid][3], polys[pid][0], polys[pid][4], polys[pid][7]],
                        [polys[pid][0], polys[pid][1], polys[pid][5], polys[pid][4]],
                        [polys[pid][2], polys[pid][3], polys[pid][7], polys[pid][6]]]

        
        for f_idx in range(len(faces_tmp)):

            face_original = (faces_tmp[f_idx][0], faces_tmp[f_idx][1], faces_tmp[f_idx][2], faces_tmp[f_idx][3])
            faces_tmp[f_idx].sort()
            f = (faces_tmp[f_idx][0], faces_tmp[f_idx][1], faces_tmp[f_idx][2], faces_tmp[f_idx][3])
            fid = 0
            if f not in faces_map:
                fid = len(faces_list)-1
                faces_list.append(face_original)
                faces_map[f] = fid
            else:
                fid = faces_map[f]

            adj_pid = -1
            if fid in face_poly_map:
                adj_pid = face_poly_map[fid]
            else:
                face_poly_map[fid] = pid

            #face2poly
            idx_to_append = np.where(face2poly[fid] == -1)[0][0]
            face2poly[fid][idx_to_append] = pid 

            #poly2face
            idx_to_append = np.where(poly2face[pid] == -1)[0][0]
            poly2face[pid][idx_to_append] = fid 

            if adj_pid != -1:
                idx_to_append1 = np.where(poly2poly[pid] == -1)[0][0]
                idx_to_append2 = np.where(poly2poly[adj_pid] == -1)[0][0]
                poly2poly[pid][idx_to_append1] = adj_pid 
                poly2poly[adj_pid][idx_to_append2] = pid 
                
        for vid in polys[pid]:
            #vtx2poly
            if(vtx2poly[vid][0] == -1):
                vtx2poly[vid][0] = pid
            else:
               vtx2poly[vid].append(pid) 

    faces = np.array(faces_list[1:])
    face2poly = face2poly[:faces.shape[0]]

    edges, vtx2vtx, vtx2edge, vtx2face, edges, edge2edge, edge2face, faces, face2edge, face2face = get_connectivity_info_volume_faces(num_vertices, faces)

    for i in range(edges.shape[0]):
        tmp_2 = L()
        tmp_2.append(-1)
        edge2poly.append(tmp_2)
        
    for pid in range(polys.shape[0]):
        adj_edges = []
        for fid in poly2face[pid]:
            for eid in face2edge[fid]:
                adj_edges.append(eid)
        
        unique = np.unique(np.array(adj_edges))
        poly2edge[pid] = unique
        for eid in unique:
            if(edge2poly[eid][0] == -1):
                edge2poly[eid][0] = pid
            else:
               edge2poly[eid].append(pid) 

    return faces, edges, vtx2vtx, vtx2edge, vtx2face, vtx2poly, edges, edge2edge, edge2face, edge2poly, faces, face2edge, face2face, face2poly, polys, poly2edge, poly2face, poly2poly


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
    np.errstate(divide='ignore',invalid='ignore')
    tmp = tri_soup[0::3]
    a = tri_soup[1::3] - tmp
    b = tri_soup[2::3] - tmp
    cross = np.cross(a,b)
    face_normals = cross / np.linalg.norm(cross, axis=1, keepdims=True)
    vtx_normals = np.repeat(face_normals, 3, axis=0)
    
    return vtx_normals

