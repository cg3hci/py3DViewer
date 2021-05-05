import numpy as np
from scipy.spatial.distance import directed_hausdorff

def __tet_barycentric_coords(vertices, tets, points):
    
    tets.shape = (-1,4)
    points.shape = (-1,3)
    
    tmp_vertices = np.copy(vertices[tets])
    tmp_vertices = np.append(tmp_vertices,np.ones((tmp_vertices.shape[0],4,1)),axis=2)
    points = np.append(points,np.ones((points.shape[0],1)),axis=1)

    
    m0 = np.c_[points, tmp_vertices[:,1], tmp_vertices[:,2], tmp_vertices[:,3]]
    m1 = np.c_[tmp_vertices[:,0], points, tmp_vertices[:,2], tmp_vertices[:,3]]
    m2 = np.c_[tmp_vertices[:,0], tmp_vertices[:,1], points, tmp_vertices[:,3]]
    m3 = np.c_[tmp_vertices[:,0], tmp_vertices[:,1], tmp_vertices[:,2], points]
    
    m0.shape = (-1, 4, 4)
    m1.shape = (-1, 4, 4)
    m2.shape = (-1, 4, 4)
    m3.shape = (-1, 4, 4)
    
    det_m0 = np.linalg.det(m0)
    det_m1 = np.linalg.det(m1)
    det_m2 = np.linalg.det(m2)
    det_m3 = np.linalg.det(m3)

    sum_ = det_m0 + det_m1 + det_m2 + det_m3
     
    res = np.array([det_m0/sum_, det_m1/sum_, det_m2/sum_, det_m3/sum_])
    res.shape = (-1,4)
    return res

def volumetric_barycentric_coords(vertices, polys, points):
    
    if polys.shape == (4,) or polys.shape[1] == 4:
        return __tet_barycentric_coords(vertices, polys, points)
    else:
        raise Exception('Implemented only for tetrahedra')



def pca(P):
    
    B       = np.mean(P, axis=0)
    p       = P-B
    C       = np.matmul(np.transpose(p) , p)
    U, S, V = np.linalg.svd(C)
    return B, np.transpose(V)


def angle_between_vectors(a, b, rad=False):
    
    assert(a.shape==b.shape)
    
    if a.size == 3:
        a = np.expand_dims(a, axis=0)
        b = np.expand_dims(b, axis=0)
        
    dot = np.einsum("ij,ij->i", a, b)
    la = np.linalg.norm(a, axis=1)
    lb = np.linalg.norm(b, axis=1)
    alpha  =  np.arccos(dot / (la*lb))
    axis = np.cross(a, b)
    if rad: 
        return alpha, axis
    else:
        return alpha * 180 / np.pi, axis

def solid_angle(v0,v1,v2,p):
    
    a = v0-p
    b = v1-p
    c = v2-p
    
    al = np.linalg.norm(a,axis=1)
    bl = np.linalg.norm(b,axis=1)
    cl = np.linalg.norm(c, axis=1)
    
    ab = np.einsum("ij,ij->i", a, b)
    ac = np.einsum("ij,ij->i", a, c)
    bc = np.einsum("ij,ij->i", b, c)
    
    cross = np.cross(b,c)
    det = np.einsum("ij,ij->i", a, cross)
    res = np.arctan2(det, (al*bl*cl + ab*cl + ac*bl + bc*al))/(2*np.pi)
    return res 


def winding_number(mesh, p):
    
    assert(mesh.mesh_is_surface)

    p = np.array(p, dtype=np.float64)
    tris = mesh.vertices[mesh.tessellate()]
    sa = solid_angle(tris[:,0], tris[:,1], tris[:,2], p)
    w = np.sum(sa)
    return np.int(np.round(w))

def hausdorff_distance(A, B, directed=True):
    
    if(directed):
        return directed_hausdorff(A, B)[0]
    else:
        return np.maximum(directed_hausdorff(A, B)[0], directed_hausdorff(B, A)[0])


def compactness(P):
    P=np.array(P, dtype=np.float64)
    barycenter = P.mean(axis=0)
    return np.std(np.power(np.linalg.norm(P-barycenter, axis=1),2), axis=0)