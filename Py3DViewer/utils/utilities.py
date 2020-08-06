import numpy as np

def mesh_is_volumetric(mesh):
    return hasattr(mesh, 'tets') or hasattr(mesh, 'hexes')

def mesh_is_surface(mesh):
    return not mesh_is_volumetric(mesh)


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