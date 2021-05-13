import numpy as np
from numba import njit, float64, int64
from numba.types import Tuple
from numba.typed import List as L

hex_split_scheme = [[[0], [0,1], [0,1,2,3], [0,3], [0,4], [0,1,4,5], [0,1,2,3,4,5,6,7], [0,3,4,7]],\
                    [[0,1], [1], [1,2], [0,1,2,3], [0,1,4,5], [1,5], [1,2,5,6], [0,1,2,3,4,5,6,7]],\
                    [[0,1,2,3], [1,2], [2], [2,3], [0,1,2,3,4,5,6,7], [1,2,5,6], [2,6], [2,3,6,7]],\
                    [[0,3], [0,1,2,3], [2,3], [3], [0,3,4,7], [0,1,2,3,4,5,6,7], [2,3,6,7], [3,7]],\
                    [[0,4], [0,1,4,5], [0,1,2,3,4,5,6,7], [0,3,4,7], [4], [4,5], [4,5,6,7], [4,7]],\
                    [[0,1,4,5], [1,5], [1,2,5,6], [0,1,2,3,4,5,6,7], [4,5], [5], [5,6], [4,5,6,7]],\
                    [[0,1,2,3,4,5,6,7], [1,2,5,6], [2,6], [2,3,6,7], [4,5,6,7], [5,6], [6], [6,7]],\
                    [[0,3,4,7], [0,1,2,3,4,5,6,7], [2,3,6,7], [3,7], [4,7], [4,5,6,7], [6,7],[7]],\
                   ]
tris_split_scheme = [[[0],[1],[0,1,2]],\
                     [[1], [2], [0,1,2]],\
                     [[2],[0],[0,1,2]],\
                    ]
tris_split_scheme_edges = [[[0],[0,1],[0,2]],\
                     [[0,1], [1], [1,2]],\
                     [[2],[0,2],[1,2]],\
                     [[0,1],[1,2],[0,2]],\
                    ]

quad_split_scheme = [[[0],[0,1],[0,1,2,3],[0,3]],\
                     [[0,1], [1], [1,2],[0,1,2,3]],\
                     [[0,1,2,3],[1,2],[2],[2,3]],\
                     [[0,3],[0,1,2,3],[2,3],[3]],\
                    ]

tet_split_scheme = [[[0],[1],[2],[0,1,2,3]],\
                     [[0], [1], [3],[0,1,2,3]],\
                     [[0],[2],[3],[0,1,2,3]],\
                     [[1],[2],[3],[0,1,2,3]],\
                    ]



@njit(cache=True)
def __numba_split(split_scheme, num_p, p2v, v):
    vmap = dict()
    new_verts = []
    new_polys = []
    
    for i in range(num_p):
        for scheme in split_scheme:
            verts = []
            for p in scheme:
                s = np.zeros((len(p), 3), dtype=np.float64)
                for idx, el in enumerate(p):
                    s[idx]=v[p2v[i][int(el)]]
                tmp_vert = np.sum(s, axis=0)/len(p)
                verts.append([tmp_vert[0], tmp_vert[1], tmp_vert[2]])

            new_poly=[]
            for vert in verts:
                vert_t = (vert[0], vert[1], vert[2])
                if vert_t in vmap:
                    new_poly.append(vmap[vert_t])
                else:
                    new_verts.append(vert)
                    new_poly.append(len(new_verts)-1)
                    vmap[vert_t] = new_poly[-1]

            new_polys.append(new_poly)
    
                
    return np.array(new_verts), np.array(new_polys)


def mesh_subdivision(mesh, override_mesh=False, custom_scheme=None):

    if 'Trimesh' in str(type(mesh)):
        subdivision_scheme_ = tris_split_scheme_edges if custom_scheme is None else custom_scheme
    elif 'Quadmesh' in str(type(mesh)):
        subdivision_scheme_ = quad_split_scheme if custom_scheme is None else custom_scheme
    elif 'Tetmesh' in str(type(mesh)):
        subdivision_scheme_ = tet_split_scheme if custom_scheme is None else custom_scheme
    elif 'Hexmesh' in str(type(mesh)):
        subdivision_scheme_ = hex_split_scheme if custom_scheme is None else custom_scheme
    else:
        raise Exception('Input must be a mesh')

    subdivision_scheme = L()
    for scheme in subdivision_scheme_:
        p = L()
        for poly in scheme:
            element = L()
            for el in poly:
                element.append(el)
            p.append(element)
        subdivision_scheme.append(p)


    
    v, p = __numba_split(subdivision_scheme, mesh.num_polys, mesh.polys, mesh.vertices)
    if override_mesh:
        mesh.__init__(vertices=v, polys=p)
        return
    return v, p

def catmull_clark_subdivision(mesh):
    
    @njit(cache=True)
    def compute_new_verts(new_verts, face_verts, edge_verts, v2p, v2e):
        
        for i in range(new_verts.shape[0]):
        
            n = len(v2p[i])
        
            avgF = np.zeros((1,3), dtype=np.float64)
            avgE = np.zeros((1,3), dtype=np.float64)
            for f in v2p[i]:
                avgF += face_verts[f]
            avgF/=n
            for e in v2e[i]:
                avgE += edge_verts[e]
            avgE/=n
        
            new_verts[i] = (avgF + 2*avgE + (n-3)*new_verts[i])/n
    
        return new_verts

    @njit(cache=True)
    def compute_faces(face_verts, edge_verts, new_verts, f2e, v2e, f2v):
    
        faces = []
    
        n_faces = len(face_verts)
        n_edges = len(edge_verts)
        n_overts = len(new_verts)
    
        for i in range(n_faces):
            fc = f2v[i]
            for j, v in enumerate(fc):
                v+=n_faces+n_edges
                face = [0]
                face.append(v)
                face.append(f2e[i,j]+n_faces)
                face.append(i)
                face.append(f2e[i, (j-1)%len(f2e[i])]+n_faces)
                face.pop(0)
                faces.append(face)
    
        return np.array(faces)
    
    face_verts = mesh.poly_centroids
    
    segment_centroids = face_verts[mesh.adj_edge2poly.array].mean(axis=1)
    edge_verts = (mesh.edge_centroids + segment_centroids) * 0.5
    
    new_verts = compute_new_verts(mesh.vertices.copy(), face_verts, edge_verts, mesh.adj_vtx2poly.content, mesh.adj_vtx2edge.content)
    
    polys = compute_faces(face_verts, edge_verts, new_verts, mesh.adj_poly2edge, mesh.adj_vtx2edge.content, mesh.adj_poly2vtx)
    new_verts = np.vstack((face_verts, edge_verts, new_verts))
    
    return new_verts, polys


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

def quad_to_tri_subdivision(mesh):
    
    tris = np.c_[mesh.polys[:, :3], mesh.polys[:, 2:] , mesh.polys[:,0]]
    tris.shape = (-1,3)
    return tris