import numpy as np


#______________________________________Triangles__________________________________________________

def triangle_area(vertices, triangles):
    
    b = vertices[triangles][:,1] - vertices[triangles][:,0]
    h = vertices[triangles][:,2] - vertices[triangles][:,1]
    return 0.5 * np.linalg.norm(np.cross(b, h), axis = 1)


def triangle_aspect_ratio(vertices, triangles):
    
    l1 = np.linalg.norm(vertices[triangles][:,0] - vertices[triangles][:,1], axis = 1)
    l2 = np.linalg.norm(vertices[triangles][:,1] - vertices[triangles][:,2], axis = 1)
    l3 = np.linalg.norm(vertices[triangles][:,2] - vertices[triangles][:,0], axis = 1)

    a = triangle_area(vertices, triangles)
    r = (2 * a) / (l1 + l2 + l3)
    l_max = np.max(np.c_[l1, l2, l3], axis = 1)

    return l_max / (2 * np.sqrt(3) * r)




#______________________________________Quads__________________________________________________

def quad_area(vertices, quads):
    
    tris = np.c_[quads[:,:3], quads[:,2:], quads[:,0]]
    tris.shape = (-1, 3)
    
    idx_1 = range(0, tris.shape[0], 2)
    idx_2 = range(1, tris.shape[0], 2)
    
    a_tri1 = triangle_area(vertices, tris[idx_1])
    a_tri2 = triangle_area(vertices, tris[idx_2])
    
    return a_tri1+a_tri2


def quad_aspect_ratio(vertices, quads):
    
    l1 = np.linalg.norm(vertices[quads][:,0] - vertices[quads][:,1], axis = 1)
    l2 = np.linalg.norm(vertices[quads][:,1] - vertices[quads][:,2], axis = 1)
    l3 = np.linalg.norm(vertices[quads][:,2] - vertices[quads][:,3], axis = 1)
    l4 = np.linalg.norm(vertices[quads][:,3] - vertices[quads][:,0], axis = 1)

    a = quad_area(vertices, quads)
    
    l_max = np.max(np.c_[l1, l2, l3, l4], axis = 1)

    return l_max / (4*a)

