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

def booo(vertices, faces):
    
    pass

    