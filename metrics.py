import numpy as np


#______________________________________Triangles__________________________________________________

def triangle_area(vertices, triangles):
    
    b = vertices[triangles][:,1] - vertices[triangles][:,0]
    h = vertices[triangles][:,2] - vertices[triangles][:,1]
    return ((None, None), 0.5 * np.linalg.norm(np.cross(b, h), axis = 1))


def triangle_aspect_ratio(vertices, triangles):
    
    l1 = np.linalg.norm(vertices[triangles][:,0] - vertices[triangles][:,1], axis = 1)
    l2 = np.linalg.norm(vertices[triangles][:,1] - vertices[triangles][:,2], axis = 1)
    l3 = np.linalg.norm(vertices[triangles][:,2] - vertices[triangles][:,0], axis = 1)

    a = triangle_area(vertices, triangles)[1]
    r = (2 * a) / (l1 + l2 + l3)
    l_max = np.max(np.c_[l1, l2, l3], axis = 1)

    return ((None, None), l_max / (2 * np.sqrt(3) * r))




#______________________________________Quads__________________________________________________

def quad_area(vertices, quads):
    
    tris = np.c_[quads[:,:3], quads[:,2:], quads[:,0]]
    tris.shape = (-1, 3)
    
    idx_1 = range(0, tris.shape[0], 2)
    idx_2 = range(1, tris.shape[0], 2)
    
    a_tri1 = triangle_area(vertices, tris[idx_1])
    a_tri2 = triangle_area(vertices, tris[idx_2])
    
    return ((None, None), a_tri1+a_tri2)


def quad_aspect_ratio(vertices, quads):
    
    l1 = np.linalg.norm(vertices[quads][:,0] - vertices[quads][:,1], axis = 1)
    l2 = np.linalg.norm(vertices[quads][:,1] - vertices[quads][:,2], axis = 1)
    l3 = np.linalg.norm(vertices[quads][:,2] - vertices[quads][:,3], axis = 1)
    l4 = np.linalg.norm(vertices[quads][:,3] - vertices[quads][:,0], axis = 1)

    a = quad_area(vertices, quads)
    
    l_max = np.max(np.c_[l1, l2, l3, l4], axis = 1)

    return ((None, None), l_max / (4*a))

#______________________________________Tets______________________________________________________

def tet_scaled_jacobian(vertices, tets):
    
        p0 = vertices[tets[:,0]]
        p1 = vertices[tets[:,1]]
        p2 = vertices[tets[:,2]]
        p3 = vertices[tets[:,3]]
        
        
        l0 = p1 - p0
        l1 = p2 - p1
        l2 = p0 - p2
        l3 = p3 - p0
        l4 = p3 - p1
        l5 = p3 - p2

        l0_length = np.linalg.norm(l0, axis=1)
        l1_length = np.linalg.norm(l1, axis=1)
        l2_length = np.linalg.norm(l2, axis=1)
        l3_length = np.linalg.norm(l3, axis=1)
        l4_length = np.linalg.norm(l4, axis=1)
        l5_length = np.linalg.norm(l5, axis=1)

        J = np.einsum('ij,ij->i', np.cross(l2, l0, axis= 1), l3)
        lambda_1 = np.expand_dims(l0_length * l2_length * l3_length, axis = 0).transpose()
        lambda_2 = np.expand_dims(l0_length * l1_length * l4_length, axis = 0).transpose()
        lambda_3 = np.expand_dims(l1_length * l2_length * l5_length, axis = 0).transpose()
        lambda_4 = np.expand_dims(l3_length * l4_length * l5_length, axis = 0).transpose()
        lambda_5 = np.expand_dims(J, axis = 0).transpose()
        lambda_ = np.concatenate((lambda_1, lambda_2, lambda_3, lambda_4, lambda_5), axis = 1)
        max_el = np.amax(lambda_, axis=1)

        return ((-1., 1.), (J * np.sqrt(2)) / max_el)
    
    
    
def tet_volume(vertices, tets):
    
    p0 = vertices[tets[:,0]]
    p1 = vertices[tets[:,1]]
    p2 = vertices[tets[:,2]]
    p3 = vertices[tets[:,3]]
        
        
    l0 = p1 - p0
    l2 = p0 - p2
    l3 = p3 - p0
    
    return ((None, None), (np.cross(l2, l0, axis=1)* l3)/6)
    
#______________________________________Hexes______________________________________________________


def hex_scaled_jacobian(vertices, hexes):
        """Compute quality of hexa Scaled Jacobian in tensor mode

        Paramenters:
        -----
            p?: array of vertices

        Return:
        -----
            array of Scaled Jacobian values
        """
        
        p0 = vertices[hexes[:,0]]
        p1 = vertices[hexes[:,1]]
        p2 = vertices[hexes[:,2]]
        p3 = vertices[hexes[:,3]]
        p4 = vertices[hexes[:,4]]
        p5 = vertices[hexes[:,5]]
        p6 = vertices[hexes[:,6]]
        p7 = vertices[hexes[:,7]]
        
        l0  = p1 - p0    
        l1  = p2 - p1
        l2  = p3 - p2
        l3  = p3 - p0
        l4  = p4 - p0
        l5  = p5 - p1
        l6  = p6 - p2
        l7  = p7 - p3
        l8  = p5 - p4
        l9  = p6 - p5
        l10 = p7 - p6
        l11 = p7 - p4

        # cross-derivatives
        x1  = (p1 - p0) + (p2 - p3) + (p5 - p4) + (p6 - p7)
        x2  = (p3 - p0) + (p2 - p1) + (p7 - p4) + (p6 - p5)
        x3  = (p4 - p0) + (p5 - p1) + (p6 - p2) + (p7 - p3)

        l0norm = l0/np.linalg.norm(l0, axis=1).reshape(-1,1)          
        l1norm = l1/np.linalg.norm(l1, axis=1).reshape(-1,1)  
        l2norm = l2/np.linalg.norm(l2, axis=1).reshape(-1,1)  
        l3norm = l3/np.linalg.norm(l3, axis=1).reshape(-1,1)   
        l4norm = l4/np.linalg.norm(l4, axis=1).reshape(-1,1)   
        l5norm = l5/np.linalg.norm(l5, axis=1).reshape(-1,1)  
        l6norm = l6/np.linalg.norm(l6, axis=1).reshape(-1,1)   
        l7norm = l7/np.linalg.norm(l7, axis=1).reshape(-1,1)   
        l8norm = l8/np.linalg.norm(l8, axis=1).reshape(-1,1)   
        l9norm = l9/np.linalg.norm(l9, axis=1).reshape(-1,1)   
        l10norm = l10/np.linalg.norm(l10, axis=1).reshape(-1,1)  
        l11norm = l11/np.linalg.norm(l11, axis=1).reshape(-1,1)  
        x1norm = x1/np.linalg.norm(x1, axis=1).reshape(-1,1)   
        x2norm = x2/np.linalg.norm(x2, axis=1).reshape(-1,1)   
        x3norm = x3/np.linalg.norm(x3, axis=1).reshape(-1,1)   

        # normalized jacobian matrices determinants
        alpha_1 = np.expand_dims(np.einsum('ij,ij->i', l0norm, np.cross(l3norm, l4norm, axis= 1)), axis = 0).transpose()
        alpha_2 = np.expand_dims(np.einsum('ij,ij->i', l1norm, np.cross(-l0norm, l5norm, axis=1)), axis = 0).transpose()
        alpha_3 = np.expand_dims(np.einsum('ij,ij->i', l2norm, np.cross(-l1norm, l6norm, axis=1)), axis = 0).transpose()
        alpha_4 = np.expand_dims(np.einsum('ij,ij->i', -l3norm, np.cross(-l2norm, l7norm, axis=1)), axis = 0).transpose()
        alpha_5 = np.expand_dims(np.einsum('ij,ij->i', l11norm, np.cross(l8norm, -l4norm, axis=1)), axis = 0).transpose()
        alpha_6 = np.expand_dims(np.einsum('ij,ij->i', -l8norm, np.cross(l9norm, -l5norm, axis=1)), axis = 0).transpose()
        alpha_7 = np.expand_dims(np.einsum('ij,ij->i', -l9norm, np.cross(l10norm, -l6norm, axis=1)), axis = 0).transpose()
        alpha_8 = np.expand_dims(np.einsum('ij,ij->i', -l10norm, np.cross(-l11norm, -l7norm, axis=1)), axis = 0).transpose()
        alpha_9 = np.expand_dims(np.einsum('ij,ij->i', x1norm, np.cross(x2norm, x3norm, axis=1)), axis = 0).transpose()
        alpha_ = np.concatenate((alpha_1, alpha_2, alpha_3, alpha_4, alpha_5, alpha_6, alpha_7, alpha_8, alpha_9), axis = 1)
        min_el = np.amin(alpha_, axis=1)

        min_el[min_el > 1.1] = -1

        return ((-1., 1.), min_el)
    
    
def hex_volume(vertices, hexes):
    
        p0 = vertices[hexes[:,0]]
        p1 = vertices[hexes[:,1]]
        p2 = vertices[hexes[:,2]]
        p3 = vertices[hexes[:,3]]
        p4 = vertices[hexes[:,4]]
        p5 = vertices[hexes[:,5]]
        p6 = vertices[hexes[:,6]]
        p7 = vertices[hexes[:,7]]

        # cross-derivatives
        x1  = (p1 - p0) + (p2 - p3) + (p5 - p4) + (p6 - p7)
        x2  = (p3 - p0) + (p2 - p1) + (p7 - p4) + (p6 - p5)
        x3  = (p4 - p0) + (p5 - p1) + (p6 - p2) + (p7 - p3)
        
        alpha8 = np.linalg.det(np.c_[x1,x2,x3].reshape(-1,3,3))
        return ((None, None), alpha8/64)