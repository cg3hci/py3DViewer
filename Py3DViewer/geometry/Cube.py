import numpy as np

class Cube:
    """
    This class stores information about the geometry and the topology of a cube of given size (default 1).
    A cube object can be easily turned into a surface or volumetric mesh (see vertices and topology_ properties).
    """

    def __init__(self, scale=1):
        
        self.scale = scale
        self.__vertices = np.array([[-0.5,-0.5,-0.5], \
                         [-0.5,-0.5,0.5], \
                         [0.5,-0.5,0.5],  \
                         [0.5,-0.5,-0.5], \
                         [-0.5,0.5,-0.5], \
                         [-0.5,0.5,0.5],  \
                         [0.5,0.5,0.5],   \
                         [0.5,0.5,-0.5],  \
                        ])
        

    @property
    def vertices(self):
    """
    Return:
        Array(8x3): The eight vertices of the cube
    """
        return self.__vertices*self.scale
        
    @property
    def topology_tris(self):
    """
    Return:
        Array(12x3): The 12 triangles composing the cube mesh
    """
        quads = self.topology_quad
        tris = np.c_[quads[:,:3],quads[:,2:],quads[:,0]]
        tris.shape = (-1,3)
        return tris
    
    @property
    def topology_quad(self):
    """
    Return:
        Array(6x4): The six quads composing the cube mesh
    """
        return np.array([[0,3,2,1],[4,5,6,7],[1,2,6,5],[2,3,7,6],[3,0,4,7],[0,1,5,4]])
    
    @property
    def topology_tet(self):
    """
    Return:
        Array(1x8): The five tetrahedra composing the cube mesh
    """
        split_rules = np.array([[0,1,2,5], [0,2,7,5], [0,2,3,7], [0,5,7,4], [2,7,5,6]], dtype=np.int)
        tets = np.ascontiguousarray(self.topology_hex[:,split_rules])
        tets.shape = (-1,4)
        return tets
        
    @property
    def topology_hex(self):
    """
    Return:
        Array(1x8): The hexaedron representing the cube mesh
    """
        return np.array([[0,1,2,3,4,5,6,7]]) 

