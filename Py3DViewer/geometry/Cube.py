import numpy as np

class Cube:

    def __init__(self, scale=1):
        
        self.scale = scale
        self.__vertices = np.array([[-0.5,-0.5,-0.5], \
                         [-0.5,-0.5,0.5],  \
                         [0.5,-0.5,0.5],  \
                         [0.5,-0.5,-0.5],  \
                         [-0.5,0.5,-0.5], \
                         [-0.5,0.5,0.5],  \
                         [0.5,0.5,0.5],  \
                         [0.5,0.5,-0.5],  \
                        ])
        

    @property
    def vertices(self):
        return self.__vertices*self.scale
        
    @property
    def topology_tris(self):
        quads = self.topology_quad
        tris = np.c_[quads[:,:3],quads[:,2:],quads[:,0]]
        tris.shape = (-1,3)
        return tris
    
    @property
    def topology_quad(self):
        return np.array([[0,3,2,1],[4,5,6,7],[1,2,6,5],[2,3,7,6],[3,0,4,7],[0,1,5,4]])
    
    @property
    def topology_tet(self):
        split_rules = np.array([[0,1,2,5], [0,2,7,5], [0,2,3,7], [0,5,7,4], [2,7,5,6]], dtype=np.int)
        tets = np.ascontiguousarray(self.topology_hex[:,split_rules])
        tets.shape = (-1,4)
        return tets
        
    @property
    def topology_hex(self):
        return np.array([[0,1,2,3,4,5,6,7]]) 

