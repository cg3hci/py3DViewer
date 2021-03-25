import numpy as np

class Plane:
    
    def __init__(self, tile=(2,2), scale=(1,1), direction='z'):
        
        self.scale  = (*scale,1)
        self.tile   = tile
        self.direction = direction
        
    def __compute_geometry_and_topology(self):
        
        x, y = np.meshgrid(np.linspace(-0.5, 0.5, self.tile[0]+1), np.linspace(-0.5, 0.5, self.tile[1]+1))
        c = np.concatenate((x, y, np.zeros_like(x)))
        c.shape = (3,-1,3)
        X, Y, Z = c
        m = self.tile[0]+1
        n = self.tile[1]+1
        P = np.concatenate((np.expand_dims(np.transpose(X).flatten(), 1), \
                            np.expand_dims(np.transpose(Y).flatten(), 1),\
                            np.expand_dims(np.transpose(Z).flatten(), 1)), axis=1).reshape(-1,3)
        P[:,[0,1,2]] = P[:,[1,0,2]]

        P  =  P[np.lexsort((P[:,2], P[:,0], P[:,1]))]
        q = np.linspace(1, m * n - n, m * n - n, dtype=np.int)
        q = np.expand_dims(q[(q % n) != 0], 1)
        T = np.concatenate((q, q + 1, q + n + 1, q + n), axis=1) - 1
        return P, T
        
    @property
    def vertices(self):
        verts = self.__compute_geometry_and_topology()[0]*np.array(self.scale)
        if self.direction == 'x':
            verts[:,[0,1,2]] = verts[:,[2,1,0]]
        elif self.direction == 'y':
            verts[:,[0,1,2]] = verts[:,[0,2,1]]
        return verts
        
    @property
    def topology_tris(self):
        quads = self.topology_quad
        tris = np.c_[quads[:,:3],quads[:,2:],quads[:,0]]
        tris.shape = (-1,3)
        return tris
    
    @property
    def topology_quad(self):
        return self.__compute_geometry_and_topology()[1]  