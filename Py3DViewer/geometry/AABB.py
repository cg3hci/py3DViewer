import numpy as np
from numba import float64
from numba.experimental import jitclass
from numba.typed import List
from numba import njit

spec = [('min', float64[:]),
       ('max', float64[:]),
       ('delta_x', float64),
       ('delta_y', float64),
       ('delta_z', float64)]

@jitclass(spec)
class AABB(object):
    def __init__(self, vertices):
        xmin = vertices[:,0].min()
        ymin = vertices[:,1].min()
        zmin = vertices[:,2].min()
        
        xmax = vertices[:,0].max()
        ymax = vertices[:,1].max()
        zmax = vertices[:,2].max()
        
        self.min = np.array([xmin,ymin,zmin],dtype=np.float64)
        self.max = np.array([xmax,ymax,zmax],dtype=np.float64)
        self.delta_x = self.max[0]-self.min[0]
        self.delta_y = self.max[1]-self.min[1]
        self.delta_z = self.max[2]-self.min[2]
        
    @property
    def center(self):
        return (self.min + self.max)*0.5
    
    @property
    def corners(self):
        verts = np.zeros((8,3), dtype=np.float64)
        verts[0] = self.min
        verts[1] = self.min + np.array([0.,0.,self.delta_z])
        verts[2] = self.min + np.array([self.delta_x,0.,self.delta_z])
        verts[3] = self.min + np.array([self.delta_x,0.,0.])
        verts[4] = self.min + np.array([0.,self.delta_y, 0.])
        verts[5] = self.min + np.array([0.,self.delta_y,self.delta_z])
        verts[6] = self.max
        verts[7] = self.min + np.array([self.delta_x,self.delta_y,0.])
        
        
        return verts

    
    
    def contains(self, points, strict=False):
        points=np.array(points).reshape(-1,3)
        
        if(strict):
            x_check =  np.logical_and(points[:,0] > self.min[0], points[:,0] < self.max[0])
            y_check =  np.logical_and(points[:,1] > self.min[1], points[:,1] < self.max[1])
            z_check =  np.logical_and(points[:,2] > self.min[2], points[:,2] < self.max[2])
            return np.logical_and(x_check, y_check, z_check)
            
        else:
            x_check =  np.logical_and(points[:,0] >= self.min[0], points[:,0] <= self.max[0])
            y_check =  np.logical_and(points[:,1] >= self.min[1], points[:,1] <= self.max[1])
            z_check =  np.logical_and(points[:,2] >= self.min[2], points[:,2] <= self.max[2])
            return np.logical_and(x_check, y_check, z_check)
        
        
    def intersects_box(self, aabb):
        if(self.max[0] <= aabb.min[0] or self.min[0] >= aabb.max[0]):
            return False;
        if(self.max[1] <= aabb.min[1] or self.min[1] >= aabb.max[1]):
            return False;
        if(self.max[2] <= aabb.min[2] or self.min[2] >= aabb.max[2]):
            return False;
        return True
    
    
    def push_aabb(self,aabb):
        if (self.min is not None or self.max is not None):
            self.min = np.minimum(self.min,aabb.min)
            self.max = np.maximum(self.max,aabb.max)
        else:
            self.min = aabb.min
            self.max = aabb.max
            
        self.delta_x = self.max[0]-self.min[0]
        self.delta_y = self.max[1]-self.min[1]
        self.delta_z = self.max[2]-self.min[2]
    
    def get_vertices_and_edges(self):
        
        verts = self.corners
        edges = np.array([[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]], dtype=np.int64)
        
        return verts,edges
    
    #@staticmethod
    #def get_all_vertices_and_edges(aabbs):
    #    vertices=np.zeros((len(aabbs)*8, 3), dtype=np.float64)
    #    edges=np.zeros((len(aabbs)*12, 2), dtype=np.int64)
    #    
    #    for idx, a in enumerate(aabbs):
    #        verts ,edges = a.get_vertices_and_edges()
    #        e+=idx*8
    #        
    #        for i, v in enumerate(verts):
    #            vertices[idx*8+i] = v
    #        
    #        for i, e in enumerate(edges):
    #            edges[idx*12+i] = e
    #        #vertices=np.vstack(vertices)
    #        #e=np.array(edges)
    #    return vertices,edges