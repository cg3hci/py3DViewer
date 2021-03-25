import numpy as np

class Sphere:
    
    def __init__(self, tile=(20,20), origin=(0.0,0.0,0.0), radius=0.5):
        
        self.radius  = radius
        self.tile    = tile
        self.origin  = np.array(origin)
        
    def __compute_geometry_and_topology(self):
        
        
        n               = self.tile[0] + 1
        m               = self.tile[1] + 1
        theta           = np.transpose(np.linspace(-1, 1, n)) * np.pi
        phi             = np.linspace(-1, 1, m) * (np.pi/2)
        cosphi          = np.cos(phi)
        cosphi[ 0]      = 0
        cosphi[-1]      = 0
        sintheta        = np.sin(theta)
        sintheta[0] = 0
        sintheta[-1] = 0
        x = np.matmul(cosphi.reshape(-1,1),  np.cos(theta).reshape(1,-1))
        y = np.matmul(cosphi.reshape(-1,1), sintheta.reshape(1,-1))
        z = np.matmul(np.sin(phi).reshape(-1,1), np.ones(n, dtype=np.float).reshape(1,-1))
        c = np.concatenate((x, y, z))
        c.shape = (3,m,n)
        X, Y, Z = c
        m = X.shape[0]
        n = X.shape[1]
        
        P = np.concatenate((np.expand_dims(np.transpose(X).flatten(), 1), \
                            np.expand_dims(np.transpose(Y).flatten(), 1),\
                            np.expand_dims(np.transpose(Z).flatten(), 1)), axis=1).reshape(-1,3)
        P[:,[0,1,2]] = P[:,[1,0,2]]
        
        q = np.linspace(1, m * n - n, m * n - n, dtype=np.int)
        q = np.expand_dims(q[(q % n) != 0], 1)
        T = np.concatenate((q, q + 1, q + n + 1, q + n), axis=1) - 1
        
        return P, T
        
    @property
    def vertices(self):
        verts = self.__compute_geometry_and_topology()[0]*(self.diameter)
        verts += self.origin
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
    
    @property
    def diameter(self):
        return self.radius*2
    
    @property
    def area(self):
        return 4*np.pi*(self.radius*self.radius)
    
    @property
    def volume(self):
        return (4.0/3.0)*np.pi*(self.radius*self.radius*self.radius)
    
    def point_is_inside(self, point, strict=True):
        
        point = np.array(point)
        dist = np.linalg.norm(self.origin-point, axis=1)
        
        if strict:
            return (dist < self.radius)
        else:
            return (dist <= self.radius)
        
    def ray_intersect(self, ray):
        oc = ray.origin - self.origin
        a = np.dot(ray.direction, ray.direction)
        b = 2.0 * np.dot(oc, ray.direction)
        c = np.dot(oc,oc) - self.radius*self.radius
        discriminant = b*b - 4*a*c

        if(discriminant < 0):
            return (False, -1)
        
        else:
            return (True, (-b - np.sqrt(discriminant)) / (2.0*a))
    

    def line_segment_intersect(self, segment):
        
        origin = segment.a
        direction = segment.b - segment.a
        direction = direction / np.linalg.norm(direction)
        
        oc = origin - self.origin
        a = np.dot(direction, direction)
        b = 2.0 * np.dot(oc, direction)
        c = np.dot(oc,oc) - self.radius*self.radius
        discriminant = b*b - 4*a*c
        
        if(discriminant < 0):
            return (False, -1)
        
        else:
            dist = (-b - np.sqrt(discriminant)) / (2.0*a)
            intersection_point = origin + (direction*np.abs(dist))
            eps = 1e-6
            s = np.linalg.norm(origin-intersection_point)+np.linalg.norm(segment.b-intersection_point)
            if np.abs(s-segment.length) <= eps:
                return (True, dist)
            
            return (False, -1)