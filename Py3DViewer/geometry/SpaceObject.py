import numpy as np
from numba import float64, njit
from numba.experimental import jitclass
import math


spec = [('vertices', float64[:,:])]


#Generic class that represents every object in the space (points, triangles, quad, tet, etc)
@jitclass(spec)
class SpaceObject:
    def __init__(self, vertices):
        self.vertices = vertices
    
        
    #Method to check if all vertices are different in order to find a point inside a polygon or solid
    def all_vertices_are_different(self):
        for i,v in enumerate(self.vertices):
            for vert in self.vertices[i+1:]:
                if np.array_equal(vert, v):
                    return False
        return True
    
    
    #Method to check if a point is inside a triangle calculating for every edge the determinant of:
    # ax ay 1
    # bx by 1
    # px py 1
    # a and b: the two vertices of the edge; p: the point we want to know the position relative to the edge
    # If the point is on the left or on the right for every edge we know the point is inside the triangle.
    @staticmethod
    def triangle_contains_point2D (pt, v1, v2, v3):
        
        a1 = (v1[0] - pt[0]) * (v2[1] - pt[1]) - (v1[1] - pt[1]) * (v2[0] - pt[0])
        a2 = (v2[0] - pt[0]) * (v3[1] - pt[1]) - (v2[1] - pt[1]) * (v3[0] - pt[0])
        a3 = (v3[0] - pt[0]) * (v1[1] - pt[1]) - (v3[1] - pt[1]) * (v1[0] - pt[0])
        
        if (a1 >= 0 and a2 >= 0 and a3 >= 0) or (a1 <= 0 and a2 <= 0 and a3 <= 0):
            return True
        else:
            return False
    
    
    #Method to check if a triangle in a space contains a point.
    #To make less operations we check if the point lies on the same position of the 3 vertices or lies.
    #Then we check the projection of the triangle for three couple of dimensions. The point lies inside the triangle
    #if it lies in every 2D projection of the triangle.
    def triangle_contains_point(self, point):
        if len(self.vertices) == 3:
            if self.all_vertices_are_different():
                a = self.vertices[0]
                b = self.vertices[1]
                c = self.vertices[2]
                
                if np.array_equal(point, a) or np.array_equal(point, b) or np.array_equal(point, c):
                    return True;
                
                p  = np.array([point[1], point[2]], dtype = 'float64')
                t0 = np.array([a[1], a[2]], dtype = 'float64')
                t1 = np.array([b[1], b[2]], dtype = 'float64')
                t2 = np.array([c[1], c[2]], dtype = 'float64')
                
                if(self.triangle_contains_point2D(p, t0, t1, t2) == False):
                    return False;
                
                p[0],p[1]  = point[0],  point[2]
                t0[0],t0[1] = a[0], a[2]
                t1[0],t1[1] = b[0], b[2]
                t2[0],t2[1] = c[0], c[2]
                
                if(self.triangle_contains_point2D(p, t0, t1, t2) == False):
                    return False;

                p[0],p[1]  = point[0], point[1];
                t0[0],t0[1] = a[0], a[1]
                t1[0],t1[1] = b[0], b[1]
                t2[0],t2[1] = c[0], c[1]
                
                if(self.triangle_contains_point2D(p, t0, t1, t2) == False):
                    return False;
                
                return True
        
        
    #We split the quad in two triangles and check in both triangles.
    def quad_contains_point(self, point):
        if len(self.vertices) == 4:
            if self.all_vertices_are_different():
                
                a = self.vertices[0]
                b = self.vertices[1]
                c = self.vertices[2]
                d = self.vertices[3]
                
                new_vertices = np.zeros((3, 3), dtype = 'float64')
                new_vertices[0] = a
                new_vertices[1] = b
                new_vertices[2] = d
                
                tmp = self.vertices
                self.vertices = new_vertices
                tri = self.triangle_contains_point(point)
                if tri:
                    self.vertices = tmp
                    return True
                
                new_vertices[0] = b
                new_vertices[1] = c
                new_vertices[2] = d
                
                self.vertices = new_vertices
                tri = self.triangle_contains_point(point)
                self.vertices = tmp
                
                return tri
            else:
                print('Due o piu vertici sono nella stessa posizione')
    
    
    #Method to check if a point is in on the same side of the remaining vertex.
    #We calculate the normal with a as origin point and we do the dot product between the normal and 'ad' and 'ap' to find 
    #if the sign of the dot products is the same. If the dot point between the normal and ap is 0 the point lies on the plane.
    @staticmethod
    def same_side(a, b, c, d, point):
        normal = np.cross(b - a, c - a)
        dot_d = np.dot(normal, d - a)
        dot_point = np.dot(normal, point - a)
        return np.sign(dot_d) == np.sign(dot_point) or dot_point == 0
       
        
    #Method to check if a point is inside a tetrahedron.
    #We split the tetrahedron in 4 triangles and for every triangle we check if the point lies on the same side of the 
    #remaining vertex
    def tet_contains_point(self, point):
        v1 = self.vertices[0]
        v2 = self.vertices[1]
        v3 = self.vertices[2]
        v4 = self.vertices[3]
        return (self.same_side(v1, v2, v3, v4, point)
                and self.same_side(v2, v3, v4, v1, point)
                and self.same_side(v3, v4, v1, v2, point)
                and self.same_side(v4, v1, v2, v3, point))
    
    
    #Method to check if a point is inside a hexahedron.
    #We split the hexahedron in 5 tetrahedron, and check if the point lies in one of them.
    def hex_contains_point(self, point):
        if len(self.vertices) == 8:
            if self.all_vertices_are_different():
                a = self.vertices[0]
                b = self.vertices[1]
                c = self.vertices[2]
                d = self.vertices[3]
                e = self.vertices[4]
                f = self.vertices[5]
                g = self.vertices[6]
                h = self.vertices[7]
                
                new_vertices = np.zeros((4, 3),dtype='float64')
                new_vertices[0] = a
                new_vertices[1] = b
                new_vertices[2] = c
                new_vertices[3] = f
                
                tmp = self.vertices
                self.vertices = new_vertices
                quad = self.tet_contains_point(point)
                if quad:
                    self.vertices = tmp
                    return True
                
                new_vertices[0] = a
                new_vertices[1] = c
                new_vertices[2] = h
                new_vertices[3] = f
                
                self.vertices = new_vertices
                quad = self.tet_contains_point(point)
                if quad:
                    self.vertices = tmp
                    return True
                
                new_vertices[0] = a
                new_vertices[1] = c
                new_vertices[2] = d
                new_vertices[3] = h
                
                self.vertices = new_vertices
                quad = self.tet_contains_point(point)
                if quad:
                    self.vertices = tmp
                    return True
                
                new_vertices[0] = a
                new_vertices[1] = f
                new_vertices[2] = h
                new_vertices[3] = e
                
                self.vertices = new_vertices
                quad = self.tet_contains_point(point)
                if quad:
                    self.vertices = tmp
                    return True

                new_vertices[0] = c
                new_vertices[1] = h
                new_vertices[2] = f
                new_vertices[3] = g
                
                self.vertices = new_vertices
                quad = self.tet_contains_point(point)
                self.vertices = tmp
                
                return quad
            else:
                print('Due o piu vertici sono nella stessa posizione')


    #Möller–Trumbore intersection algorithm to check if a ray intersects a triangle
    #We calculate the determinant, if it's close to 0 the ray is parallel to the triangle.
    #Then we calculate the barycentric coordinates of the point on the triangle and check if they are valid.
    #t is the distance from the ray origin to the point, if it's positive the ray hit the triangle, otherwise the direction is 
    #opposite.
    def ray_interesects_triangle(self, r_origin, r_dir):
        a = self.vertices[0]
        b = self.vertices[1]
        c = self.vertices[2]
    
        EPSILON = 0.0000001

        e0 = b - a;
        e1 = c - a;
        p = np.cross(r_dir, e1)
        det = np.dot(e0, p)
        
        if(abs(det) < EPSILON):
            return False
    
        invDet = 1.0 / det
        t = r_origin - a
        u = np.dot(t, p) * invDet
        if(u < 0.0 or u > 1.0):
            return False

        q = np.cross(t, e0)
        v = np.dot(r_dir, q) * invDet
        
        if(v < 0.0 or u + v > 1.0):
            return False
    
        t = np.dot(e1, q) * invDet
        
        if t >= 0:
            return True
        else:
            return False
     
    
    #Method to check if a ray intersects a quad.
    #We split the quad in two triangles and check if the ray hit one of the triangles
    def ray_interesects_quad(self, r_origin, r_dir):
        a = self.vertices[0]
        b = self.vertices[1]
        c = self.vertices[2]
        d = self.vertices[3]
        
        new_vertices = np.zeros((3, 3), dtype = 'float64')
        new_vertices[0] = a
        new_vertices[1] = b
        new_vertices[2] = d
        
        tmp = self.vertices
        self.vertices = new_vertices
        tri = self.ray_interesects_triangle(r_origin, r_dir)
        
        if tri:
            self.vertices = tmp
            return True
        
        new_vertices[0] = b
        new_vertices[1] = c
        new_vertices[2] = d
        
        self.vertices = new_vertices
        tri = self.ray_interesects_triangle(r_origin, r_dir)
        self.vertices = tmp
        
        return tri
    
    
    #Method to check if a ray intersects a tet.
    #We check if ray intersects one of the faces
    def ray_interesects_tet(self, r_origin, r_dir):
        a = self.vertices[0]
        b = self.vertices[1]
        c = self.vertices[2]
        d = self.vertices[3]
        
        new_vertices = np.zeros((3, 3), dtype = 'float64')
        new_vertices[0] = a
        new_vertices[1] = b
        new_vertices[2] = c
        
        tmp = self.vertices
        self.vertices = new_vertices
        tri = self.ray_interesects_triangle(r_origin, r_dir)    
        
        if tri:
            self.vertices = tmp
            return True
        
        new_vertices[0] = a
        new_vertices[1] = b
        new_vertices[2] = d
        
        self.vertices = new_vertices
        tri = self.ray_interesects_triangle(r_origin, r_dir)    
        
        if tri:
            self.vertices = tmp
            return True
        
        new_vertices[0] = b
        new_vertices[1] = c
        new_vertices[2] = d
        
        self.vertices = new_vertices
        tri = self.ray_interesects_triangle(r_origin, r_dir)    
        
        if tri:
            self.vertices = tmp
            return True
        
        new_vertices[0] = a
        new_vertices[1] = c
        new_vertices[2] = d
        
        self.vertices = new_vertices
        tri = self.ray_interesects_triangle(r_origin, r_dir)    
        self.vertices = tmp

        return tri
    
    
    #Method to check if a ray intersects a hex.
    #We check if ray intersects one of the faces
    def ray_interesects_hex(self, r_origin, r_dir):
        a = self.vertices[0]
        b = self.vertices[1]
        c = self.vertices[2]
        d = self.vertices[3]
        e = self.vertices[4]
        f = self.vertices[5]
        g = self.vertices[6]
        h = self.vertices[7]
        
        new_vertices = np.zeros((4, 3), dtype = 'float64')
        new_vertices[0] = a
        new_vertices[1] = b
        new_vertices[2] = c
        new_vertices[3] = d

        tmp = self.vertices
        self.vertices = new_vertices
        quad = self.ray_interesects_quad(r_origin, r_dir)    
        
        if quad:
            self.vertices = tmp
            return True
        
        new_vertices[0] = e
        new_vertices[1] = f
        new_vertices[2] = g
        new_vertices[3] = h

        self.vertices = new_vertices
        quad = self.ray_interesects_quad(r_origin, r_dir)    
        
        if quad:
            self.vertices = tmp
            return True
        
        new_vertices[0] = a
        new_vertices[1] = b
        new_vertices[2] = f
        new_vertices[3] = e

        self.vertices = new_vertices
        quad = self.ray_interesects_quad(r_origin, r_dir)    
        
        if quad:
            self.vertices = tmp
            return True
        
        new_vertices[0] = c
        new_vertices[1] = d
        new_vertices[2] = h
        new_vertices[3] = g

        self.vertices = new_vertices
        quad = self.ray_interesects_quad(r_origin, r_dir)  
        
        if quad:
            self.vertices = tmp
            return True
        
        new_vertices[0] = a
        new_vertices[1] = e
        new_vertices[2] = h
        new_vertices[3] = d
        
        self.vertices = new_vertices
        quad = self.ray_interesects_quad(r_origin, r_dir)    
        
        if quad:
            self.vertices = tmp
            return True
        
        new_vertices[0] = b
        new_vertices[1] = f
        new_vertices[2] = g
        new_vertices[3] = c
        
        self.vertices = new_vertices
        quad = self.ray_interesects_quad(r_origin, r_dir)    
        self.vertices = tmp

        return quad