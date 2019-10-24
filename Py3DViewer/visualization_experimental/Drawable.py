import pythreejs as three
import numpy as np
from time import time
from .Colors import colors
from ..utils import Observer
from ..structures import *

class Drawable(Observer):
    
    def __init__(self, geometry, mesh_color = None, reactive = False):
        super(Drawable, self).__init__()
        self.geometry = geometry
        if reactive:
            self.geometry.attach(self)
        self.geometry_color = self.__initialize_geometry_color(mesh_color)
        self.drawable_mesh, self.__buffer_geometry = self.__initialize_drawable_mesh()
        self.wireframe = self.__initialize_wireframe()
        self.__last_update_time = time()

    def __initialize_geometry_color(self, mesh_color):
        if mesh_color is None:
            mesh_type = type(self.geometry)
            if mesh_type == Trimesh or mesh_type == Tetmesh:
                return np.repeat(colors.teal,
                                 self.geometry.num_faces,
                                 axis=0)
            elif mesh_type == Quadmesh or mesh_type == Hexmesh:
                return np.repeat(colors.teal,
                                 self.geometry.num_faces * 2,
                                 axis=0)

    def __initialize_wireframe(self):
        mesh_type = type(self.geometry)
        if mesh_type == Trimesh or mesh_type == Tetmesh:
            edges_material = three.MeshBasicMaterial(color='#686868',
                                                 side='FrontSide',
                                                 polygonOffset=True,
                                                 polygonOffsetFactor=1,
                                                 polygonOffsetUnits=1,
                                                 #shininess=0.5,
                                                 wireframe=True,
                                                 linewidth = 1,
                                                 opacity=0.2,
                                                 depthTest=True,
                                                 transparent=True)
            return three.Mesh(
                geometry=self.__buffer_geometry,
                material=edges_material,
                position=[0, 0, 0]   
            )
        elif mesh_type == Quadmesh or mesh_type == Hexmesh:
            edges_material = three.LineBasicMaterial(color='#686868', 
                                                        linewidth = 1, 
                                                        depthTest=True, 
                                                        opacity=.2,
                                                        transparent=True)
            boundaries = self.geometry.boundary()[0]
            edges = np.c_[boundaries[:,:2], boundaries[:,1:3], boundaries[:,2:4], boundaries[:,3], boundaries[:,0]].flatten()
            surface_wireframe = self.geometry.vertices[edges].tolist()
            wireframe = three.BufferGeometry(attributes={'position': three.BufferAttribute(surface_wireframe, normalized=False)})
            return three.LineSegments(wireframe, material = edges_material, type = 'LinePieces')


    def __get_drawable_from_boundary(self):
        mesh_type = type(self.geometry)     
        boundaries = self.geometry.boundary()[0]
        n_vertices_per_simplex = 3
        if mesh_type == Quadmesh or mesh_type == Hexmesh:
            boundaries = np.c_[boundaries[:,:3], boundaries[:,2:], boundaries[:,0]]
            boundaries.shape = (-1, 3)
            n_vertices_per_simplex = 6
            
        geometry_attributes = {
            'position': three.BufferAttribute(self.geometry.vertices[boundaries.flatten()], normalized=False),
            'color': three.BufferAttribute(self.geometry_color[np.repeat(self.geometry.boundary()[1], n_vertices_per_simplex)], normalized=False),}
        drawable_geometry = three.BufferGeometry(attributes = geometry_attributes)
        drawable_geometry.exec_three_obj_method("computeVertexNormals")
        return drawable_geometry
    
    
    def __initialize_drawable_mesh(self):
        drawable_geometry = self.__get_drawable_from_boundary()
        material = three.MeshLambertMaterial(
                                           polygonOffset=True,
                                           polygonOffsetFactor=1,
                                           polygonOffsetUnits=1,
                                           flatShading = True,
                                           color = "white",
                                           opacity = 1.,
                                           transparent = False,
                                           side = 'FrontSide',
                                           wireframe=False,
                                           vertexColors = 'FaceColors',
                                          )
        return three.Mesh(
            geometry=drawable_geometry,
            material=material,
            position=[0, 0, 0]
        ), drawable_geometry

    def update(self):
        #current_time = time()
        #if (current_time - self.__last_update_time) > 1/10:
        new_drawable_geometry = self.__get_drawable_from_boundary()
        self.drawable_mesh.geometry = new_drawable_geometry
        self.wireframe.geometry = new_drawable_geometry
        #self.__last_update_time = current_time
        
    @property
    def center(self):
        return self.geometry.center
    
    @property
    def scale(self):
        return self.geometry.scale