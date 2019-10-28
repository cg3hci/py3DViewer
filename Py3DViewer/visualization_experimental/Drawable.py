import pythreejs as three
import numpy as np
from time import time, sleep
from .Colors import colors
from ..utils import Observer
import threading

    
class Drawable(Observer):
    
    def __init__(self, geometry, mesh_color = None, reactive = False):
        super(Drawable, self).__init__()
        self.geometry = geometry
        if reactive:
            self.geometry.attach(self)
        self.geometry_color = self.__initialize_geometry_color(mesh_color)
        self.drawable_mesh = self.__initialize_drawable_mesh()
        self.wireframe = self.__initialize_wireframe()
        self.updating = False
        self.queue = False
        
    def __initialize_geometry_color(self, mesh_color):
        if mesh_color is None:
            return np.repeat(colors.teal,
                             self.geometry.num_triangles,
                             axis=0)

    def __initialize_wireframe(self):
        edges_material = three.LineBasicMaterial(color='#686868', 
                                                        linewidth = 1, 
                                                        depthTest=True, 
                                                        opacity=.2,
                                                        transparent=False)
        wireframe = self.__get_wireframe_from_boundary()
        return three.LineSegments(wireframe, material = edges_material)


    def __get_wireframe_from_boundary(self): 
        surface_wireframe = self.geometry.as_edges_flat()
        buffer_wireframe = three.BufferAttribute(surface_wireframe, normalized=False, dynamic=True)
        wireframe = three.BufferGeometry(attributes={'position': buffer_wireframe})
        return wireframe
        
    def __get_drawable_from_boundary(self):
        geometry_attributes = {
            #'position': three.BufferAttribute(self.geometry.as_triangles_flat(), normalized = False, dynamic=True),
            'position': three.BufferAttribute(self.geometry.vertices, normalized = False, dynamic = True),
            'index' : three.BufferAttribute(self.geometry.as_triangles(), normalized = False, dynamic = True),
            'color': three.BufferAttribute(self.geometry_color[self.geometry._as_threejs_colors()], normalized = False, dynamic=True)}
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
                                           #color = "white",
                                           #opacity = 1.,
                                           #transparent = False,
                                           side = 'DoubleSide',
                                           #wireframe=False,
                                           vertexColors = 'FaceColors',
                                          )
        return three.Mesh(
            geometry=drawable_geometry,
            material=material,
            position=[0, 0, 0]
        )

    def run(self):
        edges = self.geometry.as_edges_flat()
        tris = self.geometry.as_triangles()
        colors = self.geometry._as_threejs_colors()
        self.wireframe.geometry.attributes['position'].array = edges
        self.drawable_mesh.geometry.attributes['color'].array = self.geometry_color[colors]
        self.drawable_mesh.geometry.attributes['index'].array = tris
        #self.drawable_mesh.geometry.exec_three_obj_method("computeVertexNormals") Why does this cause index attribute to disappear!?
        if self.queue:
            self.queue = False
            self.updating = False
            self.update()
        else:
            self.updating = False
        
    def update(self):
        if (not self.updating):
            self.updating=True
            thread = threading.Thread(target=self.run, args=())
            thread.daemon = True
            thread.start()
        else:
            self.queue = True
        
    @property
    def center(self):
        return self.geometry.center
    
    @property
    def scale(self):
        return self.geometry.scale