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
                                                        transparent=True)
            wireframe = self.__get_wireframe_from_boundary()
            return three.LineSegments(wireframe, material = edges_material, type = 'LinePieces')


    def __get_wireframe_from_boundary(self): 
        surface_wireframe = self.geometry.as_edges_flat()
        buffer_wireframe = three.BufferAttribute(surface_wireframe, normalized=False, dynamic=True)
        wireframe = three.BufferGeometry(attributes={'position': buffer_wireframe})
        return wireframe
        
    def __get_drawable_from_boundary(self):
        geometry_attributes = {
            'position': three.BufferAttribute(self.geometry.as_triangles_flat(), normalized = False, dynamic=True),
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
        )

    def __dispose_buffers(self, buffer_dict):
        keys = list(buffer_dict.keys())
        length = len(keys)
        for i in range(length-5):
            el = buffer_dict.pop(keys[i])
            print(el)
            
    def run(self):
        self.wireframe.geometry.attributes['position'].array = self.geometry.as_edges_flat()
        self.drawable_mesh.geometry.attributes['color'] .array = self.geometry_color[self.geometry._as_threejs_colors()]
        self.drawable_mesh.geometry.attributes['position'] .array = self.geometry.as_triangles_flat()
        self.drawable_mesh.geometry.exec_three_obj_method("computeVertexNormals")
        if self.queue:
            self.queue = False
            self.updating = False
            self.update()
        else:
            self.updating = False
        
    def update(self):
        if (not self.updating):
            self.geometry.boundary() #Why will this line make everything work?
            self.updating=True
            thread = threading.Thread(target=self.run, args=())
            thread.daemon = True                            # Daemonize thread
            thread.start()                                  # Start the execution
        else:
            self.queue = True
        
    @property
    def center(self):
        return self.geometry.center
    
    @property
    def scale(self):
        return self.geometry.scale