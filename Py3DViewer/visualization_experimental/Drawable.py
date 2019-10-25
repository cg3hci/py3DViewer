import pythreejs as three
import numpy as np
from time import time
from .Colors import colors
from ..utils import Observer

class Drawable(Observer):
    
    def __init__(self, geometry, mesh_color = None, reactive = False):
        super(Drawable, self).__init__()
        self.geometry = geometry
        if reactive:
            self.geometry.attach(self)
        self.geometry_color = self.__initialize_geometry_color(mesh_color)
        self.drawable_mesh = self.__initialize_drawable_mesh()
        self.wireframe = self.__initialize_wireframe()
        self.__last_update_time = time()

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
            surface_wireframe = self.geometry.as_edges_flat()
            buffer_wireframe = three.BufferAttribute(surface_wireframe, normalized=False)
            wireframe = three.BufferGeometry(attributes={'position': buffer_wireframe})
            return three.LineSegments(wireframe, material = edges_material, type = 'LinePieces')


    def __get_drawable_from_boundary(self):
        geometry_attributes = {
            'position': three.BufferAttribute(self.geometry.as_triangles_flat(), normalized = False),
            'color': three.BufferAttribute(self.geometry_color[self.geometry._as_threejs_colors()], normalized = False)}
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
        for key in buffer_dict.keys():
            item = buffer_dict[key]
            item.array = np.zeros(0)
            del item
    
    def update(self):
        #current_time = time()
        #if (current_time - self.__last_update_time) > 1/10:
        new_drawable_geometry = self.__get_drawable_from_boundary()
        old_geometry = self.drawable_mesh.geometry
        self.drawable_mesh.geometry = new_drawable_geometry
        self.__dispose_buffers(old_geometry.attributes)
        old_geometry.exec_three_obj_method("dispose")
        
        old_wireframe_geometry = self.wireframe.geometry
        self.wireframe.geometry = new_drawable_geometry
        self.__dispose_buffers(old_wireframe_geometry.attributes)
        old_wireframe_geometry.exec_three_obj_method("dispose")
        #self.__last_update_time = current_time
        
    @property
    def center(self):
        return self.geometry.center
    
    @property
    def scale(self):
        return self.geometry.scale