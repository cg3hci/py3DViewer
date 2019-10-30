import pythreejs as three
import numpy as np
from time import time, sleep
from .Colors import colors
from ..utils import Observer
import threading
import copy
    
class Drawable(Observer):
    
    def __init__(self, geometry, mesh_color = None, reactive = False):
        super(Drawable, self).__init__()
        self._external_color = colors.teal
        self._internal_color = colors.orange
        self.geometry = geometry
        self.tri_soup = geometry._three_triangle_soup
        if reactive:
            self.geometry.attach(self)
        self.geometry_color = self.__initialize_geometry_color(mesh_color)
        self.drawable_mesh = self.__initialize_drawable_mesh()
        
        self.wireframe = self.__initialize_wireframe()
        self.updating = False
        self.queue = False
        
    def __initialize_geometry_color(self, mesh_color):
        if mesh_color is None:
            color = np.repeat(self._external_color,
                             self.geometry.num_triangles,
                             axis=0)
            if hasattr(self.geometry, "internals"):
                internal_color = self.geometry.internal_triangles_idx()
                color[internal_color] = self._internal_color[0]
        
        return color
        
    def update_wireframe_color(self, new_color):
        self.wireframe.material.color = new_color
        
    def update_wireframe_opacity(self, new_opacity):
        self.wireframe.material.opacity = new_opacity
        
    def update_internal_color(self, new_color):
        if hasattr(self.geometry, "internals"):
            internal_color = self.geometry.internal_triangles_idx()
            self.geometry_color[internal_color] = new_color
            colors = self.geometry._as_threejs_colors()
            new_colors = self.geometry_color[colors]
            if (self.tri_soup):
                interleaved = np.concatenate((self.geometry.as_triangles_flat(), new_colors), axis=1)
                self.drawable_mesh.geometry.attributes['color'].data.array = interleaved
            else:
                self.drawable_mesh.geometry.attributes['color'].array = new_colors

    def update_external_color(self, new_color):
        if hasattr(self.geometry, "internals"):
            internal_color = self.geometry.internal_triangles_idx()
            self.geometry_color[np.logical_not(internal_color)] = new_color
        else:
            self.geometry_color[:] = new_color
        colors = self.geometry._as_threejs_colors()
        new_colors = self.geometry_color[colors]
        if (self.tri_soup):
            interleaved = np.concatenate((self.geometry.as_triangles_flat(), new_colors), axis=1)
            self.drawable_mesh.geometry.attributes['color'].data.array = interleaved
        else:
            self.drawable_mesh.geometry.attributes['color'].array = new_colors        
            
    def __initialize_wireframe(self):
        edges_material = three.LineBasicMaterial(color='#686868', 
                                                        linewidth = 1, 
                                                        depthTest=True, 
                                                        opacity=.2,
                                                        transparent=True)
        wireframe = self.__get_wireframe_from_boundary()
        return three.LineSegments(wireframe, material = edges_material)

    def __get_drawable_from_boundary(self):
        geometry_attributes = {}
        if (self.tri_soup):
            tris = self.geometry.as_triangles_flat().astype(np.float32)
            new_colors = self.geometry_color[self.geometry._as_threejs_colors()].astype(np.float32)
            interleaved_array = np.concatenate((tris, new_colors), axis=1)
            buffer = three.InterleavedBuffer(array = interleaved_array, stride = 3)
            geometry_attributes['position'] = three.InterleavedBufferAttribute(data=buffer, itemSize=3, dynamic = True)
            geometry_attributes['color'] = three.InterleavedBufferAttribute(data=buffer, itemSize=3, offset=3, dynamic=True)
            drawable_geometry = three.BufferGeometry(attributes = geometry_attributes)
            #drawable_geometry.exec_three_obj_method("computeVertexNormals")
            return drawable_geometry
        else:
            geometry_attributes['position'] = self.__as_buffer_attr(self.geometry.vertices)
            geometry_attributes['index'] = self.__as_buffer_attr(self.geometry.as_triangles())
            geometry_attributes['normal'] = self.__as_buffer_attr(self.geometry.vtx_normals)
            geometry_attributes['color'] = self.__as_buffer_attr(self.geometry_color[self.geometry._as_threejs_colors()])
            drawable_geometry = three.BufferGeometry(attributes = geometry_attributes)
            return drawable_geometry
            
        return drawable_geometry
    
    def __as_buffer_attr(self, array):
        return three.BufferAttribute(array, normalized = False, dynamic = True)

    def __get_wireframe_from_boundary(self): 
        surface_wireframe = self.geometry.as_edges_flat()
        buffer_wireframe = three.BufferAttribute(surface_wireframe, normalized=False, dynamic=True)
        wireframe = three.BufferGeometry(attributes={'position': buffer_wireframe})
        return wireframe
    
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

    def run(self, geometry):
        edges = geometry.as_edges_flat()
        colors = geometry._as_threejs_colors()
        new_colors = self.geometry_color[colors]
        self.wireframe.geometry.attributes['position'].array = edges
        if (self.tri_soup):
            tris = geometry.as_triangles_flat()
            interleaved_array = np.concatenate((tris, new_colors), axis=1)
            self.drawable_mesh.geometry.attributes['position'].data.array = interleaved_array
            #self.drawable_mesh.geometry.exec_three_obj_method("computeVertexNormals") 
        else:
            tris = geometry.as_triangles()
            self.drawable_mesh.geometry.attributes['index'].array = tris
            self.drawable_mesh.geometry.attributes['color'].array = new_colors
            #self.drawable_mesh.geometry.exec_three_obj_method("computeVertexNormals") #Why does this cause index attribute to disappear!?
        if self.queue:
            self.queue = False
            self.updating = False
            self.update()
        else:
            self.updating = False
        
        
    def update(self):
        if (not self.updating):
            self.updating=True
            thread = threading.Thread(target=self.run, args=(self.geometry.copy(),))
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