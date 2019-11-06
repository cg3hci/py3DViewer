import pythreejs as three
import numpy as np
from time import time, sleep
from .Colors import colors
from ..utils import Observer, ColorMap
import threading
import copy
    
class DrawableMesh(Observer):
    
    def __init__(self, geometry, mesh_color = None, reactive = False):
        super(DrawableMesh, self).__init__()
        self._external_color = colors.teal
        self._internal_color = colors.orange
        self._color_map      = None
        self._metric_string   = None
        self._c_map_string    = None
        self._label_colors    = None
        
        self.geometry = geometry
        if reactive:
            self.geometry.attach(self)
        self.geometry_color = self.__initialize_geometry_color(mesh_color)
        self.mesh = self.__initialize_mesh()
        self.wireframe = self.__initialize_wireframe()
        self.threejs_items = [self.mesh, self.wireframe]
        self.updating = False
        self.queue = False
        
    def __initialize_geometry_color(self, mesh_color, geometry = None):
        
        if geometry is None:
            geometry = self.geometry
        if mesh_color is None:
            color = np.repeat(self._external_color.reshape(1, 3),
                             geometry.num_triangles*3, axis=0
                             )
            if hasattr(self.geometry, "internals"):
                internal_color = geometry.internal_triangles_idx()
                color[internal_color] = self._internal_color
        
        return color
        
    def update_wireframe_color(self, new_color):
        self.wireframe.material.color = new_color
        
    def update_wireframe_opacity(self, new_opacity):
        self.wireframe.material.opacity = new_opacity
        
    def update_internal_color(self, new_color, geometry = None):
        
        if geometry is None:
            geometry = self.geometry
        self._internal_color = np.array(new_color)
        if hasattr(geometry, "internals"):
            internal_color = geometry.internal_triangles_idx()
            self.geometry_color[internal_color] = new_color
            colors = geometry._as_threejs_colors()
            new_colors = self.geometry_color[colors]
            tris, vtx_normals = geometry._as_threejs_triangle_soup()
            interleaved = np.concatenate((tris, new_colors, vtx_normals), axis=1)
            self.mesh.geometry.attributes['color'].data.array = interleaved


    def update_external_color(self, new_color, geometry = None):
        
        if geometry is None:
            geometry = self.geometry
        self._external_color = np.array(new_color)
        if hasattr(geometry, "internals"):
            internal_color = geometry.internal_triangles_idx()
            self.geometry_color[np.logical_not(internal_color)] = new_color
        else:
            self.geometry_color[:] = new_color
        colors = geometry._as_threejs_colors()
        new_colors = self.geometry_color[colors]
        tris, vtx_normals = geometry._as_threejs_triangle_soup()
        interleaved = np.concatenate((tris, new_colors, vtx_normals), axis=1)
        self.mesh.geometry.attributes['color'].data.array = interleaved
        
    
    def update_color_map(self, new_colors, geometry = None):
        
        if geometry is None:
            geometry = self.geometry
        
        self.geometry_color[:] = geometry._as_threejs_colors(colors= new_colors)
        colors = geometry._as_threejs_colors()
        new_colors = self.geometry_color[colors]
        tris, vtx_normals = geometry._as_threejs_triangle_soup()
        interleaved = np.concatenate((tris, new_colors, vtx_normals), axis=1)
        self.mesh.geometry.attributes['color'].data.array = interleaved
        
    
    def compute_color_map(self, metric_string, c_map_string, geometry=None):
        
        if geometry is None:
            geometry = self.geometry
        
        self._metric_string = metric_string
        self._c_map_string  = c_map_string
        
        (min_range, max_range), metric = self.geometry.simplex_metrics[metric_string]
        c_map = ColorMap.color_maps[c_map_string]
        
        if min_range is None or max_range is None:
            
            min_range = np.min(metric)
            max_range = np.max(metric)
            
            if (np.abs(max_range-min_range) > 1e-7):
                normalized_metric = ((metric - np.min(metric))/np.ptp(metric)) * (c_map.shape[0]-1)
            else:
                normalized_metric = np.repeat(np.mean(metric), metric.shape[0])
        else:
            normalized_metric = np.clip(metric, min_range, max_range)
            normalized_metric = (normalized_metric - min_range)/(max_range-min_range) * (c_map.shape[0]-1)
            
        normalized_metric = 1-normalized_metric
            
        metric_to_colormap = np.rint(normalized_metric).astype(np.int)
        
        mesh_color = c_map[metric_to_colormap]
        
        self._color_map = mesh_color
        self.update_color_map(mesh_color, geometry)
        
    
    def update_color_label(self, geometry = None):
        
        if geometry is None:
            geometry = self.geometry
            
        mesh_color = np.zeros((self.geometry.labels.size,3), dtype=np.float)
        
        for idx, i in enumerate(self.geometry.labels.reshape(-1)):
            mesh_color[idx] = self._label_colors[i]
        
        self._color_map = mesh_color
        self.update_color_map(mesh_color)
            
            
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
        tris, vtx_normals = self.geometry._as_threejs_triangle_soup()
        new_colors = self.geometry_color[self.geometry._as_threejs_colors()].astype(np.float32)
        interleaved_array = np.concatenate((tris, new_colors, vtx_normals), axis=1)
        buffer = three.InterleavedBuffer(array = interleaved_array, stride = 3)
        geometry_attributes['position'] = three.InterleavedBufferAttribute(data=buffer, itemSize=3, dynamic = True)
        geometry_attributes['color'] = three.InterleavedBufferAttribute(data=buffer, itemSize=3, offset=3, dynamic=True)
        geometry_attributes['normal'] = three.InterleavedBufferAttribute(data=buffer, itemSize=3, offset=6, dynamic=True)
        drawable_geometry = three.BufferGeometry(attributes = geometry_attributes)
            
        return drawable_geometry
    
    def __as_buffer_attr(self, array):
        return three.BufferAttribute(array, normalized = False, dynamic = True)

    def __get_wireframe_from_boundary(self): 
        edges = self.geometry.vertices[self.geometry.as_edges_flat()].astype(np.float32)
        buffer = np.empty((int(edges.shape[0] * 3), 3), dtype=np.float32).reshape(-1, 3)
        buffer[:edges.shape[0]] = edges
        vertices = self.__as_buffer_attr(buffer)
        wireframe = three.BufferGeometry(attributes={'position': vertices})
        wireframe.exec_three_obj_method("setDrawRange", 0, edges.shape[0])
        return wireframe
    
    def __initialize_mesh(self):
        drawable_geometry = self.__get_drawable_from_boundary()
        material = three.MeshLambertMaterial(
                                           polygonOffset=True,
                                           polygonOffsetFactor=1,
                                           polygonOffsetUnits=1,
                                           flatShading = True,
                                           color = "white",
                                           opacity = 1.,
                                           transparent = False,
                                           side = 'DoubleSide',
                                           wireframe=False,
                                           vertexColors = 'FaceColors',
                                          )
        return three.Mesh(
            geometry=drawable_geometry,
            material=material,
            position=[0, 0, 0]
        )

    def run(self, geometry):
        edges = self.geometry.vertices[self.geometry.as_edges_flat()].astype(np.float32)
        self.wireframe.geometry.attributes['position'].array[:edges.shape[0]] = edges
        self.wireframe.geometry.exec_three_obj_method('setDrawRange', 0, edges.shape[0])
        self.wireframe.geometry.attributes['position'].array = self.wireframe.geometry.attributes['position'].array
        self.geometry_color = self.__initialize_geometry_color(None, geometry)
        if self._color_map is None:
            self.update_internal_color(self._internal_color, geometry)
            self.update_external_color(self._external_color, geometry)
        
        elif self._label_colors is not None:
            self.update_color_label(geometry)
        else:
            self.compute_color_map(self._metric_string, self._c_map_string, geometry)


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