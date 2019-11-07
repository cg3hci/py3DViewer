import pythreejs as three
import numpy as np
from time import time, sleep
from .Colors import colors
from ..utils import Observer, ColorMap
import threading
import copy
    
class DrawableSkeleton(Observer):
    
    def __init__(self, geometry, skeleton_color = None, reactive = False):
        super(DrawableSkeleton, self).__init__()        
        self.geometry = geometry
        if reactive:
            self.geometry.attach(self)
        self.wireframe = self.__initialize_wireframe()
        self.color = skeleton_color
        self.threejs_items = [self.wireframe]
        self.updating = False
        self.queue = False
        
    def update_wireframe_color(self, new_color):
        self.wireframe.material.color = new_color
        
    def update_wireframe_opacity(self, new_opacity):
        self.wireframe.material.opacity = new_opacity
    
    def __initialize_wireframe(self):
        edges_material = three.LineBasicMaterial(color='#ff0000', 
                                                        linewidth = 1, 
                                                        depthTest=False, 
                                                        opacity=.2,
                                                        transparent=True)
        wireframe = self.__get_wireframe_from_boundary()
        return three.LineSegments(wireframe, material = edges_material)
    
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
    
    def run(self, geometry):
        edges = self.geometry.vertices[self.geometry.as_edges_flat()].astype(np.float32)
        self.wireframe.geometry.attributes['position'].array[:edges.shape[0]] = edges
        self.wireframe.geometry.exec_three_obj_method('setDrawRange', 0, edges.shape[0])
        self.wireframe.geometry.attributes['position'].array = self.wireframe.geometry.attributes['position'].array
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