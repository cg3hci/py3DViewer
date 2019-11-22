import pythreejs as three
import numpy as np
from time import time, sleep
from .Colors import colors
from ..utils import Observer, ColorMap
import threading
import copy
    
class DrawablePointCloud(Observer):
    
    def __init__(self, geometry, point_size = None, point_color = None, reactive = False):
        super(DrawablePointCloud, self).__init__()        
        self.geometry = geometry
        if reactive:
            self.geometry.attach(self)
        self.color = point_color
        self.size = point_size if point_size is not None else self.geometry.scale/100
        self.pointcloud = self.__initialize_pointcloud()
        self.threejs_items = [self.pointcloud]
        self.updating = False
        self.queue = False
        
    def update_points_color(self, new_color):
        self.pointcloud.material.color = new_color
        
    def __initialize_pointcloud(self):
        points_material = three.PointsMaterial(color='#ff0000' if self.color is None else self.color, 
                                                        size=self.size)
        points = self.__get_points_from_pointcloud()
        return three.Points(points, material = points_material)
    
    def __as_buffer_attr(self, array):
        return three.BufferAttribute(array, normalized = False, dynamic = True)

    def __get_points_from_pointcloud(self): 
        vertices = self.geometry.vertices.astype(np.float32)
        buffer = np.empty((int(vertices.shape[0] * 3), 3), dtype=np.float32).reshape(-1, 3)
        buffer[:vertices.shape[0]] = vertices
        vertex_buffer = self.__as_buffer_attr(buffer)
        points = three.BufferGeometry(attributes={'position': vertex_buffer})
        points.exec_three_obj_method("setDrawRange", 0, vertices.shape[0])
        return points
    
    def run(self, geometry):
        points = self.geometry.vertices.astype(np.float32)
        self.pointcloud.geometry.attributes['position'].array[:points.shape[0]] = points
        self.pointcloud.geometry.exec_three_obj_method('setDrawRange', 0, points.shape[0])
        self.pointcloud.geometry.attributes['position'].array = self.pointcloud.geometry.attributes['position'].array
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