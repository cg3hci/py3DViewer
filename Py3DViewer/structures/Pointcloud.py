from ..visualization.Viewer import Viewer
import numpy as np
import copy
from ..utils import Observer, Subject, IO, ObservableArray

class PointCloud(Observer, Subject):
    
    """
    This class represents a Point Cloud. 

    Parameters:

        vertices (np.array nx3): vertices of the point cloud
    
    """
    
    def __init__(self, vertices=None):
        
        if vertices is not None:
            self.vertices = self.__make_observable(vertices)

        Observer.__init__(self)
        Subject.__init__(self)
        
    def copy(self):
        new = type(self)()
        for key in self.__dict__.keys():
            if "observer" not in key:
                setattr(new, key, copy.deepcopy(getattr(self, key)))
        return new
    
    def __make_observable(self, array):
        tmp = ObservableArray(array.shape)
        tmp[:] = array
        tmp.attach(self)
        return tmp
        
    def update(self):
        self._notify()
        
    """
    def show(self, UI = False, width = 700, height = 700):
        view = Viewer(self, UI=UI, width = width, height = height).show()
        return view
    """
    
    @property
    def center(self):
        return np.mean(self.vertices, axis=0)
    
    @property
    def scale(self):
        return np.linalg.norm(np.min(self.vertices, axis=0)-np.max(self.vertices, axis=0))