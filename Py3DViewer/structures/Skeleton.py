from ..visualization.Viewer import Viewer
import numpy as np
from ..utils import Observer, Subject, IO, ObservableArray

class Skeleton(Observer, Subject):
    
    """
    This class represent a skeleton composed of joints and bones. It is possible to load the mesh from a file (.skel) or
    from raw geometry and topology data.

    Parameters:

        filename (string): The name of the file to load 
        joints (Array (Nx3) type=float): The list of joints of the skeleton
        radius (float): The radius of the joints of the skeleton
        bones (Array (Nx2) type=int): The list of bones of the skeleton 
        load_associated_mesh (boolean): True if you want to automatically load the mesh associated to the skeleton, False otherwise
    
    """
    
    def __init__(self, filename=None, vertices=None, radius=1.0, edges=None):
        
        self.vertices = None
        self.radius = None
        self.edges = None
        
        if filename is not None:
            
            self.__load_from_file(filename)
            
        elif vertices is not None and edges is not None:
            
            self.radius = self.__make_observable(radius)
            self.vertices = self.__make_observable(vertices)
            self.edges = self.__make_observable(edges)

        Observer.__init__(self)
        Subject.__init__(self)
        
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
    
    def __load_from_file(self, filename):
        if 'skel' in filename.split('.')[-1]:
            self.vertices, self.radius, self.edges = IO.read_skeleton(filename)
            self.radius = self.__make_observable(self.radius)
            self.vertices = self.__make_observable(self.vertices)
            self.edges = self.__make_observable(self.edges)
 
    def as_edges_flat(self):
        return self.edges.astype(np.int).flatten()
    
    @property
    def center(self):
        return np.mean(self.vertices, axis=0)
    
    @property
    def scale(self):
        return np.linalg.norm(np.min(self.vertices, axis=0)-np.max(self.vertices, axis=0))