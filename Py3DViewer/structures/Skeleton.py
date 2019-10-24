from .Trimesh import Trimesh
from ..visualization.Viewer import Viewer
import numpy as np
from ..utils import IO

class Skeleton(object):
    
    def __init__(self, filename=None, nodes=None, radius=1.0, bones=None, load_associated_mesh=False):
        
        self.nodes = None
        self.radius = None
        self.bones = None
        
        if filename is not None:
            
            self.__load_from_file(filename)
            if load_associated_mesh:
                self.associated_mesh = Trimesh(filename.replace('skel', 'obj'))
        
        elif nodes is not None and edges is not None:
            
            self.nodes = nodes
            self.radius = radius
            self.bones = edges
            
    

        super(Skeleton,self).__init__()
        
        
    def show(self, UI = False, width = 700, height = 700):
        view = Viewer(self, UI=UI, width = width, height = height).show()
        return view
        
        
    
    
    
    def __load_from_file(self, filename):
        
        if 'skel' in filename.split('.')[-1]:
            
            self.nodes, self.radius, self.bones = IO.read_skeleton(filename)
 