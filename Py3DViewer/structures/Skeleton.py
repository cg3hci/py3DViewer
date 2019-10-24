from .Trimesh import Trimesh
from ..visualization.Viewer import Viewer
import numpy as np
from ..utils import IO

class Skeleton(object):
    
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
    
    def __init__(self, filename=None, joints=None, radius=1.0, bones=None, load_associated_mesh=False):
        
        self.joints = None
        self.radius = None
        self.bones = None
        
        if filename is not None:
            
            self.__load_from_file(filename)
            if load_associated_mesh:
                self.associated_mesh = Trimesh(filename.replace('skel', 'obj'))
        
        elif joints is not None and bones is not None:
            
            self.joints = joints
            self.radius = radius
            self.bones = bones
            

        super(Skeleton,self).__init__()
        
        
    def show(self, UI = False, width = 700, height = 700):
        view = Viewer(self, UI=UI, width = width, height = height).show()
        return view
        
        
    
    
    
    def __load_from_file(self, filename):
        
        if 'skel' in filename.split('.')[-1]:
            
            self.joits, self.radius, self.bones = IO.read_skeleton(filename)
 