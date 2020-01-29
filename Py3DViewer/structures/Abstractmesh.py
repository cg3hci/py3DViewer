import numpy as np
from ..visualization import Viewer
from ..utils import Subject, Observer
import copy

class Clipping(object):
            
    class __Flip(object):
        def __init__(self):
            self.x = False
            self.y = False
            self.z = False

    def __init__(self):
        self.min_x = None
        self.max_x = None
        self.min_y = None
        self.max_y = None
        self.min_z = None
        self.max_z = None
        self.flip = self.__Flip()
        super(Clipping, self).__init__()
        
    def __repr__(self):
        return ("Clipping:\n" +
                f"min_x: {self.min_x} \tmax_x: {self.max_x} \t{('flipped' if self.flip.x else '')}\n" +
                f"min_y: {self.min_y} \tmax_y: {self.max_y} \t{('flipped' if self.flip.y else '')}\n" +
                f"min_z: {self.min_z} \tmax_z: {self.max_z} \t{('flipped' if self.flip.z else '')}\n")

class AbstractMesh(Observer, Subject):

    """
    This class represents a generic mesh. It must be estended by a specific mesh class. It stores all the information
    shared among the different kind of supported meshes.
    """
    
    def __init__(self):
        
        self.__finished_loading = False
        self.vertices            = None #npArray (Nx3)
        self.vtx_normals         = None #npArray (Nx3) ## Is this used by volumetric meshes? Consider moving it inside surface meshes only
        self.faces               = None #npArray (NxM)
        self._dont_update        = False
        self.__vtx2face          = None #npArray (NxM)
        self.__vtx2vtx           = None #npArray (Nx1)
        self.__bounding_box      = None #npArray (2x3)
        self.simplex_metrics     = dict() #dictionary[propertyName : ((min, max), npArray (Nx1))]
        self.__simplex_centroids = None #npArray (Nx1)
        self.__clipping          = Clipping()
        self.__boundary_needs_update = True
        self.__boundary_cached = None
        Observer.__init__(self)
        Subject.__init__(self)
        
     
    # ==================== METHODS ==================== #
    
    def __setattr__(self, key, value):
        self.__dict__[key] = value
        if key[0] != "_" and self.__finished_loading:
            self.update()
        
    def copy(self):
        """
        Remember to add that this doesn't copy observer, vtx2vtx and vtx2face, and this is a value copy"""
        new = type(self)()
        for key in self.__dict__.keys():
            if "observer" not in key and "vtx2vtx" not in key and "vtx2face" not in key and "vtx2tet" not in key and "vtx2hex" not in key:
                setattr(new, key, copy.deepcopy(getattr(self, key)))
        return new
        
    def update(self):
        self.__boundary_needs_update = True
        self.__update_bounding_box()
        self.reset_clipping()
        if (not self._dont_update):
            self._notify()
        
    def show(self, width = 700, height = 700, mesh_color = None, reactive =  False):

        """
        Show the mesh within the current cell. It is possible to manipulate the mesh through the UI.

        Parameters:

            UI (bool): Show or not show the graphic user interface of the viewer
            width (int): The width of the canvas
            height (int): The height of thne canvas
        
        Return:

            Viewer: The viewer object
        """

        view = Viewer(self, mesh_color=mesh_color, width = width, height = height, reactive=reactive)
        view.show()
        return view
        
    @property
    def clipping(self):
        
        return self.__clipping
    
    
    def set_clipping(self, min_x = None, max_x = None, 
                      min_y = None, max_y = None, 
                      min_z = None, max_z = None,
                      flip_x = None, flip_y = None, flip_z = None):
        """
        clipping the mesh along x, y and z axes. It doesn't affect the geometry of the mesh.

        Parameters:

            min_x (float): The minimum value of x
            max_x (float): The maximum value of x
            min_y (float): The minimum value of y
            max_y (float): The maximum value of y
            min_z (float): The minimum value of z
            max_z (float): The maximum value of z
    
        """
        if min_x is not None:
            self.__clipping.min_x = min_x
        if max_x is not None:
            self.__clipping.max_x = max_x
        if min_y is not None:
            self.__clipping.min_y = min_y
        if max_y is not None:
            self.__clipping.max_y = max_y
        if min_z is not None:
            self.__clipping.min_z = min_z
        if max_z is not None:
            self.__clipping.max_z = max_z
        if flip_x is not None:
            self.__clipping.flip.x = flip_x
        if flip_y is not None:
            self.__clipping.flip.y = flip_y
        if flip_z is not None:
            self.__clipping.flip.z = flip_z
        
        self.__boundary_needs_update = True
        self.update()
        
    def reset_clipping(self):

        """
        Set the clippings to the bounding box in order to show the whole mesh.
        """        
        
        self.set_clipping(min_x = self.bbox[0,0], max_x = self.bbox[1,0], 
                     min_y = self.bbox[0,1], max_y = self.bbox[1,1],
                     min_z = self.bbox[0,2], max_z = self.bbox[1,2])
        self.__boundary_needs_update = True
        self.update()

    def load_from_file(filename):
        
        raise NotImplementedError('This method must be implemented in the subclasses')
            
    def __compute_adjacencies(self):
        
        raise NotImplementedError('This method must be implemented in the subclasses')

    
    def save_file(self, filename):
        
        raise NotImplementedError('This method must be implemented in the subclasses')


    def get_metric(self, property_name, id_element):
        """
        Get a specific metric element from the dictionary of metrics 'simplex_metrics'.

        Parameters:

            property_name (string): The name of the wanted metric
            id_element (int): The index of a specific element of the metric

        Returns:
            object: The specific metric element. The return type depends on the metric
    
        """
        return self.simplex_metrics[property_name][id_element]
    
    @property
    def simplex_centroids(self):
        
        raise NotImplementedError('This method must be implemented in the subclasses')
        
    
    def __compute_metrics(self): 
        
        raise NotImplementedError('This method must be implemented in the subclasses')
        
    def as_triangles_flat(self):
        
        raise NotImplementedError('This method must be implemented in the subclasses')
        
    def as_edges_flat(self):
        
        raise NotImplementedError('This method must be implemented in the subclasses')
        
    def _as_threejs_colors(self):
        
        raise NotImplementedError('This method must be implemented in the subclasses')
        
    @property
    def num_triangles(self):
    
        raise NotImplementedError('This method must be implemented in the subclasses')
        
    def boundary(self):
        
        """
        Compute the boundary of the current mesh. It only returns the faces that are inside the clipping
        """
        min_x = self.clipping.min_x
        max_x = self.clipping.max_x
        min_y = self.clipping.min_y
        max_y = self.clipping.max_y
        min_z = self.clipping.min_z
        max_z = self.clipping.max_z
        flip_x = self.clipping.flip.x
        flip_y = self.clipping.flip.y
        flip_z = self.clipping.flip.z
        centroids = np.array(self.simplex_centroids)
        x_range = np.logical_xor(flip_x,((centroids)[:,0] >= min_x) & (centroids[:,0] <= max_x))
        y_range = np.logical_xor(flip_y,((centroids[:,1] >= min_y) & (centroids[:,1] <= max_y)))
        z_range = np.logical_xor(flip_z,((centroids[:,2] >= min_z) & (centroids[:,2] <= max_z)))
        clipping_range = x_range & y_range & z_range
        return clipping_range
        
        
    def add_vertex(self, x, y, z): 
        
        """
        Add a new vertex to the current mesh. It affects the mesh geometry. 

        Parameters:

            x (float): The x coordinate of the new vertex
            y (float): The y coordinate of the new vertex
            z (float): The z coordinate of the new vertex
    
        """
        self._dont_update = True
        new_vertex = np.array([x,y,z], dtype=np.float)
        new_vertex.shape = (1,3)
        
        self.vertices = np.concatenate([self.vertices, new_vertex])
        self._dont_update = False
        self.update()
    
    
    def add_vertices(self, new_vertices):

        """
        Add a list of new vertices to the current mesh. It affects the mesh geometry. 

        Parameters:

            new_vertices (Array (Nx3) type=float): List of vertices to add. Each vertex is in the form [float,float,float]
    
        """
        
        self._dont_update = True
        new_vertices = np.array(new_vertices)
        self.vertices = np.concatenate([self.vertices, new_vertices])
        self._dont_update = False
        self.update()
        
        
    @property
    def vtx2vtx(self):
        
        return self.__vtx2vtx
        
        
    @property
    def vtx2face(self):
        
        return self.__vtx2face  

    
    @property
    def bbox(self):

        return self.__bounding_box
    
    
    @property
    def num_vertices(self):
        
        return self.vertices.shape[0]
    
    @property
    def center(self):
        
        x1, x2 = self.__bounding_box[0][0], self.__bounding_box[1][0]
        y1, y2 = self.__bounding_box[0][1], self.__bounding_box[1][1]
        z1, z2 = self.__bounding_box[0][2], self.__bounding_box[1][2]
    
        return np.array([(x1+x2)/2, (y1+y2)/2, (z1+z2)/2])
    
    @property
    def scale(self):
        
        return np.linalg.norm(self.__bounding_box[0]-self.__bounding_box[1])
    
    def __update_bounding_box(self):
        
        min_x_coord = self.vertices[:,0].min()
        max_x_coord = self.vertices[:,0].max()
        min_y_coord = self.vertices[:,1].min()
        max_y_coord = self.vertices[:,1].max()
        min_z_coord = self.vertices[:,2].min()
        max_z_coord = self.vertices[:,2].max()
        
        self.__bounding_box = np.array([[min_x_coord, min_y_coord, min_z_coord],
                                        [max_x_coord, max_y_coord, max_z_coord]])
       
    def __repr__(self):
        return f"Mesh of {self.num_faces} polygons."
