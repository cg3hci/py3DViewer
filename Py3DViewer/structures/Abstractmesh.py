import numpy as np
from ..visualization.Viewer import Viewer


class AbstractMesh(object):

    """
    This class represents a generic mesh. It must be estended by a specific mesh class. It stores all the information
    shared among the different kind of supported meshes.
    """
    
    def __init__(self):
        
        self.vertices            = None #npArray (Nx3)
        self.vtx_normals         = None #npArray (Nx3) ## Is this used by volumetric meshes? Consider moving it inside surface meshes only
        self.faces               = None #npArray (NxM)
        self.__vtx2face          = None #npArray (NxM)
        self.__vtx2vtx           = None #npArray (Nx1)
        self.__bounding_box      = None #npArray (2x3)
        self.simplex_metrics     = dict() #dictionary[propertyName : ((min, max), npArray (Nx1))]
        self.__simplex_centroids = None #npArray (Nx1)
        self.__cut            = {'min_x':None, 
                                 'max_x':None, 
                                 'min_y':None, 
                                 'max_y':None, 
                                 'min_z':None, 
                                 'max_z':None}  #dictionary
        
        super(AbstractMesh, self).__init__()
        
     
    # ==================== METHODS ==================== #
    
        
    def show(self, UI = False, width = 700, height = 700, mesh_color = None):

        """
        Show the mesh within the current cell. It is possible to manipulate the mesh through the UI.

        Parameters:

            UI (bool): Show or not show the graphic user interface of the viewer
            width (int): The width of the canvas
            height (int): The height of thne canvas
        
        Return:

            Viewer: The viewer object
        """

        view = Viewer(self, UI=UI, mesh_color=mesh_color, width = width, height = height).show()
        return view
        
    
    @property
    def cut(self):
        
        return self.__cut
    
    
    def set_cut(self, min_x = None, max_x = None, 
                      min_y = None, max_y = None, 
                      min_z = None, max_z = None):
        """
        Cut the mesh along x, y and z axes. It doesn't affect the geometry of the mesh.

        Parameters:

            min_x (float): The minimum value of x
            max_x (float): The maximum value of x
            min_y (float): The minimum value of y
            max_y (float): The maximum value of y
            min_z (float): The minimum value of z
            max_z (float): The maximum value of z
    
        """
        if min_x is not None:
            self.__cut['min_x'] = min_x
        if max_x is not None:
            self.__cut['max_x'] = max_x
        if min_y is not None:
            self.__cut['min_y'] = min_y
        if max_y is not None:
            self.__cut['max_y'] = max_y
        if min_z is not None:
            self.__cut['min_z'] = min_z
        if max_z is not None:
            self.__cut['max_z'] = max_z
            
    def reset_cut(self):

        """
        Set the cuts to the bounding box in order to show the whole mesh.
        """        
        
        self.set_cut(min_x = self.bbox[0,0], max_x = self.bbox[1,0], 
                     min_y = self.bbox[0,1], max_y = self.bbox[1,1],
                     min_z = self.bbox[0,2], max_z = self.bbox[1,2])
           

    def load_from_file(filename):
        
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
        
        
    def boundary(self, flip_x = None, flip_y = None, flip_z = None):
        
        raise NotImplementedError('This method must be implemented in the subclasses')
        
        
    def add_vertex(self, x, y, z): 
        
        """
        Add a new vertex to the current mesh. It affects the mesh geometry. 

        Parameters:

            x (float): The x coordinate of the new vertex
            y (float): The y coordinate of the new vertex
            z (float): The z coordinate of the new vertex
    
        """
        
        new_vertex = np.array([x,y,z], dtype=np.float)
        new_vertex.shape = (1,3)
        
        self.vertices = np.concatenate([self.vertices, new_vertex])
    
    
    def add_vertices(self, new_vertices):

        """
        Add a list of new vertices to the current mesh. It affects the mesh geometry. 

        Parameters:

            new_vertices (Array (Nx3) type=float): List of vertices to add. Each vertex is in the form [float,float,float]
    
        """
        
        new_vertices = np.array(new_vertices)
        self.vertices = np.concatenate([self.vertices, new_vertices])
        
        
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
        self.show()
        return f"Showing {self.num_faces} polygons."
