import numpy as np

class AbstractMesh(object):
    
    def __init__(self):
        
        self.vertices            = None #npArray (Nx3)
        self.vtx_normals         = None #npArray (Nx3) ## Is this used by volumetric meshes? Consider moving it inside surface meshes only
        self.faces               = None #npArray (NxM)
        self.__vtx2face          = None #npArray (NxM)
        self.__vtx2vtx           = None #npArray (Nx1)
        self.__bounding_box      = None #npArray (2x3)
        self.simplex_metrics     = dict() #dictionary[propertyName : npArray (Nx1)]
        self.__simplex_centroids = None #npArray (Nx1)
        self.__cut            = {'min_x':None, 
                                 'max_x':None, 
                                 'min_y':None, 
                                 'max_y':None, 
                                 'min_z':None, 
                                 'max_z':None}  #dictionary
        
        super(AbstractMesh, self).__init__()
        
     
    # ==================== METHODS ==================== #
    
    @property
    def cut(self):
        
        return self.__cut
    
    
    def set_cut(self, min_x = None, max_x = None, 
                      min_y = None, max_y = None, 
                      min_z = None, max_z = None):
        
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
        
        self.set_cut(min_x = self.bbox[0,0], max_x = self.bbox[1,0], 
                     min_y = self.bbox[0,1], max_y = self.bbox[1,1],
                     min_z = self.bbox[0,2], max_z = self.bbox[1,2])
           

    def load_from_file(filename):
        
        raise NotImplementedError('This method must be implemented in the subclasses')
        
    
    def save_file(self, filename):
        
        raise NotImplementedError('This method must be implemented in the subclasses')


    def get_metric(self, property_name, id_element):
        
        return self.simplex_metrics[property_name][id_element]
    
    @property
    def simplex_centroids(self):
        
        raise NotImplementedError('This method must be implemented in the subclasses')
        
    
    def __compute_metrics(self): 
        
        raise NotImplementedError('This method must be implemented in the subclasses')
        
        
    def boundary(self, flip_x = None, flip_y = None, flip_z = None):
        
        raise NotImplementedError('This method must be implemented in the subclasses')
        
        
    def add_vertex(self, x, y, z): 
        
        new_vertex = np.array([x,y,z], dtype=np.float)
        new_vertex.shape = (1,3)
        
        self.vertices = np.concatenate([self.vertices, new_vertex])
    
    
    def add_vertices(self, new_vertices):
        
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
        
        