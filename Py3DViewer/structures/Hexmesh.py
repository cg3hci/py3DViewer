from .Abstractmesh import AbstractMesh
from ..visualization.Viewer import Viewer
import numpy as np
from ..utils import IO
from ..utils.metrics import hex_scaled_jacobian, hex_volume


class Hexmesh(AbstractMesh):
    
    """
    This class represent a volumetric mesh composed of hexahedra. It is possible to load the mesh from a file (.mesh) or
    from raw geometry and topology data.

    Parameters:

        filename (string): The name of the file to load 
        vertices (Array (Nx3) type=float): The list of vertices of the mesh
        hexes (Array (Nx8) type=int): The list of hexahedra of the mesh
        labels (Array (Nx1) type=int): The list of labels of the mesh (Optional)

    
    """
    
    def __init__(self, filename= None, vertices = None, hexes = None, labels = None):
        
        self.hexes            = None #npArray (Nx8) 
        self.labels           = None #npArray (Nx1) 
        self.hex2hex          = None #npArray (Nx4?) 
        self.face2hex         = None #npArray (Nx2?)
        self.hex2face         = None #npArray (Nx6)
        self.vtx2hex          = None #npArray (NxM)
        self.__internal_hexes = None
        
        super(Hexmesh, self).__init__()
        
        if filename is not None:
            
            self.__load_from_file(filename)
        
        elif vertices is not None and hexes is not None:
            
            self.vertices = np.array(vertices) 
            self.hexes = np.array(hexes)
            
            if labels:
                self.labels = np.array(labels)
            
            self.__load_operations()
        
        else:
            print('Warning: Empty Hexmesh object')
         
    
    # ==================== METHODS ==================== #
    
    
    @property
    def num_faces(self):
        
        return self.faces.shape[0]

    @property
    def num_hexes(self):
        
        return self.hexes.shape[0]

    def add_hex(self, hex_id0, hex_id1, hex_id2, hex_id3, hex_id4, hex_id5, hex_id6, hex_id7):
        """
        Add a new hexahedron to the current mesh. It affects the mesh topology. 

        Parameters:

            hex_id0 (int): The index of the first vertex composing the new hexahedron
            hex_id1 (int): The index of the second vertex composing the new hexahedron
            hex_id2 (int): The index of the third vertex composing the new hexahedron
            hex_id3 (int): The index of the fourth vertex composing the new hexahedron
            hex_id4 (int): The index of the fifth vertex composing the new hexahedron
            hex_id5 (int): The index of the sixth vertex composing the new hexahedron
            hex_id6 (int): The index of the seventh vertex composing the new hexahedron
            hex_id7 (int): The index of the eighth vertex composing the new hexahedron
            
        """
        
        self.add_hexes([hex_id0, hex_id1, hex_id2, hex_id3, hex_id4, hex_id5, hex_id6, hex_id7])
        
        
    def add_hexes(self, new_hexes):
        """
        Add a list of new hexahedra to the current mesh. It affects the mesh topology. 

        Parameters:

            new_hexes (Array (Nx8) type=int): List of hexahedra to add. Each hexahedron is in the form [int,int,int,int,int,int,int,int]
    
        """
            
        new_hexes = np.array(new_hexes)
        new_hexes.shape = (-1,8)
                
        if new_hexes[(new_hexes[:,0] > self.num_vertices) | 
                     (new_hexes[:,1] > self.num_vertices) | 
                     (new_hexes[:,2] > self.num_vertices) | 
                     (new_hexes[:,3] > self.num_vertices) |
                     (new_hexes[:,4] > self.num_vertices) |
                     (new_hexes[:,5] > self.num_vertices) |
                     (new_hexes[:,6] > self.num_vertices) |
                     (new_hexes[:,7] > self.num_vertices)].shape[0] > self.num_vertices:
            raise Exception('The ID of a vertex must be less than the number of vertices')

        self.hexes = np.concatenate([self.hexes, new_hexes])
        self.__load_operations()
        
    
    def remove_hex(self, hex_id):
        """
        Remove a hexahedron from the current mesh. It affects the mesh topology. 

        Parameters:

            hex_id (int): The index of the hexahedron to remove 
    
        """
        
        self.remove_hexes([hex_id])
        
        
    def remove_hexes(self, hex_ids):
        """
        Remove a list of hexahedra from the current mesh. It affects the mesh topology. 

        Parameters:

            hex_ids (Array (Nx1 / 1xN) type=int): List of hexahedra to remove. Each hexahedron is in the form [int]
    
        """
        
        hex_ids = np.array(hex_ids)
        mask = np.ones(self.num_hexes)
        mask[hex_ids] = 0
        mask = mask.astype(np.bool)
        
        self.hexes = self.hexes[mask]
        self.__load_operations()
        
    
    def remove_vertex(self, vtx_id):
        """
        Remove a vertex from the current mesh. It affects the mesh geometry. 

        Parameters:

            vtx_id (int): The index of the vertex to remove 
    
        """
        
        self.remove_vertices([vtx_id])
    
    
    def remove_vertices(self, vtx_ids):
        """
        Remove a list of vertices from the current mesh. It affects the mesh geoemtry. 

        Parameters:

            vtx_ids (Array (Nx1 / 1xN) type=int): List of vertices to remove. Each vertex is in the form [int]
    
        """ 
        
        vtx_ids = np.array(vtx_ids)
        
        for v_id in vtx_ids:
                        
            self.vertices = np.delete(self.vertices, v_id, 0)
            self.hexes = self.hexes[(self.hexes[:,0] != v_id) & 
                                    (self.hexes[:,1] != v_id) & 
                                    (self.hexes[:,2] != v_id) & 
                                    (self.hexes[:,3] != v_id) &
                                    (self.hexes[:,4] != v_id) &
                                    (self.hexes[:,5] != v_id) &
                                    (self.hexes[:,6] != v_id) &
                                    (self.hexes[:,7] != v_id)]
            
            self.hexes[(self.hexes[:,0] > v_id)] -= np.array([1, 0, 0, 0, 0, 0, 0, 0])
            self.hexes[(self.hexes[:,1] > v_id)] -= np.array([0, 1, 0, 0, 0, 0, 0, 0])
            self.hexes[(self.hexes[:,2] > v_id)] -= np.array([0, 0, 1, 0, 0, 0, 0, 0])
            self.hexes[(self.hexes[:,3] > v_id)] -= np.array([0, 0, 0, 1, 0, 0, 0, 0])
            self.hexes[(self.hexes[:,4] > v_id)] -= np.array([0, 0, 0, 0, 1, 0, 0, 0])
            self.hexes[(self.hexes[:,5] > v_id)] -= np.array([0, 0, 0, 0, 0, 1, 0, 0])
            self.hexes[(self.hexes[:,6] > v_id)] -= np.array([0, 0, 0, 0, 0, 0, 1, 0])
            self.hexes[(self.hexes[:,7] > v_id)] -= np.array([0, 0, 0, 0, 0, 0, 0, 1])
            
            vtx_ids[vtx_ids > v_id] -= 1;
            
        self.__load_operations()
        
        
    def __load_operations(self):
        
        self.__compute_faces()
        self.__compute_adjacencies()
        self._AbstractMesh__update_bounding_box()
        self.set_cut(self.bbox[0,0], self.bbox[1,0], 
                     self.bbox[0,1], self.bbox[1,1], 
                     self.bbox[0,2], self.bbox[1,2])
        self.__compute_metrics()
    
    def __compute_faces(self):
        self.faces = np.c_[self.hexes[:,0], self.hexes[:,3], self.hexes[:, 2], self.hexes[:, 1], 
                           self.hexes[:,1], self.hexes[:,2], self.hexes[:, 6], self.hexes[:, 5], 
                           self.hexes[:,4], self.hexes[:,5], self.hexes[:, 6], self.hexes[:, 7], 
                           self.hexes[:,3], self.hexes[:,0], self.hexes[:, 4], self.hexes[:, 7], 
                           self.hexes[:,0], self.hexes[:,1], self.hexes[:, 5], self.hexes[:, 4], 
                           self.hexes[:,2], self.hexes[:,3], self.hexes[:, 7], self.hexes[:, 6]].reshape(-1,4)
    
    def __compute_adjacencies(self):
        
        map_ = dict()
        adjs = np.zeros((self.num_hexes, 6), dtype=np.int)-1
        #adjs = [[] for i in range(self.num_hexes)]
        vtx2hex = [[] for i in range(self.num_vertices)]
        hexes_idx = np.repeat(np.array(range(self.num_hexes)), 6)
        self.hex2face = np.array(range(self.num_faces)).reshape(-1,6)
        
        for f, t in zip(self.faces, hexes_idx):
            
            vtx2hex[f[0]].append(t)
            vtx2hex[f[1]].append(t)
            vtx2hex[f[2]].append(t)
            vtx2hex[f[3]].append(t)
            
            f =  (f[0], f[1], f[2], f[3])
            f1 = (f[3], f[2], f[1], f[0])
            f2 = (f[2], f[1], f[0], f[3])
            f3 = (f[1], f[0], f[3], f[2])
            f4 = (f[0], f[3], f[2], f[1])

            try:
                tmp = map_[f]
            except KeyError:
                tmp = None

            if tmp is None:
                map_[f1] = t
                map_[f2] = t
                map_[f3] = t
                map_[f4] = t
            else:
                
                idx_to_append1 = np.where(adjs[t] == -1)[0][0]
                idx_to_append2 = np.where(adjs[map_[f]] == -1)[0][0]
                adjs[t][idx_to_append1] = map_[f]
                adjs[map_[f]][idx_to_append2] = t
        
        self.hex2hex = adjs#np.array([np.array(a) for a in adjs])
        self.vtx2hex = np.array([np.unique(np.array(a)) for a in vtx2hex])
        
        
    def __load_from_file(self, filename):
        
        ext = filename.split('.')[-1]
        
        if ext == 'mesh':
            self.vertices, self.hexes, self.labels = IO.read_mesh(filename)
        else:
            raise Exception("File Extension unknown")
    
    
        self.__load_operations()
        
        return self
        
    
    def save_file(self, filename):
        """
        Save the current mesh in a file. Currently it supports the .mesh extension. 

        Parameters:

            filename (string): The name of the file
    
        """
        
        ext = filename.split('.')[-1]
        
        if ext == 'mesh':
            IO.save_mesh(self, filename)
        
    
    def __compute_metrics(self): 
        self.simplex_metrics['scaled_jacobian'] = hex_scaled_jacobian(self.vertices, self.hexes)
        self.simplex_metrics['volume'] = hex_volume(self.vertices, self.hexes)
        
    @property
    def internals(self):
        
        if self.__internal_hexes is None:
            self.__internal_hexes = np.all(self.hex2hex != -1, axis = 1)
        
        return self.__internal_hexes
        
    
    def boundary(self, flip_x = False, flip_y = False, flip_z = False):
        """
        Compute the boundary of the current mesh. It only returns the faces that respect
        the cut and the flip conditions.

        Parameters:

            flip_x (bool): Flip the cut condition for the x axis
            flip_y (bool): Flip the cut condition for the y axis
            flip_z (bool): Flip the cut condition for the z axis
    
        """
        if (self._AbstractMesh__boundary_needs_update):
            clipping_range = super(Hexmesh, self).boundary()
            indices = np.where(self.internals)[0]
            clipping_range[indices[np.all(clipping_range[self.hex2hex[indices]], axis=1)]] = False
            clipping_range = np.repeat(clipping_range, 6)
            self._AbstractMesh__boundary_cached = clipping_range
            self._AbstractMesh__boundary_needs_update = False
        
        return self.faces[self._AbstractMesh__boundary_cached], self._AbstractMesh__boundary_cached
        

        
    @property
    def simplex_centroids(self):
        
        if self._AbstractMesh__simplex_centroids is None:
            self._AbstractMesh__simplex_centroids = self.vertices[self.hexes].mean(axis = 1)
        
        return self._AbstractMesh__simplex_centroids
