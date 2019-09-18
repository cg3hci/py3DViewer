from .Abstractmesh import AbstractMesh
from ..visualization.Viewer import Viewer
import numpy as np
from ..utils import IO
from ..utils.metrics import hex_scaled_jacobian, hex_volume


class Hexmesh(AbstractMesh):
    
    def __init__(self, vertices = None, tets = None, labels = None):
        
        self.hexes            = None #npArray (Nx4) 
        self.labels           = None #npArray (Nx1) 
        self.hex2hex          = None #npArray (Nx4?) 
        self.face2hex         = None #npArray (Nx2?)
        self.hex2face         = None #npArray (Nx6)
        self.vtx2hex          = None #npArray (NxM)
        self.__internal_hexes = None
        
        super(Hexmesh, self).__init__()
        
        if vertices and hexes:
            
            self.vertices = np.array(vertices) 
            self.tets = np.array(hexes)
            
            if labels:
                self.labels = np.array(labels)
            
            self.__load_operations()
         
    
    # ==================== METHODS ==================== #
    
    
    @property
    def num_faces(self):
        
        return self.faces.shape[0]

    @property
    def num_hexes(self):
        
        return self.hexes.shape[0]

    def add_hex(self, tet_id0, tet_id1, tet_id2, tet_id3, tet_id4, tet_id5, tet_id6, tet_id7):
        
        self.add_tets([tet_id0, tet_id1, tet_id2, tet_id3, tet_id4, tet_id5, tet_id6, tet_id7])
        
        
    def add_hexes(self, new_hexes):
            
        new_hexes = np.array(new_hexes)
        new_hexes.shape = (-1,3)
                
        if new_hexes[(new_hexes[:,0] > self.num_vertices) | 
                     (new_hexes[:,1] > self.num_vertices) | 
                     (new_hexes[:,2] > self.num_vertices) | 
                     (new_hexes[:,3] > self.num_vertices) |
                     (new_hexes[:,4] > self.num_vertices) |
                     (new_hexes[:,5] > self.num_vertices) |
                     (new_hexes[:,6] > self.num_vertices) |
                     (new_hexes[:,7] > self.num_vertices)].shape[0] > self.num_vertices:
            raise Exception('The ID of a vertex must be less than the number of vertices')

        self.tets = np.concatenate([self.tets, new_hexes])
        self.__load_operations()
        
    
    def remove_tet(self, hex_id):
        
        self.remove_tets([hex_id])
        
        
    def remove_tets(self, hex_ids):
        
        hex_ids = np.array(hex_ids)
        mask = np.ones(self.num_hexes)
        mask[hex_ids] = 0
        mask = mask.astype(np.bool)
        
        self.hexes = self.hexes[mask]
        self.__load_operations()
        
    
    def remove_vertex(self,vtx_id):
        
        self.remove_vertices([vtx_id])
    
    
    def remove_vertices(self, vtx_ids):
        
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
        
        
    def load_from_file(self, filename):
        
        ext = filename.split('.')[-1]
        
        if ext == 'mesh':
            self.vertices, self.hexes, self.labels = IO.read_mesh(filename)
        else:
            raise Exception("File Extension unknown")
    
    
        self.__load_operations()
        
        return self
        
    
    def save_file(self, filename):
        
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
        
        min_x = self.cut['min_x']
        max_x = self.cut['max_x']
        min_y = self.cut['min_y']
        max_y = self.cut['max_y']
        min_z = self.cut['min_z']
        max_z = self.cut['max_z']
            
        x_range = np.logical_xor(flip_x,((self.simplex_centroids[:,0] >= min_x) & (self.simplex_centroids[:,0] <= max_x)))
        y_range = np.logical_xor(flip_y,((self.simplex_centroids[:,1] >= min_y) & (self.simplex_centroids[:,1] <= max_y)))
        z_range = np.logical_xor(flip_z,((self.simplex_centroids[:,2] >= min_z) & (self.simplex_centroids[:,2] <= max_z)))
        cut_range = x_range & y_range & z_range
        
        indices = np.where(self.internals)[0]
        cut_range[indices[np.all(cut_range[self.hex2hex[indices]], axis=1)]] = False
        cut_range = np.repeat(cut_range, 6)
        return self.faces[cut_range], cut_range
        

        
    @property
    def simplex_centroids(self):
        
        if self._AbstractMesh__simplex_centroids is None:
            self._AbstractMesh__simplex_centroids = self.vertices[self.hexes].mean(axis = 1)
        
        return self._AbstractMesh__simplex_centroids
