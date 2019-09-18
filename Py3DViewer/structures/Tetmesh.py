from .Abstractmesh import AbstractMesh
from ..visualization.Viewer import Viewer
import numpy as np
from ..utils import IO
from ..utils.metrics import tet_scaled_jacobian, tet_volume

class Tetmesh(AbstractMesh):
    
    def __init__(self, vertices = None, tets = None, labels = None):
        
        self.tets             = None #npArray (Nx4) 
        self.labels       = None #npArray (Nx1) 
        self.tet2tet          = None #npArray (Nx4?) 
        self.face2tet         = None #npArray (Nx2?)
        self.tet2face         = None #npArray (Nx4)
        self.vtx2tet          = None #npArray (NxM)
        self.__internal_tets = None
        
        super(Tetmesh, self).__init__()
        
        if vertices and tets:
            
            self.vertices = np.array(vertices) 
            self.tets = np.array(tets)
            
            if labels:
                self.labels = np.array(labels)
            
            self.__load_operations()
         
    
    # ==================== METHODS ==================== #
    
    @property
    def num_faces(self):
        
        return self.faces.shape[0]

    @property
    def num_tets(self):
        
        return self.tets.shape[0]

    def add_tet(self, tet_id0, tet_id1, tet_id2, tet_id3):
        
        self.add_tets([tet_id0, tet_id1, tet_id2, tet_id3])
        
        
    def add_tets(self, new_tets):
            
        new_tets = np.array(new_tets)
        new_tets.shape = (-1,3)
                
        if new_tets[(new_tets[:,0] > self.num_vertices) | 
                     (new_tets[:,1] > self.num_vertices) | 
                     (new_tets[:,2] > self.num_vertices) | 
                     (new_tets[:,3] > self.num_vertices)].shape[0] > self.num_vertices:
            raise Exception('The ID of a vertex must be less than the number of vertices')

        self.tets = np.concatenate([self.tets, new_tets])
        self.__load_operations()
        
    
    def remove_tet(self, tet_id):
        
        self.remove_tets([tet_id])
        
        
    def remove_tets(self, tet_ids):
        
        tet_ids = np.array(tet_ids)
        mask = np.ones(self.num_tets)
        mask[tet_ids] = 0
        mask = mask.astype(np.bool)
        
        self.tets = self.tets[mask]
        self.__load_operations()
        
    
    def remove_vertex(self,vtx_id):
        
        self.remove_vertices([vtx_id])
    
    
    def remove_vertices(self, vtx_ids):
        
        vtx_ids = np.array(vtx_ids)
        
        for v_id in vtx_ids:
                        
            self.vertices = np.delete(self.vertices, v_id, 0)
            self.tets = self.tets[(self.tets[:,0] != v_id) & 
                                    (self.tets[:,1] != v_id) & 
                                    (self.tets[:,2] != v_id) & 
                                    (self.tets[:,3] != v_id)]
            
            self.tets[(self.tets[:,0] > v_id)] -= np.array([1, 0, 0, 0])
            self.tets[(self.tets[:,1] > v_id)] -= np.array([0, 1, 0, 0])
            self.tets[(self.tets[:,2] > v_id)] -= np.array([0, 0, 1, 0])
            self.tets[(self.tets[:,3] > v_id)] -= np.array([0, 0, 0, 1])
            
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
        self.faces = np.c_[self.tets[:,0], self.tets[:,2], self.tets[:, 1], 
                           self.tets[:,0], self.tets[:,1], self.tets[:,3],
                           self.tets[:,1], self.tets[:,2], self.tets[:,3],
                           self.tets[:,0], self.tets[:,3], self.tets[:,2]].reshape(-1,3)
    
    def __compute_adjacencies(self):
        
        map_ = dict()
        adjs = np.zeros((self.num_tets, 4), dtype=np.int)-1
        vtx2tet = [[] for i in range(self.num_vertices)]
        tets_idx = np.repeat(np.array(range(self.num_tets)), 4)
        self.tet2face = np.array(range(self.num_faces)).reshape(-1,4)
        
        for f, t in zip(self.faces, tets_idx):
            
            vtx2tet[f[0]].append(t)
            vtx2tet[f[1]].append(t)
            vtx2tet[f[2]].append(t)
            
            f =  (f[0], f[1], f[2])
            f1 = (f[2], f[1], f[0])
            f2 = (f[0], f[2], f[1])
            f3 = (f[1], f[0], f[2])

            try:
                tmp = map_[f]
            except KeyError:
                tmp = None

            if tmp is None:
                map_[f1] = t
                map_[f2] = t
                map_[f3] = t
            else:
                
                idx_to_append1 = np.where(adjs[t] == -1)[0][0]
                idx_to_append2 = np.where(adjs[map_[f]] == -1)[0][0]
                adjs[t][idx_to_append1] = map_[f]
                adjs[map_[f]][idx_to_append2] = t
        
        self.tet2tet = adjs#np.array([np.array(a) for a in adjs])
        self.vtx2tet = np.array([np.unique(np.array(a)) for a in vtx2tet])
        
        
    def load_from_file(self, filename):
        
        ext = filename.split('.')[-1]
        
        
        if ext == 'mesh':
            self.vertices, self.tets, self.labels = IO.read_mesh(filename)
        else:
            raise Exception("File Extension unknown")
        
    
        self.__load_operations()
        
        return self
    
    def save_file(self, filename):
        
        ext = filename.split('.')[-1]
        
        if ext == 'mesh':
            IO.save_mesh(self, filename)
        
    
    def __compute_metrics(self): 
        self.simplex_metrics['scaled_jacobian'] = tet_scaled_jacobian(self.vertices, self.tets)
        self.simplex_metrics['volume'] = tet_volume(self.vertices, self.tets)
        
    @property
    def internals(self):
        
        if self.__internal_tets is None:
            self.__internal_tets = np.all(self.tet2tet != -1, axis = 1)
        
        return self.__internal_tets
        
    
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
                
        
        #naive
        #for id_tet in np.nonzero(cut_range)[0]: 
        #    if self.tet2tet[id_tet].shape[0] == 4: # se ha quattro adiacenti vuol dire che è un tet interno
        #        if np.count_nonzero(cut_range[self.tet2tet[id_tet]]) == 4: # se tutti i tet adiacenti sono nel cut range vuol dire che è un tet interno
        #            cut_range[id_tet] = False
                            
        #tets = self.tets[ cut_range ]
        
        cut_range = x_range & y_range & z_range
        indices = np.where(self.internals)[0]
        cut_range[indices[np.all(cut_range[self.tet2tet[indices]], axis=1)]] = False
        cut_range = np.repeat(cut_range, 4)
        
        return self.faces[cut_range], cut_range

        
    @property
    def simplex_centroids(self):
        
        if self._AbstractMesh__simplex_centroids is None:
            self._AbstractMesh__simplex_centroids = self.vertices[self.tets].mean(axis = 1)
        
        return self._AbstractMesh__simplex_centroids

