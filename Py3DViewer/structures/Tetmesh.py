from .Abstractmesh import AbstractMesh
from ..visualization.Viewer import Viewer
import numpy as np
from ..utils import IO
from ..utils.metrics import tet_scaled_jacobian, tet_volume

class Tetmesh(AbstractMesh):
    
    """
    This class represent a volumetric mesh composed of tetrahedra. It is possible to load the mesh from a file (.mesh) or
    from raw geometry and topology data.

    Parameters:

        filename (string): The name of the file to load 
        vertices (Array (Nx3) type=float): The list of vertices of the mesh
        tets (Array (Nx4) type=int): The list of tetrahedra of the mesh
        labels (Array (Nx1) type=int): The list of labels of the mesh (Optional)

    
    """
    
    def __init__(self, filename = None, vertices = None, tets = None, labels = None):
        
        self.tets             = None #npArray (Nx4) 
        self.labels           = None #npArray (Nx1) 
        self.tet2tet          = None #npArray (Nx4?) 
        self.face2tet         = None #npArray (Nx2?)
        self.tet2face         = None #npArray (Nx4)
        self.vtx2tet          = None #npArray (NxM)
        self.__internal_tets = None
        
        super(Tetmesh, self).__init__()
        
        if filename is not None:
            
            self.__load_from_file(filename)
        
        elif vertices and tets:
            
            self.vertices = np.array(vertices) 
            self.tets = np.array(tets)
            
            if labels:
                self.labels = np.array(labels)
            
            self.__load_operations()
        
        else:
            
            print('Warning: Empty Tetmesh object')
         
    
    # ==================== METHODS ==================== #
    
    @property
    def num_faces(self):
        
        return self.faces.shape[0]

    @property
    def num_tets(self):
        
        return self.tets.shape[0]

    def add_tet(self, tet_id0, tet_id1, tet_id2, tet_id3):
        """
        Add a new tetrahedron to the current mesh. It affects the mesh topology. 

        Parameters:

            tet_id0 (int): The index of the first vertex composing the new tetrahedron
            tet_id1 (int): The index of the second vertex composing the new tetrahedron
            tet_id2 (int): The index of the third vertex composing the new tetrahedron
            tet_id3 (int): The index of the fourth vertex composing the new tetrahedron
            
        """
        
        self.add_tets([tet_id0, tet_id1, tet_id2, tet_id3])
        
        
    def add_tets(self, new_tets):
        """
        Add a list of new tetrahedra to the current mesh. It affects the mesh topology. 

        Parameters:

            new_tets (Array (Nx4) type=int): List of tetrahedra to add. Each tetrahedron is in the form [int,int,int,int]
    
        """
            
        new_tets = np.array(new_tets)
        new_tets.shape = (-1,4)
                
        if new_tets[(new_tets[:,0] > self.num_vertices) | 
                     (new_tets[:,1] > self.num_vertices) | 
                     (new_tets[:,2] > self.num_vertices) | 
                     (new_tets[:,3] > self.num_vertices)].shape[0] > self.num_vertices:
            raise Exception('The ID of a vertex must be less than the number of vertices')

        self.tets = np.concatenate([self.tets, new_tets])
        self.__load_operations()
        
    
    def remove_tet(self, tet_id):
        """
        Remove a tetrahedron from the current mesh. It affects the mesh topology. 

        Parameters:

            tet_id (int): The index of the tetrahedron to remove 
    
        """
        
        self.remove_tets([tet_id])
        
        
    def remove_tets(self, tet_ids):
        """
        Remove a list of tetrahedra from the current mesh. It affects the mesh topology. 

        Parameters:

            tet_ids (Array (Nx1 / 1xN) type=int): List of tethrahedra to remove. Each tetrahedron is in the form [int]
    
        """
        
        tet_ids = np.array(tet_ids)
        mask = np.ones(self.num_tets)
        mask[tet_ids] = 0
        mask = mask.astype(np.bool)
        
        self.tets = self.tets[mask]
        self.__load_operations()
        
    
    def remove_vertex(self,vtx_id):
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
        vtx2vtx = [[] for i in range(self.num_vertices)]
        tets_idx = np.repeat(np.array(range(self.num_tets)), 4)
        self.tet2face = np.array(range(self.num_faces)).reshape(-1,4)
        
        for f, t in zip(self.faces, tets_idx):

            vtx2vtx[f[0]].append(f[1])
            vtx2vtx[f[0]].append(f[2])



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
        self._AbstractMesh__vtx2vtx = np.array([np.array(a) for a in vtx2vtx])
        
        
    def __load_from_file(self, filename):
        
        ext = filename.split('.')[-1]
        
        
        if ext == 'mesh':
            self.vertices, self.tets, self.labels = IO.read_mesh(filename)
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
        self.simplex_metrics['scaled_jacobian'] = tet_scaled_jacobian(self.vertices, self.tets)
        self.simplex_metrics['volume'] = tet_volume(self.vertices, self.tets)
        
    @property
    def internals(self):
        
        if self.__internal_tets is None:
            self.__internal_tets = np.all(self.tet2tet != -1, axis = 1)
        
        return self.__internal_tets
        
    
    def boundary(self, flip_x = False, flip_y = False, flip_z = False):
        """
        Compute the boundary of the current mesh. It only returns the faces that respect
        the clipping.

        """
        if (self._AbstractMesh__boundary_needs_update):
            clipping_range = super(Tetmesh, self).boundary()
            indices = np.where(self.internals)[0]
            clipping_range[indices[np.all(clipping_range[self.tet2tet[indices]], axis=1)]] = False
            clipping_range = np.repeat(clipping_range, 4)
            self._AbstractMesh__boundary_cached = clipping_range
            self._AbstractMesh__boundary_needs_update = False
        
        return self.faces[self._AbstractMesh__boundary_cached], self._AbstractMesh__boundary_cached

        
    @property
    def simplex_centroids(self):
        
        if self._AbstractMesh__simplex_centroids is None:
            self._AbstractMesh__simplex_centroids = self.vertices[self.tets].mean(axis = 1)
        
        return self._AbstractMesh__simplex_centroids

