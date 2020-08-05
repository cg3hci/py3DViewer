from .Abstractmesh import AbstractMesh
from .Trimesh import Trimesh
from ..algorithms.cleaning import remove_isolated_vertices as rm_isolated
import numpy as np
from ..utils import IO, ObservableArray
from ..utils.load_operations import compute_tet_mesh_adjs as compute_adjacencies
from ..utils.load_operations import _compute_three_vertex_normals as compute_three_normals, compute_adj_f2f_volume as compute_f2f
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
        
        super(Tetmesh, self).__init__()
        self.tets             = None #npArray (Nx4) 
        self.labels           = None #npArray (Nx1) 
        self.__tet2tet          = None #npArray (Nx4?) 
        self.face2tet         = None #npArray (Nx2?)
        self.tet2face         = None #npArray (Nx4)
        self.__vtx2tet          = None #npArray (NxM)
        self.__internal_tets = None
        
        
        if filename is not None:
            
            self.__load_from_file(filename)
        
        elif vertices is not None and tets is not None:
            
            vertices = np.array(vertices)
            tets = np.array(tets)
            self.vertices = ObservableArray(vertices.shape)
            self.vertices[:] = vertices
            self.vertices.attach(self)
            self.tets = ObservableArray(tets.shape, dtype=np.int)
            self.tets[:] = tets
            self.tets.attach(self)
            self.__load_operations()
        
            if labels is not None:
                labels = np.array(labels)
                assert(labels.shape[0] == self.tets.shape[0])
                self.labels = ObservableArray(labels.shape, dtype=np.int)
                self.labels[:] = labels
                self.labels.attach(self)
            else:
                self.labels = ObservableArray(tets.shape[0], dtype=np.int)
                self.labels[:] = np.zeros(self.labels.shape, dtype=np.int)
                self.labels.attach(self)
         
        self._AbstractMesh__finished_loading = True
    
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
        self._dont_update = True
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
        self._dont_update = True
        tet_ids = np.array(tet_ids)
        mask = np.ones(self.num_tets)
        mask[tet_ids] = 0
        mask = mask.astype(np.bool)
        
        self.tets = self.tets[mask]
                    
        if self.labels is not None:
            self.labels = self.labels[mask]
        
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

            vtx_ids (Array (Nx1 / 1xN) type=int): List of vertices to remove. Each vertex is ifn the form [int]
    
        """ 
        self._dont_update = True
        vtx_ids = np.array(vtx_ids)
        
        for v_id in vtx_ids:
                        
            self.vertices = np.delete(self.vertices, v_id, 0)
            
            condition = ((self.tets[:,0] != v_id) & 
                        (self.tets[:,1] != v_id) & 
                        (self.tets[:,2] != v_id) & 
                        (self.tets[:,3] != v_id))
            
            if self.labels is not None:
                self.labels = self.labels[condition]
                
            self.tets = self.tets[condition]
            
            self.tets[(self.tets[:,0] > v_id)] -= np.array([1, 0, 0, 0])
            self.tets[(self.tets[:,1] > v_id)] -= np.array([0, 1, 0, 0])
            self.tets[(self.tets[:,2] > v_id)] -= np.array([0, 0, 1, 0])
            self.tets[(self.tets[:,3] > v_id)] -= np.array([0, 0, 0, 1])
            
            vtx_ids[vtx_ids > v_id] -= 1;
            
        self.__load_operations()
        
        
    def __load_operations(self):
        self._dont_update = True
        self._AbstractMesh__boundary_needs_update = True
        self._AbstractMesh__simplex_centroids = None
        self.__internal_tets = None

        self.__compute_faces()
        self.__tet2tet, self._AbstractMesh__vtx2vtx, self.__vtx2tet, self._AbstractMesh__vtx2face = compute_adjacencies(self.faces, self.num_vertices)
        self._AbstractMesh__update_bounding_box()
        self.reset_clipping()
        self.__compute_metrics()
        self._AbstractMesh__face2face = None
        self._dont_update = False
        self.update()

    
    def __compute_faces(self):
        self.faces = np.c_[self.tets[:,0], self.tets[:,2], self.tets[:, 1], 
                           self.tets[:,0], self.tets[:,1], self.tets[:,3],
                           self.tets[:,1], self.tets[:,2], self.tets[:,3],
                           self.tets[:,0], self.tets[:,3], self.tets[:,2]].reshape(-1,3)
        tmp = ObservableArray(self.faces.shape, dtype=np.int)
        tmp[:] = self.faces
        self.faces = tmp
        self.faces.attach(self)
    
    
        
        
    def __load_from_file(self, filename):
        
        ext = filename.split('.')[-1]
        
        
        if ext == 'mesh':
            self.vertices, self.tets, self.labels = IO.read_mesh(filename)
            self.vertices.attach(self)
            self.tets.attach(self)
            self.labels.attach(self)
            
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
            self.__internal_tets = np.all(self.__tet2tet != -1, axis = 1)
        
        return self.__internal_tets
        
    
    def boundary(self):
        """
        Compute the boundary of the current mesh. It only returns the faces that respect
        the clipping.

        """
        if (self._AbstractMesh__boundary_needs_update):
            clipping_range = super(Tetmesh, self).boundary()
            indices = np.where(self.internals)[0]
            clipping_range[indices[np.all(clipping_range[self.__tet2tet[indices]], axis=1)]] = False
            clipping_range = np.repeat(clipping_range, 4)
            self._AbstractMesh__boundary_cached = clipping_range
            self._AbstractMesh__boundary_needs_update = False
        
        return self.faces[self._AbstractMesh__boundary_cached], self._AbstractMesh__boundary_cached

    def as_edges_flat(self):
        
        boundaries = self.boundary()[0]
        edges = np.c_[boundaries[:,:2], boundaries[:,1:], boundaries[:,2], boundaries[:,0]].flatten()
        #edges_flat = self.vertices[edges].tolist()
        return edges
    
    def _as_threejs_triangle_soup(self):
        tris = self.vertices[self.boundary()[0].flatten()]
        return tris.astype(np.float32), compute_three_normals(tris).astype(np.float32)
    
    def as_triangles(self):
        return self.boundary()[0].flatten().astype("uint32")
    
    def internal_triangles_idx(self):
        internal_triangles = np.repeat(self.internals, 4*3, axis=0)
        return internal_triangles
    
    def _as_threejs_colors(self, colors=None):
        
        if colors is not None:
            return np.repeat(colors, 4*3, axis=0)
        
        return np.repeat(self.boundary()[1], 3)
    
    @property
    def num_triangles(self):
        return self.num_faces
    
    @property
    def simplex_centroids(self):
        
        if self._AbstractMesh__simplex_centroids is None:
            self._AbstractMesh__simplex_centroids = np.asarray(self.vertices[self.tets].mean(axis = 1))
        
        return self._AbstractMesh__simplex_centroids
    
    @property
    def tet2tet(self):
        
        return self.__tet2tet
    
    @property
    def vtx2tet(self):
        
        return self.__vtx2tet
    
    @property
    def face2face(self):
        if self._AbstractMesh__face2face is None: 
            self._AbstractMesh__face2face = compute_f2f(self.faces) 
        return self._AbstractMesh__face2face
    
    @property
    def edges(self):

        edges =  np.c_[self.faces[:,:2], self.faces[:,1:], self.faces[:,2], self.faces[:,0]]
        edges.shape = (-1,2)

        return edges

    @property
    def surface_faces(self):
        return np.where(self.face2face == -1)[0]
    
    def face_is_on_surface(self, face_ids):
        res = self.face2face[face_ids] == -1
        return res if res.size > 1 else res.item()
    
    def vert_is_on_surface(self, vert_id):
        verts = np.where((self.faces[:,0] == vert_id) | 
        (self.faces[:,1] == vert_id) |
        (self.faces[:,2] == vert_id))

        return np.intersect1d(verts, self.surface_faces).size > 0
    
    def extract_surface_mesh(self, remove_isolated_vertices=False):
        faces = self.faces[self.surface_faces]
        vertices = self.vertices
        result = Trimesh(vertices=vertices, faces=faces)
        if remove_isolated_vertices:
            rm_isolated(result)
        return result
