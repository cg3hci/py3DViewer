from .Abstractmesh import AbstractMesh
import numpy as np
from ..utils import IO, ObservableArray
from ..utils.load_operations import compute_hex_mesh_adjs as compute_adjacencies, _compute_three_vertex_normals as compute_three_normals
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
        self.__hex2hex          = None #npArray (Nx4?) 
        self.face2hex         = None #npArray (Nx2?)
        self.hex2face         = None #npArray (Nx6)
        self.__vtx2hex          = None #npArray (NxM)
        self.__internal_hexes = None
        
        super(Hexmesh, self).__init__()
        
        if filename is not None:
            
            self.__load_from_file(filename)
        
        elif vertices is not None and hexes is not None:
            
            self.vertices = ObservableArray(vertices.shape)
            self.vertices[:] = vertices
            self.vertices.attach(self)
            self.hexes = ObservableArray(hexes.shape, dtype=np.int)
            self.hexes[:] = hexes
            self.hexes.attach(self)
            self.__load_operations()
            
            if labels:
                self.labels = np.array(labels)
                self.labels[:] = labels
                self.labels.attach(self)
            
            self.__load_operations()
        
    
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
        self._dont_update = True
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
       
        self._dont_update = True
        hex_ids = np.array(hex_ids)
        mask = np.ones(self.num_hexes)
        mask[hex_ids] = 0
        mask = mask.astype(np.bool)
        
        self.hexes = self.hexes[mask]
        if self.labels is not None:
            self.labels = self.labels[mask]
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
        
        self._dont_update = True
        vtx_ids = np.array(vtx_ids)
        
        for v_id in vtx_ids:
                        
            self.vertices = np.delete(self.vertices, v_id, 0)
            
            condition = ((self.hexes[:,0] != v_id) & 
                                    (self.hexes[:,1] != v_id) & 
                                    (self.hexes[:,2] != v_id) & 
                                    (self.hexes[:,3] != v_id) &
                                    (self.hexes[:,4] != v_id) &
                                    (self.hexes[:,5] != v_id) &
                                    (self.hexes[:,6] != v_id) &
                                    (self.hexes[:,7] != v_id))
            
            if self.labels is not None:
                self.labels = self.labels[condition]
            
            self.hexes = self.hexes[condition]
            
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
        self._dont_update = True
        self._AbstractMesh__boundary_needs_update = True
        self._AbstractMesh__simplex_centroids = None
        self.__internal_hexes = None
        
        self.__compute_faces()
        self.__hex2hex, self._AbstractMesh__vtx2vtx, self.__vtx2hex = compute_adjacencies(self.faces, self.num_vertices)
        self._AbstractMesh__update_bounding_box()
        self.__compute_metrics()
        self.reset_clipping()
        self._dont_update = False
        self.update()

    
    def __compute_faces(self):
        self.faces = np.c_[self.hexes[:,0], self.hexes[:,3], self.hexes[:, 2], self.hexes[:, 1], 
                           self.hexes[:,1], self.hexes[:,2], self.hexes[:, 6], self.hexes[:, 5], 
                           self.hexes[:,4], self.hexes[:,5], self.hexes[:, 6], self.hexes[:, 7], 
                           self.hexes[:,3], self.hexes[:,0], self.hexes[:, 4], self.hexes[:, 7], 
                           self.hexes[:,0], self.hexes[:,1], self.hexes[:, 5], self.hexes[:, 4], 
                           self.hexes[:,2], self.hexes[:,3], self.hexes[:, 7], self.hexes[:, 6]].reshape(-1,4)
        tmp = ObservableArray(self.faces.shape, dtype=np.int)
        tmp[:] = self.faces
        self.faces = tmp
        self.faces.attach(self)
    
        
        
    def __load_from_file(self, filename):
        
        ext = filename.split('.')[-1]
        
        if ext == 'mesh':
            self.vertices, self.hexes, self.labels = IO.read_mesh(filename)
            self.vertices.attach(self)
            self.hexes.attach(self)
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
        self.simplex_metrics['scaled_jacobian'] = hex_scaled_jacobian(self.vertices, self.hexes)
        self.simplex_metrics['volume'] = hex_volume(self.vertices, self.hexes)
        
    @property
    def internals(self):
        
        if self.__internal_hexes is None:
            self.__internal_hexes = np.all(self.__hex2hex != -1, axis = 1)
        
        return self.__internal_hexes
        
    
    def boundary(self):
        """
        Compute the boundary of the current mesh. It only returns the faces that respect
        the cut and the flip conditions.
        """
        if (self._AbstractMesh__boundary_needs_update):
            clipping_range = super(Hexmesh, self).boundary()
            indices = np.where(self.internals)[0]
            clipping_range[indices[np.all(clipping_range[self.__hex2hex[indices]], axis=1)]] = False
            clipping_range = np.repeat(clipping_range, 6)
            self._AbstractMesh__boundary_cached = clipping_range
            self._AbstractMesh__boundary_needs_update = False
            
        return self.faces[self._AbstractMesh__boundary_cached], self._AbstractMesh__boundary_cached
    
    def as_edges_flat(self):
        boundaries = self.boundary()[0]
        edges = np.c_[boundaries[:,:2], boundaries[:,1:3], boundaries[:,2:4], boundaries[:,3], boundaries[:,0]].reshape(-1, 2)
        edges = np.sort(edges, axis=1)
        if edges.size > 0:
            edges = np.unique(edges, axis=0)
        return edges.flatten().astype(np.uint32)
    
    def as_edges_debug(self):
        boundaries = self.boundary()[0]
        edges = np.c_[boundaries[:,:2], boundaries[:,1:3], boundaries[:,2:4], boundaries[:,3], boundaries[:,0]]
        return edges
    
    def _as_threejs_triangle_soup(self):
        boundaries = self.boundary()[0]
        boundaries = np.c_[boundaries[:,:3], boundaries[:,2:], boundaries[:,0]]
        boundaries.shape = (-1, 3)
        tris = self.vertices[boundaries.flatten()]
        vtx_normals = compute_three_normals(tris)
        return tris.astype(np.float32), vtx_normals.astype(np.float32)
    
    def as_triangles(self):
        boundaries = self.boundary()[0]
        boundaries = np.c_[boundaries[:,:3], boundaries[:,2:], boundaries[:,0]]
        boundaries.shape = (-1, 3)
        return boundaries.astype("uint32").flatten()
    
    def internal_triangles_idx(self):
        internal_triangles = np.repeat(self.internals, 12*3, axis=0)
        return internal_triangles
    
    def _as_threejs_colors(self, colors=None):
        
        if colors is not None:
            return np.repeat(colors, 6*2*3, axis=0)
        
        return np.repeat(self.boundary()[1], 6)
    
    @property
    def num_triangles(self):
        return self.num_faces*2
    
    @property
    def simplex_centroids(self):
        
        if self._AbstractMesh__simplex_centroids is None:
            self._AbstractMesh__simplex_centroids = np.asarray(self.vertices[self.hexes].mean(axis = 1))
        
        return self._AbstractMesh__simplex_centroids
    
    
    @property
    def hex2hex(self):
        
        return self.__hex2hex
    
    @property
    def vtx2hex(self):
        
        return self.__vtx2hex
