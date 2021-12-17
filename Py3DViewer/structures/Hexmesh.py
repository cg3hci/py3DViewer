from .Abstractmesh import AbstractMesh
from .Quadmesh import Quadmesh
import numpy as np
from ..utils import IO, ObservableArray, deprecated, NList
from ..algorithms.cleaning import remove_isolated_vertices as rm_isolated
from ..utils.load_operations import get_connectivity_info_volume_hex, get_connectivity_info_volume_faces as get_connectivity_info_surf
from ..utils.load_operations import _compute_three_vertex_normals as compute_three_normals
from ..utils.metrics import hex_scaled_jacobian, hex_volume

class Hexmesh(AbstractMesh):
    
    """
    This class represents a volumetric mesh composed of hexahedra. It is possible to load the mesh from a file (.mesh) or
    from raw geometry and topology data.

    Parameters:

        filename (string): The name of the file to load 
        vertices (Array (Nx3) type=float): The list of vertices of the mesh
        polys (Array (Nx8) type=int): The list of hexahedra of the mesh
        labels (Array (Nx1) type=int): The list of labels of the mesh (Optional)

    
    """
    
    def __init__(self, filename= None, vertices = None, polys = None, labels = None):
        
        super(Hexmesh, self).__init__()
        
        self.__adj_vtx2face          = None #npArray (Nx4?) 
        self.__adj_edge2face         = None #npArray (NxM)
        self.__adj_poly2face         = None #npArray (Nx2?) NOT IMPLEMENTED YET
        self.__adj_face2vtx          = None #npArray (Nx6) NOT IMPLEMENTED YET
        self.__adj_face2edge         = None
        self.__adj_face2face         = None
        self.__adj_face2poly         = None

        self.__faces                 = None
        self.__threejs_faces         = None
        self.__face_centroids        = None

        self.__internal_hexes = None
        self.__map_poly_indices  = None
   
        
        if filename is not None:
            
            self.__load_from_file(filename)
            self._AbstractMesh__filename = filename.split('/')[-1]
        
        elif vertices is not None and polys is not None:
            
            vertices = np.array(vertices)
            polys = np.array(polys)
            self.vertices = ObservableArray(vertices.shape)
            self.vertices[:] = vertices
            self.vertices.attach(self)
            self._AbstractMesh__polys = ObservableArray(polys.shape, dtype=np.int)
            self._AbstractMesh__polys[:] = polys
            self._AbstractMesh__polys.attach(self)
            self.__load_operations()
            
            if labels is not None:
                labels = np.array(labels)
                assert(self.polys.shape[0] == labels.shape[0])
                self.labels = ObservableArray(labels.shape, dtype=np.int)
                self.labels[:] = labels
                self.labels.attach(self)
            else:
                self.labels = ObservableArray(polys.shape[0], dtype=np.int)
                self.labels[:] = np.zeros(self.labels.shape, dtype=np.int)
                self.labels.attach(self)
            
            self.__load_operations()
        
        self._AbstractMesh__poly_size = 8
        self._AbstractMesh__finished_loading = True
    
    # ==================== METHODS ==================== #
    
    
    @property
    def num_faces(self):
        return self.__faces.shape[0]
        
        
    def __load_operations(self):
        self._dont_update = True
        self._AbstractMesh__boundary_needs_update = True
        self._AbstractMesh__simplex_centroids = None
        self.__internal_hexes = None
        
        self.__faces, \
        self._AbstractMesh__edges, \
        self._AbstractMesh__adj_vtx2vtx, \
        self._AbstractMesh__adj_vtx2edge, \
        self.__adj_vtx2face, \
        self._AbstractMesh__adj_vtx2poly, \
        self._AbstractMesh__adj_edge2vtx, \
        self._AbstractMesh__adj_edge2edge, \
        self.__adj_edge2face, \
        self._AbstractMesh__adj_edge2poly, \
        self.__adj_face2vtx, \
        self.__adj_face2edge, \
        self.__adj_face2face, \
        self.__adj_face2poly, \
        self._AbstractMesh__adj_poly2vtx, \
        self._AbstractMesh__adj_poly2edge, \
        self.__adj_poly2face,\
        self._AbstractMesh__adj_poly2poly = get_connectivity_info_volume_hex(self.num_vertices, self.polys) 

        self._AbstractMesh__update_bounding_box()
        self.__compute_metrics()
        self._AbstractMesh__simplex_centroids = None
        self.__face_centroids = None
        self.reset_clipping()
        self._dont_update = False
        self.update()

    
        
        
    def __load_from_file(self, filename):
        
        ext = filename.split('.')[-1]
        
        if ext == 'mesh':
            self.vertices, self._AbstractMesh__polys, self.labels = IO.read_mesh(filename)
            self.vertices.attach(self)
            self._AbstractMesh__polys.attach(self)
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
        self.simplex_metrics['scaled_jacobian'] = hex_scaled_jacobian(self.vertices, self.polys)
        self.simplex_metrics['volume'] = hex_volume(self.vertices, self.polys)
    
    def update_metrics(self):
        self.__compute_metrics()
        
    @property
    def internals(self):
        
        if self.__internal_hexes is None:
            self.__internal_hexes = np.all(self.adj_poly2poly != -1, axis = 1)
        
        return self.__internal_hexes

    @property
    def _threejs_faces(self):
        if self.__threejs_faces is None:
            self.__threejs_faces = np.c_[self.polys[:,0], self.polys[:,3], self.polys[:, 2], self.polys[:, 1], 
                           self.polys[:,1], self.polys[:,2], self.polys[:, 6], self.polys[:, 5], 
                           self.polys[:,4], self.polys[:,5], self.polys[:, 6], self.polys[:, 7], 
                           self.polys[:,3], self.polys[:,0], self.polys[:, 4], self.polys[:, 7], 
                           self.polys[:,0], self.polys[:,1], self.polys[:, 5], self.polys[:, 4], 
                           self.polys[:,2], self.polys[:,3], self.polys[:, 7], self.polys[:, 6]].reshape(-1,4)

        return self.__threejs_faces


        
    
    def boundary(self):
        """
        Compute the boundary of the current mesh. It only returns the faces that respect
        the cut and the flip conditions.
        """
        if (self._AbstractMesh__boundary_needs_update):
            clipping_range = super(Hexmesh, self).boundary()
            indices = np.where(self.internals)[0]
            adjs = self._AbstractMesh__adj_poly2poly
            clipping_range[indices[np.all(clipping_range[adjs[indices]], axis=1)]] = False

            self.__map_poly_indices  = []
            counter = 0
            for c in clipping_range:
                if c:
                    self.__map_poly_indices.append(counter)
                else:
                    counter = counter + 1
            self._AbstractMesh__visible_polys = clipping_range
            clipping_range = np.repeat(clipping_range, 6)
            self._AbstractMesh__boundary_cached = clipping_range
            self._AbstractMesh__boundary_needs_update = False
            
            
        return self._threejs_faces[self._AbstractMesh__boundary_cached], self._AbstractMesh__boundary_cached
    
    def as_edges_flat(self):
        boundaries = self.boundary()[0]
        edges = np.c_[boundaries[:,:2], boundaries[:,1:3], boundaries[:,2:4], boundaries[:,3], boundaries[:,0]].reshape(-1, 2)
        edges = np.sort(edges, axis=1)
        if edges.size > 0:
            edges = np.unique(edges, axis=0)
        return edges.flatten().astype(np.uint32)
    
    
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
    
    def internal_triangles_idx(self):
        internal_triangles = np.repeat(self.internals, 12*3, axis=0)
        return internal_triangles
    
    def _as_threejs_colors(self, colors=None):
        
        if colors is not None:
            return np.repeat(colors, 6*2*3, axis=0)
        
        return np.repeat(self.boundary()[1], 6)


    @property
    def face_centroids(self):

        if self.__face_centroids is None:
            self.__face_centroids = np.asarray(self.vertices[self.faces].mean(axis=1))
        return self.__face_centroids
    
    @property
    def volume(self):
        return np.sum(self.simplex_metrics['volume'][1])

    def pick_face(self, point):
        point = np.repeat(np.asarray(point).reshape(-1,3), self.num_faces, axis=0)
        idx = np.argmin(np.linalg.norm(self.face_centroids - point, axis=1), axis=0)
        return idx
    
    
    @property
    def num_triangles(self):
        return self.num_polys*12

    

    
    def vertex_remove(self, vtx_id):
        """
        Remove a vertex from the current mesh. It affects the mesh geometry. 
        Parameters:
            vtx_id (int): The index of the vertex to remove 
    
        """
        
        self.vertices_remove([vtx_id])
    
    
    def vertices_remove(self, vtx_ids):
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
                                    (self._AbstractMesh__polys[:,1] != v_id) & 
                                    (self._AbstractMesh__polys[:,2] != v_id) & 
                                    (self._AbstractMesh__polys[:,3] != v_id) &
                                    (self._AbstractMesh__polys[:,4] != v_id) &
                                    (self._AbstractMesh__polys[:,5] != v_id) &
                                    (self._AbstractMesh__polys[:,6] != v_id) &
                                    (self._AbstractMesh__polys[:,7] != v_id))
            
            if self.labels is not None:
                self.labels = self.labels[condition]
            
            self._AbstractMesh__polys = self._AbstractMesh__polys[condition]
            
            self._AbstractMesh__polys[(self._AbstractMesh__polys[:,0] > v_id)] -= np.array([1, 0, 0, 0, 0, 0, 0, 0])
            self._AbstractMesh__polys[(self._AbstractMesh__polys[:,1] > v_id)] -= np.array([0, 1, 0, 0, 0, 0, 0, 0])
            self._AbstractMesh__polys[(self._AbstractMesh__polys[:,2] > v_id)] -= np.array([0, 0, 1, 0, 0, 0, 0, 0])
            self._AbstractMesh__polys[(self._AbstractMesh__polys[:,3] > v_id)] -= np.array([0, 0, 0, 1, 0, 0, 0, 0])
            self._AbstractMesh__polys[(self._AbstractMesh__polys[:,4] > v_id)] -= np.array([0, 0, 0, 0, 1, 0, 0, 0])
            self._AbstractMesh__polys[(self._AbstractMesh__polys[:,5] > v_id)] -= np.array([0, 0, 0, 0, 0, 1, 0, 0])
            self._AbstractMesh__polys[(self._AbstractMesh__polys[:,6] > v_id)] -= np.array([0, 0, 0, 0, 0, 0, 1, 0])
            self._AbstractMesh__polys[(self._AbstractMesh__polys[:,7] > v_id)] -= np.array([0, 0, 0, 0, 0, 0, 0, 1])
            
            vtx_ids[vtx_ids > v_id] -= 1;
            
        self.__load_operations()

    def poly_add(self, new_poly):
        """
        Add a new face to the current mesh. It affects the mesh topology. 

        Parameters:

            new_poly (Array (Nx1) type=int): Poly to add in the form [int, ..., int]

        """
        self.polys_add(new_poly)

    def polys_add(self, new_polys):

        """
        Add a list of new faces to the current mesh. It affects the mesh topology. 

        Parameters:

            new_polys (Array (NxM) type=int): List of faces to add. Each face is in the form [int, ..., int]
        """

        AbstractMesh.polys_add(self, new_polys)
        self.__load_operations()
       


    def poly_remove(self, poly_id):

        """
        Remove a poly from the current mesh. It affects the mesh topology. 

        Parameters:

            poly_id (int): The index of the face to remove 

        """

        self.polys_remove([poly_id])

    
    def polys_remove(self, poly_ids):

        """
        Remove a list of polys from the current mesh. It affects the mesh topology. 

        Parameters:

            poly_ids (Array (Nx1 / 1xN) type=int): List of polys to remove. Each face is in the form [int]

        """
        AbstractMesh.polys_remove(self, poly_ids)
        self.__load_operations()

    def face_id(self, verts):
    
        verts = np.sort(verts)
        faces = np.sort(self.faces, axis=1)
        result = (faces == verts).all(axis=1).nonzero()[0]
    
        return result.item() if result.size == 1 else result

    @property
    def edge_is_manifold(self):
    
        surface_edges = self.edge_is_on_surface.nonzero()[0]
        surface_faces = self.face_is_on_surface.nonzero()[0]
        face_edges = self.adj_face2edge[surface_faces]
        result = np.array(list(map(lambda x : np.count_nonzero(face_edges == x), surface_edges)))
        all_ = np.ones(self.num_edges, dtype=np.bool)
        all_[surface_edges] = np.logical_or(result == 0, result == 2)
        return all_

    @property
    def num_faces_per_poly(self):
        return 6

    @property
    def poly_is_on_surface(self):
        return np.logical_not(np.all(self.adj_poly2poly != -1, axis = 1))
    
    @property
    def face_is_on_surface(self):
        return np.logical_not(np.all(self.adj_face2poly != -1, axis = 1))
    
    @property
    def edge_is_on_surface(self):
        surf_edges = self.adj_face2edge[self.face_is_on_surface]
        bool_vec = np.zeros((self.num_edges), dtype=np.bool)
        bool_vec[surf_edges] = True
        return bool_vec
    
    @property
    def vert_is_on_surface(self):
        surf_verts = self.adj_face2vtx[self.face_is_on_surface]
        bool_vec = np.zeros((self.num_vertices), dtype=np.bool)
        bool_vec[surf_verts] = True
        return bool_vec

    @property
    def volume(self):
        return np.sum(self.simplex_metrics['volume'][1])

    def normalize_volume(self):
        scale_factor = 1.0/np.power(self.volume, 1.0/3.0)
        self.transform_scale([scale_factor, scale_factor, scale_factor])
        self.simplex_metrics['volume'] = hex_volume(self.vertices, self.polys)
        
    def extract_surface(self, keep_original_vertices=True):
        
        polys = self.faces[self.face_is_on_surface]
        surface = Quadmesh(vertices=self.vertices, polys=polys)
        if not keep_original_vertices:
            rm_isolated(surface)
        return surface

    @property
    def _map_poly_indices (self):
        return self.__map_poly_indices 
    
    @property
    def faces(self):
        return self.__faces
    
        #adjacencies
    @property
    def adj_vtx2face(self):
       return NList.NList(self.__adj_vtx2face)
    
    @property
    def adj_edge2face(self):
       return NList.NList(self.__adj_edge2face)
    
    @property
    def adj_poly2face(self):
       return self.__adj_poly2face
    
    @property
    def adj_face2vtx(self):
       return self.__adj_face2vtx
    
    @property
    def adj_face2edge(self):
       return self.__adj_face2edge
    
    @property
    def adj_face2face(self):
       return NList.NList(self.__adj_face2face)

    @property
    def adj_face2poly(self):
       return self.__adj_face2poly


    #deprecated

    @property
    @deprecated("Use the method adj_poly2poly instead")
    def hex2hex(self):
        return self.adj_poly2poly
    
    @property
    @deprecated("Use the method adj_vtx2poly instead")
    def vtx2hex(self):
        return self.adj_vtx2poly

    @property
    @deprecated("Use the method adj_face2face instead")
    def face2face(self):
        return self.adj_face2face