from .Abstractmesh import AbstractMesh
import numpy as np
from ..utils import IO, ObservableArray, deprecated
from ..utils.load_operations import get_connectivity_info_surface as get_connectivity_info 
from ..utils.load_operations import compute_vertex_normals, compute_face_normals
from ..utils.load_operations import _compute_three_vertex_normals as compute_three_normals
from ..utils.metrics import quad_area, quad_aspect_ratio


class Quadmesh(AbstractMesh):
    """
    This class represent a mesh composed of quadrilaterals. It is possible to load the mesh from a file (.obj) or
    from raw geometry and topology data.

    Parameters:

        filename (string): The name of the file to load 
        vertices (Array (Nx3) type=float): The list of vertices of the mesh
        polys (Array (Nx4) type=int): The list of faces of the mesh
        labels (Array (Nx1) type=int): The list of labels of the mesh (Optional)


    """

    def __init__(self, filename=None, vertices=None, polys=None, labels=None, texture=None, mtl=None, smoothness=False):
        super(Quadmesh, self).__init__()

        self.vtx_normals  = None # npArray (Nx3)
        self.poly_normals = None  # npArray (Nx3)
        self.texture = texture
        self.material = {}
        self.groups = {}
        self.smoothness = smoothness

        if mtl is not None:
            self.__load_from_file(mtl)

        if filename is not None:
            self.__load_from_file(filename)
            self._AbstractMesh__filename = filename.split('/')[-1]

        elif vertices is not None and polys is not None:

            vertices = np.array(vertices)
            polys = np.array(polys)
            self.vertices = ObservableArray(vertices.shape)
            self.vertices[:] = vertices
            self.vertices.attach(self)
            self._AbstractMesh__polys = ObservableArray(polys.shape, dtype=np.int64)
            self._AbstractMesh__polys[:] = polys
            self._AbstractMesh__polys.attach(self)
            self.__load_operations()

            if labels is not None:
                labels = np.array(labels)
                assert(labels.shape[0] == polys.shape[0])
                self.labels = ObservableArray(labels.shape, dtype=np.int)
                self.labels[:] = labels
                self.labels.attach(self)
            else:
                self.labels = ObservableArray(polys.shape[0], dtype=np.int)
                self.labels[:] = np.zeros(self.labels.shape, dtype=np.int)
                self.labels.attach(self)
        
        self._AbstractMesh__poly_size = 4
        self._AbstractMesh__finished_loading = True

    # ==================== METHODS ==================== #     


    def __load_operations(self):

        self._dont_update = True
        self._AbstractMesh__boundary_needs_update = True
        self._AbstractMesh__simplex_centroids = None

        self._AbstractMesh__edges, \
        self._AbstractMesh__adj_vtx2vtx, \
        self._AbstractMesh__adj_vtx2edge, \
        self._AbstractMesh__adj_vtx2poly, \
        self._AbstractMesh__adj_edge2vtx, \
        self._AbstractMesh__adj_edge2edge, \
        self._AbstractMesh__adj_edge2poly, \
        self._AbstractMesh__adj_poly2vtx, \
        self._AbstractMesh__adj_poly2edge, \
        self._AbstractMesh__adj_poly2poly = get_connectivity_info(self.num_vertices, self.polys)

        self._AbstractMesh__update_bounding_box()
        self.reset_clipping()
        self.poly_normals = compute_face_normals(self.vertices, self.polys, quad=True)
        self.vtx_normals = compute_vertex_normals(self.poly_normals, self.adj_vtx2poly._NList__list)
        self.__compute_metrics()
        self._AbstractMesh__simplex_centroids = None

        self._dont_update = False
        self.update()

    def __load_from_file(self, filename):

        ext = filename.split('.')[-1]

        if ext == 'obj':
            self.vertices, self._AbstractMesh__polys, self.poly_normals, self.uvcoords, self.coor, self.groups = IO.read_obj(filename)
            # self.vertices, self.faces, self.face_normals = IO.read_obj(filename)
            self.vertices.attach(self)
            self._AbstractMesh__polys.attach(self)
            self.poly_normals.attach(self)
            self.uvcoords.attach(self)
            self.coor.attach(self)
        elif ext == 'mtl':
            self.material = IO.read_mtl(filename)
            return

        elif ext == 'off':
            self.vertices, self._AbstractMesh__polys = IO.read_off(filename)
            self.vertices.attach(self)
            self._AbstractMesh__polys.attach(self)

        else:
            raise Exception("Only .obj and .off files are supported")

        self.labels = ObservableArray(self.num_polys, dtype=np.int)
        self.labels[:] = np.zeros(self.labels.shape, dtype=np.int)
        self.labels.attach(self)

        self.__load_operations()

        return self

    def save_file(self, filename):

        """
        Save the current mesh in a file. Currently it supports the .obj extension. 

        Parameters:

            filename (string): The name of the file

        """

        ext = filename.split('.')[-1]

        if ext == 'obj':
            IO.save_obj(self, filename)
        elif ext == 'off':
            IO.save_off(self, filename)
        else:
            raise Exception("Only .obj and .off files are supported")

    def __compute_metrics(self):

        self.simplex_metrics['area'] = quad_area(self.vertices, self.polys)
        self.simplex_metrics['aspect_ratio'] = quad_aspect_ratio(self.vertices, self.polys)
    
    def update_metrics(self):
        self.__compute_metrics()

    def boundary(self):

        """
        Compute the boundary of the current mesh. It only returns the faces that are inside the clipping
        """
        if (self._AbstractMesh__boundary_needs_update):
            clipping_range = super(Quadmesh, self).boundary()
            self._AbstractMesh__visible_polys = clipping_range
            self._AbstractMesh__boundary_cached = clipping_range
            self._AbstractMesh__boundary_needs_update = False

        return self.polys[self._AbstractMesh__boundary_cached], self._AbstractMesh__boundary_cached

    def as_edges_flat(self):
        boundaries = self.boundary()[0]
        edges = np.c_[
            boundaries[:, :2], boundaries[:, 1:3], boundaries[:, 2:4], boundaries[:, 3], boundaries[:, 0]].flatten()
        # edges_flat = self.vertices[edges].tolist()
        return edges

    def _as_threejs_triangle_soup(self):
        boundaries = self.boundary()[0]
        boundaries = np.c_[boundaries[:, :3], boundaries[:, 2:], boundaries[:, 0]]
        boundaries.shape = (-1, 3)
        tris = self.vertices[boundaries.flatten()]
        return tris.astype(np.float32), compute_three_normals(tris).astype(np.float32)

    def as_triangles(self):
        boundaries = self.boundary()[0]
        boundaries = np.c_[boundaries[:, :3], boundaries[:, 2:], boundaries[:, 0]]
        boundaries.shape = (-1, 3)
        return boundaries.astype("uint32").flatten()

    def _as_threejs_colors(self, colors=None):
        if colors is not None:
            return np.repeat(colors, 6, axis=0)
        return np.repeat(self.boundary()[1], 6)

    @property
    def num_triangles(self):
        return self.num_polys * 2

    
    
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
            condition = ((self._AbstractMesh__polys[:, 0] != v_id) &
                         (self._AbstractMesh__polys[:, 1] != v_id) &
                         (self._AbstractMesh__polys[:, 2] != v_id) &
                         (self._AbstractMesh__polys[:, 3] != v_id))

            if self.labels is not None:
                self.labels = self.labels[condition]

            self._AbstractMesh__polys = self._AbstractMesh__polys[condition]

            self._AbstractMesh__polys[(self._AbstractMesh__polys[:, 0] > v_id)] -= np.array([1, 0, 0, 0])
            self._AbstractMesh__polys[(self._AbstractMesh__polys[:, 1] > v_id)] -= np.array([0, 1, 0, 0])
            self._AbstractMesh__polys[(self._AbstractMesh__polys[:, 2] > v_id)] -= np.array([0, 0, 1, 0])
            self._AbstractMesh__polys[(self._AbstractMesh__polys[:, 3] > v_id)] -= np.array([0, 0, 0, 1])

            vtx_ids[vtx_ids > v_id] -= 1

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

    @property
    def edge_is_manifold(self):
        val = self.edge_valence
        return np.logical_and(val > 0, val < 3)
    
    @property
    def poly_is_on_boundary(self):
        return np.logical_not(np.all(self.adj_poly2poly != -1, axis = 1))
    
    @property
    def edge_is_on_boundary(self):
        boundary_edges = self.adj_poly2edge[self.poly_is_on_boundary].reshape(-1)
        boundary_edges = [e for e in boundary_edges if len(self.adj_edge2poly[e]) == 1]
        bool_vec = np.zeros((self.num_edges), dtype=np.bool)
        bool_vec[boundary_edges] = True
        return bool_vec
    
    @property
    def vert_is_on_boundary(self):
        boundary_verts = self.edges[self.edge_is_on_boundary].reshape(-1)
        bool_vec = np.zeros((self.num_vertices), dtype=np.bool)
        bool_vec[boundary_verts] = True
        return bool_vec

    @property
    def area(self):
        return np.sum(self.simplex_metrics['area'][1])
    
    def normalize_area(self):
        scale_factor = 1.0/np.sqrt(self.area)
        self.transform_scale([scale_factor, scale_factor, scale_factor])
        self.simplex_metrics['area'] = quad_area(self.vertices, self.polys)


    #deprecated
    @property
    @deprecated("Use the method adj_poly2poly instead")
    def face2face(self):
        return self._AbstractMesh__adj_poly2poly
