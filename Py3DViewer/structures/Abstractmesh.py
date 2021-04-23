import numpy as np
from ..visualization import Viewer
from ..utils import Subject, Observer, deprecated, matrices, NList
import copy
from numba import njit, int64, float64
from numba.types import ListType as LT

@njit(int64[:](LT(LT(int64))), cache=True)
def _valence(adj_x2y):
    valences = np.zeros(len(adj_x2y), dtype=np.int64)
    for idx, row in enumerate(adj_x2y):
        valences[idx] = len(row)
    return valences

class Clipping(object):

    class __Flip(object):
        def __init__(self):
            self.x = False
            self.y = False
            self.z = False

    def __init__(self):
        self.min_x = None
        self.max_x = None
        self.min_y = None
        self.max_y = None
        self.min_z = None
        self.max_z = None
        self.flip = self.__Flip()
        super(Clipping, self).__init__()

    def __repr__(self):
        return ("Clipping:\n" +
                f"min_x: {self.min_x} \tmax_x: {self.max_x} \t{('flipped' if self.flip.x else '')}\n" +
                f"min_y: {self.min_y} \tmax_y: {self.max_y} \t{('flipped' if self.flip.y else '')}\n" +
                f"min_z: {self.min_z} \tmax_z: {self.max_z} \t{('flipped' if self.flip.z else '')}\n")

class AbstractMesh(Observer, Subject):

    """
    This class represents a generic mesh. It must be estended by a specific mesh class. It stores all the information
    shared among the different kind of supported meshes.
    """

    def __init__(self):

        self.__boundary_needs_update = True
        self.__boundary_cached   = None
        self.__finished_loading  = False        
        self._dont_update        = False
        self.__poly_size         = None

        self.vertices              = None #npArray (Nx3)
        self.__edges               = None #npArray (Nx2)
        self.__polys               = None #npArray (NxM)
        self.labels                = None # npArray (Nx1)
        
        self.uvcoords            = None
        self.coor                = [] #Mappatura indici coordinate uv per faccia
        self.texture             = None
        self.material            = {}
        self.smoothness          = False

        self.__adj_vtx2vtx           = None
        self.__adj_vtx2edge          = None
        self.__adj_vtx2poly          = None #npArray (NxM)
        self.__adj_edge2vtx          = None
        self.__adj_edge2edge         = None
        self.__adj_edge2poly         = None
        self.__adj_poly2vtx          = None
        self.__adj_poly2edge         = None
        self.__adj_poly2poly         = None

        self.__bounding_box      = None #npArray (2x3)
        self.__simplex_centroids = None #npArray (Nx1)
        self.__clipping          = Clipping()
        self.__visible_polys     = None

        self.simplex_metrics     = dict() #dictionary[propertyName : ((min, max), npArray (Nx1))]

        self.__filename          = ''
        
       
        Observer.__init__(self)
        Subject.__init__(self)


    # ==================== METHODS ==================== #

    def __setattr__(self, key, value):
        self.__dict__[key] = value
        if key[0] != "_" and self.__finished_loading:
            self.update()

    def copy(self):

        new = type(self)()
        for key in self.__dict__.keys():
            if "observer" not in key and ("adj" not in key or "poly2poly" in key):
                setattr(new, key, copy.deepcopy(getattr(self, key)))
        return new

    def update(self):
        """
            Update the mesh manually when the Viewer is set as not reactive.
        """
        self.__boundary_needs_update = True
        self.__update_bounding_box()
        if (not self._dont_update):
            self._notify()

    def show(self, width = 700, height = 700, mesh_color = None, reactive =  False):

        """
        Show the mesh within the current cell. It is possible to manipulate the mesh through the UI.

        Parameters:

            UI (bool): Show or not show the graphic user interface of the viewer
            width (int): The width of the canvas
            height (int): The height of thne canvas

        Return:

            Viewer: The viewer object
        """

        view = Viewer(self, width = width, height = height, reactive=reactive)
        view.show()
        return view

    
    def set_clipping(self, min_x = None, max_x = None,
                      min_y = None, max_y = None,
                      min_z = None, max_z = None,
                      flip_x = None, flip_y = None, flip_z = None):
        """
        clipping the mesh along x, y and z axes. It doesn't affect the geometry of the mesh.

        Parameters:

            min_x (float): The minimum value of x
            max_x (float): The maximum value of x
            min_y (float): The minimum value of y
            max_y (float): The maximum value of y
            min_z (float): The minimum value of z
            max_z (float): The maximum value of z

        """
        if min_x is not None:
            self.__clipping.min_x = min_x
        if max_x is not None:
            self.__clipping.max_x = max_x
        if min_y is not None:
            self.__clipping.min_y = min_y
        if max_y is not None:
            self.__clipping.max_y = max_y
        if min_z is not None:
            self.__clipping.min_z = min_z
        if max_z is not None:
            self.__clipping.max_z = max_z
        if flip_x is not None:
            self.__clipping.flip.x = flip_x
        if flip_y is not None:
            self.__clipping.flip.y = flip_y
        if flip_z is not None:
            self.__clipping.flip.z = flip_z

        self.__boundary_needs_update = True
        self.update()

    def reset_clipping(self):

        """
        Set the clippings to the bounding box in order to show the whole mesh.
        """

        self.set_clipping(min_x = self.bbox[0,0], max_x = self.bbox[1,0],
                     min_y = self.bbox[0,1], max_y = self.bbox[1,1],
                     min_z = self.bbox[0,2], max_z = self.bbox[1,2])
        self.__boundary_needs_update = True
        self.update()

    def load_from_file(filename):

        raise NotImplementedError('This method must be implemented in the subclasses')

    def __compute_adjacencies(self):

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
    def clipping(self):
        """
            Return the clipping region of the current mesh.
        """
        return self.__clipping

    @property
    def visible_polys(self):
        return self.__visible_polys


    def __compute_metrics(self):

        raise NotImplementedError('This method must be implemented in the subclasses')

    def as_triangles_flat(self):

        raise NotImplementedError('This method must be implemented in the subclasses')

    def as_edges_flat(self):

        raise NotImplementedError('This method must be implemented in the subclasses')

    def _as_threejs_colors(self):

        raise NotImplementedError('This method must be implemented in the subclasses')


    def boundary(self):

        """
        Compute the boundary of the current mesh. It only returns the faces that are inside the clipping
        """
        min_x = self.clipping.min_x
        max_x = self.clipping.max_x
        min_y = self.clipping.min_y
        max_y = self.clipping.max_y
        min_z = self.clipping.min_z
        max_z = self.clipping.max_z
        flip_x = self.clipping.flip.x
        flip_y = self.clipping.flip.y
        flip_z = self.clipping.flip.z
        centroids = np.array(self.poly_centroids)
        x_range = np.logical_xor(flip_x,((centroids)[:,0] >= min_x) & (centroids[:,0] <= max_x))
        y_range = np.logical_xor(flip_y,((centroids[:,1] >= min_y) & (centroids[:,1] <= max_y)))
        z_range = np.logical_xor(flip_z,((centroids[:,2] >= min_z) & (centroids[:,2] <= max_z)))
        clipping_range = x_range & y_range & z_range
        return clipping_range


    def vertex_add(self, x, y, z):

        """
        Add a new vertex to the current mesh. It affects the mesh geometry.

        Parameters:

            x (float): The x coordinate of the new vertex
            y (float): The y coordinate of the new vertex
            z (float): The z coordinate of the new vertex

        """
        self._dont_update = True
        new_vertex = np.array([x,y,z], dtype=np.float)
        new_vertex.shape = (1,3)

        self.vertices = np.concatenate([self.vertices, new_vertex])
        self._dont_update = False
        self.update()


    def vertices_add(self, new_vertices):

        """
        Add a list of new vertices to the current mesh. It affects the mesh geometry.

        Parameters:

            new_vertices (Array (Nx3) type=float): List of vertices to add. Each vertex is in the form [float,float,float]

        """

        self._dont_update = True
        new_vertices = np.array(new_vertices)
        self.vertices = np.concatenate([self.vertices, new_vertices])
        self._dont_update = False
        self.update()



    def transform_translation(self, t):
        """
        Translate the mesh by a given value for each axis. It affects the mesh geometry.

        Parameters:

            t (Array (3x1) type=float): Translation value for each axis.

        """
        self._dont_update = True
        matrix = np.identity(4)
        t = np.resize(t, (1, 4))
        t[:, -1] = 1
        matrix[:, -1] = t

        #crea un array colonna con il vettore vertices e nell'ultima colonna un vettore di soli 1
        a = np.hstack((self.vertices, np.ones((self.vertices.shape[0], 1))))#(nx3)->(nx4)
        #moltiplica l'array appena creato con la matrice di trasformazione trasposta (per non trasporre tutte le righe di vertices)
        self.vertices = a.dot(matrix.T)[:,:-1]

        self._dont_update = False
        self.update()

    def  transform_scale(self, t):
        """
        Scale the mesh by a given value for each axis. It affects the mesh geometry.

        Parameters:

            t (Array (3x1) type=float): Scale value for each axis.

        """

        self._dont_update = True
        t = np.append(t, 1)
        matrix = np.diag(t)

        #crea un array colonna con il vettore vertices e nell'ultima colonna un vettore di soli 1
        a = np.hstack((self.vertices, np.ones((self.vertices.shape[0], 1))))#(nx3)->(nx4)
        #moltiplica l'array appena creato con la matrice di trasformazione trasposta (per non trasporre tutte le righe di vertices)
        self.vertices = a.dot(matrix.T)[:,:-1]

        self._dont_update = False
        self.update()


    

    def transform_rotation(self, angle, axis):
        """
        Rotate the mesh by a given angle for a given axis. It affects the mesh geometry.

        Parameters:

            angle (float): Rotation angle, degrees.
            axis  (string): Rotation axis. It can be 'x', 'y' or 'z' or an array of size 3

        """
        self._dont_update = True

        axis_tmp = np.zeros(3, dtype=np.float)
        if axis == 'x' or axis == 0:
            axis_tmp[0] = 1
            axis = axis_tmp
        elif axis == 'y' or axis == 1:
            axis_tmp[1] = 1
            axis = axis_tmp
        elif axis == 'z' or axis == 2:
            axis_tmp[2] = 1
            axis = axis_tmp

        matrix = matrices.rotation_matrix(angle, axis)
        #crea un array colonna con il vettore vertices e nell'ultima colonna un vettore di soli 1
        a = np.hstack((self.vertices, np.ones((self.vertices.shape[0], 1))))#(nx3)->(nx4)
        #moltiplica l'array appena creato con la matrice di trasformazione trasposta (per non trasporre tutte le righe di vertices)
        self.vertices = a.dot(matrix.T)[:,:-1]
        self._dont_update = False
        self.update()

    def transform_reflection(self, axis):
        """
        Reflect the mesh with respect a given axis or plane. It affects the mesh geometry.

        Parameters:

            axis  (string): Reflection plane. It can be 'x', 'y', 'z' or xy', 'xz', 'yz'

        """

        self._dont_update = True
        matrix = np.array([[-1,0,0],[0,-1,0],[0,0,-1]]);
        if 'x' in axis:
            matrix[0][0] = 1
        if 'y' in axis:
            matrix[1][1] = 1
        if 'z' in axis:
            matrix[2][2] = 1
        
        self.vertices = self.vertices.dot(matrix).reshape(-1, 3)
        self._dont_update = False
        self.update()

    def vert_id(self, vert, strict=False):

        """
        Return the id of a vertex given its coordinates. If the vertex doesn't exist 
        or there are multiple matches then an array with all the matches is returned.

        Parameters:

            vert (Array (,3) type=float): The coordinates of the vertex
            strict (bool): if False there is a tolerance of 1e-5 

        """
        
        if strict:
            result = (self.vertices == vert).all(axis=1).nonzero()[0]
        else:
            verts = np.around(self.vertices, decimals=5)
            vert  = np.around(vert, decimals=5) 
            result = (verts == vert).all(axis=1).nonzero()[0]
    
        return result.item() if result.size == 1 else result

    def edge_id(self, v0, v1):

        """
        Return the id of an edge given its 2 vertices. If the edge doesn't exist 
        or there are multiple matches then an array with all the matches is returned.

        Parameters:

            v0 (int): index of the first vertex
            v1 (int): index of the second vertex
        """
    
        or_cond_1 = np.logical_or(self.edges[:,0] == v0 , self.edges[:,1] == v0)
        or_cond_2 = np.logical_or(self.edges[:,0] == v1 , self.edges[:,1] == v1)
        condition = np.logical_and(or_cond_1, or_cond_2)
        result = np.nonzero(condition)[0]
        return result.item() if result.size == 1 else result

    def poly_id(self, verts):

        """
        Return the id of a poly given its vertices. If the poly doesn't exist 
        or there are multiple matches then an array with all the matches is returned.

        Parameters:

            verts (Array (n, m) type=int): indices of the vertices composing the poly
        """
    
        verts = np.sort(verts)
        polys = np.sort(self.polys, axis=1)
        result = (polys == verts).all(axis=1).nonzero()[0]
    
        return result.item() if result.size == 1 else result

    @property
    def bbox(self):
        """
        Return the axis aligned bounding box of the current mesh.
        """
        return self.__bounding_box


    @property
    def num_vertices(self):
        """
        Return the number of vertices of the current mesh.
        """
        return self.vertices.shape[0]

    @property
    def num_edges(self):
        """
        Return the number of edges of the current mesh.
        """

        return self.__edges.shape[0]
    
    @property
    def num_polys(self):
        """
        Return the number of polys of the current mesh.
        """
        return self.__polys.shape[0]

    @property
    def edges(self):
        """
        Return the edges of the current mesh as an Array (n, 2).
        """
        return self.__edges

    @property
    def polys(self):
        """
        Return the edges of the current mesh as an Array (n, m).
        """
        return self.__polys

    

    @property
    def center(self):
        """
        Return the center of the bounding box as an Array (n, 3).
        """

        x1, x2 = self.__bounding_box[0][0], self.__bounding_box[1][0]
        y1, y2 = self.__bounding_box[0][1], self.__bounding_box[1][1]
        z1, z2 = self.__bounding_box[0][2], self.__bounding_box[1][2]

        return np.array([(x1+x2)/2, (y1+y2)/2, (z1+z2)/2])

    @property
    def scale(self):
        """
        Return the scale of the current mesh calculated as the distance between the minimun and maximum point of the bounding box.
        """
        return np.linalg.norm(self.__bounding_box[0]-self.__bounding_box[1])

    def __update_bounding_box(self):

        min_x_coord = self.vertices[:,0].min()
        max_x_coord = self.vertices[:,0].max()
        min_y_coord = self.vertices[:,1].min()
        max_y_coord = self.vertices[:,1].max()
        min_z_coord = self.vertices[:,2].min()
        max_z_coord = self.vertices[:,2].max()

        self.__bounding_box = np.array([[min_x_coord, min_y_coord, min_z_coord],
                                        [max_x_coord, max_y_coord, max_z_coord]])



    def polys_add(self, new_polys):



        self._dont_update = True
        new_polys = np.array(new_polys)
        new_polys.shape = (-1, self.__poly_size)

        if new_polys.max() >= self.num_vertices:
            raise Exception('The id of a vertex must be less than the number of vertices')

        self.__polys = np.concatenate([self.__polys, new_polys])
        #it should update the mesh locally



    def polys_remove(self, poly_ids):

        self._dont_update = True
        poly_ids = np.array(poly_ids)
        mask = np.ones(self.num_polys)
        mask[poly_ids] = 0
        mask = mask.astype(np.bool)

        self.__polys = self.__polys[mask]
        if self.labels is not None:
            self.labels = self.labels[mask]

    
    @property
    def filename(self):
        """
        Return the filename of the current mesh.
        """
        return self.__filename

    @property
    def poly_centroids(self):
        """
        Return the centroids of the polys of the current mesh as an Array (n,3).
        """
        if self.__simplex_centroids is None:
            self.__simplex_centroids = np.asarray(self.vertices[self.polys].mean(axis=0))
        return self.__simplex_centroids

    @property
    def mesh_centroid(self):
        """
        Return the centroid of the current mesh as an Array (,3).
        """
        return np.asarray(self.vertices.mean(axis=0))
    
    @property
    def edge_centroids(self):
        """
        Return the centroids of the current mesh edges as an Array (n,3).
        """
        return self.vertices[self.edges].mean(axis=0)


    def edges_sample_at(self, value):
        """
        Sample the edges at a given value.
        
        Parameters:

            value (float): The value used to sample the point. It must be a value in the range (0,1)
        """
        assert(value >= 0 and value <=1)
        return (1.0-value)*self.vertices[self.edges[:,0]] + value*self.vertices[self.edges[:,1]]
    
    @property
    def edge_length(self):
        """
        Return the length of the edges of the current mesh as an Array (n,1).
        """
        return np.linalg.norm(self.vertices[self.edges[:,0]]-self.vertices[self.edges[:,1]], axis=1)


    def __repr__(self):
        return f"Mesh of {self.num_vertices} vertices and {self.num_polys} polygons."

    @property
    def mesh_is_volumetric(self):
        """
        Return True if the current mesh is a Hexmesh or a Tetmesh.
        """
        return hasattr(self, 'faces')

    @property
    def mesh_is_surface(self):
        """
        Return True if the current mesh is a Trimesh or a Quadmesh.
        """
        return not self.mesh_is_volumetric

    
    @property
    def euler_characteristic(self):
        """
        Return the Euler characteristic of the current mesh.
        """
        if self.mesh_is_volumetric:
            return self.num_vertices - self.num_edges + self.num_faces - self.num_polys
        else:
            return self.num_vertices - self.num_edges + self.num_polys
    
    @property
    def genus(self):
        """
        Return the genus of the current mesh.
        """
        if self.mesh_is_volumetric:
            return int(1-self.euler_characteristic)
        else:
            return int((2-self.euler_characteristic)*0.5)

    @property
    def edge_valence(self):
        """
        Return the valence of each edge.
        """
        return _valence(self.adj_edge2poly)

    @property
    def vert_valence(self):
        """
        Return the valence of each vertex.
        """
        return _valence(self.adj_vtx2vtx)

    def pick_vertex(self, point):
        """
        Return the nearest vertex id given a point.

        Parameters:

            point (Array (, 3) type=float): Coordinates of the point
        """
        point = np.repeat(np.asarray(point).reshape(-1,3), self.num_vertices, axis=0)
        idx = np.argmin(np.linalg.norm(self.vertices - point, axis=1), axis=0)
        return idx
    
    def pick_edge(self, point):
        """
        Return the nearest edge id given a point.

        Parameters:

            point (Array (, 3) type=float): Coordinates of the point
        """
        point = np.repeat(np.asarray(point).reshape(-1,3), self.num_edges, axis=0)
        idx = np.argmin(np.linalg.norm(self.edge_centroids - point, axis=1), axis=0)
        return idx
    
    def pick_poly(self, point):
        """
        Return the nearest poly id given a point.

        Parameters:

            point (Array (, 3) type=float): Coordinates of the point
        """
        point = np.repeat(np.asarray(point).reshape(-1,3), self.num_polys, axis=0)
        idx = np.argmin(np.linalg.norm(self.poly_centroids - point, axis=1), axis=0)
        return idx
    
    def normalize_bbox(self):
        diag = np.linalg.norm(self.bbox[0]-self.bbox[1])
        s = 1.0/diag
        self.transform_scale([s,s,s])


    #adjacencies

    @property
    def adj_vtx2vtx(self):
        """
        Return the adjacencies between vertex and vertex 
        """
        return NList.NList(self.__adj_vtx2vtx)

    @property
    def adj_vtx2edge(self):
        """
        Return the adjacencies between vertex and edge 
        """
        return NList.NList(self.__adj_vtx2edge)

    @property
    def adj_vtx2poly(self):
        """
        Return the adjacencies between vertex and poly 
        """
        return NList.NList(self.__adj_vtx2poly)

    @property
    def adj_edge2vtx(self):
        """
        Return the adjacencies between edge and vertex 
        """
        return self.__adj_edge2vtx
    
    @property
    def adj_edge2edge(self):
        """
        Return the adjacencies between edge and edge 
        """
        return NList.NList(self.__adj_edge2edge)
    
    @property
    def adj_edge2poly(self):
        """
        Return the adjacencies between edge and poly 
        """
        return NList.NList(self.__adj_edge2poly)

    @property
    def adj_poly2vtx(self):
        """
        Return the adjacencies between poly and vertex 
        """
        return self.__adj_poly2vtx
    
    @property
    def adj_poly2edge(self):
        """
        Return the adjacencies between poly and edge 
        """
        return self.__adj_poly2edge

    @property
    def adj_poly2poly(self):
        """
        Return the adjacencies between poly and poly 
        """
        return self.__adj_poly2poly 
    
    

    #deprecated

    @property
    @deprecated("Use the method adj_vtx2vtx instead")
    def vtx2vtx(self):
                
        return self.__adj_vtx2vtx

    @property
    @deprecated("Use the method adj_vtx2face instead")
    def vtx2face(self):
        return self.__adj_vtx2face
