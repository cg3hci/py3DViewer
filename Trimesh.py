from AbstractMesh import AbstractMesh
from Visualization import Viewer
import numpy as np
import utils
from metrics import triangle_aspect_ratio, triangle_area

class Trimesh(AbstractMesh):
    
    def __init__(self, vertices = None, faces = None, labels = None):
        
        self.face_normals     = None #npArray (Nx3)
        self.face_labels      = None #npArray (Nx1)
        self.__face2face      = None #npArray (Nx3?)
        
        super(Trimesh, self).__init__()
        
        if vertices and faces:
            
            self.vertices = np.array(vertices) 
            self.faces = np.array(faces)
            
            if labels:
                self.labels = np.array(labels)
            
            self.__load_operations()
         
    
    # ==================== METHODS ==================== #
    
    def show(self, width = 700, height = 700, mesh_color = None):
        
        Viewer(self, UI=False, mesh_color=mesh_color).show(width = width , height = height)
    
    
    @property
    def num_faces(self):
        
        return self.faces.shape[0]


    def add_face(self,face_id0, face_id1, face_id2):
        
        self.add_faces([face_id0, face_id1, face_id2])
        
        
    def add_faces(self, new_faces):
            
        new_faces = np.array(new_faces)
        new_faces.shape = (-1,3)
                
        if new_faces[(new_faces[:,0] > self.num_vertices) | 
                     (new_faces[:,1] > self.num_vertices) | 
                     (new_faces[:,2] > self.num_vertices)].shape[0] > self.num_vertices:
            raise Exception('The Id of a vertex must be lesser than the number of vertices')

        self.faces = np.concatenate([self.faces, new_faces])
        self.__load_operations()
        
    
    def remove_face(self,face_id):
        
        self.remove_faces([face_id])
        
        
    def remove_faces(self, face_ids):
        
        face_ids = np.array(face_ids)
        mask = np.ones(self.num_faces)
        mask[face_ids] = 0
        mask = mask.astype(np.bool)
        
        self.faces = self.faces[mask]
        self.__load_operations()
        
    
    def remove_vertex(self,vtx_id):
        
        self.remove_vertices([vtx_id])
    
    
    def remove_vertices(self, vtx_ids):
        
        vtx_ids = np.array(vtx_ids)
        
        for v_id in vtx_ids:
                        
            self.vertices = np.delete(self.vertices, v_id, 0)
            self.faces = self.faces[(self.faces[:,0] != v_id) & 
                                    (self.faces[:,1] != v_id) & 
                                    (self.faces[:,2] != v_id)]
            
            self.faces[(self.faces[:,0] > v_id)] -= np.array([1, 0, 0])
            self.faces[(self.faces[:,1] > v_id)] -= np.array([0, 1, 0])
            self.faces[(self.faces[:,2] > v_id)] -= np.array([0, 0, 1])
            
            vtx_ids[vtx_ids > v_id] -= 1;
            
        self.__load_operations()
        
        
    def __load_operations(self):
        
        self.__compute_adjacencies()
        self._AbstractMesh__update_bounding_box()
        self.set_cut(self.bbox[0,0], self.bbox[1,0], 
                     self.bbox[0,1], self.bbox[1,1], 
                     self.bbox[0,2], self.bbox[1,2])
        self.__compute_face_normals()
        self.__compute_vertex_normals()
        self.__compute_metrics()
    
    
    def __compute_adjacencies(self):
        
        map_ = dict()
        adjs =  np.zeros((self.num_faces, 3))-1
        vtx2vtx = [[] for i in range(self.num_vertices)]
        vtx2face = [[] for i in range(self.num_vertices)]


        edges = np.c_[self.faces[:,0], self.faces[:,1], 
                      self.faces[:,1], self.faces[:,2], 
                      self.faces[:,2], self.faces[:,0]]
        edges.shape = (-1, 2)
        faces_idx = np.repeat(np.array(range(self.num_faces)), 3)
        
        for e, f in zip(edges, faces_idx):
            
            vtx2vtx[e[0]].append(e[1])
            vtx2face[e[0]].append(f)
            vtx2face[e[1]].append(f)
            
            e = (e[0], e[1])

            try:
                tmp = map_[e]
            except KeyError:
                tmp = None

            if tmp is None:
                map_[(e[1], e[0])] = f
            else:
                idx_to_append1 = np.where(adjs[f] == -1)[0][0]
                idx_to_append2 = np.where(adjs[map_[e]] == -1)[0][0]
                adjs[f][idx_to_append1] = map_[e]
                adjs[map_[e]][idx_to_append2] = f
                


        self.__face2face =  adjs
        self._AbstractMesh__vtx2vtx = np.array([np.array(a) for a in vtx2vtx])
        self._AbstractMesh__vtx2face = np.array([np.unique(np.array(a)) for a in vtx2face])

        
    def __compute_face_normals(self):
        
        e1_v = self.vertices[self.faces][:,1] - self.vertices[self.faces][:,0]
        e2_v = self.vertices[self.faces][:,2] - self.vertices[self.faces][:,1]
        
        self.face_normals = np.cross(e1_v, e2_v)
        norm = np.linalg.norm(self.face_normals, axis=1)
        norm.shape = (-1,1)
        self.face_normals = self.face_normals / norm
        
    
    def __compute_vertex_normals(self):
        
        self.vtx_normals = np.array([np.mean(self.face_normals[v2f], axis = 0) for v2f in self.vtx2face])
        norm = np.linalg.norm(self.vtx_normals, axis=1)
        norm.shape = (-1,1)
        self.vtx_normals = self.vtx_normals / norm
        
        
    def load_from_file(self, filename):
        
        ext = filename.split('.')[-1]
        
        if ext == 'obj':
            self.vertices, self.faces, self.face_normals = utils.read_obj(filename)
            
        self.__load_operations()
        
    
    def save_file(self, filename):
        
        ext = filename.split('.')[-1]
        
        if ext == 'obj':
            utils.save_obj(self, filename)
        
    
    def __compute_metrics(self): 
        
        self.simplex_metrics['area'] = triangle_area(self.vertices, self.faces)
        self.simplex_metrics['aspect_ratio'] = triangle_aspect_ratio(self.vertices, self.faces)
        
    
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
        
        return self.faces[cut_range], cut_range
    
        
    @property
    def face2face(self):
        return self.__face2face

        
    @property
    def simplex_centroids(self):
        
        if self._AbstractMesh__simplex_centroids is None:
            self._AbstractMesh__simplex_centroids = self.vertices[self.faces].mean(axis = 1)
        
        return self._AbstractMesh__simplex_centroids
    
    @property
    def edges(self):
        
        edges =  np.c_[self.faces[:,:2], self.faces[:,1:], self.faces[:,2], self.faces[:,0]]
        edges.shape = (-1,2)
        
        return edges
       
    def __repr__(self):
        self.show()
        return f"Showing {self.boundary()[0].shape[0]} polygons."
    
    
    