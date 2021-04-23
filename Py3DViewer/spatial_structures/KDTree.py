from scipy.spatial import KDTree as kdt
class KDTree:
    
    def __init__(self, mesh, leaf_size=10):
        self.__mesh = mesh
        self.__tree = kdt(mesh.vertices, leaf_size)
        
    def knn(self, point=None, idx=None, k=1):
        """
        Return the distance and the indices of the k-nearest neighbors of the given point(s) 
        """
        assert(point is not None or idx is not None)
        if idx is not None:
            point = self.__mesh.vertices[idx]
            
        return self.__tree.query(point, k)
    
    """
    def contains(self, point):
        d, i = self.knn(point)
        neighs = self.__mesh.adj_vtx2poly[i]
    """    