import numpy as np

class AbstractMesh(object):
    
    def __init__(self):
        
        self.vertices         = None #npArray (Nx3)
        self.vtx_normals       = None #npArray (Nx3)
        self.boundary         = None #TriMesh or QuadMesh
        self.faces            = None #npArray (NxM)
        self.vtx2face         = None #npArray (NxM)
        self.__vtx2vtx        = None
        self.__bounding_box    = None #npArray (2x3)
        self.subspace         = None #npArray (3x2)
        self.simplex_metrics   = None #dictionary[propertyName : npArray (Nx1)]
        self.simplex_centroids = None #npArray (Nx1)
        self.is_dirty          = None #Bool
        
        #CUT
        self.__cut            = {'min_x':None, 
                                 'max_x':None, 
                                 'min_y':None, 
                                 'max_y':None, 
                                 'min_z':None, 
                                 'max_z':None}  
        
        super().__init__()
     
    @property
    def cut(self):
        return self.__cut
    
    def setCut(self, min_x = None, max_x = None, 
            min_y = None, max_y = None, 
            min_z = None, max_z = None):
        
        
        if min_x:
            self.__cut['min_x'] = min_x
        if max_x:
            self.__cut['max_x'] = max_x
        if min_y:
            self.__cut['min_y'] = min_y
        if max_y:
            self.__cut['max_y'] = max_y
        if min_z:
            self.__cut['min_z'] = min_z
        if max_z:
            self.__cut['max_z'] = max_z
           

    def loadFromFile(filename):
        
        raise NotImplementedError('This method must be implemented in the subclasses')
        
    
    def saveFile(self, filename):
        raise NotImplementedError('This method must be implemented in the subclasses')


    def getMetric(self, propertyName, idElement):
        
        return self.simplexMetrics[propertyName][idElement]
    
    @property
    def __simplexCentroids(self):
        raise NotImplementedError('This method must be implemented in the subclasses')
    
    def __computeMetrics(self): 
        raise NotImplementedError('This method must be implemented in the subclasses')
        
    def addVertex(self, x, y, z): 
        
        newVertex = np.array([x,y,z], dtype=np.float)
        
        self.vertices = np.concatenate([self.vertices, newVertex])
    
    
    def addVertexList(newVertices):
        
        newVertices = np.array(newVertices)
        self.vertices = np.concatenate([self.vertices, newVertices])
        
        
    @property
    def vtx2vtx(self, vtx):
        
        return self.__vtx2vtx[vtx]
        
    
    @property
    def bbox(self):

        return self.__boundingBox
    
    
    def __updateBoundingBox(self):
        
        minXCoord = self.vertices[:,0].min()
        maxXCoord = self.vertices[:,0].max()
        minYCoord = self.vertices[:,1].min()
        maxYCoord = self.vertices[:,1].max()
        minZCoord = self.vertices[:,2].min()
        maxZCoord = self.vertices[:,2].max()
        
        self.__boundingBox = np.array([[minXCoord, maxXCoord],[minYCoord, maxXCoord],[minZCoord, maxZCoord]])
        
        