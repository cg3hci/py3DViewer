import numpy as np

class Segment:
    
    def __init__(self, a, b):
        
        self.a = np.array(a, dtype=np.float)
        self.b = np.array(b, dtype=np.float)
        
    @property
    def length(self):
        return np.linalg.norm(self.a-self.b)

    def point_on_segment(self, point):
        eps = 1e-6
        s = np.linalg.norm(self.a-point)+np.linalg.norm(self.b-point)
        return np.abs(s-self.length) <= eps
        