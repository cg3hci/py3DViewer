import numpy as np


class Ray:
    
    def __init__(self, origin=(0.0,0.0,0.0), direction=(1.0,0,0)):
        
        self.origin    = np.array(origin, dtype=np.float)
        self.direction = np.array(direction, dtype=np.float)