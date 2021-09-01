import numpy as np

class Quaternion:
    
    def __init__(self, w,x,y,z):
        
        self.x = x
        self.y = y
        self.z = z
        self.w = w
    
    def to_euler_angles(self):
        
        sinr_cosp = 2 * (self.w * self.x + self.y * self.z)
        cosr_cosp = 1 - 2 * (self.x * self.x + self.y * self.y)
        x=np.arctan2(sinr_cosp, cosr_cosp)
        
        sinp = 2 * (self.w * self.y - self.z * self.x);
        if  np.abs(sinp) >= 1:
            y = np.copysign(np.pi/2, sinp)
        else:
            y = np.arcsin(sinp)
        
        siny_cosp = 2 * (self.w * self.z + self.x * self.y)
        cosy_cosp = 1 - 2 * (self.y * self.y + self.z * self.z)
        z = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.array([x,y,z], dtype=np.float64)
    
    @staticmethod
    def from_euler_angles(x,y,z):
         
        cy = np.cos(z*0.5)
        sy = np.sin(z*0.5)
        cp = np.cos(y*0.5)
        sp = np.sin(y*0.5)
        cr = np.cos(x*0.5)
        sr = np.sin(x*0.5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return Quaternion(w,x,y,z)
        
        
    @property
    def array(self):
        return np.array([self.w, self.x, self.y, self.z], dtype=np.float64)
    
    def __repr__(self):
        return f"w: {self.w} x:{self.x} y:{self.y} z:{self.z}"