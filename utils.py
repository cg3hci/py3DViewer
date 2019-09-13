import numpy as np

def read_mesh(filename):
    """
    Imports the data from the given MESH file
    
    Parameters
    ----------
    filename : str
        name of the file
    
    Returns
    -------
    (Array,Array)
        the mesh vertices, normals and the topology arrays
    """


def read_obj(filename):
    """
    Imports the data from the given OBJ file
    
    Parameters
    ----------
    filename : str
        name of the file
    
    Returns
    -------
    (Array,Array,Array,Array)
        the mesh vertices, normals and the topology arrays
    """
    
    with open(filename) as f:
        
        tmpVtx     = []
        tmpFaces   = []
        tmpNormals = []
        
        for line in f.readlines():
            if line[0:2] == 'v ':
                vtx = line.split()
                tmpVtx.append([float(vtx[1]), float(vtx[2]), float(vtx[3])])
            
            if line[0:2] == 'f ':
                face = line.split()
                
                tmpFaces.append([int(f) -1 for f in face[1:]])
            
            
#            if line[0:2] == 'vn':
#                    face_normal = line.split()
#                    tmpNormals.append([float(face_normal[1]), float(face_normal[2]), float(face_normals[3])])
                    
        tmpVtx = np.array(tmpVtx)
        tmpFaces = np.array(tmpFaces)
        tmpNormals = np.array(tmpNormals)
            
        return tmpVtx, tmpFaces, tmpNormals
    
    
    
def save_obj(mesh, filename):
    """
    Writes the data from the given mesh object to a file
    
    Parameters
    ----------
    mesh : Trimesh / Quadmesh
        the mesh you want to serialize
    filename : str
        name of the file
    
    Returns
    -------
    void
    """
    
    with open(filename, 'w') as f:
        
        for vtx in mesh.vertices:
            f.write(f"v {vtx[0]} {vtx[1]} {vtx[2]}\n")
            
        for face in mesh.faces:
            
            if 'Trimesh' in str(type(mesh)):
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
            
            if 'Quadmesh' in str(type(mesh)):
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1} {face[3]+1}\n")

            