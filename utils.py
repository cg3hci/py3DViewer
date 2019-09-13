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
    (Array,Array,Array)
        the mesh vertex, tet and label arrays
    """
    
    assert filename.split(".")[-1] == "mesh" # Maybe throw exception?
        
    
    with open(filename) as f:
        reading_vertices = False
        tmp_vtx          = []
        tmp_simplices    = []
        tmp_labels   = []
        num_vtx          = 0
        num_simplices         = 0
        
        line             = f.readline()
        
        while line != "" and "Vertices" not in line:        
            line = f.readline()
             
        assert line != ""

        num_vtx = int(f.readline())
        
        for i in range(num_vtx):
            line = f.readline()
            x, y, z = list(map(lambda x : float(x), line.split()[:-1]))
            tmp_vtx += [(x, y, z)]
        
        line = f.readline()
        
        while "Tetrahedra" not in line and "Hexahedra" not in line and line != "":
            line = f.readline()
            
        assert line != ""
        
        num_simplices = int(f.readline())
        
        if "Tetrahedra" in line:
            for i in range(num_simplices):
                line = f.readline()
                a, b, c, d, label = list(map(lambda x : int(x)-1, line.split()))
                tmp_simplices += [(a, b, c, d)]
                tmp_labels += [label]
        else:
            for i in range(num_simplices):
                line = f.readline()
                a, b, c, d, e, f, label = list(map(lambda x : int(x)-1, line.split()))
                tmp_simplices += [(a, b, c, d, e, f)]
                tmp_labels += [label]

        
        tmp_vtx = np.array(tmp_vtx)
        tmp_simplices = np.array(tmp_simplices)
        tmp_labels = np.array(tmp_labels)
        
        return tmp_vtx, tmp_simplices, tmp_labels
    

def save_mesh(mesh, filename):
    """
    Writes the data from the given mesh object to a file
    
    Parameters
    ----------
    mesh : Tetmesh / Hexmesh
        the mesh you want to serialize
    filename : str
        name of the file
    
    Returns
    -------
    void
    """
    
    with open(filename, 'w') as f:
        
        f.write('MeshVersionFormatted 1\nDimension 3\n')
        f.write('Vertices\n')
        f.write(f'{mesh.num_vertices}\n')
        
        for v in mesh.vertices:
            f.write(f'{v[0]} {v[1]} {v[2]} 0\n')
        
        if mesh.tets.shape[1] == 4:
            f.write('Tetrahedra\n')
            f.write(f'{mesh.num_tets}\n')
            for idx, t in enumerate(mesh.tets):
                f.write(f'{t[0]+1} {t[1]+1} {t[2]+1} {t[3]+1} {mesh.labels[idx]}\n')
        
        else:
            f.write('Hexahedra\n')
            f.write(f'{mesh.num_hexes}\n')
            for idx, h in enumerate(mesh.hexes):
                f.write(f'{h[0]+1} {h[1]+1} {h[2]+1} {h[3]+1} {h[4]+1} {h[5]+1} {h[6]+1} {h[7]+1} {mesh.labels[idx]}\n')
        
        f.write('End')

 

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

            