import numpy as np
from .ObservableArray import *
import copy


def read_mesh(filename):
    """
    Imports the data from the given .mesh file

    Parameters:

        filename (string): The name of the .mesh file

    Return:

        (Array, Array, Array): The mesh vertices, the mesh simplices and the mesh labels

    """

    assert filename.split(".")[-1] == "mesh"  # Maybe throw exception?

    with open(filename) as f:
        reading_vertices = False
        tmp_vtx = []
        tmp_simplices = []
        tmp_labels = []
        num_vtx = 0
        num_simplices = 0

        line = f.readline()

        while line != "" and "Vertices" not in line:
            line = f.readline()

        assert line != ""

        num_vtx = int(f.readline())

        for i in range(num_vtx):
            line = f.readline()
            x, y, z = list(map(lambda x: float(x), line.split()[:-1]))
            tmp_vtx += [(x, y, z)]

        line = f.readline()

        while "Tetrahedra" not in line and "Hexahedra" not in line and line != "":
            line = f.readline()

        assert line != ""

        num_simplices = int(f.readline())

        if "Tetrahedra" in line:
            for i in range(num_simplices):
                line = f.readline()
                a, b, c, d = list(map(lambda x: int(x) - 1, line.split()[:-1]))
                label = float(line.split()[-1])
                tmp_simplices += [(a, b, c, d)]
                tmp_labels += [label]
        else:
            for i in range(num_simplices):
                line = f.readline()
                a, b, c, d, e, f_, g, h = list(map(lambda x: int(x) - 1, line.split()[:-1]))
                label = float(line.split()[-1])
                tmp_simplices += [(a, b, c, d, e, f_, g, h)]
                tmp_labels += [label]

        tmp_vtx = np.array(tmp_vtx)
        tmp_simplices = np.array(tmp_simplices)
        tmp_labels = np.array(tmp_labels, dtype=np.int64)

        vtx = ObservableArray(tmp_vtx.shape)
        vtx[:] = tmp_vtx
        simplices = ObservableArray(tmp_simplices.shape, dtype=np.int64)
        simplices[:] = tmp_simplices
        labels = ObservableArray(tmp_labels.shape, dtype=np.int64)
        labels[:] = tmp_labels

        return vtx, simplices, labels


def save_mesh(mesh, filename):
    """
    Writes the data from the given mesh object to a .mesh file

    Parameters : 

        mesh (Tetmesh / Hexmesh): The mesh to serialize to the file
        filename (string): The name of the .mesh file

    """

    with open(filename, 'w') as f:

        f.write('MeshVersionFormatted 1\nDimension 3\n')
        f.write('Vertices\n')
        f.write(f'{mesh.num_vertices}\n')

        for v in np.asarray(mesh.vertices):
            f.write(f'{float(v[0])} {float(v[1])} {float(v[2])} 0\n')

        if mesh.polys.shape[1] == 4:
            f.write('Tetrahedra\n')
            f.write(f'{mesh.num_polys}\n')
            for idx, t in enumerate(np.asarray(mesh.polys)):
                f.write(f'{int(t[0]) + 1} {t[1] + 1} {int(t[2]) + 1} {int(t[3]) + 1} {np.asarray(mesh.labels)[idx]}\n')

        else:
            f.write('Hexahedra\n')
            f.write(f'{mesh.num_polys}\n')
            for idx, h in enumerate(np.asarray(mesh.polys)):
                f.write(
                    f'{int(h[0]) + 1} {int(h[1]) + 1} {int(h[2]) + 1} {int(h[3]) + 1} {int(h[4]) + 1} {int(h[5]) + 1} {int(h[6]) + 1} {int(h[7]) + 1} {np.asarray(mesh.labels)[idx]}\n')

        f.write('End')


def read_mtl(filename):
    with open(filename) as f:
        material_aux = {
            "kd": [],
            "ks": [],
            "ke": 1,
            "ns": 0.0,
            "transparence": False,
            "opacity": 0.0,
            "map_kd": None,
            "map_ke": None,
            "map_d": None,
            "map_ks": None,
            "bump": None,
            "norm": None,
            "illum": 0.0,
            "ni": 0.98
        }
        material_queue = {}
        s = ""
        flag = False

        for line in f.readlines():

            if line[0:6] == 'newmtl':
                # The first time in this branch we don't want to copy the material aux because it's empty
                if flag:
                    material_queue[s] = copy.copy(material_aux)
                flag = True
                s = line.split()[1]

            aux = line.split()

            if line[0:2] == 'Kd':
                material_aux["kd"] = [float(aux[1]), float(aux[2]), float(aux[3])]

            if line[0:2] == 'Ks':
                material_aux["ks"] = [float(aux[1]), float(aux[2]), float(aux[3])]

            if line[0:2] == 'Ke':
                material_aux["ke"] = float(aux[1])

            if line[0:2] == 'Ns':
                material_aux["ns"] = float(aux[1])

            if line[0:5] == 'illum':
                material_aux["illum"] = float(aux[1])

            if line[0] == 'd':
                material_aux['transparence'] = True
                material_aux["opacity"] = float(aux[1])

            if line[0:2] == 'tr':
                material_aux['transparence'] = True
                material_aux["opacity"] = 1 - float(aux[1])

            if line[0:6] == 'map_Kd':
                texture = line.split()[1]
                material_aux["map_kd"] = texture

            if line[0:6] == 'map_Ke':
                texture = line.split()[1]
                material_aux["map_ke"] = texture

            if line[0:6] == 'map_d':
                texture = line.split()[1]
                material_aux["map_d"] = texture

            if line[0:6] == 'map_Ks':
                texture = line.split()[1]
                material_aux["map_ks"] = texture

            if line[0:4] == 'bump' or line[0:8] == 'map_Bump':
                texture = line.split()[1]
                material_aux["bump"] = texture

            if line[0:4] == 'norm':
                texture = line.split()[1]
                material_aux["norm"] = texture

            if line[0:4] == 'Ni':
                texture = line.split()[1]
                material_aux["ni"] = texture

        # This represents the last material
        material_queue[s] = copy.copy(material_aux)

    return material_queue


def read_obj(filename):
    """
    Imports the data from the given .obj file

    Parameters:

        filename (string): The name of the .obj file

    Return:

        (Array, Array, Array): The mesh vertices, the mesh simplices and the mesh labels"""

    with open(filename) as f:

        tmp_vtx = []
        tmp_faces = []
        tmp_normals = []
        tmp_uvcoords = []
        tmp_coor = []
        coor_row = []
        groups = {}
        normals_row = []
        tmp_polytex = []
        tmp_polypos = []
        cont = 0
        s = ""
        sgroup = False

        for line in f.readlines():
            if line[0:6] == 'usemtl':
                if sgroup:
                    groups[s] = copy.copy(cont)
                s = line.split()[1]
                sgroup = True

            if line[0:2] == 'v ':
                if (line[0:2]):
                    vtx = line.split()
                    tmp_vtx.append([float(vtx[1]), float(vtx[2]), float(vtx[3])])
                # The slashes after the split up here are temporary, we need to improve this parser

            if line[0:2] == 'vt':
                uvcoords = line.split()
                if (len(uvcoords) == 3):
                    tmp_uvcoords.append([float(uvcoords[1]), float(uvcoords[2]), 0])
                else:
                    tmp_uvcoords.append([float(uvcoords[1]), float(uvcoords[2]), float(uvcoords[3])])

            if line[0:2] == 'vn':
                if (line[0:2]):
                    normals = line.split()

                tmp_normals.append([float(normals[1]), float(normals[2]), float(normals[3])])

            if line[0:2] == 'f ':
                # count is
                cont = cont + 1
                faces = line.split()
                format = len(faces[1].split("/"))
                faces = faces[1:]
                tmp_faces.append([int(f.split("/")[0]) - 1 for f in faces])

                if format == 2:  # v/vt
                    for c in faces:
                        coor_row.append(int(c.split("/")[1]))
                    tmp_coor.append(coor_row)
                    coor_row = []
                elif format == 3:  # v/vt/vn
                    for c in faces:
                        coor_row.append(int(c.split("/")[1]))
                        normals_row.append((int(c.split("/")[2])))
                    tmp_coor.append(coor_row)
                    tmp_normals.append(normals_row)
                    coor_row = []
                    normals_row = []

        groups[s] = copy.copy(cont)

        tmp_vtx = np.array(tmp_vtx)
        tmp_faces = np.array(tmp_faces)

        tmp_normals = np.array(tmp_normals, dtype=object)
        # Uv coords
        tmp_uvcoords = np.array(tmp_uvcoords)
        # Correspondences between uv coords and faces
        tmp_coor = np.array(tmp_coor)

        vtx = ObservableArray(tmp_vtx.shape)
        vtx[:] = tmp_vtx
        faces = ObservableArray(tmp_faces.shape, dtype=np.int64)
        faces[:] = tmp_faces
        normals = ObservableArray(tmp_normals.shape)
        normals[:] = normals
        uvcoords = ObservableArray(tmp_uvcoords.shape)
        uvcoords[:] = tmp_uvcoords
        coor = ObservableArray(tmp_coor.shape, dtype=np.int64)
        coor[:] = tmp_coor
        return vtx, faces, normals, uvcoords, coor, groups


def save_obj(mesh, filename):
    """
    Writes the data from the given mesh object to a .obj file

    Parameters : 

        mesh (Trimesh / Quadmesh): The mesh to serialize to the file
        filename (string): the name of the .obj file

    """

    with open(filename, 'w') as f:

        for vtx in mesh.vertices:
            f.write(f"v {float(vtx[0])} {float(vtx[1])} {float(vtx[2])}\n")

        for face in mesh.polys:

            if 'Trimesh' in str(type(mesh)):
                f.write(f"f {int(face[0]) + 1} {int(face[1]) + 1} {int(face[2]) + 1}\n")

            if 'Quadmesh' in str(type(mesh)):
                f.write(f"f {int(face[0]) + 1} {int(face[1]) + 1} {int(face[2]) + 1} {int(face[3]) + 1}\n")


def read_skeleton(filename):
    """
    Imports the data from the given .skel file

    Parameters:

        filename (string): The name of the .skel file

    Return:

        (Array, Array, Array): The skeleton joints, the joints radius and the skeleton bones

    """

    with open(filename) as file:

        joint_list = []
        bones = []
        radius = []

        for line in file.readlines():

            try:

                splitted_line = line.split()
                idx = int(splitted_line[0])
                x = float(splitted_line[1])
                y = float(splitted_line[2])
                z = float(splitted_line[3])
                rad = float(splitted_line[4])
                num_neighbors = int(splitted_line[5])

                joint_list.append([x, y, z])
                radius.append(rad)

                for i in range(num_neighbors):
                    neighbor = int(splitted_line[6 + i])
                    bones.append([idx, neighbor])


            except Exception:
                continue

        return np.array(joint_list), np.array(radius), np.array(bones, dtype=np.int64)


def read_off(filename):
    with open(filename) as file:

        vtx_list = []
        face_list = []
        edge_list = []

        num_vertices = 0
        num_faces = 0
        num_edges = 0

        lines = file.readlines()

        idx = 0
        for line in lines:

            idx += 1

            if line[0] == '#':
                continue

            if len(line.split()) == 3:
                head = line.split()
                num_vertices = int(head[0])
                num_faces = int(head[1])
                num_edges = int(head[2])

                break

        for line in lines[idx:idx + num_vertices]:
            vtx = line.split()
            vtx_list += [[float(vtx[0]), float(vtx[1]), float(vtx[2])]]

        if num_faces > 0:
            for line in lines[idx + num_vertices:]:
                face = line.split()
                face_list += [[int(f) for f in face[1:int(face[0]) + 1]]]

        vtx_list = np.array(vtx_list)
        face_list = np.array(face_list)

        vtx = ObservableArray(vtx_list.shape)
        vtx[:] = vtx_list
        faces = ObservableArray(face_list.shape, dtype=np.int64)
        faces[:] = face_list

        return vtx, faces


def save_off(mesh, filename):
    with open(filename, 'w') as f:

        f.write('OFF\n')
        f.write('#Created with Py3DViewer\n\n')
        f.write(f'{mesh.num_vertices} {mesh.num_polys} 0\n')
        for vtx in mesh.vertices:
            f.write(f'{float(vtx[0])} {float(vtx[1])} {float(vtx[2])}\n')

        for face in mesh.polys:
            if face.size == 3:
                f.write(f'3 {int(face[0])} {int(face[1])} {int(face[2])}\n')
            elif face.size == 4:
                f.write(f'4 {int(face[0])} {int(face[1])} {int(face[2])} {int(face[3])}\n')







