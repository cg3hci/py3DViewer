import pythreejs as three
import numpy as np
from time import time, sleep
from .Colors import colors
from ..utils import Observer, ColorMap
import threading
import copy
import math
import re


class DrawableMesh (Observer):


    def __init__(self, geometry, mesh_color = None, reactive = False):
        super(DrawableMesh, self).__init__()
        self._external_color = colors.teal
        self._internal_color = colors.orange
        self._color_map = None
        self._metric_string = None
        self._c_map_string = None
        self._label_colors = None
        self.texture = geometry.texture
        self.faceVertexUvs = []
        self.geometry = geometry
        self.type = str(type(self.geometry))

        if reactive:
            self.geometry.attach(self)

        #Methods for initializing meshes' attributes (color, mesh, wireframe, threeks_items, flags like: updating and queue)
        self.geometry_color = self.__initialize_geometry_color(mesh_color)
        self.mesh = self.__initialize_mesh()

        ##reminder. wireframe is when the object is projected into screen space and rendered by drawing lines at the location of each edge.
        self.wireframe = self.__initialize_wireframe()
        self.threejs_items = [self.mesh, self.wireframe]
        self.updating = False
        self.queue = False




    def __initialize_geometry_color(self, mesh_color, geometry = None):

        if geometry is None:
            geometry = self.geometry
        if mesh_color is None:
            #External color is teal, initialized in __init__ and represented with a numpy array 1x3
            color = np.repeat(self._external_color.reshape(1, 3), geometry.num_triangles*3, axis=0)
            # This condition is for initializing the color of the internal part of volumetric meshes
            if hasattr(self.geometry, "internals"):
                internal_color = geometry.internal_triangles_idx()
                color[internal_color] = self._internal_color
        else:
            mesh_color = np.array(mesh_color, dtype=np.float)/255
            color = np.repeat(mesh_color.reshape(1,3), geometry.num_triangles*3, axis=0) 
            self._external_color = mesh_color

        return color

    def update_wireframe_color(self, new_color):
        self.wireframe.material.color = new_color

    def update_wireframe_opacity(self, new_opacity):
        self.wireframe.material.opacity = new_opacity

    def update_internal_color(self, new_color, geometry = None):

        if geometry is None:
            geometry = self.geometry
        self._internal_color = np.array(new_color)

        # This condition is for updating the color of the internal part of volumetric meshes
        if hasattr(geometry, "internals"):
            internal_color = geometry.internal_triangles_idx()
            self.geometry_color[internal_color] = new_color
            #_as_threejs_colors has been recalled without passing nothing, it returns true
            colors = geometry._as_threejs_colors()
            new_colors = self.geometry_color[colors]
            tris, vtx_normals = geometry._as_threejs_triangle_soup()
            interleaved = np.concatenate((tris, new_colors, vtx_normals), axis=1).astype(np.float32)
            #interleaved is made up of the triangle soup, the new colors and the normals of this vertices
            self.mesh.geometry.attributes['color'].data.array = interleaved

    def update_poly_color(self, new_color, poly_index, num_triangles=None, geometry=None):

        if geometry is None:
            geometry = self.geometry
        
        if num_triangles is None:
            if "Quadmesh" in str(type(self.geometry)):
                num_triangles = 2
            elif "Tetmesh" in str(type(self.geometry)):
                num_triangles = 4
            elif "Hexmesh" in str(type(self.geometry)):
                num_triangles = 12
            else:
                num_triangles = 1

        start = poly_index*num_triangles*3
        end   = start+num_triangles*3
        indices = np.arange(start, end);

        self.geometry_color[indices] = new_color
        colors = geometry._as_threejs_colors()
        new_colors = self.geometry_color[colors].astype(np.float32)
        tris, vtx_normals = geometry._as_threejs_triangle_soup()
        interleaved = np.c_[tris, new_colors, vtx_normals].astype(np.float32)
       

        self.mesh.geometry.attributes['color'].data.array = interleaved

    def update_external_color(self, new_color, geometry = None):


        if geometry is None:
            geometry = self.geometry
        self._external_color = np.array(new_color)

        #This condition is for initializing the color of the external part of volumetric meshes
        if hasattr(geometry, "internals"):
            internal_color = geometry.internal_triangles_idx()
            self.geometry_color[np.logical_not(internal_color)] = new_color
        else:
            self.geometry_color[:] = new_color

        colors = geometry._as_threejs_colors()
        new_colors = self.geometry_color[colors]
        tris, vtx_normals = geometry._as_threejs_triangle_soup()
        
        if len(self.geometry.material) > 0 or self.texture is not None:
            interleaved = np.concatenate((tris, new_colors, vtx_normals, self.geometry.uvcoords), axis=1).astype(np.float32)
        else:
            interleaved = np.concatenate((tris, new_colors, vtx_normals), axis=1).astype(np.float32)
        self.mesh.geometry.attributes['color'].data.array = interleaved

    def update_color_map(self, new_colors, geometry = None):

        if geometry is None:
            geometry = self.geometry

        self.geometry_color[:] = geometry._as_threejs_colors(colors = new_colors)
        colors = geometry._as_threejs_colors()
        new_colors = self.geometry_color[colors]
        tris, vtx_normals = geometry._as_threejs_triangle_soup()

        if(tris.shape != new_colors.shape):
            return

        if len(self.geometry.material) > 0 or self.texture is not None:
            interleaved = np.concatenate((tris, new_colors, vtx_normals, self.faceVertexUvs), axis=1).astype(np.float32)
        else:
            interleaved = np.concatenate((tris, new_colors, vtx_normals), axis=1).astype(np.float32)
        self.mesh.geometry.attributes['color'].data.array = interleaved
        

    def compute_color_map(self, metric_string, c_map_string, geometry=None):

        if geometry is None:
            geometry = self.geometry

        self._metric_string = metric_string
        self._c_map_string  = c_map_string

        #simplex metrics is a dictionary inherited by Abstract Mesh
        #[propertyName : ((min, max), npArray (Nx1))]
        (min_range, max_range), metric = self.geometry.simplex_metrics[metric_string]
        #virdis, parula, jet or red_blue
        c_map = ColorMap.color_maps[c_map_string]

        if min_range is None or max_range is None:

            min_range = np.min(metric)
            max_range = np.max(metric)
            #ptp = peak to peak "Range of values (maximum - minimum) along an axis"
            if (np.abs(max_range-min_range) > 1e-7):
                normalized_metric = ((metric - np.min(metric))/np.ptp(metric)) * (c_map.shape[0]-1)
            else:
                normalized_metric = np.repeat(np.mean(metric), metric.shape[0])
        else:
            #Clip (limit) the values in an array.
            #Given an interval, values outside the interval are clipped to the interval edges.
            #ex. a = [0,1,2,3,4,5,6,7] clip(a, 1, 5) a = [1,1,2,3,4,5,5,5]
            normalized_metric = np.clip(metric, min_range, max_range)
            normalized_metric = (normalized_metric - min_range)/(max_range-min_range) * (c_map.shape[0]-1)

        normalized_metric = 1-normalized_metric
        #rint round elements to the nearest integer number
        metric_to_colormap = np.rint(normalized_metric).astype(np.int)

        mesh_color = c_map[metric_to_colormap]

        self._color_map = mesh_color
        self.update_color_map(mesh_color, geometry)


    def update_color_label(self, geometry = None):

        if geometry is None:
            geometry = self.geometry


        mesh_color = np.zeros((geometry.num_polys,3), dtype=np.float)

        
        for idx, value in enumerate(self.geometry.labels):

            if(int(value) not in self._label_colors):
                self._label_colors[int(value)] = colors.random_color()
            mesh_color[idx] = self._label_colors[int(value)]
            

        self._color_map = mesh_color
        self.update_color_map(mesh_color)


    def __initialize_wireframe(self):

        #LineBasicMaterial: A material for drawing wireframe-style geometries
        edges_material = three.LineBasicMaterial(color='#686868',
                                                        linewidth = 1,
                                                        depthTest=True,
                                                        opacity=.2,
                                                        transparent=True)
        wireframe = self.__get_wireframe_from_boundary()
        return three.LineSegments(wireframe, material = edges_material)

    def __get_drawable_from_boundary(self):
        geometry_attributes = {}
        tris, vtx_normals = self.geometry._as_threejs_triangle_soup()
        new_colors = self.geometry_color[self.geometry._as_threejs_colors()].astype(np.float32)

        if self.geometry.uvcoords is not None:
            uvcoords =  self.geometry.uvcoords.astype(np.float32)
            coor = self.geometry.coor


            uv = []
            if len(uvcoords) != 0:
            #corrispondenza delle coordinate uv in triangle soup
            #Uv coords in triangle soup
                if 'Quadmesh' in self.type:
                    coor = np.c_[coor[:,:3], coor[:,2:], coor[:,0]]


            coor = coor.flatten()
            for c in coor:
                uv.append(uvcoords[c - 1])


            self.faceVertexUvs = np.array(uv).astype(np.float32)

        if  len(self.faceVertexUvs) != 0 :
            interleaved_array = np.concatenate((tris, new_colors, vtx_normals, self.faceVertexUvs), axis=1)
            buffer = three.InterleavedBuffer(array=interleaved_array, stride=4)
        else:
            interleaved_array = np.concatenate((tris, new_colors, vtx_normals), axis=1)
            buffer = three.InterleavedBuffer(array=interleaved_array, stride=3)


        #Making the interleavedBuffer using the interleaved_array made up of the triangle soup, the color and the vertices' normals, with a stride of 3

        #itemsize = item size, dynamic = (is the normalized attribute o f the super class)?, offset = it's the offset from the start item,
        geometry_attributes['position'] = three.InterleavedBufferAttribute(data=buffer, itemSize=3, dynamic = True)
        geometry_attributes['color'] = three.InterleavedBufferAttribute(data=buffer, itemSize=3, offset=3, dynamic=True)
        geometry_attributes['normal'] = three.InterleavedBufferAttribute(data=buffer, itemSize=3, offset=6, dynamic=True)
        if self.geometry.material is not {} or self.geometry.texture is not None :
            geometry_attributes['uv'] = three.InterleavedBufferAttribute(data=buffer, itemSize=2, offset=9, dynamic=True)
        #         # Buffer Geometry = an efficient representation of mesh, line, or point geometry
        # Includes vertex positions, face indices, normals, colors, UVs, and custom attributes within buffers
        drawable_geometry = three.BufferGeometry(attributes=geometry_attributes)

        #The multiplier is used because groups need faces' indices in triangle soup and 'count' counts only the number of faces per material
        mult = 1
        if 'Trimesh' in self.type:
            mult = 3
        elif 'Quadmesh' in self.type:
            mult = 6

        if len(self.geometry.material) != 0:
            '''
             group = { start: Integer, count: Integer, materialIndex: Integer } where:
             - start : the first triangle index of the group
             - count : how many indices are included
             - materialIndex : the material array index to use for this group 
            '''

            i = 0
            for g in self.geometry.groups:
                if i == 0:
                    n = copy.copy(g)
                    drawable_geometry.exec_three_obj_method("addGroup", 0, mult * self.geometry.groups[g], self.search_key(g))
                else:
                    drawable_geometry.exec_three_obj_method("addGroup", mult * self.geometry.groups[n], mult * self.geometry.groups[g],
                                                            self.search_key(g))
                    n = copy.copy(g)

                i = i + 1

        return drawable_geometry

    # Search key returns the position of the group corresponding the position of the material in the material array
    def search_key(self, s):
        i = 0
        for k in self.geometry.material.keys():
            if s == k:
                return i
            i = i + 1

    def __as_buffer_attr(self, array):
        #BufferAttribute stores data for an attribute (such as vertex positions, face indices etc) associated with a BufferGeometry,
        return three.BufferAttribute(array, normalized = False, dynamic = True)

    def __get_wireframe_from_boundary(self):
        #edges in the boundary box
        edges = self.geometry.vertices[self.geometry.as_edges_flat()].astype(np.float32)
        #The function empty returns an array without values initialized
        buffer = np.empty((int(edges.shape[0] * 3), 3), dtype=np.float32).reshape(-1, 3)
        buffer[:edges.shape[0]] = edges
        vertices = self.__as_buffer_attr(buffer)
        wireframe = three.BufferGeometry(attributes={'position': vertices})
        #Excute the method specified by `method_name` on the three object, with arguments `args`
        #SetDrawRange is a function that sets the attribute DrawRange which determines the part of the geometry to render. (start, end)
        wireframe.exec_three_obj_method("setDrawRange", 0, edges.shape[0])
        return wireframe

    def getTexture(self, filename):
        tex = None
        if filename is not None:
           tex = three.ImageTexture(filename)
        return tex

    def color (self, array):
        #From rgb (0 to 1) to rgb (0 to 255) and from rgb (0 to 255) to html color
        r = (int)(array[0] * 255.999)
        g = (int)(array[1] * 255.999)
        b = (int)(array[2] * 255.999)
        return '#%02x%02x%02x' % (r, g, b)


    def __initialize_mesh(self):
        drawable_geometry = self.__get_drawable_from_boundary()

        #No color under textures or materials
        if len(self.geometry.material) != 0 or self.geometry.texture is not None:
            vertexEnum = 'NoColors'
        else:
            vertexEnum = 'FaceColors'
        materials = []

        #LambertMaterial is a material for non-shiny surfaces, without specular highlights.
        if len(self.geometry.material) == 0: #No material or texture

            material_geometry = three.MeshLambertMaterial(
                                           map = self.getTexture(self.texture),
                                           polygonOffset=True,
                                           polygonOffsetFactor=1,
                                           polygonOffsetUnits=1,
                                           flatShading = True,
                                           opacity = 1.,
                                           transparent = False,
                                           side = 'DoubleSide',
                                           wireframe=False,
                                           vertexColors = vertexEnum)
            materials = material_geometry


        else:

            for m in self.geometry.material:
                if self.geometry.smoothness:
                    material_geometry = three.MeshLambertMaterial(
                        map=self.getTexture(self.geometry.material[m]["map_kd"]),
                        color=self.color(self.geometry.material[m]["kd"]),
                        emissiveIntensity=self.geometry.material[m]["ke"],
                        specular=self.color(self.geometry.material[m]["ks"]),
                        shininess=self.geometry.material[m]["ns"],
                        transparence=self.geometry.material[m]["transparence"],
                        opacity=self.geometry.material[m]["opacity"],
                        emissiveMap=self.getTexture(self.geometry.material[m]["map_ke"]),
                        alphaMap=self.getTexture(self.geometry.material[m]["map_d"]),
                        specularMap=self.getTexture(self.geometry.material[m]["map_ks"]),
                        bumpMap=self.getTexture(self.geometry.material[m]["bump"]),
                        normalMap=self.getTexture(self.geometry.material[m]["norm"]),
                        refractionRatio=self.geometry.material[m]["ni"]
                    )
                else:
                    material_geometry = three.MeshPhongMaterial(
                        map = self.getTexture(self.geometry.material[m]["map_kd"]),
                        color = self.color(self.geometry.material[m]["kd"]),
                        emissiveIntensity = self.geometry.material[m]["ke"],
                        specular = self.color(self.geometry.material[m]["ks"]),
                        shininess  =self.geometry.material[m]["ns"],
                        transparence = self.geometry.material[m]["transparence"],
                        opacity = self.geometry.material[m]["opacity"],
                        emissiveMap = self.getTexture(self.geometry.material[m]["map_ke"]),
                        alphaMap = self.getTexture(self.geometry.material[m]["map_d"]),
                        specularMap = self.getTexture(self.geometry.material[m]["map_ks"]),
                        bumpMap = self.getTexture(self.geometry.material[m]["bump"]),
                        normalMap = self.getTexture(self.geometry.material[m]["norm"]),
                        refractionRatio = self.geometry.material[m]["ni"]
                    )

                materials.append(material_geometry)

        mesh1 = three.Mesh(
                geometry=drawable_geometry,
                material=materials,
                position=[0, 0, 0]
            )

        return mesh1


    def run(self, geometry):

        edges = self.geometry.vertices[self.geometry.as_edges_flat()].astype(np.float32)
        self.wireframe.geometry.attributes['position'].array[:edges.shape[0]] = edges
        self.wireframe.geometry.exec_three_obj_method('setDrawRange', 0, edges.shape[0])
        self.wireframe.geometry.attributes['position'].array = self.wireframe.geometry.attributes['position'].array

        #initilization of the color
        self.geometry_color = self.__initialize_geometry_color(None, geometry)
        if self._color_map is None:
            self.update_internal_color(self._internal_color, geometry)
            self.update_external_color(self._external_color, geometry)

        elif self._label_colors is not None:
            self.update_color_label(geometry)
        else:
            self.compute_color_map(self._metric_string, self._c_map_string, geometry)

        if self.queue:
            self.queue = False
            self.updating = False
            self.update()
        else:
            self.updating = False

    def update(self):
        if (not self.updating):
            self.updating = True
            thread = threading.Thread(target=self.run, args=(self.geometry.copy(),))
            thread.daemon = True
            thread.start()
        else:
            self.queue = True

    @property
    def center(self):
        return self.geometry.center

    @property
    def scale(self):
        return self.geometry.scale
