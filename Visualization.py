import numpy as np
from pythreejs import *
import ipywidgets as widgets
import math


class Viewer(object):
    
    def __init__(self, mesh, UI = True):
        
        self.mesh = mesh
        self.scene = None
        self.mesh_color = np.array([[0.88, 0.9, 0.94],[0.88, 0.9, 0.94],[0.88, 0.9, 0.94]])
        self.mesh_color = np.repeat(self.mesh_color, self.mesh.boundary.shape[0]*3, axis=0)
        self.center = list(mesh.vertices.mean(axis = 0))
        
        if UI:
            self.__create_UI()
        
        super(Viewer, self).__init__()
        
    
    
    def __create_UI(self):
        pass
    
    def show(self, width=700, height=700, mesh_color=None):
        
        if mesh_color is not None:
            self.mesh_color = np.array([mesh_color, mesh_color, mesh_color], dtype=np.float)
            self.mesh_color = np.repeat(self.mesh_color, self.mesh.boundary.shape[0]*3, axis=0)
        
        
        renderer = self.initialize_camera(self.center, width, height)
        
        if 'Trimesh' in str(type(self.mesh)) or 'Tetmesh' in str(type(self.mesh)):
            
            self.__draw_trimesh()
            
        if 'Quadmesh' in str(type(self.mesh)):
            
            self.__draw_quadmesh()
        
        
        display(renderer)
    
    def __draw_trimesh(self, color=None):
        
        
        tri_properties = {
            'position': BufferAttribute(self.mesh.vertices[self.mesh.boundary(flip_x=False, flip_y=False, flip_z=False).flatten()], normalized=False),
            #'index' : BufferAttribute(np.asarray(self.surface, dtype='uint32').ravel(), normalized=False),
            'color' : BufferAttribute(self.mesh_color, normalized=False),
        }
        
        mesh_geometry = BufferGeometry(attributes=tri_properties)
        mesh_geometry.exec_three_obj_method('computeVertexNormals')
        
        mesh_material = MeshLambertMaterial(#shininess=25,
                                         #emissive = '#aaaaaa',#phong
                                         #specular = '#aaaaaa',#phong
                                           polygonOffset=True,
                                           polygonOffsetFactor=1,
                                           polygonOffsetUnits=1,
                                           flatShading = True,
                                           side = 'FrontSide',
                                           #color = '#550000',
                                           wireframe=False,
                                           vertexColors = 'FaceColors',
                                          )
        
        edges_material = MeshBasicMaterial(color='black',
#                                           side= 'FrontSide'
                                           polygonOffset=True,
                                           polygonOffsetFactor=1,
                                           polygonOffsetUnits=1,
                                           #shininess=0.5,
                                           wireframe=True,
                                           linewidth = 1,
                                           opacity=1,
                                           depthTest=True,
                                           transparent=True)
        
        mesh_ = Mesh(
            geometry=mesh_geometry,
            material=mesh_material,
            position=[0, 0, 0]   # Center in 0
        )
        
        line_ = Mesh(
            geometry=mesh_geometry,
            material=edges_material,
            position=[0, 0, 0]   # Center in 0
        )


        #aggiunge la mesh alla scena
        self.scene.add(mesh_)
        self.scene.add(line_)

    def __draw_quadmesh(self, color=None):
        
        boundaries = self.mesh.boundary(flip_x=False, flip_y=False, flip_z=False)
        tris = np.c_[boundaries[:,:3], boundaries[:,2:], boundaries[:,0]]
        tris.shape = (-1, 3)
        
        
        quad_properties = {
            'position': BufferAttribute(self.mesh.vertices[tris.flatten()], normalized=False),
            #'index' : BufferAttribute(np.asarray(self.surface, dtype='uint32').ravel(), normalized=False),
            'color' : BufferAttribute(self.mesh_color, normalized=False),
        }
        
        mesh_geometry = BufferGeometry(attributes=quad_properties)
        mesh_geometry.exec_three_obj_method('computeVertexNormals')
        
        
        edges = np.c_[self.mesh.boundary()[:,:2], self.mesh.boundary()[:,1:3], self.mesh.boundary()[:,2:4], self.mesh.boundary()[:,3], self.mesh.boundary()[:,0]].flatten()
        surface_wireframe = self.mesh.vertices[edges].tolist()
        
        wireframe = BufferGeometry(attributes={'position': BufferAttribute(surface_wireframe, normalized=False)})
        
        mesh_material = MeshLambertMaterial(#shininess=25,
                                         #emissive = '#aaaaaa',#phong
                                         #specular = '#aaaaaa',#phong
                                           polygonOffset=True,
                                           polygonOffsetFactor=1,
                                           polygonOffsetUnits=1,
                                           flatShading = True,
                                           side = 'FrontSide',
                                           #color = '#550000',
                                           wireframe=False,
                                           vertexColors = 'FaceColors',
                                          )
        
        edges_material = MeshBasicMaterial(color='black',
#                                           side= 'FrontSide'
                                           polygonOffset=True,
                                           polygonOffsetFactor=1,
                                           polygonOffsetUnits=1,
                                           #shininess=0.5,
                                           wireframe=True,
                                           linewidth = 1,
                                           opacity=1,
                                           depthTest=True,
                                           transparent=True)
        
        mesh_ = Mesh(
            geometry=mesh_geometry,
            material=mesh_material,
            position=[0, 0, 0]   # Center in 0
        )
        
        line_ = LineSegments(wireframe,
                             material=LineBasicMaterial(color='black', 
                                                        linewidth = 1, 
                                                        depthTest=True, 
                                                        opacity=1,
                                                        transparent=True), 
                             type = 'LinePieces')


        #aggiunge la mesh alla scena
        self.scene.add(mesh_)
        self.scene.add(line_)
    
    
    def initialize_camera(self, center_target, width, height):
        camera_target = center_target  # the point to look at
        camera_position = [0, 10., 4.] # the camera initial position
        key_light = DirectionalLight(color='#ffffff',position=[0,10,30], intensity=0.5)
        #key_light2 = SpotLight(position=[0, 0, 0], angle = 0.3, penumbra = 0.1, target = tetraObj,castShadow = True)

        camera_t = PerspectiveCamera(
            position=camera_position, lookAt=camera_target, fov=50,
            children=[key_light]
        )
        self.scene = Scene(children=[camera_t, AmbientLight(color='white')], background='#ffffff')
        controls_c = OrbitControls(controlling=camera_t)
        controls_c.enableDamping = False
        controls_c.dumping = 0.01
        controls_c.dampingFactor = 0.1 #friction
        controls_c.rotateSpeed = 0.5 #mouse sensitivity
        controls_c.target = center_target # centro dell'oggetto
        controls_c.zoomSpeed = 0.5
        

        return Renderer(camera=camera_t, background_opacity=1,
                        scene = self.scene, controls=[controls_c], width=width, height=height,antialias=True)