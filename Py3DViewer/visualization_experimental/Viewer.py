import pythreejs as three
import ipywidgets as widgets
from IPython.display import display as ipydisplay

class Viewer(object):

    def __init__(self, geometry):
        super(Viewer, self).__init__()
        self.geometry = geometry
        self.camera = self.__initialize_camera()
        self.scene = self.__initialize_scene()
        self.renderer = self.__initialize_renderer()
    
    def __initialize_camera(self):
        camera_target = [0, 0, 0]
        camera_position = [0, 10., 4.] 
        directional_light = three.DirectionalLight(color = '#ffffff', position = [0,10,0], intensity = 0.5)
        camera = three.PerspectiveCamera(
            position=camera_position, lookAt=camera_target, fov=50, near=.1, far=10000,
            children=[directional_light]
        )
        return camera
    
    def __initialize_scene(self):
        boundaries = self.geometry.boundary()[0]
        geometry_attributes = {
            'position': three.BufferAttribute(self.geometry.vertices[boundaries.flatten()], normalized=False),
            }
        drawable_geometry = three.BufferGeometry(attributes = geometry_attributes)
        drawable_geometry.exec_three_obj_method('computeVertexNormals')
        material = three.MeshLambertMaterial(
                                           polygonOffset=True,
                                           polygonOffsetFactor=1,
                                           polygonOffsetUnits=1,
                                           flatShading = True,
                                           color = "white",
                                           opacity = 1.,
                                           transparent = False,
                                           side = 'FrontSide',
                                           wireframe=False,
                                           vertexColors = 'FaceColors',
                                          )
        drawable_mesh = three.Mesh(
            geometry=drawable_geometry,
            material=material,
            position=[0, 0, 0] 0
            
            
            
            
            
        )
        scene = three.Scene(children=[self.camera, three.AmbientLight(color='white')])
        scene.add(drawable_mesh)
        return scene
    
    def __initialize_renderer(self):
        
        controls = three.OrbitControls(controlling=self.camera)
        controls.enableDamping = False
        controls.damping = 0.01 ##TODO: Check if this is a typo
        controls.dampingFactor = 0.1 #friction
        controls.rotateSpeed = 0.5 #mouse sensitivity
        controls.target = [0, 0, 0] # centro dell'oggetto
        controls.zoomSpeed = 0.5

        return three.Renderer(camera = self.camera, background_opacity=1,
                        scene = self.scene, controls=[controls], width=1000, height=700,
                        antialias=True)

    def show(self):
        ipydisplay(self.renderer)        
    
    def __repr__(self):
        self.show()
        return "Drawing Mesh"