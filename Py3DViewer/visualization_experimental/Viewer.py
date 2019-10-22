import pythreejs as three
import ipywidgets as widgets
from IPython.display import display as ipydisplay
from .Drawable import Drawable

class Viewer(object):

    def __init__(self, geometry, mesh_color = None, width=1000, height=700):
        super(Viewer, self).__init__()
        self.drawable = Drawable(geometry, mesh_color)
        self.camera = self.__initialize_camera(width, height)
        self.scene = self.__initialize_scene()
        self.renderer = self.__initialize_renderer(width, height)
    
    def __initialize_camera(self, width, height):
        camera_target = [0, 0, 0]
        camera_position = [0, 10., 4.] 
        directional_light = three.DirectionalLight(color = '#ffffff', position = [0,10,0], intensity = 0.5)
        camera = three.PerspectiveCamera(
            position=camera_position, aspect=width/height, lookAt=camera_target, fov=50, near=.1, far=10000,
            children=[directional_light]
        )
        return camera
    
    def __initialize_scene(self):
        scene = three.Scene(children=[self.camera, three.AmbientLight(color='white')])
        scene.add(self.drawable.drawable_mesh)
        scene.add(self.drawable.wireframe)
        return scene
    
    def __initialize_renderer(self, width, height):
        controls = three.OrbitControls(controlling=self.camera)
        controls.enableDamping = False
        controls.damping = 0.01 ##TODO: Check if this is a typo
        controls.dampingFactor = 0.1 #friction
        controls.rotateSpeed = 0.5 #mouse sensitivity
        controls.target = [0, 0, 0] # centro dell'oggetto
        controls.zoomSpeed = 0.5

        return three.Renderer(camera = self.camera, background_opacity=1,
                        scene = self.scene, controls=[controls], width=width, height=height,
                        antialias=True)

    def show(self):
        ipydisplay(self.renderer)        
    
    def __repr__(self):
        self.show()
        return "Drawing Mesh"