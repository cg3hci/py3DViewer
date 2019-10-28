import pythreejs as three
import ipywidgets as widgets
from IPython.display import display as ipydisplay
from .Drawable import Drawable
from .GUI import GUI

class Viewer(object):

    def __init__(self, geometry, mesh_color = None, width=1000, height=700, reactive=False, with_gui=False):
        super(Viewer, self).__init__()
        self.drawable = Drawable(geometry, mesh_color = mesh_color, reactive = reactive)
        self.camera = self.__initialize_camera(width, height)
        self.scene = self.__initialize_scene()
        self.controls = self.__initialize_controls()
        self.renderer = self.__initialize_renderer(width, height)
        if with_gui:
            self.UI = self.__initialize_GUI(self.drawable)
        self.controls.exec_three_obj_method("update")
        
    def __initialize_GUI(self, geometry):
        return GUI(geometry)
        
    def __initialize_camera(self, width, height):
        camera_target = self.drawable.center
        camera_position = tuple(camera_target + [0, 0, self.drawable.scale])
        directional_light = three.DirectionalLight(color = '#ffffff', position = [0, 10, 0], intensity = 0.5)
        camera = three.PerspectiveCamera(
            position=camera_position, aspect=width/height, lookAt=camera_target, fov=50, near=.1, far=10000,
            children=[directional_light]
        )
        return camera
    
    def __initialize_scene(self):
        scene = three.Scene(children=[three.AmbientLight(color='white'),
                                      self.camera,
                                      self.drawable.drawable_mesh,
                                      self.drawable.wireframe])
        return scene
    
    def __initialize_controls(self):
        controls = three.OrbitControls(controlling=self.camera)
        controls.target = tuple(self.drawable.center) # centro dell'oggetto
        return controls
        
    def __initialize_renderer(self, width, height):
        return three.Renderer(camera = self.camera, background_opacity=1,
                        scene = self.scene, controls=[self.controls], width=width, height=height,
                        antialias=True)

    def update(self):
        self.drawable.update()
    
    def show(self):
        ipydisplay(self.renderer)
    
    def __repr__(self):
        self.show()
        return ""