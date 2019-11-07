import pythreejs as three
import ipywidgets as widgets
import numpy as np
from IPython.display import display as ipydisplay
from .DrawableMesh import DrawableMesh
from .DrawableSkeleton import DrawableSkeleton
from .GUI import GUI

class Viewer(object):

    def __init__(self, geometries, mesh_color = None, width=1000, height=700, reactive=False, with_gui=False):
        super(Viewer, self).__init__()
        self.drawables=[]
        if type(geometries) is not list:
            self.drawables += [self.__get_drawable_from_geometry(geometries, mesh_color, reactive or with_gui)]
        else:
            self.drawables = [self.__get_drawable_from_geometry(geometry, mesh_color, reactive or with_gui) for geometry in geometries]
        self.camera = self.__initialize_camera(width, height)
        self.scene = self.__initialize_scene()
        self.controls = self.__initialize_controls()
        self.renderer = self.__initialize_renderer(width, height)
        if with_gui:
            if len(self.drawables) > 1:
                print("ERROR: GUI only supports one geometry at a time, so far.")
            else:
                self.UI = self.__initialize_GUI(self.drawables[0])
        self.controls.exec_three_obj_method("update")
        
    
    def __get_drawable_from_geometry(self, geometry, color, reactive):
        geometry_type = str(type(geometry))
        if "mesh" in geometry_type: #TODO: Find a more clever way of doing this
            return DrawableMesh(geometry, mesh_color = color, reactive = reactive)
        elif "Skeleton" in geometry_type:
            return DrawableSkeleton(geometry, skeleton_color = color, reactive = reactive)
        
    def __initialize_GUI(self, geometry):
        return GUI(geometry)
        
    def __initialize_camera(self, width, height):
        camera_target = np.mean([drawable.center for drawable in self.drawables], axis=0)
        camera_position = tuple(camera_target + [0, 0, np.mean([drawable.scale for drawable in self.drawables])])
        directional_light = three.DirectionalLight(color = '#ffffff', position = [0, 10, 0], intensity = 1)
        camera = three.PerspectiveCamera(
            position=camera_position, aspect=width/height, lookAt=camera_target, fov=50, near=.1, far=10000,
            children=[directional_light]
        )
        return camera
    
    def __initialize_scene(self):
        threejs_items = []
        for drawable in self.drawables:
            threejs_items+=drawable.threejs_items
        scene = three.Scene(children=[three.AmbientLight(color='white', intensity=1),
                                      self.camera,
                                      *threejs_items
                                      ])
        return scene
    
    def __initialize_controls(self):
        controls = three.OrbitControls(controlling=self.camera)
        controls.target = tuple(np.mean([drawable.center for drawable in self.drawables], axis=0))
        return controls
        
    def __initialize_renderer(self, width, height):
        return three.Renderer(camera = self.camera, background_opacity=1,
                        scene = self.scene, controls=[self.controls], width=width, height=height,
                        antialias=True)

    def update(self):
        [drawable.update() for drawable in self.drawables]
    
    def show(self):
        ipydisplay(self.renderer)
    
    def __repr__(self):
        self.show()
        return ""