import numpy as np
from pythreejs import *
import ipywidgets as widgets
import math


class Viewer:
    
    def __init__(self, mesh, UI = True, mesh_color=None):
        
        self.mesh = mesh
        self.scene = None
        if mesh_color is None:
            self.mesh_color = np.array([[0.88, 0.7, 0.94],[0.88, 0.7, 0.94],[0.88, 0.7, 0.94]])
        else:
            self.mesh_color = np.array([mesh_color,mesh_color,mesh_color])
        
        self.mesh_color = np.repeat(self.mesh_color, self.mesh.boundary().shape[0]*3, axis=0)
            
        self.center = list(mesh.vertices[self.mesh.boundary().flatten()].mean(axis = 0))
        
        self.flip_x = False
        self.flip_y = False
        self.flip_z = False
        
        if UI:
            self.__create_UI()
                
    def __create_UI(self):
        """Creates user interface
        """
        # -----------------------   M E N U   S L I C E   -----------------------------

        #style = {width: '50px'}
        #titax = widgets.Label(value='Slice from axes', layout=widgets.Layout(padding='1px 1px 1px 1px', margin='1px 1px 1px 1px'))
        row_layout = {'width':'50px','padding':'1px 1px 1px 1px', 'margin':'1px 1px 1px 1px'}
        wireframe_layout = {'width':'100px','padding':'1px 1px 1px 1px', 'margin':'1px 1px 1px 1px'}
        self.invisibleLayout = {'display':'none'}
        self.visibleLayout = {'display':''}

        self.flip_x = widgets.ToggleButton(
                    value=False,
                    description='Flip x',
                    disabled=False,
                    button_style='info', # 'success', 'info', 'warning', 'danger' or ''
                    tooltip='Flip the visualization range on x axis',
                    icon='check',
                    layout=row_layout
                )

        self.flip_y = widgets.ToggleButton(
                value=False,
                description='Flip y',
                disabled=False,
                button_style='info', # 'success', 'info', 'warning', 'danger' or ''
                tooltip='IFlip the visualization range on y axis',
                icon='check',
                layout=row_layout
            )
        self.flip_z = widgets.ToggleButton(
                value=False,
                description='Flip z',
                disabled=False,
                button_style='info', # 'success', 'info', 'warning', 'danger' or ''
                tooltip='Flip the visualization range on z axis',
                icon='check',
                layout=row_layout
            )
        
        self.percXp = widgets.FloatRangeSlider(
            value=[self.round_down(self.mesh.cut['min_x'],3)-0.001, self.round_up(self.mesh.cut['max_x'],3)+0.001],
            min=self.round_down(self.mesh.cut['min_x'],3)-0.001,
            max=self.round_up(self.mesh.cut['max_x'],3)+0.001,
            step=0.001,
            description='X:',
            disabled=False,
            continuous_update=True,
            orientation='horizontal',
            readout=True,
            readout_format=".3f",
            layout=widgets.Layout(width='30%')

        )


        self.percYp = widgets.FloatRangeSlider(
            value=[self.round_down(self.mesh.cut['min_y'],3)-0.001, self.round_up(self.mesh.cut['max_y'],3)+0.001],
            min=self.round_down(self.mesh.cut['min_y'],3)-0.001,
            max=self.round_up(self.mesh.cut['max_y'],3)+0.001,
            step=0.001,
            description='Y:',
            disabled=False,
            continuous_update=True,
            orientation='horizontal',
            readout=True,
            readout_format=".3f",
            layout=widgets.Layout(width='30%')
        )

        self.percZp = widgets.FloatRangeSlider(
            value=[self.round_down(self.mesh.cut['min_z'],3)-0.001,self.round_up(self.mesh.cut['max_z'],3)+0.001],
            min=self.round_down(self.mesh.cut['min_z'],3)-0.001,
            max=self.round_up(self.mesh.cut['max_z'],3)+0.001,
            step=0.001,
            description='Z:',
            disabled=False,
            continuous_update=True,
            orientation='horizontal',
            readout=True,
            readout_format=".3f",
            layout=widgets.Layout(width='30%')
        )
        
        self.singleColorSurface = widgets.ColorPicker(
            concise=True,
            description='Pick a color',
            value='blue',
            disabled=False
        )
        
        hbox1 = widgets.HBox([self.percXp,self.flip_x])
        hbox2 = widgets.HBox([self.percYp,self.flip_y])
        hbox3 = widgets.HBox([self.percZp,self.flip_z])
        vbox=widgets.VBox([hbox1,hbox2,hbox3],
                     layout=widgets.Layout(width='100%'))
        """
        self.wireSlider = widgets.FloatSlider(
                        value=1,
                        min=0.,
                        max=1.,
                        step=0.1,
                        continuous_update=True,
                        readout_format=".1f",
                        layout=widgets.Layout(width='30%'),
                        description = 'Wireframe',
                        disable = False,
                )

        self.colorWireframe = widgets.ColorPicker(
                            concise=True,
                            description='Color',
                            value='black',
                            disabled=False,
                        )
        
        self.colorMap = widgets.Dropdown(
            options=[('Parula',0),('Jet', 1), ('Red-Blue', 2), ('Virdis', 3)],
            value=0,
            description='Color-Map:',
        )

        self.typeColorSurface = widgets.Dropdown(
            options=[('Hex Quality',0),('Easy', 1),('Label',2)],
            value=0,
            description='Type Color:',
        )
        
        self.colorSurface = widgets.ColorPicker(
                            concise=True,
                            description='Color surface',
                            value='#FF0000',
                            disabled=False,
                            layout= self.invisibleLayout
                        )

        self.colorInside = widgets.ColorPicker(
                            concise=True,
                            description='Color inside',
                            value='#0000FF',
                            disabled=False,
                            layout= self.invisibleLayout
                        )
        
        self.itemsColorsLabel = [widgets.ColorPicker(
                                            concise=True,
                                            description='Label ' + str(self.mesh.labels[i]),
                                            value= self.listColor(int(i)),
                                            disabled=False,
                                            layout= self.invisibleLayout
                                            ) for i in range(len(self.mesh.labels))]
        
        self.sideView = widgets.ToggleButtons(options=['Front', 'Back','Double'],
                            description='View Side:',
                            disabled=False,
                            button_style='info', # 'success', 'info', 'warning', 'danger' or ''
                            tooltips=['View front side of shape', 
                                      'View back side of shape',
                                      'View both front and back side of shape'],
                            #icons=['check'] * 3,
                            style = {'button_width':'65px'}
#                            layout=Layout(width='50', height='80px')                            
                            )
        """
        self.flip_x.observe(self.__update_draw, names='value')
        self.percXp.observe(self.slicing, names='value')
        self.flip_y.observe(self.__update_draw, names='value')
        self.percYp.observe(self.slicing, names='value')
        self.flip_z.observe(self.__update_draw, names='value')
        self.percZp.observe(self.slicing, names='value')

        #self.wireSlider.observe(self.drawWireframeWidth, names='value')
        #self.colorWireframe.observe(self.changeColorWireframe, names='value')
        #self.sideView.observe(self.changeSideView, names='value')
        
        #self.colorMap.observe(self.changeColorMap, names='value')
        #self.colorSurface.observe(self.changeColorSurface, names='value')
        #self.colorInside.observe(self.changeColorInside, names='value')
        #[i.observe(self.changeColorByLabel,names='value') for i in self.itemsColorsLabel]
        
        #self.typeColorSurface.observe(self.changeTypeColor, names='value')
        
    
        #menu slice
        vvbox=widgets.VBox([vbox])
        #menu rendering
        #boxRendering = widgets.HBox([self.wireSlider,self.colorWireframe, self.sideView])
        #boxRendering01 = widgets.HBox([self.typeColorSurface,self.colorMap, self.colorSurface, self.colorInside])
        #boxRendering02 = widgets.HBox(self.itemsColorsLabel)
        #boxRendering1 = widgets.HBox([boxRendering01,boxRendering02])
        #verticalRendering = widgets.VBox([boxRendering,boxRendering1])


        self.accordion = widgets.Accordion(children=[vvbox])
        self.accordion.set_title(0,"Slice from axes")
        #self.accordion.set_title(1,"Rendering")
        display(self.accordion)
        
    
    def listColor(self,n):
        """based on value n return a color

        Parameter:
        ---
            n: int
                number for color
        Return
        ----
            return a color based a number
        """

        if n == 0:
            color = '#ff0000'
        elif n == 1:
            color = '#ffff00'
        elif n == 2:
            color = '#00ffff'
        elif n == 3:
            color = '#ff00ff'
        elif n == 4:
            color = '#0000ff'
        elif n == 5:
            color = '#af0fa0'
        elif n == 6:
            color = '#f0a0f0'
        else:
            color = '#ffffff'
        return color
        
        
        
    def round_up(self,n, decimals=0):

        multiplier = 10 ** decimals
        return math.ceil(n * multiplier) / multiplier        

    def round_down(self,n, decimals=0):
       
        multiplier = 10 ** decimals
        return math.floor(n * multiplier) / multiplier  
        
    
    
    
    
    
    
#============================================================================SHOW===========================================================================================================================  
    
    
    def show(self, width=700, height=700):
        
        
        
        renderer = self.initialize_camera(self.center, width, height)
        
        
        self.__draw(mesh_color)
        
        display(renderer)
        
        
    def __draw(self):
        
        if 'Trimesh' in str(type(self.mesh)) or 'Tetmesh' in str(type(self.mesh)):
            
            self.__draw_trimesh()
            
        if 'Quadmesh' in str(type(self.mesh)) or 'Hexmesh' in str(type(self.mesh)):
            
            self.__draw_quadmesh()
            
    def __update_draw(self):
        
        if 'Trimesh' in str(type(self.mesh)) or 'Tetmesh' in str(type(self.mesh)):
            
            self.__update_draw_tri()
            
        if 'Quadmesh' in str(type(self.mesh)) or 'Hexmesh' in str(type(self.mesh)):
            
            self.__update_draw_quad()
            
    
    def __update_draw_tri(self):
        pass
        
        
    def __update_draw_quad(self):
        
        boundaries = self.mesh.boundary(flip_x=self.flip_x.value, flip_y=self.flip_y.value, flip_z=self.flip_z.value)
        tris = np.c_[boundaries[:,:3], boundaries[:,2:], boundaries[:,0]]
        tris.shape = (-1, 3)
        
        
        quad_properties = {
            'position': BufferAttribute(self.mesh.vertices[tris.flatten()], normalized=False),
            #'index' : BufferAttribute(np.asarray(self.surface, dtype='uint32').ravel(), normalized=False),
            'color' : BufferAttribute(self.mesh_color, normalized=False),
        }
        
        self.mesh_.geometry = BufferGeometry(attributes=quad_properties)
        self.mesh_.geometry.exec_three_obj_method('computeVertexNormals')
    
        edges = np.c_[self.mesh.boundary()[:,:2], self.mesh.boundary()[:,1:3], 
                      self.mesh.boundary()[:,2:4], self.mesh.boundary()[:,3],
                      self.mesh.boundary()[:,0]].flatten()
        
        surface_wireframe = self.mesh.vertices[edges].tolist()
        
        wireframe = BufferGeometry(attributes={'position': BufferAttribute(surface_wireframe, normalized=False)})
        
        self.line_.geometry = wireframe
        
        
        
    
    def __draw_trimesh(self):
        
        
        tri_properties = {
            'position': BufferAttribute(self.mesh.vertices[self.mesh.boundary(flip_x=self.flip_x.value, flip_y=self.flip_y.value, flip_z=self.flip_z.value).flatten()], normalized=False),
            #'index' : BufferAttribute(np.asarray(self.surface, dtype='uint32').ravel(), normalized=False),
            'color' : BufferAttribute(self.mesh_color, normalized=False),
        }
        
        self.mesh_geometry = BufferGeometry(attributes=tri_properties)
        mesh_geometry.exec_three_obj_method('computeVertexNormals')
        
        self.mesh_material = MeshLambertMaterial(#shininess=25,
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
        
        self.edges_material = MeshBasicMaterial(color='black',
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
        
        self.mesh_ = Mesh(
            geometry=mesh_geometry,
            material=mesh_material,
            position=[0, 0, 0]   # Center in 0
        )
        
        self.line_ = Mesh(
            geometry=mesh_geometry,
            material=edges_material,
            position=[0, 0, 0]   # Center in 0
        )


        #aggiunge la mesh alla scena
        self.scene.add(self.mesh_)
        self.scene.add(self.line_)

    def __draw_quadmesh(self):
        
        boundaries = self.mesh.boundary(flip_x=self.flip_x.value, flip_y=self.flip_y.value, flip_z=self.flip_z.value)
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
        
        self.mesh_ = Mesh(
            geometry=mesh_geometry,
            material=mesh_material,
            position=[0, 0, 0]   # Center in 0
        )
        
        self.line_ = LineSegments(wireframe,
                             material=LineBasicMaterial(color='black', 
                                                        linewidth = 1, 
                                                        depthTest=True, 
                                                        opacity=1,
                                                        transparent=True), 
                             type = 'LinePieces')


        #aggiunge la mesh alla scena
        self.scene.add(self.mesh_)
        self.scene.add(self.line_)
        

        
        
#===========================================================================================================================================================================



    def slicing(self,change):
        """Update visible mesh based on value of cut

        Parameter
        ----
            change: not used
        """
        
        self.mesh.set_cut(self.percXp.value[0], self.percXp.value[1], 
                          self.percYp.value[0], self.percYp.value[1],
                          self.percZp.value[0], self.percZp.value[1])
        
        
        self.__update_draw()
    
    
    def initialize_camera(self, center_target, width, height):
        camera_target = center_target  # the point to look at
        camera_position = [0, 10., 4.] # the camera initial position
        key_light = DirectionalLight(color='#ffffff',position=[0,10,30], intensity=0.5)
        #key_light2 = SpotLight(position=[0, 0, 0], angle = 0.3, penumbra = 0.1, target = tetraObj,castShadow = True)

        camera_t = PerspectiveCamera(
            position=camera_position, lookAt=camera_target, fov=50, near=.5, ##careful with this near clipping plane...
            children=[key_light]
        )
        self.scene = Scene(children=[camera_t, AmbientLight(color='white')], background='#ffffff')
        controls_c = OrbitControls(controlling=camera_t)
        controls_c.enableDamping = False
        controls_c.dumping = 0.01 ##TODO: Check if this is a typo
        controls_c.dampingFactor = 0.1 #friction
        controls_c.rotateSpeed = 0.5 #mouse sensitivity
        controls_c.target = center_target # centro dell'oggetto
        controls_c.zoomSpeed = 0.5
        

        return Renderer(camera=camera_t, background_opacity=1,
                        scene = self.scene, controls=[controls_c], width=width, height=height,antialias=True)