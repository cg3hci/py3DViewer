import numpy as np
from pythreejs import *
from ipywidgets import Layout
import ipywidgets as widgets
import math
from ..utils import ColorMap

class Viewer:
    
    def __init__(self, mesh, UI = True, mesh_color=None, width=700, height=700):
        
        self.mesh = mesh
        self.scene = None
        self.__UI = UI
        self.width = widt
        self.height = height
        
        if 'Skeleton' in str(type(mesh)):
            self.skel = mesh
            self.center = list(self.skel.nodes.mean(axis=0))
            if hasattr(self.skel, 'associated_mesh'):
                self.mesh = self.skel.associated_mesh
           
        if 'Skeleton' not in str(type(self.mesh)):
            if mesh_color is None:
                self.mesh_color = np.array([[1, 212, 180],[1, 212, 180],[1, 212, 180]], dtype=np.float) / 255
            else:
                self.mesh_color = np.array([mesh_color,mesh_color,mesh_color])

            if 'Hexmesh' in str(type(self.mesh)) or 'Quadmesh' in str(type(self.mesh)):
                self.mesh_color = np.repeat(self.mesh_color, self.mesh.num_faces*2, axis=0)
            else:
                self.mesh_color = np.repeat(self.mesh_color, self.mesh.num_faces, axis=0)


            self.center = list(self.mesh.vertices[self.mesh.boundary()[0].flatten()].mean(axis = 0))

            self.flip_x_value = False
            self.flip_y_value = False
            self.flip_z_value = False

            if UI:
                self.__create_UI()
        

            
        
                
    def __create_UI(self):
        """Creates user interface
        """
        # -----------------------   M E N U   S L I C E   -----------------------------

        #style = {width: '50px'}
        #titax = widgets.Label(value='Slice from axes', layout=widgets.Layout(padding='1px 1px 1px 1px', margin='1px 1px 1px 1px'))
        row_layout = {'width':'100px', 'padding':'1px 1px 1px 1px', 'margin':'1px 1px 1px 1px'}
        wireframe_layout = {'width':'100px','padding':'1px 1px 1px 1px', 'margin':'1px 1px 1px 1px'}
 
        
        
        self.invisibleLayout = {'display':'none'}
        self.visibleLayout = {'display':''}
        self.label_layout = {'display':'block', 'max_width' : '80px'}

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
        
        self.clipping_slider_x = widgets.FloatRangeSlider(
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


        self.clipping_slider_y = widgets.FloatRangeSlider(
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

        self.clipping_slider_z = widgets.FloatRangeSlider(
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
        
        self.external_color = widgets.ColorPicker(
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
        
        self.wireframe_thickness_slider = widgets.FloatSlider(
                        value=0.2,
                        min=0.,
                        max=1.,
                        step=0.1,
                        continuous_update=True,
                        readout_format=".1f",
                        layout=widgets.Layout(width='30%'),
                        description = 'Wireframe',
                        disable = False,
                )

        self.wireframe_color_picker = widgets.ColorPicker(
                            concise=True,
                            description='Color',
                            value='#686868',
                            disabled=False,
                        )
        
        self.color_map = widgets.Dropdown(
            options=[(i, idx) for idx, i in enumerate(ColorMap.color_maps.keys())],
            value=0,
            description='Color-Map:',
            layout=self.invisibleLayout
        )

        self.coloring_type_menu = widgets.Dropdown(
            options=[('Default', 0), ('Simplex Quality', 1), ('Label',2)],
            value=0,
            description='Type Color:',
        )
        
        self.metric_menu = widgets.Dropdown(
            options= [(i, idx) for idx, i in enumerate(self.mesh.simplex_metrics.keys())],
            value=0,
            description='Metric:',
            layout=self.invisibleLayout
        )
        
        self.color_internal = widgets.ColorPicker(
                            concise=True,
                            description='Color inside',
                            value='#FF9C00',
                            disabled=False,
                        )
        
        self.color_label_pickers = [widgets.ColorPicker(
                                            concise=True,
                                            description='Label ' + str(i),
                                            value= self.listColor(int(i)),
                                            disabled=False,
                                            layout= self.invisibleLayout
                                            ) for i in range(len(np.unique(self.mesh.labels)))]
        
        
        self.flip_x.observe(self.__slicing, names='value')
        self.percXp.observe(self.__slicing, names='value')
        self.flip_y.observe(self.__slicing, names='value')
        self.percYp.observe(self.__slicing, names='value')
        self.flip_z.observe(self.__slicing, names='value')
        self.percZp.observe(self.__slicing, names='value')

        self.wireSlider.observe(self.__set_wireframe_width, names='value')
        self.colorWireframe.observe(self.__set_wireframe_color, names='value')
        
        self.colorMap.observe(self.change_color_map, names='value')
        self.colorSurface.observe(self.change_color_surface, names='value')
        self.colorInside.observe(self.change_color_inside, names='value')
        self.chosen_metric.observe(self.change_color_map, names='value')
        [i.observe(self.change_color_label,names='value') for i in self.itemsColorsLabel]
        
        self.typeColorSurface.observe(self.change_type_color, names='value')
        
    
        #menu slice
        vvbox=widgets.VBox([vbox], layout={'height': '100px'})
        #menu rendering
        box_rendering = widgets.HBox([self.wireSlider,self.colorWireframe], layout={'height': '100px'})
        box_rendering01 = widgets.HBox([self.colorSurface], layout={'height': '100px'})
        if 'Hexmesh' in str(type(self.mesh)) or 'Tetmesh' in str(type(self.mesh)):
            box_rendering01 = widgets.HBox([self.typeColorSurface,self.colorMap, self.chosen_metric, self.colorSurface, self.colorInside] + self.itemsColorsLabel, layout={'height': '130px'})
        else:
            box_rendering01 = widgets.HBox([self.typeColorSurface,self.colorMap, self.chosen_metric, self.colorSurface] + self.itemsColorsLabel, layout={'height': '100px'})
        #boxRendering02 = widgets.HBox(self.itemsColorsLabel)
        #boxRendering1 = widgets.HBox([boxRendering01,boxRendering02])
        vertical_rendering = widgets.VBox([box_rendering, box_rendering01], layout={'height': '130px'})


        self.accordion = widgets.Accordion(children=[vvbox, vertical_rendering])
        self.accordion.set_title(0,"Slice from axes")
        self.accordion.set_title(1,"Rendering")
        display(self.accordion)
        
    
    def __set_wireframe_color(self, change=None):
        
        self.line_.material.color = self.colorWireframe.value
        
    def __set_wireframe_width(self, change=None):
        
        self.line_.material.opacity = self.wireSlider.value
        
    
    def listColor(self,n):
       
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
        
    
    def change_color_label(self, change=None):
        
        if self.mesh_color.shape[0] != self.mesh.labels.shape[0]:
            self.mesh_color = np.zeros((self.mesh.labels.shape[0], 3))
        
        for idx, color in enumerate(self.itemsColorsLabel):
            self.mesh_color[self.mesh.labels == idx] = [int(color.value[1:3],16)/255,int(color.value[3:5],16)/255,int(color.value[5:7],16)/255]
        
        if 'Hexmesh' in str(type(self.mesh)):
            self.mesh_color = np.repeat(self.mesh_color, 6*2*3, axis=0)
        elif 'Quadmesh' in str(type(self.mesh)):
            self.mesh_color = np.repeat(self.mesh_color, 2*3, axis=0)
        elif 'Tetmesh' in str(type(self.mesh)):
            self.mesh_color = np.repeat(self.mesh_color, 4*3, axis=0)
        else:
            self.mesh_color = np.repeat(self.mesh_color, 3, axis=0)        
        
        self.__update_draw()
        

    def change_color_surface(self, change=None):
        
        faces_per_poly = 0
        faces_in_face  = 2
        if 'Tetmesh' in str(type(self.mesh)):
            faces_per_poly = 4
            faces_in_face  = 1
        elif 'Hexmesh' in str(type(self.mesh)):
            faces_per_poly = 6
        elif 'Quadmesh' in str(type(self.mesh)):
            faces_per_poly = 1
        elif 'Trimesh' in str(type(self.mesh)):
            faces_per_poly = 1
            faces_in_face  = 1
        
        mesh_color = [int(self.colorSurface.value[1:3],16)/255,int(self.colorSurface.value[3:5],16)/255,int(self.colorSurface.value[5:7],16)/255]
        if 'Trimesh' in str(type(self.mesh)) or 'Quadmesh' in str(type(self.mesh)):
            indices = np.repeat(self.mesh.boundary()[1], faces_per_poly*faces_in_face*3)
            
        else:
            indices = np.logical_not(np.repeat(self.mesh.internals, faces_per_poly*faces_in_face*3))
        
        self.mesh_color[indices] = np.array(mesh_color)
        self.__update_draw()
        
        
        
    def change_color_inside(self, change=None):
        
        faces_per_poly = 0
        faces_in_face  = 2
        if 'Tetmesh' in str(type(self.mesh)):
            faces_per_poly = 4
            faces_in_face  = 1
            
        elif 'Hexmesh' in str(type(self.mesh)):
            faces_per_poly = 6
            
        mesh_color = [int(self.colorInside.value[1:3],16)/255,int(self.colorInside.value[3:5],16)/255,int(self.colorInside.value[5:7],16)/255]
        indices = np.repeat(self.mesh.internals, faces_per_poly*faces_in_face*3)
        self.mesh_color[indices] = np.array(mesh_color)
        self.__update_draw()
        

    def change_side_view(self,change=None):
        if change.new == 'Front':
            self.view_fromt_side()
        elif change.new == 'Back':
            self.view_back_side()
        elif change.new == 'Double':
            self.view_double_side()
            


    def change_color_map(self, change=None):
        
        metric_keys = list(self.mesh.simplex_metrics.keys())
        metric_idx = metric_keys[self.chosen_metric.value]
        metric = self.mesh.simplex_metrics[metric_idx][1]

        color_map_keys = list(ColorMap.color_maps.keys())
        color_map_idx = color_map_keys[self.colorMap.value]
        color_map = ColorMap.color_maps[color_map_idx]
        
        min_range = self.mesh.simplex_metrics[metric_idx][0][0]
        max_range = self.mesh.simplex_metrics[metric_idx][0][1]
        if ( min_range is None or max_range is None):
            min_range = np.min(metric)
            max_range = np.max(metric)
            if (np.abs(max_range-min_range) > 1e-7):
                normalized_metric = ((metric - np.min(metric))/np.ptp(metric)) * (color_map.shape[0]-1)
            else:
                normalized_metric = np.repeat(np.mean(metric), metric.shape[0])
        else:
            normalized_metric = np.clip(metric, min_range, max_range)
            normalized_metric = (normalized_metric - min_range)/(max_range-min_range) * (color_map.shape[0]-1)
            
        normalized_metric = 1-normalized_metric
            
        metric_to_colormap = np.rint(normalized_metric).astype(np.int)

        mesh_color = color_map[metric_to_colormap]
        
        if 'Hexmesh' in str(type(self.mesh)):
            self.mesh_color = np.repeat(mesh_color, 6*2*3, axis=0)
        elif 'Quadmesh' in str(type(self.mesh)):
            self.mesh_color = np.repeat(mesh_color, 2*3, axis=0)
        elif 'Tetmesh' in str(type(self.mesh)):
            self.mesh_color = np.repeat(mesh_color, 4*3, axis=0)
        else:
            self.mesh_color = np.repeat(mesh_color, 3, axis=0)        

        self.__update_draw()

        
    def change_type_color(self,change=None):
        
        if self.typeColorSurface.value==1:
            self.colorSurface.layout = self.invisibleLayout
            self.colorInside.layout = self.invisibleLayout
            self.colorMap.layout = self.visibleLayout
            self.chosen_metric.layout = self.visibleLayout
            for i in self.itemsColorsLabel:
                i.layout = self.invisibleLayout
            self.change_color_map()
        elif self.typeColorSurface.value==0:
            self.colorInside.layout = self.visibleLayout
            self.colorSurface.layout = self.visibleLayout
            self.colorMap.layout = self.invisibleLayout
            self.chosen_metric.layout = self.invisibleLayout
            for i in self.itemsColorsLabel:
                i.layout = self.invisibleLayout
            self.change_color_surface()
            self.change_color_inside()
        elif self.typeColorSurface.value==2:
            self.colorInside.layout = self.invisibleLayout
            self.colorSurface.layout = self.invisibleLayout
            self.colorMap.layout = self.invisibleLayout
            self.chosen_metric.layout = self.invisibleLayout
            for i in self.itemsColorsLabel:
                i.layout = self.label_layout
            self.change_color_label()
    
    
    
#============================================================================SHOW===========================================================================================================================  
    
    
    def show(self):
        
        
        
        renderer = self.initialize_camera(self.center, self.width, self.height)
        
        self.__draw()
        
        if self.__UI :
            if 'Tetmesh' in str(type(self.mesh)) or 'Hexmesh' in str(type(self.mesh)):
                self.change_color_inside()
            self.__set_wireframe_width()
            self.__set_wireframe_color()
        
        display(renderer)
        
        
    def __draw(self):
        
        if hasattr(self, 'skel'):
            
            self.__draw_skeleton()
        
        elif 'Trimesh' in str(type(self.mesh)) or 'Tetmesh' in str(type(self.mesh)):
            
            self.__draw_trimesh()
            
        elif 'Quadmesh' in str(type(self.mesh)) or 'Hexmesh' in str(type(self.mesh)):
            
            self.__draw_quadmesh()
            
        
            
    def __update_draw(self):
        
        if 'Trimesh' in str(type(self.mesh)) or 'Tetmesh' in str(type(self.mesh)):
            
            self.__update_draw_tri()
            
        if 'Quadmesh' in str(type(self.mesh)) or 'Hexmesh' in str(type(self.mesh)):
            
            self.__update_draw_quad()
            
    
    def __update_draw_tri(self):
        
        boundaries = self.mesh.boundary(flip_x=self.flip_x_value, flip_y=self.flip_y_value, flip_z=self.flip_z_value)[0]
        tris_properties = {
            'position': BufferAttribute(self.mesh.vertices[boundaries.flatten()], normalized=False),
            #'index' : BufferAttribute(np.asarray(self.surface, dtype='uint32').ravel(), normalized=False),
            'color' : BufferAttribute(self.mesh_color[np.repeat(self.mesh.boundary()[1], 3)], normalized=False),
        }
        self.mesh_.geometry = BufferGeometry(attributes=tris_properties)
        self.mesh_.geometry.exec_three_obj_method('computeVertexNormals')
        
        self.line_.geometry = self.mesh_.geometry
        
        
    def __update_draw_quad(self):
        
        boundaries = self.mesh.boundary(flip_x=self.flip_x_value, flip_y=self.flip_y_value, flip_z=self.flip_z_value)[0]
        tris = np.c_[boundaries[:,:3], boundaries[:,2:], boundaries[:,0]]
        tris.shape = (-1, 3)
        
        
        quad_properties = {
            'position': BufferAttribute(self.mesh.vertices[tris.flatten()], normalized=False),
            #'index' : BufferAttribute(np.asarray(self.surface, dtype='uint32').ravel(), normalized=False),
            'color' : BufferAttribute(self.mesh_color[np.repeat(self.mesh.boundary()[1], 2*3)], normalized=False),
        }
        
        self.mesh_.geometry = BufferGeometry(attributes=quad_properties)
        self.mesh_.geometry.exec_three_obj_method('computeVertexNormals')
    
        edges = np.c_[boundaries[:,:2], boundaries[:,1:3], 
                      boundaries[:,2:4], boundaries[:,3],
                      boundaries[:,0]].flatten()
        
        surface_wireframe = self.mesh.vertices[edges].tolist()
        
        wireframe = BufferGeometry(attributes={'position': BufferAttribute(surface_wireframe, normalized=False)})
        
        self.line_.geometry = wireframe
        
    

    
    def __draw_skeleton(self):
        
    
        bonesGeometry = BufferGeometry(attributes={'position': BufferAttribute(self.skel.nodes[self.skel.bones], normalized=False)})
            
        bones = LineSegments( bonesGeometry, 
            MeshBasicMaterial(color = '#FF9C00' ))
            
        self.scene.add(bones)
        
        for point, radius in zip(self.skel.nodes, self.skel.radius):
            nodeGeometry = SphereGeometry(radius=radius/100)
            sphere = Mesh(nodeGeometry, 
            MeshBasicMaterial(color = '#ff0000' ))
            sphere.position = list(point)
            self.scene.add(sphere)
            
        
        if hasattr(self.skel, 'associated_mesh'):
            
            self.__draw_trimesh()
            self.mesh_.material.transparent = True
            self.mesh_.material.opacity = 0.05
            self.line_.material.opacity = 0.1
            
            self.__update_draw()
    
    def __draw_trimesh(self):
        
        
        tri_properties = {
            'position': BufferAttribute(self.mesh.vertices[self.mesh.boundary(flip_x=self.flip_x_value, flip_y=self.flip_y_value, flip_z=self.flip_z_value)[0].flatten()], normalized=False),
            #'index' : BufferAttribute(np.asarray(self.surface, dtype='uint32').ravel(), normalized=False),
            'color' : BufferAttribute(self.mesh_color[np.repeat(self.mesh.boundary()[1], 3)], normalized=False),
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
                                           opacity = 1.,
                                           transparent = False,
                                           side = 'FrontSide',
                                           #color = '#550000',
                                           wireframe=False,
                                           vertexColors = 'FaceColors',
                                          )
        
        edges_material = MeshBasicMaterial(color='#686868',
#                                           side= 'FrontSide'
                                           polygonOffset=True,
                                           polygonOffsetFactor=1,
                                           polygonOffsetUnits=1,
                                           #shininess=0.5,
                                           wireframe=True,
                                           linewidth = 1,
                                           opacity=0.2,
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
        
        boundaries = self.mesh.boundary(flip_x=self.flip_x_value, flip_y=self.flip_y_value, flip_z=self.flip_z_value)[0]
        tris = np.c_[boundaries[:,:3], boundaries[:,2:], boundaries[:,0]]
        tris.shape = (-1, 3)
        
        
        quad_properties = {
            'position': BufferAttribute(self.mesh.vertices[tris.flatten()], normalized=False),
            #'index' : BufferAttribute(np.asarray(self.surface, dtype='uint32').ravel(), normalized=False),
            'color' : BufferAttribute(self.mesh_color[np.repeat(self.mesh.boundary()[1], 2*3)], normalized=False),
        }
        
        mesh_geometry = BufferGeometry(attributes=quad_properties)
        mesh_geometry.exec_three_obj_method('computeVertexNormals')
        
        
        edges = np.c_[boundaries[:,:2], boundaries[:,1:3], boundaries[:,2:4], boundaries[:,3], boundaries[:,0]].flatten()
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
        
        edges_material = MeshBasicMaterial(color='#686868',
#                                           side= 'FrontSide'
                                           polygonOffset=True,
                                           polygonOffsetFactor=1,
                                           polygonOffsetUnits=1,
                                           #shininess=0.5,
                                           wireframe=True,
                                           linewidth = 1,
                                           opacity=0.2,
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



    def __slicing(self,change):
               
        self.mesh.set_cut(self.percXp.value[0], self.percXp.value[1], 
                          self.percYp.value[0], self.percYp.value[1],
                          self.percZp.value[0], self.percZp.value[1])
        
        self.flip_x_value = self.flip_x.value
        self.flip_y_value = self.flip_y.value
        self.flip_z_value = self.flip_z.value


        self.__update_draw()
    
    
    def initialize_camera(self, center_target, width, height):
        camera_target = center_target  # the point to look at
        camera_position = [0, 10., 4.] # the camera initial position
        key_light = DirectionalLight(color='#ffffff',position=[0,10,0], intensity=0.5)
        #key_light2 = SpotLight(position=[0, 0, 0], angle = 0.3, penumbra = 0.1, target = tetraObj,castShadow = True)

        camera_t = PerspectiveCamera(
            position=camera_position, lookAt=camera_target, fov=50, near=.1, far=10000, ##careful with this near clipping plane...
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
    
    def __repr__(self):
        
        self.show()
        return "Displaying Mesh"
    
    
