import ipywidgets as widgets
import pythreejs as three
from ..utils import ColorMap, Observer, utilities
from ..visualization import colors
from IPython.display import display as ipydisplay
import threading
from time import sleep
import numpy as np

class GUI(Observer):
    
    def __init__(self, drawable_mesh):
        self.drawable = drawable_mesh
        self.mesh = drawable_mesh.geometry
        self.mesh.attach(self)
        self.widgets = []
        self.click_picker = self.__initialize_picker()
        self.old_picked_face = None
        self.old_picked_face_internal = False
        self.__clipping_in_queue = False
        self.__dont_update_clipping = False
        
        self.__create_UI()
        #self.wireframe_thickness_slider.observe(self.__update_wireframe_thickness, names='value')
        
        for widget in self.widgets:
            ipydisplay(widget)
    
    def __create_UI(self):
        
        self.invisible_layout = {'display':'none', 'height' : 'auto'}
        self.visible_layout = {'display':'', 'height' : 'auto'}
        self.flip_button_layout = {'width': 'auto', 
                                   'margin': '0px 0px 0px 10px'}
        self.picking_info_layout = {'width': 'auto', 
                                   'margin': '50px 50px 50px 0px'}
        self.slider_layout = {
            
        }
        self.flip_x_button = widgets.ToggleButton(
                    value=False,
                    description='Flip x',
                    disabled=False,
                    button_style='info',
                    tooltip='Flip the visualization range on x axis',
                    layout=self.flip_button_layout
                )

        self.flip_y_button = widgets.ToggleButton(
                value=False,
                description='Flip y',
                disabled=False,
                button_style='info', # 'success', 'info', 'warning', 'danger' or ''
                tooltip='IFlip the visualization range on y axis',
                layout=self.flip_button_layout
            )
        
        self.flip_z_button = widgets.ToggleButton(
                value=False,
                description='Flip z',
                disabled=False,
                button_style='info', # 'success', 'info', 'warning', 'danger' or ''
                tooltip='Flip the visualization range on z axis',
                layout=self.flip_button_layout
            )
       
        x_range = self.mesh.bbox[0][0], self.mesh.bbox[1][0]
        x_step = abs(x_range[0]-x_range[1])/100
        self.clipping_slider_x = widgets.FloatRangeSlider(
            value=x_range,
            min=x_range[0]-x_step,
            max=x_range[1]+x_step,
            step=x_step,
            description='X Clipping:',
            disabled=False,
            continuous_update=True,
            orientation='horizontal',
            readout=True,
            readout_format=".1f",
            layout=self.slider_layout
        )


        y_range = self.mesh.bbox[0][1], self.mesh.bbox[1][1]
        y_step = abs(y_range[0]-y_range[1])/100
        self.clipping_slider_y = widgets.FloatRangeSlider(
            value = y_range,
            min=y_range[0]-y_step,
            max=y_range[1]+y_step,
            step=y_step,
            description='Y Clipping:',
            disabled=False,
            continuous_update=True,
            orientation='horizontal',
            readout=True,
            readout_format=".1f",
            layout=self.slider_layout
        )

        z_range = self.mesh.bbox[0][2], self.mesh.bbox[1][2]
        z_step = abs(z_range[0]-z_range[1])/100
        self.clipping_slider_z = widgets.FloatRangeSlider(
            value = z_range,
            min = z_range[0]-z_step,
            max = z_range[1]+z_step,
            step=z_step,
            description='Z Clipping:',
            disabled=False,
            continuous_update=True,
            orientation='horizontal',
            readout=True,
            readout_format=".1f",
            layout=self.slider_layout
        )
        
        self.wireframe_opacity_slider = widgets.FloatSlider(
                        value=0.4,
                        min=0.,
                        max=1.,
                        step=0.1,
                        continuous_update=True,
                        readout_format=".1f",
                        description = 'Wireframe',
                        disable = False,
                )
        
        self.color_wireframe = widgets.ColorPicker(
                            concise=True,
                            value=self.drawable.wireframe.material.color,
                            disabled=False,
                            layout={'margin': '0 0 0 10px'}
                        )
        

        self.enable_picking_button = widgets.ToggleButton(
                value=False,
                description='Show Picking Info',
                disabled=False,
                button_style='info', # 'success', 'info', 'warning', 'danger' or ''
                tooltip='Enable the picking functionality',
                layout=self.picking_info_layout
        )


        self.picking_label = widgets.Label(
            llayout=self.invisible_layout,
            disabled=False,
            continuous_update=True
        )

        tab_titles = ['Poly', 'Vertex', 'Face', 'Edge'] if self.drawable.geometry.mesh_is_volumetric else ['Poly', 'Vertex', 'Edge']
        children = [
            widgets.HTML(
                value="",
                layout={'width': '100','margin': '50 0 0 0'},
                disabled=False,
                continuous_update=True
            ) for title in tab_titles]
        self.picking_tab = widgets.Tab(layout=self.invisible_layout,
                                       disabled=True,
                                       width=300,
                                       height=400)
        self.picking_tab.children = children
        for i in range(len(children)):
            self.picking_tab.set_title(i, tab_titles[i])
        self.color_picking_label = widgets.Label(
            value="Click Color  ",
            layout=self.invisible_layout,
            disabled=False,
            continuous_update=True
        )


        
        self.color_map = widgets.Dropdown(
            options=[(i, idx) for idx, i in enumerate(ColorMap.color_maps.keys())],
            value=0,
            description='Color-Map:',
            layout = self.invisible_layout,
        )
        self.widgets += [self.color_map]
        
        self.metric_menu = widgets.Dropdown(
            options= [(i, idx) for idx, i in enumerate(self.mesh.simplex_metrics.keys())],
            value=0,
            description='Metric:',
            layout = self.invisible_layout,
        )
        self.widgets += [self.metric_menu]

        self.coloring_type_menu = widgets.Dropdown(
            options=[('Default', 0), ('Simplex Quality', 1), ('Label',2)],
            value=0,
            description='Color Type:',
        )
        self.widgets += [
        widgets.HBox([
        
        widgets.VBox([
            widgets.HBox([
                self.clipping_slider_x, self.flip_x_button
            ]),
            widgets.HBox([
                self.clipping_slider_y, self.flip_y_button
            ]),
            widgets.HBox([
                self.clipping_slider_z, self.flip_z_button
            ]),
            widgets.HBox([
                self.wireframe_opacity_slider, self.color_wireframe
            ]),
            
            widgets.HBox([
            self.coloring_type_menu
            ]),

            widgets.VBox([

            widgets.HBox(
                [
                    self.enable_picking_button
                ]
            ),
            widgets.HBox([
                self.picking_label
            ]),
            widgets.HBox([
                self.picking_tab
            ]),

            ])

        ]),

        ]),
       
        ]
                

        mesh_colors = []
        if hasattr(self.mesh, "internals"):
            self.color_internal = widgets.ColorPicker(
                concise=True,
                description='Internal',
                value=colors.rgb2hex(self.drawable._internal_color),
                disabled=False,
            )
            mesh_colors += [self.color_internal]

        #self.color_picking = widgets.ColorPicker(
        #    concise=True,
        #    description="Click Color",
        #    value=colors.rgb2hex(colors.purple),
        #    layout = self.invisible_layout,
        #    disabled=False,
        #)
        self.color_external = widgets.ColorPicker(
            concise=True,
            description='External',
            value=colors.rgb2hex(self.drawable._external_color),
            disabled=False,
        )
        mesh_colors += [self.color_external]
        #mesh_colors += [self.color_picking]
        
        self.widgets += [widgets.HBox(mesh_colors)]
        
        self.color_label_pickers = [widgets.ColorPicker(
                                            concise=True,
                                            description='Label ' + str(i),
                                            value= colors.random_color(return_hex=True),
                                            disabled=False,
                                            layout = self.visible_layout,
                                            ) for i in np.unique(self.mesh.labels)]
        
        self.color_label_pickers = widgets.HBox(self.color_label_pickers, layout=self.invisible_layout)
        self.widgets += [self.color_label_pickers]
        
        
        
        self.flip_x_button.observe(self.__update_clipping, names='value')
        self.flip_y_button.observe(self.__update_clipping, names='value')
        self.flip_z_button.observe(self.__update_clipping, names='value')
        self.clipping_slider_x.observe(self.__update_clipping, names='value')
        self.clipping_slider_y.observe(self.__update_clipping, names='value')
        self.clipping_slider_z.observe(self.__update_clipping, names='value')
        if hasattr(self.mesh, "internals"): 
            self.color_internal.observe(self.__update_internal_color, names='value')
        self.color_external.observe(self.__update_external_color, names='value')
        self.color_wireframe.observe(self.__update_wireframe_color, names='value')
        self.wireframe_opacity_slider.observe(self.__update_wireframe_opacity, names='value')
        self.coloring_type_menu.observe(self.__change_color_type, names='value')
        self.color_map.observe(self.__change_color_map, names='value')
        self.metric_menu.observe(self.__change_metric, names='value')

        self.enable_picking_button.observe(self.__toggle_picking, names='value')
        self.click_picker.observe(self.on_click, names=['point'])

        [i.observe(self.__change_color_label,names='value') for i in self.color_label_pickers.children]

    def __update_wireframe_color(self, change): 
        self.drawable.update_wireframe_color(self.color_wireframe.value)
            
    def __update_wireframe_opacity(self, change): 
        self.drawable.update_wireframe_opacity(self.wireframe_opacity_slider.value)
        
    def __update_internal_color(self, change): 
        if hasattr(self.mesh, "internals"): 
            self.drawable.update_internal_color(colors.hex2rgb(self.color_internal.value))
            
    def __update_external_color(self, change): 
        self.drawable.update_external_color(colors.hex2rgb(self.color_external.value))
        
    def __change_color_type(self, change):
        
        if self.coloring_type_menu.value == 0:
            
            self.color_map.layout = self.invisible_layout
            self.metric_menu.layout = self.invisible_layout
            self.color_external.layout = self.visible_layout
            if hasattr(self.mesh, "internals"):
                self.color_internal.layout = self.visible_layout
            self.color_label_pickers.layout = self.invisible_layout
            self.drawable._label_colors = None
            self.drawable._color_map = None
            self.__update_external_color(None)
            self.__update_internal_color(None)
            
        elif self.coloring_type_menu.value == 1:
        
            self.color_map.layout = self.visible_layout
            self.metric_menu.layout = self.visible_layout
            self.color_external.layout = self.invisible_layout
            if hasattr(self.mesh, "internals"):
                self.color_internal.layout = self.invisible_layout
            self.color_label_pickers.layout = self.invisible_layout
            self.drawable._label_colors = None
            self.__change_color_map(None)
        
        elif self.coloring_type_menu.value == 2:
            
            self.color_external.layout = self.invisible_layout
            if hasattr(self.mesh, "internals"):
                self.color_internal.layout = self.invisible_layout
            self.color_map.layout = self.invisible_layout
            self.metric_menu.layout = self.invisible_layout
            
            if self.mesh.labels is not None:
            
                self.color_label_pickers.layout = self.visible_layout
                self.__change_color_label(None)

    def __initialize_picker(self):
        pickable_objects = self.drawable.mesh
        picker = three.Picker(controlling=pickable_objects, event='click')
        return picker
        
    def __change_metric(self, change):
        
        self.__change_color_map(None)
        
    def __change_color_label(self, change):
        
        self.drawable._label_colors = {int(i.description.split()[1]): colors.hex2rgb(i.value) for i in self.color_label_pickers.children}
        
        self.drawable.update_color_label()
    
    def __change_color_map(self, change):
        
        metric_string = list(self.mesh.simplex_metrics.keys())[self.metric_menu.value]
        
        c_map_string = list(ColorMap.color_maps.keys())[self.color_map.value]
         
        self.drawable.compute_color_map(metric_string, c_map_string)
        
    def __clipping_updater(self):
            
        self.__dont_update_clipping = True
        flip_x = self.flip_x_button.value
        flip_y = self.flip_y_button.value
        flip_z = self.flip_z_button.value
        min_x, max_x = self.clipping_slider_x.value
        min_y, max_y = self.clipping_slider_y.value
        min_z, max_z = self.clipping_slider_z.value
        self.mesh.set_clipping(min_x = min_x, max_x = max_x, 
                               min_y = min_y, max_y = max_y, 
                               min_z = min_z, max_z = max_z,
                               flip_x = flip_x, flip_y = flip_y, flip_z = flip_z)
        if self.__clipping_in_queue:
            self.__clipping_in_queue = False
            self.__dont_update_clipping = False
            self.__update_clipping(None)
        else:
            self.__dont_update_clipping = False

    def on_click(self, change):

        if not self.enable_picking_button.value:
            return

        geometry_type = str(type(self.drawable.geometry))

        # if nothing is clicked
        if change.owner.object is None:
            self.picking_label.value = "Nothing found"
            self.picking_tab.children[0].value = ' '
            self.picking_tab.children[1].value = ' '
        else:
            # click_operations is called based on the number of triangles per poly of the geometry
            if "Quadmesh" in geometry_type:
                self.click_operations(change, 2)
            elif "Tetmesh" in geometry_type:
                self.click_operations(change, 4)
            elif "Hexmesh" in geometry_type:
                self.click_operations(change, 12)
            else:
                # Trimesh
                self.click_operations(change, 1)

    def click_operations(self, change, num_triangles):
        poly_index = change.owner.faceIndex // num_triangles
        coords = change['new']
        
        internal = False

        if num_triangles <= 2:
            vids = self.drawable.geometry.polys[poly_index].astype("int32")
        elif num_triangles == 4:
            poly_index = poly_index + self.drawable.geometry._map_poly_indices[poly_index]
            if self.drawable.geometry.internals[poly_index]:
                internal = True
            vids = self.drawable.geometry.polys[poly_index].astype("int32")
        else:
            poly_index = poly_index + self.drawable.geometry._map_poly_indices[poly_index]
            if self.drawable.geometry.internals[poly_index]:
                internal = True
            vids = self.drawable.geometry.polys[poly_index].astype("int32")
        

        vertex_coords = self.drawable.geometry.vertices[vids]
        nearest_vertex, nearest_vertex_coords = self.find_nearest_vertex(vids, vertex_coords, change.owner.point)
        

        nearest_polys = np.array(self.drawable.geometry.adj_vtx2poly[nearest_vertex]).astype("int32")
        
        self.picking_label.value = 'Clicked on (%.3f, %.3f, %.3f)' % tuple(coords)
        self.picking_tab.children[0].value = 'Poly index: %d' % poly_index + '<br>'
        self.picking_tab.children[0].value += 'Vertex indices: '
        self.picking_tab.children[0].value += ', '.join([str(v) for v in vids]) + '<br>'
        self.picking_tab.children[0].value += ''.join(
            'Vertex ' + str(a) + ' coords: (%.3f, %.3f, %.3f)' % tuple(b) + '<br>' for a, b in
            zip(vids, vertex_coords))

        self.picking_tab.children[1].value = 'Vertex index: %d' % nearest_vertex + '<br>'
        self.picking_tab.children[1].value += 'Vertex coords: (%.3f, %.3f, %.3f)' % tuple(nearest_vertex_coords) + '<br>'
        self.picking_tab.children[1].value += 'Nearest Polys: '
        self.picking_tab.children[1].value += ', '.join([str(v) for v in nearest_polys]) + '<br>'


        #if self.old_picked_face is not None:
        #    self.__change_color_type(None)
           
        self.old_picked_face = poly_index
        self.old_picked_face_internal = internal
        #self.drawable.update_poly_color(colors.hex2rgb(self.color_picking.value), poly_index=poly_index,num_triangles=num_triangles)

        

        
    
    def __toggle_picking(self, change):

        if self.enable_picking_button.value:
            self.picking_tab.layout = {'margin': '0 0 0 20px'}
            self.picking_label.layout = {'margin': '0 0 0 20px'}
            self.enable_picking_button.description = 'Hide Picking Info'
            #self.color_picking.layout = self.visible_layout
        else:
            self.picking_tab.layout = self.invisible_layout
            self.picking_label.layout = self.invisible_layout
            #self.color_picking.layout = self.invisible_layout
            self.enable_picking_button.description = 'Show Picking Info'
            self.__change_color_type(None)



    def __update_clipping(self, change): 
       
        if self.__dont_update_clipping:
            self.__clipping_in_queue = True
        else:
            thread = threading.Thread(target=self.__clipping_updater, args=())
            thread.daemon = True
            thread.start()

    def update(self):
        clipping = self.mesh.clipping
        flips = clipping.flip
        flip_x = flips.x
        flip_y = flips.y
        flip_z = flips.z
        x_range = clipping.min_x, clipping.max_x
        y_range = clipping.min_y, clipping.max_y
        z_range = clipping.min_z, clipping.max_z
        
        self.flip_x_button.value = flip_x
        self.flip_y_button.value = flip_y
        self.flip_z_button.value = flip_z
        self.clipping_slider_x.value = x_range
        self.clipping_slider_y.value = y_range
        self.clipping_slider_z.value = z_range

    @staticmethod
    def find_nearest_vertex(vertices, vertex_coords, click_coords):
        click_coords = np.array(click_coords).reshape(-1,3)
        click_coords = np.repeat(click_coords, vertices.size, axis=0)
        dist = np.linalg.norm(click_coords-vertex_coords, axis=1)
        idx_arg_min = np.argmin(dist)
        return vertices[idx_arg_min], vertex_coords[idx_arg_min]