import ipywidgets as widgets
from ..utils import ColorMap, Observer
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
        self.__clipping_in_queue = False
        self.__dont_update_clipping = False
        
        self.invisible_layout = {'display':'none'}
        self.visible_layout = {'display':''}
        
        self.flip_x_button = widgets.ToggleButton(
                    value=False,
                    description='Flip x',
                    disabled=False,
                    button_style='info',
                    tooltip='Flip the visualization range on x axis',
                    icon='check'
                )
        self.widgets += [self.flip_x_button]

        self.flip_y_button = widgets.ToggleButton(
                value=False,
                description='Flip y',
                disabled=False,
                button_style='info', # 'success', 'info', 'warning', 'danger' or ''
                tooltip='IFlip the visualization range on y axis',
                icon='check',
            )
        self.widgets += [self.flip_y_button]
        
        self.flip_z_button = widgets.ToggleButton(
                value=False,
                description='Flip z',
                disabled=False,
                button_style='info', # 'success', 'info', 'warning', 'danger' or ''
                tooltip='Flip the visualization range on z axis',
                icon='check',
            )
        self.widgets += [self.flip_z_button]
       
        x_range = self.mesh.bbox[0][0], self.mesh.bbox[1][0]
        self.clipping_slider_x = widgets.FloatRangeSlider(
            value=x_range,
            min=x_range[0],
            max=x_range[1],
            step=abs(x_range[0]-x_range[1])/100,
            description='X:',
            disabled=False,
            continuous_update=True,
            orientation='horizontal',
            readout=True,
            readout_format=".2f",
        )
        self.widgets += [self.clipping_slider_x]


        y_range = self.mesh.bbox[0][1], self.mesh.bbox[1][1]
        self.clipping_slider_y = widgets.FloatRangeSlider(
            value = y_range,
            min = y_range[0],
            max = y_range[1],
            step=abs(y_range[0]-y_range[1])/100,
            description='Y:',
            disabled=False,
            continuous_update=True,
            orientation='horizontal',
            readout=True,
            readout_format=".2f",
        )
        self.widgets += [self.clipping_slider_y]

        z_range = self.mesh.bbox[0][2], self.mesh.bbox[1][2]
        self.clipping_slider_z = widgets.FloatRangeSlider(
            value = z_range,
            min = z_range[0],
            max = z_range[1],
            step=abs(z_range[0]-z_range[1])/100,
            description='Z:',
            disabled=False,
            continuous_update=True,
            orientation='horizontal',
            readout=True,
            readout_format=".2f",
        )
        self.widgets += [self.clipping_slider_z]
        
        self.wireframe_opacity_slider = widgets.FloatSlider(
                        value=0.4,
                        min=0.,
                        max=1.,
                        step=0.1,
                        continuous_update=True,
                        readout_format=".1f",
                        description = 'Wireframe Opacity',
                        disable = False,
                )
        self.widgets += [self.wireframe_opacity_slider]

        self.color_wireframe = widgets.ColorPicker(
                            concise=True,
                            description='Wireframe Color',
                            value=self.drawable.wireframe.material.color,
                            disabled=False,
                        )
        self.widgets += [self.color_wireframe]
        
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
            description='Type Color:',
        )
        self.widgets += [self.coloring_type_menu]
        

        
        self.color_internal = widgets.ColorPicker(
                            concise=True,
                            description='Internal Color',
                            value=colors.rgb2hex(self.drawable._internal_color),
                            disabled=False,
        )
        self.widgets += [self.color_internal]
            
        self.color_external = widgets.ColorPicker(
            concise=True,
            description='External Color',
            value=colors.rgb2hex(self.drawable._external_color),
            disabled=False
        )
        self.widgets += [self.color_external]
        
        
        self.color_label_pickers = [widgets.ColorPicker(
                                            concise=True,
                                            description='Label ' + str(i),
                                            value= colors.random_color(return_hex=True),
                                            disabled=False,
                                            layout = self.visible_layout,
                                            ) for i in range(len(np.unique(self.mesh.labels)))]
        
        self.color_label_pickers = widgets.HBox(self.color_label_pickers, layout=self.invisible_layout)
        self.widgets += [self.color_label_pickers]
        
        
        
        self.flip_x_button.observe(self.__update_clipping, names='value')
        self.flip_y_button.observe(self.__update_clipping, names='value')
        self.flip_z_button.observe(self.__update_clipping, names='value')
        self.clipping_slider_x.observe(self.__update_clipping, names='value')
        self.clipping_slider_y.observe(self.__update_clipping, names='value')
        self.clipping_slider_z.observe(self.__update_clipping, names='value')
        self.color_internal.observe(self.__update_internal_color, names='value')
        self.color_external.observe(self.__update_external_color, names='value')
        self.color_wireframe.observe(self.__update_wireframe_color, names='value')
        self.wireframe_opacity_slider.observe(self.__update_wireframe_opacity, names='value')
        self.coloring_type_menu.observe(self.__change_color_type, names='value')
        self.color_map.observe(self.__change_color_map, names='value')
        self.metric_menu.observe(self.__change_metric, names='value')
        [i.observe(self.__change_color_label,names='value') for i in self.color_label_pickers.children]
        #self.wireframe_thickness_slider.observe(self.__update_wireframe_thickness, names='value')
        
        for widget in self.widgets:
            ipydisplay(widget)
            
            
    def __update_wireframe_color(self, change): 
        self.drawable.update_wireframe_color(self.color_wireframe.value)
            
    def __update_wireframe_opacity(self, change): 
        self.drawable.update_wireframe_opacity(self.wireframe_opacity_slider.value)
        
    def __update_internal_color(self, change): 
        self.drawable.update_internal_color(colors.hex2rgb(self.color_internal.value))
            
    def __update_external_color(self, change): 
        self.drawable.update_external_color(colors.hex2rgb(self.color_external.value))
        
    def __change_color_type(self, change):
        
        if self.coloring_type_menu.value == 0:
            
            self.color_map.layout = self.invisible_layout
            self.metric_menu.layout = self.invisible_layout
            self.color_external.layout = self.visible_layout
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
            self.color_internal.layout = self.invisible_layout
            self.color_label_pickers.layout = self.invisible_layout
            self.drawable._label_colors = None
            self.__change_color_map(None)
        
        elif self.coloring_type_menu.value == 2:
            
            self.color_external.layout = self.invisible_layout
            self.color_internal.layout = self.invisible_layout
            self.color_map.layout = self.invisible_layout
            self.metric_menu.layout = self.invisible_layout
            
            if self.mesh.labels is not None:
            
                self.color_label_pickers.layout = self.visible_layout
                self.__change_color_label(None)
                    

        
    def __change_metric(self, change):
        
        self.__change_color_map(None)
        
        
    def __change_color_label(self, change):
        
        self.drawable._label_colors = [colors.hex2rgb(i.value) for i in self.color_label_pickers.children]
        
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