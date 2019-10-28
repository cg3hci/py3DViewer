import ipywidgets as widgets
from ..utils import ColorMap, Observer
from ..visualization import colors
from IPython.display import display as ipydisplay
import numpy as np

class GUI(Observer):
    
    def __init__(self, drawable_mesh):
        self.drawable = drawable_mesh
        self.mesh = drawable_mesh.geometry
        self.mesh.attach(self)
        self.widgets = []
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
       
        x_range = (round(self.mesh.bbox[0][0], 2), round(self.mesh.bbox[1][0],2))
        self.clipping_slider_x = widgets.FloatRangeSlider(
            value=x_range,
            min=x_range[0],
            max=x_range[1],
            step=0.01,
            description='X:',
            disabled=False,
            continuous_update=True,
            orientation='horizontal',
            readout=True,
            readout_format=".3f",
        )
        self.widgets += [self.clipping_slider_x]


        y_range = (round(self.mesh.bbox[0][1], 2), round(self.mesh.bbox[1][1],2))
        self.clipping_slider_y = widgets.FloatRangeSlider(
            value = y_range,
            min = y_range[0],
            max = y_range[1],
            step=0.001,
            description='Y:',
            disabled=False,
            continuous_update=True,
            orientation='horizontal',
            readout=True,
            readout_format=".3f",
        )
        self.widgets += [self.clipping_slider_y]

        z_range = (round(self.mesh.bbox[0][2], 2), round(self.mesh.bbox[1][2],2))
        self.clipping_slider_z = widgets.FloatRangeSlider(
            value = z_range,
            min = z_range[0],
            max = z_range[1],
            step=0.001,
            description='Z:',
            disabled=False,
            continuous_update=True,
            orientation='horizontal',
            readout=True,
            readout_format=".3f",
        )
        self.widgets += [self.clipping_slider_z]
        
        """
        self.wireframe_thickness_slider = widgets.FloatSlider(
                        value=0.2,
                        min=0.,
                        max=1.,
                        step=0.1,
                        continuous_update=True,
                        readout_format=".1f",
                        description = 'Wireframe',
                        disable = False,
                )
        self.widgets += [self.wireframe_thickness_slider]
        """

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
        )
        self.widgets += [self.color_map]

        self.coloring_type_menu = widgets.Dropdown(
            options=[('Default', 0), ('Simplex Quality', 1), ('Label',2)],
            value=0,
            description='Type Color:',
        )
        self.widgets += [self.coloring_type_menu]
        
        self.metric_menu = widgets.Dropdown(
            options= [(i, idx) for idx, i in enumerate(self.mesh.simplex_metrics.keys())],
            value=0,
            description='Metric:',
        )
        self.widgets += [self.metric_menu]
        
        self.color_internal = widgets.ColorPicker(
                            concise=True,
                            description='Internal Color',
                            value=colors.rgb2hex(self.drawable._internal_color[0]),
                            disabled=False,
        )
        self.widgets += [self.color_internal]
            
        self.color_external = widgets.ColorPicker(
            concise=True,
            description='External Color',
            value=colors.rgb2hex(self.drawable._external_color[0]),
            disabled=False
        )
        self.widgets += [self.color_external]
        
        """
        self.color_label_pickers = [widgets.ColorPicker(
                                            concise=True,
                                            description='Label ' + str(i),
                                            value= self.listColor(int(i)),
                                            disabled=False,
                                            ) for i in range(len(np.unique(self.mesh.labels)))]
        self.widgets += [self.color_label_pickers]
        """
        
        
        self.flip_x_button.observe(self.__update_clipping, names='value')
        self.flip_y_button.observe(self.__update_clipping, names='value')
        self.flip_z_button.observe(self.__update_clipping, names='value')
        self.clipping_slider_x.observe(self.__update_clipping, names='value')
        self.clipping_slider_y.observe(self.__update_clipping, names='value')
        self.clipping_slider_z.observe(self.__update_clipping, names='value')
        self.color_internal.observe(self.__update_internal_color, names='value')
        self.color_external.observe(self.__update_external_color, names='value')
        self.color_wireframe.observe(self.__update_wireframe_color, names='value')
        #self.wireframe_thickness_slider.observe(self.__update_wireframe_thickness, names='value')
        
        for widget in self.widgets:
            ipydisplay(widget)
            
            
    def __update_wireframe_color(self, change): 
        self.drawable.update_wireframe_color(self.color_wireframe.value)
            
    def __update_internal_color(self, change): 
        self.drawable.update_internal_color(colors.hex2rgb(self.color_internal.value))
            
    def __update_external_color(self, change): 
        self.drawable.update_external_color(colors.hex2rgb(self.color_external.value))
        
    def __update_clipping(self, change): 
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