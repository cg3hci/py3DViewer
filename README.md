![Py3DViewer Logo](https://github.com/cg3hci/py3DViewer/blob/master/docs/source/logo.png)


Fast research and prototyping, nowadays, is shifting towards languages that allow interactive execution and quick changes. Python is very widely used for rapid prototyping. Py3DViewer is a new Python library that allows researchers to quickly prototype geometry processing algorithms by interactively editing and viewing meshes. Polygonal and polyhedral meshes are both supported. The library is designed to be used in conjunction with [Jupyter environments](https://jupyter.org), which allow interactive Python code execution and data visualization in a browser, thus opening up the possibility of viewing a mesh while editing the underlying geometry and topology.

- [Installation](#installation)
- [Getting Started](#getting-started) 
- [Documentation](#documentation)
- [Deployment](#deployment)
- [Authors](#authors)
- [Contributing](#contributing)
- [Cite Us](#cite-us)
- [License](#license)
- [More Informations](#more-information)

--------------------------------------------------------------------------------

## Installation

If you have never used Jupyter, then the first step is to install and configure Jupyter on your computer. You can follow this [guide](https://jupyter.org/install.html) to do it.

Py3DViewer can be easily installed with pip:

```
pip install git+https://github.com/cg3hci/py3DViewer
```

Then, install the extension for jupyter notebooks:

```
jupyter nbextension install --py --symlink --sys-prefix pythreejs
jupyter nbextension enable --py --sys-prefix pythreejs
```

Or for jupyter lab:

```
jupyter labextension install @jupyter-widgets/jupyterlab-manager 
jupyter labextension install jupyter-threejs
```

Finally, you need to install the Jupyter widgets extension for notebooks: 

```
jupyter nbextension enable --py widgetsnbextension
```

Or for jupyter lab:
```
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

## Getting Started

The library is designed to be easy and quick to use in the context of fast prototyping. In the example below it is shown how a mesh can be loaded from a file and drawn in a canvas with just a few linesof code.

```python
from Py3DViewer import Trimesh
m = Trimesh('data/goyle.obj')
m.show()
```

More examples are available [here](https://py3dviewer.readthedocs.io/en/latest/notes/getting_started.html)!

## Documentation

You can find the complete documentation of the library [here](https://py3dviewer.readthedocs.io)!

## Deployment

We are working to extend the kinds of geometric data that the library supports, to appeal to an even broader part of the community. For the animation community, we are working to fully support skeletons, both for editing and visualizing. Ideally, similar to what some other libraries are doing, Py3DViewer will support generic polygonal or polyhedral meshes, and point clouds. By making the system as modular as possible, we are working to make it easy to extend the supported data representations by making the viewer and the interface as data agnostic as possible. 

We aim to completely integrate [PyTorch](https://pytorch.org) tensors to represent the geometry information of a mesh, so that any tensorial operation on the underlying data structure could be seamlessly executed on the GPU by simply passing GPU-instantiated tensors to our data structures or by loading a mesh file directly on the GPU. 

In the near future, the interactive Jupyter interface will be extended to more widgets and will support adding custom UI elements to better suit each different user's need. 

More generally, the library will continously improve in its documentation and examples. To facilitate prototyping for new users, we are working to implement interactive examples in the form of tutorial notebooks, by using [Google Colab](https://colab.research.google.com) platform as a mean to quickly try our library's features. 

One of the most important features we will implement in the near future, is a complete PyTorch support for the data structures and algorithms for the underlying representation, instead of [Numpy](https://numpy.org), if the user so chooses. This feature will allow the library to seamlessly run its algorithms on the GPU, to speed up parallel computations and to allow researchers to easily and efficiently prototype Geometry Processing algorithms and Deep Learning networks.

## Authors

Gianmarco Cherchi, Luca Pitzalis, Giovanni Laerte Frongia and Riccardo Scateni

University of Cagliari (Italy)

### Other Contributors

Giampaolo Perelli (University of Cagliari).

## Contributing

Pull requests are welcome! 
For major changes, please open an issue first to discuss what you would like to change. 

## Cite Us

Please cite our [paper]() if you use the Py3DViewer in your own work:

```bibtex
@inproceedings{py3dviewer2019,
  title={{The Py3DViewer project}: a Python library for fast prototyping in geometry processing},
  author={Cherchi, Gianmarco and Pitzalis, Luca and Frogia, Giovanni L. and Scateni, Riccardo},
  booktitle={Proceedings of the Conference on Smart Tools and Applications in Computer Graphics},
  organization={Eurographics Association},
  year={2019},
}
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/cg3hci/py3DViewer/blob/master/LICENSE) file for details.

## More Information

For other information you can contact one of the main developers of the library: G. Cherchi (g.cherchi@unica.it), L. Pitzalis (luca.pitzalis94@unica.it) and G. L. Frongia (giovannil.frongia@unica.it).
