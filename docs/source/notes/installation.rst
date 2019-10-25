Installation
============

If you have never used Jupyter, then the first step is to install and configure Jupyter on your computer. You can follow this `guide <https://jupyter.org/install.html>`_ to do it.

Py3DViewer can be easily installed with pip::

pip install git+https://github.com/cg3hci/py3DViewer

Then, install the extension for jupyter notebooks::

jupyter nbextension install --py --symlink --sys-prefix pythreejs
jupyter nbextension enable --py --sys-prefix pythreejs

Or for jupyter lab::

jupyter labextension install @jupyter-widgets/jupyterlab-manager
jupyter labextension install jupyter-threejs

Finally, you need to install the Jupyter widgets extension for notebooks::

jupyter nbextension enable --py widgetsnbextension

Or for jupyter lab::

jupyter labextension install @jupyter-widgets/jupyterlab-manager
