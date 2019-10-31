import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = ['numpy', 'pythreejs', 'ipywidgets', 'numba']

setuptools.setup(
    name="Py3DViewer",
    version="1.0.1",
    author="CG3HCI Lab",
    author_email="luca.pitzalis94@unica.it",
    description="A Python Library for fast prototyping in geometry processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cg3hci/py3DViewer",
    install_requires=install_requires,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
