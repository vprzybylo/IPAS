from distutils.core import setup
from setuptools import find_packages
import os


current_directory = os.path.dirname(os.path.abspath(__file__))
try:
    with open(os.path.join(current_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except Exception:
    long_description = ''

setup(
    # Name of the package
    name='ipas',
    # Packages to include into the distribution
    packages=find_packages('.'),
    version='1.3.0',
    license='MIT',
    # Short description of your library
    description='Theoretically simulates ice crystal aggregation (snow) using hexagonal prisms',

    # Long description of your library
    long_description = long_description,
    long_description_context_type = 'text/markdown',
    author='Vanessa Przybylo', 
    author_email='vprzybylo@albany.edu',     
    # Either the link to your github or to your website
    url='https://github.com/vprzybylo/IPAS',
    # Link from which the project can be downloaded
    download_url='https://github.com/vprzybylo/IPAS.git',
    python_requires='>=3.7',
    # List of packages to install with this one
    install_requires=["cytoolz",
                      "dask",
                      "dask-jobqueue",
                      "descartes",
                      "hickle",
                      "jupyterlab",
                      "jupyter-server",
                      "jupyterlab-lsp",
                      "matplotlib",
                      "numpy",
                      "pandas",
                      "plotly",
                      "pyquaternion",
                      "QtPy",
                      "Shapely",
                      "scipy"],
    extras_require={'dev': ["flakehell",
                            "ipympl",
                            "ipykernel",
                            "ipywidgets",
                            "nbconvert",
                            "nbstripout",
                            "seaborn"]
                   },
    # https://pypi.org/classifiers/
    classifiers=['Development Status :: 3 - Alpha',
                 'Programming Language :: Python :: 3.7']
)
