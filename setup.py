from setuptools import setup, find_packages
import os

# read the contents of README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    readme = f.read()

setup(
    # Name of the package
    name='ipas',
    packages=find_packages('.'),
    version='2.1.7',
    description='Theoretically simulates ice crystal aggregation (snow) using hexagonal prisms',
    long_description = readme,
    long_description_content_type='text/markdown',
    license='MIT',
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
                            "jupyterlab",
                            "jupyter-server",
                            "jupyterlab-lsp",
                            "matplotlib",
                            "nbconvert",
                            "nbstripout",
                            "seaborn"]
                   },
    entry_points={
        'console_scripts': [
            'ice_ice=ipas.executables.collection_no_db.Ice_Ice:main',
            'ice_agg=ipas.executables.collection_no_db.Ice_Agg:main'
    ]},
    include_package_data=True,
    # https://pypi.org/classifiers/
    classifiers=['Development Status :: 3 - Alpha',
                 'Programming Language :: Python :: 3.7',
                 'License :: OSI Approved :: MIT License']
)
