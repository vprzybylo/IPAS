[![Forks][forks-shield]][forks-url]
[![GitHub stars][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![GitHub release][release-shield]][release-url]

[download-shield]:https://img.shields.io/github/downloads/vprzybylo/IPAS/total?style=plastic
[download-url]: https://github.com/vprzybylo/IPAS/downloads
[release-shield]: https://img.shields.io/github/v/release/vprzybylo/IPAS?style=plastic
[release-url]:https://github.com/vprzybylo/IPAS/releases/
[forks-shield]: https://img.shields.io/github/forks/vprzybylo/IPAS?label=Fork&style=plastic
[forks-url]: https://github.com/vprzybylo/IPAS/network/members
[stars-shield]: https://img.shields.io/github/stars/vprzybylo/IPAS?style=plastic
[stars-url]: https://github.com/vprzybylo/IPAS/stargazers
[issues-shield]: https://img.shields.io/github/issues/vprzybylo/IPAS?style=plastic
[issues-url]: https://github.com/vprzybylo/IPAS/issues
[license-shield]: https://img.shields.io/github/license/vprzybylo/IPAS?style=plastic
[license-url]: https://github.com/vprzybylo/IPAS/blob/master/LICENSE.md
[![DOI](https://zenodo.org/badge/232696476.svg)](https://zenodo.org/badge/latestdoi/232696476)


<p align="center">
  <a>
    <img src="https://github.com/vprzybylo/IPAS/blob/master/rotateplot.gif" alt="Logo" width="250" height="250">
  </a>

<h1 align="center">IPAS</h1>

The [Ice Particle and Aggregate Simulator (IPAS)](http://www.carlgschmitt.com/Microphysics.html) is a statistical model in a theoretical framework that mimics simplified laboratory investigations to perform sensitivity tests, visualize, and better understand ice crystal growth via collection.  IPAS collects any number of solid hexagonal prisms that represent primary habits of plates and columns.  A detailed background description on monomer-monomer collection in IPAS can be found in [Przybylo (2019)](https://journals.ametsoc.org/view/journals/atsc/76/6/jas-d-18-0187.1.xml?tab_body=abstract-display), bulk testing of which can be found in [Sulia (2020)](https://journals.ametsoc.org/view/journals/atsc/aop/JAS-D-20-0020.1/JAS-D-20-0020.1.xml?rskey=9V3BQD&result=6).

##  Features
* Collection of any size and aspect ratio hexagonal prisms 
    * only simulates ice crystal shapes of plates and columns
    * all monomers within each aggregate have the same size and shape
* Aggregate calculations after collection:
    - [x] Aggregate axis lengths and aspect ratio (0.0-1.0) from an encompassing fit-ellipsoid 
    - [x] Change in density going from n to n+1 monomers within an aggregate
    - [x] 2-dimensional aspect ratio from a given projection (e.g., x-y plane)
    - [x] Aggregate complexity (uses aggregate perimeter and area of the smallest encompassing circle)
    - [x] 2D area ratio (aggregate area to the area of the smallest encompassing circle)
    - [] [Contact](#contact) if there is something specific you would like to add
    
## Installation

1. create a virtual environment:
``` conda create --name IPAS python=3.7.9 ```
2. activate environment:
``` conda activate IPAS ```
3. install IPAS:
``` pip install ipas ```
    * if requirements aren't satisfied, run ``` pip install -r requirements.txt ``` 
4. run ipas from any directory:
* ICE-ICE AGGREGATION:
    * examples:
        * ``` ice_ice -p 0.1 0.2 -r 100 200 --rand True -a 3 -s True -f 'ipas/instance_files/test' ```
        * ``` ice_ice -p 0.1 10 -r 50 --rand False -a 3 --use_dask True --w 5 ```
* ICE-AGG AGGREGATION:
    * examples:
        * ``` ice_agg -p 0.1 0.2 -r 10 20 --rand True -a 3 -m 2 -s True -f 'ipas/instance_files/test' ```
        * ``` ice_agg -p 10.0 20.0 -r 1000 --rand False -a 3 -m 10 --use_dask True --w 5 ```

FLAGS:
  * --help, -h:
        * show this help message and exit
  * --phi or -p:
        * (list) monomer aspect ratio(s)
  * --radius or -r:
        * (list) monomer radius/radii
  * --aggregates or -a:
        *(int) number of aggregates to create per aspect ratio-radius pairing
  * --monomers or -m:
        *(int) number of monomers per aggregate
        * only used in ice_agg
  * --rand: 
        * (bool) monomer and aggregate orientation
  * --save or -s:
        * (bool) save aggregate attributes to pickled file
    * if save is True also add filename
    * load data back in using:
        ```
        f = open('filename_with_path', 'rb')
        results = pickle.load(f)
        as = results['agg_as'] # aggregate a axis
        bs = results['agg_bs'] # aggregate a axis
        cs = results['agg_cs'] # aggregate a axis
        phi2Ds = results['phi2Ds'] # aggregate 2D aspect ratio (minor axis/major axis in x-z plane)
        cplxs = results['cplxs'] # aggregate complexity
        dds = results['dds'] # aggregate change in density
        ```
  * --filename or -f:
        * (str) filename to save data (include path from execution location)
  * --use_dask:
        * (bool) whether or not to use dask for parallelization
        **if True, requires setting up a client exclusive to your machine under start_client() in /ipas/executables/Ice_Agg.py or Ice_Ice.py**
    * if use_dask is True add the number of workers to the client as an argument
  * --workers or -w: 
        * (int) number of workers for dask client

## Deployment

* IPAS is typically scaled on a cluster using [dask](https://dask.org/) but can be run without a cluster.
* For aggregate plotting and visualization, the command line initializations cannot be used (above):
    * The executables/collection_no_db and executables/collection_from_db directories hold jupyter notebooks that act as main executables to run IPAS with or without starting a dask cluster and output figures
        * Plotting can be turned on (plot=True) under the ```if __name__ == '__main__':``` clause
        * Parallelization can be turned on (```use_dask=True```) *again, initialize the client appropriately*
    * For interactive plotting using [plotly](https://plotly.com/chart-studio/), turn to ipas/visualizations/Agg_Visualizations.ipynb


## Folder Structure
* executables:
  * holds all executables for running IPAS 
  * subdivided based on collection type
* collection_no_db:
  * src code
  * creates aggregates from 'scratch' instead of the pre-made database of aggregates
  * the command line arg parser only uses this package (i.e., does not read in the aggregate database due to size)
* collection_from_db: 
  * src code
  * creates aggregates pulling from the predined aggregate database (ice-agg and agg-agg collection). Please [contact](#contact) for acquisition (~50Gb). 
* visualization: 
  * holds plotting scripts and notebooks for publication figures and visualizations
* CPI_verification:
  * 'verify_IPAS.ipynb' shows comparisons between the Cloud Particle Imager (CPI) aggregates and IPAS aggregates

## Authors

1. Vanessa Przybylo - alterations, improvements, and packaging
2. Carl Schmitt - original rights
3. William May - conversion from IDL to Python
4. Kara Sulia - advisement
5. Zachary Lebo - advisement

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments
V. Przybylo, K. Sulia, C. Schmitt, and Z. Lebo would like to thank the Department of Energy for support under DOE Grant Number DE-SC0016354. The authors would also like to thank the ASRC ExTreme Collaboration, Innovation, and TEchnology (xCITE) Laboratory for IPAS development support.

## Contact
<vprzybylo@albany.edu>