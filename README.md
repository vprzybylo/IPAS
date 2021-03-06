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

##  Prerequisites

The requirements.txt file lists all Python libraries required to run IPAS in a virtual environment.

    pip install -r requirements.txt
    
## Installation

... TBD


## Deployment

- IPAS is typically scaled on a cluster using [dask](https://dask.org/) but can be run without a cluster.
- The collection_no_db and collection_from_db folders hold jupyter notebooks that act as main executables to run IPAS with or without starting a cluster.

## Folder Structure
- collection_no_db: creates aggregates from 'scratch' instead of the pre-made database of aggregates.  Can be run after cloning repo.
- collection_from_db: creates aggregates pulling from the predined aggregate database (~50Gb). Please contact for acquisition.
- 'ipas' subdirectories hold IPAS functionality
- analysis: holds plotting scripts and notebooks for publication figures and visualizations.
- CPI_verification: 'verify_IPAS.ipynb' shows comparisons between the Cloud Particle Imager (CPI) aggregates and IPAS aggregates.

## Authors

1. Vanessa Przybylo - improvements and alterations for publication
2. Carl Schmitt - original rights
3. William May - conversion from IDL to Python
4. Kara Sulia - advisement
5. Zachary Lebo - advisement

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments
V. Przybylo, K. Sulia, C. Schmitt, and Z. Lebo would like to thank the Department of Energy for support under DOE Grant Number DE-SC0016354. The authors would also like to thank the ASRC ExTreme Collaboration, Innovation, and TEchnology (xCITE) Laboratory for IPAS development support.

## Contact
vprzybylo@albany.edu
