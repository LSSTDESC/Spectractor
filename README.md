[![Build Status](https://travis-ci.org/LSSTDESC/Spectractor.svg?branch=master)](https://travis-ci.org/LSSTDESC/Spectractor)
[![Coverage Status](https://coveralls.io/repos/github/LSSTDESC/Spectractor/badge.svg?branch=master)](https://coveralls.io/github/LSSTDESC/Spectractor?branch=master)
[![Documentation Status](https://readthedocs.org/projects/spectractor/badge/?version=latest)](https://spectractor.readthedocs.io/en/latest/?badge=latest)
<a href="https://ascl.net/2104.004"><img src="https://img.shields.io/badge/ascl-2104.004-blue.svg?colorB=262255" alt="ascl:2104.004" /></a>

# Spectractor

The goal of Spectractor is to measure the atmospheric transmission and intempt extracting spectra from slitless spectrophotometric images. It has been optimized on CTIO images but can be configured to analyse any kind of slitless data that contains the order 0 and the order 1 of a spectrum. In particular it can  be used to estimate the atmospheric transmission of the LSST site using the dedicated  Auxiliary Telescope. 

Spectractor is structured in three subpackages: 
- `spectractor.extractor`: extracts as most information as possible from a slitless data  image, as the amplitude of the spectrum, the PSF evolution with the wavelength, the pixel to wavelength calibration, an estimate of the background under the spectrum, the position of the order 0;
- `spectractor.simulation`: contains all the tools to simulate a spectrogram, as the atmospheric transmission simulation, the inclusion of instrumental throughput, the simulation of mock  slitless  data images;
- `spectractor.fit`: compares the extracted data with simulations to estimate the atmospheric transmission and refine the spectral extraction.

Some submodules complete the structures with generic functions:
- `spectractor.parameters`: contains all the global parameters of Spectractor to set its general behaviour, the instrumental characteritics, etc;
- `spectractor.config`: tools to read config `.ini` text files and set the global parameters;
- `spectractor.logbook`: tools to read logbook `.csv` text files and get some metadata relative to the data images that are not contained in the header;
- `spectractor.tools`: contains generic functions shared by all  the subpackages (fitting procedures, plotting functions, etc).

## Installation

Spectractor is written in Python 3.9. The mandatory dependencies are listed in the `requirements.txt` file. To install Spectractor, just run
```
pip install -r requirements.txt .
```

To simulate atmospheric transmission, Spectractor needs [libradtran](http://www.libradtran.org/doku.php) (optional). There are two ways to install this software:
- use the [Rubin observatory conda package](https://anaconda.org/conda-forge/rubin-libradtran)
```
conda install -c conda-forge rubin-libradtran
```
- compile it from [sources](http://www.libradtran.org/doku.php): it needs the installation of [netcdf](https://www.unidata.ucar.edu/software/netcdf/) and a python 2 environment (for the compilation only, not the usage); `uvpsec` executable must in the user `$PATH` or the user has to set an environmental variable `$LIBRADTRAN_DIR` pointing to the install directory.

To find accurate star centroids using astrometry, one can install [astrometry.net](https://astrometrynet.readthedocs.io/en/latest/) (optional). There are two ways to install this software:
- use the anaconda package
```
conda install -c conda-forge astrometry
```
and then download and move [index files](http://astrometry.net/doc/readme.html#getting-index-files) in the `$CONDA_PREFIX/data` folder.
- compile it from [sources](https://astrometrynet.readthedocs.io/en/latest/): it needs the installation of [netpbm](https://netpbm.sourceforge.net/) and [wcslib](https://www.atnf.csiro.au/people/mcalabre/WCS/wcslib/); `solve-field` executable must in the user `$PATH` or the user has to set an environmental variable `$ASTROMETRYNET_DIR` pointing to the install directory.

For faster matrix multiplications, if available for a given computer configuration, one can install [sparse_dot_mkl](https://anaconda.org/conda-forge/sparse_dot_mkl):
```
conda install -c conda-forge sparse_dot_mkl
```

Spectractor is able to perform parameter fits using the MCMC library [emcee](https://emcee.readthedocs.io/en/stable/) (optional) with [mpi4py](https://mpi4py.readthedocs.io/en/stable/) and [h5py](https://www.h5py.org/).  The latter might be better installed using `conda install ...` command to get their own dependencies (openmp and hdf5).

## Basic extraction

The main file is `spectractor/extractor/extractor.py` with the function `Spectractor`. It extracts the spectrum from a science data image (deflatted, debiased), given:
- the path to the FITS image from which to extract the image, 
- the path of the output directory to save the extracted spectrum (created automatically if it does not exist yet),
- the rough pr exact position of the object in the image (in pixels),
- the name of the disperser (as it is named in the `spectractor/extractor/dispersers/` folder),
- the name of the config .ini file,
- optionally the name of the target (to search for the extra-atmospheric spectrum if available).

```
filename="./tests/data/reduc_20170530_134.fits"
output_directory="./outputs/"
guess = [745,643]
disperser_label = "HoloAmAg"
config = "./config/ctio.ini"
target = "HD111980"
```

Then the spectrum is simply extracted from the image and saved in a new fits file using the `Spectractor` function:
```
spectrum = Spectractor(filename, output_directory, guess=guess, target_label=target, disperser_label=disperser_label, config=config)
```

or typing the following command within a terminal:
```
python runExtractor.py ./tests/data/reduc_20170530_134.fits -o outputs --config ./config/ctio.ini --xy [745, 643] --target HD111980 --grating HoloAmAg
```

Spectractor comes with two verbosity modes, set in the `parameters.py` file:
- `VERBOSE` (or -v, --verbose) : the first level of verbosity that returns many information along the process to know what the program is doing; it also plots the output spectrum
- `DEBUG` (or -d, --debug) : in the debugging mode some intermediate plots are produced to see the performance of the program.

### Plot a spectrum

To see the result of the extraction process, it is possible to load it via the `Spectrum` class and plot it:
```
spectrum = Spectrum('./tests/data/reduc_20170530_134_spectrum.fits', config="./config/ctio.ini)
spectrum.plot_spectrum()
spectrum.plot_spectrogram()
```
This object is also returned by the `Spectractor` function.

### Tutorial notebook

A tutorial Jupyter notebook is available in the `notebooks` folder.

