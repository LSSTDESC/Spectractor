# Spectractor

The goal of Spectractor is to extract spectra from CTIO images in order to test the performance of dispersers dedicated to the LSST Auxiliary Telescope, and characterize the atmospheric transmission.

## Installation

Spectractor is written in Python 2.7. It needs the numpy, scipy, astropy, astroquery modules for science computations, and also logging and coloredlogs.

## Basic usage

The main file is `spectractor.py` with the function `Spectractor`. It extracts the spectrum from one deflatted and trimmed CTIO images, given:
- the path to the fits image from which to extract the image, 
- the path of the output directory to save the extracted spectrum (created automatically if does not exist yet),
- the rough position of the object in the image,
- optionally the name of the target (to search for the extra-atmospheric spectrum if available).

```
filename="./notebooks/fits/trim_20170605_007.fits"
output_directory="./outputs/"
guess = [745,643]
target="3C273"
```

Then the spectrum is simply extracted from the image and saved in a new fits file using the command:
```
Spectractor(filename,output_directory,guess,target)
```

Spectractor comes with two verbosity modes, ste in the `parameters.py` file:
- `VERBOSE` (or -v, --verbose) : the format of the logging message gives first the time of execution, the class involved, the class method involved, the logging level, and a message with some information to know what the program is doing; it also plots the output spectrum
- `DEBUG` (or -d, --debug) : in the debugging mode some intermediate plots are produced to check why the process failed.

### Plot an extracted spectrum

To check a posteriori the quality of a spectrum, it is possible to load it via the `Spectrum` class and plot it:
```
spec = Spectrum('./outputs/trim_20170605_007_spectrum.fits')
spec.plot_spectrum()
```

### Tutorial notebook

A tutorial Jupyter notebook is available in the `notebooks` folder.

## Detailled description

### Dispersers

### Shot noise

### Rotation

### Background extraction

### Wavelength calibration
#### First geometrical calibration
#### Second calibration with line detection

### Second order subtraction


## Description of the files and classes
