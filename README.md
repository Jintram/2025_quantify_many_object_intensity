

# Script to quantify FP expression in nuclei in demixed FLIM images

This script aims to quantify the expression of fluorescent proteins (FPs) in nuclei.

### Prerequisites

- Make sure you have Python and the correct libraries installed. 
    - Required libraries: pandas, numpy, tifffile, scikit-image, scipy, matplotlib, seaborn, openpyxl.
    - This can e.g. be done using Conda. See e.g. this [blog post](https://www.biodsc.nl/posts/installing_conda_python.html).
    - If you use conda, you can use the following line in the terminal to install: 
```
conda create -n 2025_PeekNuclei python=3.10 pandas numpy tifffile scikit-image scipy matplotlib seaborn
conda install openpyxl
```

### Quick use guide

Navigate to the directory where you installed the scripts (or otherwise make sure that that directory is the working directory) and 
use the script `quantnuclei_analyse-experiment.py` to analyze an experiment.

### Rationale behind the script

The nuclei can be of varying brightness, due to their location in Z-plane,
or due to varying degree of fluorescent protein expression.

Since brightness of these nuclei can be comparable to background levels, 
only brighter nuclei can be detected, skewing the measurement.

The choice was therefor made to specifically focus on the brightest X nuclei
in the image, as a proxy for FP expression in a specific sample.

### Detection of nuclei

Segmentation of *all* nuclei is hard because of aformentioned varying brightness.

Assuming that the brightest pixels are located in nuclei that consist of multiple
pixels, whilst there might also be stray pixels with high values, signal from nuclei
can be detected by looking for clusters of pixels with higher than background signal.

This is done by applying a convolution with a disk with a radius similar to the 
radius of the nuclei, and looking for locations with the highest signal. 

The locations of the the top X brightest pixels after convolution are then collected
(with a minimum distance such that nuclei aren't located redundantly). For each location
the brightness (either from the original image or in the convoluted image) is
then determined.

### Technical background: FLIM measurement

The source images are de-mixed FLIM (fluorescence-lifetime imaging microscopy) images.
The reason for this is that the FP of interest emits in the same spectral range as 
autofluorescent plant molecules. The lifetimes of these two signals is however 
different, such that unmixing can produce intensity images that reflect the concentration
of the FP (but not the background).


### Methods summary

(To be added.)
