# ProdSCMS
This repository contains Python3 code for the mean shift and subspace constrained mean shift (SCMS) algorithms in any Euclidean and/or directional (Cartesian) product space.

- Paper Reference: [Mode and Ridge Estimation in Euclidean and Directional Product Spaces: A Mean Shift Approach]() (2021)

### Requirements

- Python >= 3.8 (earlier version might be applicable).
- [NumPy](http://www.numpy.org/), [Matplotlib](https://matplotlib.org/) (especially the [Basemap](https://matplotlib.org/basemap/) toolkit), [pandas](https://pandas.pydata.org/), [SciPy](https://www.scipy.org/) (The speical function [scipy.special.iv](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.iv.html#scipy.special.iv) computes the modified Bessel function of the first kind of real order; [scipy.linalg.block_diag](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.block_diag.html) creates a block diagonal matrix from provided arrays), [pickle](https://docs.python.org/3/library/pickle.html), [datetime](https://docs.python.org/3/library/datetime.html) and [time](https://docs.python.org/3/library/time.html) libraries.
- [astropy](https://www.astropy.org/) 
- [Ray](https://ray.io/) (Ray is a fast and simple distributed computing API for Python and Java. We use "ray\[default\]==1.4.0" because the lastest versions (>=1.6.0) cannot be run on our Ubuntu 16.04 server.)
- We provide an [guideline](https://github.com/zhangyk8/DirMS/blob/main/Install_Basemap_Ubuntu.md) of installing the [Basemap](https://matplotlib.org/basemap/) toolkit on Ubuntu.
