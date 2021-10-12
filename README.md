## (Subspace Constrained) Mean Shift Algorithms in Euclidean and/or Directional Product Spaces
This repository contains Python3 code for the mean shift and subspace constrained mean shift (SCMS) algorithms in any Euclidean and/or directional (Cartesian) product space.

- Paper Reference: [Mode and Ridge Estimation in Euclidean and Directional Product Spaces: A Mean Shift Approach]() (2021)

### Requirements

- Python >= 3.8 (earlier version might be applicable).
- [NumPy](http://www.numpy.org/), [Matplotlib](https://matplotlib.org/) (especially the [Basemap](https://matplotlib.org/basemap/) toolkit), [pandas](https://pandas.pydata.org/), [SciPy](https://www.scipy.org/) (The speical function [scipy.special.iv](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.iv.html#scipy.special.iv) computes the modified Bessel function of the first kind of real order; [scipy.linalg.block_diag](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.block_diag.html) creates a block diagonal matrix from provided arrays), [pickle](https://docs.python.org/3/library/pickle.html), [datetime](https://docs.python.org/3/library/datetime.html) and [time](https://docs.python.org/3/library/time.html) libraries.
- [astropy](https://www.astropy.org/) ("Astropy" is a Python package for analyzing data among the astronomical community.
- [Ray](https://ray.io/) ("Ray" is a fast and simple distributed computing API for Python and Java. We use "ray\[default\]==1.4.0" because the lastest versions (>=1.6.0) cannot be run on our Ubuntu 16.04 server.)
- We provide an [guideline](https://github.com/zhangyk8/DirMS/blob/main/Install_Basemap_Ubuntu.md) of installing the [Basemap](https://matplotlib.org/basemap/) toolkit on Ubuntu.

### Descriptions

Some high-level descriptions of our Python scripts are as follows:

- **Cosmic Filament Detection with Directional-Linear SCMS Algorithm (An Example).ipynb**: This Jupyter Notebook contains detailed code and descriptions about how we process the SDSS-IV galaxy data (Ahumada et al., 2020) and detect cosmic filaments on them with our proposed SCMS algorithm in the directional-linear \[(RA,DEC)*Redshift\] product space.
- **Curves_Sphere_Torus.py**: This script simulates a circular-circular dataset and plot its points on a unit sphere and torus, respectively. (Figure 2 in the arxiv version of the paper).
- **DirLinProdSCMS_Ray.py**: This script implements the functions of KDE, component-wise/simultaneous mean shift, and subspace constrained mean shift (SCMS) algorithms with the Gaussian/von Mises product kernels in a directional/linear (mixture) product space using the parallel programming under the "Ray" environment.
- **DirLinProdSCMS_fun.py**: This script implements the functions of KDE, component-wise/simultaneous mean shift, and subspace constrained mean shift (SCMS) algorithms with the Gaussian/von Mises product kernels in a directional/linear (mixture) product space.
- **Earthquake_Modes.py**: This script contains code for applying our proposed mean shift algorithm to an Earthquake dataset (directional-linear data) (Figure 5 in the arxiv version of the paper). This script take more than 35 minutes to run on my laptop with 8 CPU cores.
- **MS_SCMS_Ray.py**: This script contains code for the parallel implementations of regular Euclidean/directional mean shift and SCMS algorithms.
- **Mode_Seeking_Examples.py**: This script contains code for mode-seeking simulation studies with our proposed mean shift algorithm (Figure 3 in the arxiv version of the paper).
- **Spherical_Cone.py**: This script contains code for comparing the results of the regular SCMS and our proposed SCMS algorithms on the simulated spherical cone data (Figure 4 in the arxiv version of the paper).
- **Spiral_Curve.py**: This script contains code for comparing the results of the regular SCMS and our proposed SCMS algorithms on the simulated spiral curve data. (Figure 1 in the arxiv version of the paper).
- **Utility_fun.py**: This script contains all the utility functions for our experiments.
- **Varying_Stepsize.py**: This script contains code for investigating the effects of varying the stepsize parameter in our proposed SCMS algorithm in Euclidean/directional product spaces. (Figures 9 and 10 in the arxiv version of the paper). The script takes more than 1.5 hours to execute due to the slow convergence of the proposed SCMS algorithm with step size "eta=1". The SCMS algorithm with our suggested choice of the step size parameter, however, does converge very fast.



 ### Additional References
 - R. Ahumada, C. A. Prieto, A. Almeida, F. Anders, S. F. Anderson, B. H. Andrews, B. Anguiano, R. Arcodia, E. Armengaud, M. Aubert, et al. The 16th data release of the sloan digital sky surveys: first release from the apogee-2 southern survey and full release of eboss spectra. _The Astrophysical Journal Supplement Series_, 249(1):3, 2020.
