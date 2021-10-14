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

### 1. Motivation: Euclidean/Directional Product Spaces

It is intuitive that the (Cartesian) product of two Euclidean spaces is again an Euclidean space whose dimension is the sum of the dimensions of two factor (Euclidean) spaces. As the topology of such product space does not mathematically change, the regular kernel density estimator (KDE) as well as mean shift and SCMS algorithms are applicable in the Euclidean-Euclidean product space (Cheng, 1995; Comaniciu and Meer, 2002; Ozertem and Erdogmus, 2011). The Euclidean-directional and directional-directional product spaces, however, are not topologically equivalent to any of its factor spaces under any dimension. Consider, for example, a dataset <img src="https://latex.codecogs.com/svg.latex?&space;\left\{(\theta_i,\phi_i\right\}_{i=1}^n"/> with <img src="https://latex.codecogs.com/svg.latex?&space;\theta_i"/> and <img src="https://latex.codecogs.com/svg.latex?\large&space;\phi_i"/> being periodic. Under some renormalizations, every such circular-circular (or periodic-periodic) observation can be viewed as a point on the sphere <img src="https://latex.codecogs.com/svg.latex?&space;\Omega_2"/>, where <img src="https://latex.codecogs.com/svg.latex?&space;(\phi_i,\theta_i)"/> represents the longitude and latitude, or a point on the torus <img src="https://latex.codecogs.com/svg.latex?&space;\Omega_1\times\Omega_1"/>. Here, 

<img src="https://latex.codecogs.com/svg.latex?\large&space;\Omega_q=\left\{\mathbf{x}\in\mathbb{R}^{q+1}:||\mathbf{x}||_2=1\right\}"/>,

where <img src="https://latex.codecogs.com/svg.latex?&space;||\cdot||_2"/> is the usual Euclidean norm in <img src="https://latex.codecogs.com/svg.latex?&space;\mathbb{R}^{q+1}"/>. The supports <img src="https://latex.codecogs.com/svg.latex?&space;\Omega_2"/> and <img src="https://latex.codecogs.com/svg.latex?&space;\Omega_1\times\Omega_1"/> of the same dataset are topological different; see Figure 1 below. Therefore, it is worthwhile to reconsider (subspace constrained) mean shift algorithms as well as the related mode and ridge estimation problems in Euclidean/directional product spaces.

<!--
<p align="center">
<img src="https://github.com/zhangyk8/EuDirSCMS/blob/main/Figures/Output.png" style="zoom:60%" />
 <br><B>Fig 1. </B>Simulated dataset <img src="https://latex.codecogs.com/svg.latex?&space;\left\{(\theta_i,\phi_i\right\}_{i=1}^n"/> on <img src="https://latex.codecogs.com/svg.latex?&space;\Omega_2"/> and <img src="https://latex.codecogs.com/svg.latex?&space;\Omega_1\times\Omega_1"/> . Each observation <img src="https://latex.codecogs.com/svg.latex?&space;(\phi_i,\theta_i)"/> is sampled uniformly from <img src="https://latex.codecogs.com/svg.latex?&space;\left[2p_1\pi,2(p_1+1)\pi\right)\times\{2p_2\pi\}"/> for some integers <img src="https://latex.codecogs.com/svg.latex?&space;p_1,p_2"/>.
 </p>
-->

Besides the aforementioned circular-circular data, there are many real-world datasets whose observations lie on a Euclidean/directional product space. For instance, in astronomical survey data, each object has its right ascension (RA) and declination (DEC) on a celestial sphere, while its redshift measures its distance to the Earth. The collection of (RA,DEC,Redshift) tuples thus forms a directional-linear dataset.

### 2. Mode and Ridge Estimation on Euclidean/directional product spaces with (Subspace Constrained) Mean Shift Algorithms

Our interested data consist of independent and identically distributed (i.i.d.) observations <img src="https://latex.codecogs.com/svg.latex?&space;\left\{\mathbf{Z}_i\right\}_{i=1}^n=\left\{(\mathbf{X}_i,\mathbf{Y}_i)\right\}_{i=1}^n"/> sampled from a distribution on <img src="https://latex.codecogs.com/svg.latex?&space;\mathcal{S}_1\times\mathcal{S}_2"/>, where 
<img src="https://latex.codecogs.com/svg.latex?&space;\mathcal{S}_j=\mathbb{R}^{D_j}\,\text{or}\,\Omega_{D_j}"/>
for <img src="https://latex.codecogs.com/svg.latex?&space;j=1,2"/>. While we only present the formulations of our proposed algorithms and related theory on (Cartesian) product spaces with two factors, our implementations (i.e., associated functions in **DirLinProdSCMS_fun.py** and **DirLinProdSCMS_Ray.py**) are adaptive to any product space with arbitrarily finte number of Euclidean/directional factor spaces.

#### 2.1 Kernel Density Estimator (KDE) on <img src="https://latex.codecogs.com/svg.latex?&space;\mathcal{S}_1\times\mathcal{S}_2"/>

It is natural to leverage a product kernel to construct a kernel density estimator (KDE) on <img src="https://latex.codecogs.com/svg.latex?&space;\mathbf{z}=(\mathbf{x},\mathbf{y})\in\mathcal{S}_1\times\mathcal{S}_2"/> as:

<img src="https://latex.codecogs.com/svg.latex?&space;\hat{f}_{\mathbf{h}}(\mathbf{x},\mathbf{y})=\frac{1}{n}\sum_{i=1}^nK_1\left(\frac{\mathbf{x}-\mathbf{X}_i}{h_1}\right)K_2\left(\frac{\mathbf{y}-\mathbf{Y}_i}{h_2}\right)"/>,

where each element of <img src="https://latex.codecogs.com/svg.latex?&space;\mathbf{h}=(h_1,h_2)"/> is a bandwidth parameter and the kernel functions <img src="https://latex.codecogs.com/svg.latex?&space;K_j:\mathcal{S}_j\to\mathbb{R}"/> for <img src="https://latex.codecogs.com/svg.latex?&space;j=1,2"/> take the form

<a href="https://latex.codecogs.com/svg.latex?&space;K_j(\mathbf{u})&space;=&space;C_{k_j,D_j}(h_j)&space;\cdot&space;k_j\left(||\mathbf{u}||_2^2&space;\right)&space;=&space;\begin{cases}&space;\frac{C_{k,D_j}}{h_i^{D_j}}\cdot&space;k\left(\frac{||\mathbf{u}||_2^2}{2}&space;\right)&space;&&space;\text{&space;if&space;}&space;\mathcal{S}_j&space;=\mathbb{R}^{D_j},\\&space;C_{L,D_j}(h_j)&space;\cdot&space;L\left(\frac{||\mathbf{u}||_2^2}{2}&space;\right)&space;&&space;\text{&space;if&space;}&space;\mathcal{S}_j&space;=\Omega_{D_j},&space;\end{cases}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?K_j(\mathbf{u})&space;=&space;C_{k_j,D_j}(h_j)&space;\cdot&space;k_j\left(||\mathbf{u}||_2^2&space;\right)&space;=&space;\begin{cases}&space;\frac{C_{k,D_j}}{h_i^{D_j}}\cdot&space;k\left(\frac{||\mathbf{u}||_2^2}{2}&space;\right)&space;&&space;\text{&space;if&space;}&space;\mathcal{S}_j&space;=\mathbb{R}^{D_j},\\&space;C_{L,D_j}(h_j)&space;\cdot&space;L\left(\frac{||\mathbf{u}||_2^2}{2}&space;\right)&space;&&space;\text{&space;if&space;}&space;\mathcal{S}_j&space;=\Omega_{D_j},&space;\end{cases}" title="K_j(\mathbf{u}) = C_{k_j,D_j}(h_j) \cdot k_j\left(||\mathbf{u}||_2^2 \right) = \begin{cases} \frac{C_{k,D_j}}{h_i^{D_j}}\cdot k\left(\frac{||\mathbf{u}||_2^2}{2} \right) & \text{ if } \mathcal{S}_j =\mathbb{R}^{D_j},\\ C_{L,D_j}(h_j) \cdot L\left(\frac{||\mathbf{u}||_2^2}{2} \right) & \text{ if } \mathcal{S}_j =\Omega_{D_j}, \end{cases}" /></a>


 ### Additional References
 - R. Ahumada, C. A.Prieto, A. Almeida, F. Anders, S. F. Anderson, B. H. Andrews, B. Anguiano, R. Arcodia, E. Armengaud, M. Aubert, et al. The 16th data release of the sloan digital sky surveys: first release from the apogee-2 southern survey and full release of eboss spectra. _The Astrophysical Journal Supplement Series_, 249(1):3, 2020.
 - Y. Cheng. Mean shift, mode seeking, and clustering. _IEEE Transactions on Pattern Analysis and Machine Intelligence_, 17(8):790–799, 1995.
 - D. Comaniciu and P. Meer. Mean shift: a robust approach toward feature space analysis. _IEEE Transactions on Pattern Analysis and Machine Intelligence_, 24(5):603–619, 2002.
 - U. Ozertem and D. Erdogmus. Locally defined principal curves and surfaces. _Journal of Machine Learning Research_, 12(34):1249–1286, 2011.
