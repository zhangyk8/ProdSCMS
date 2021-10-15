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

<p align="center">
<img src="https://github.com/zhangyk8/ProdSCMS/blob/main/Figures/curve_sph_torus.png" style="zoom:60%" />
 <br><B>Fig 1. </B>Simulated dataset <img src="https://latex.codecogs.com/svg.latex?&space;\left\{(\theta_i,\phi_i\right\}_{i=1}^n"/> on <img src="https://latex.codecogs.com/svg.latex?&space;\Omega_2"/> and <img src="https://latex.codecogs.com/svg.latex?&space;\Omega_1\times\Omega_1"/> . Each observation <img src="https://latex.codecogs.com/svg.latex?&space;(\phi_i,\theta_i)"/> is sampled uniformly from <img src="https://latex.codecogs.com/svg.latex?&space;\left[2p_1\pi,2(p_1+1)\pi\right)\times\{2p_2\pi\}"/> for some integers <img src="https://latex.codecogs.com/svg.latex?&space;p_1,p_2"/>.
 </p>

Besides the aforementioned circular-circular data, there are many real-world datasets whose observations lie on a Euclidean/directional product space. For instance, in astronomical survey data, each object has its right ascension (RA) and declination (DEC) on a celestial sphere, while its redshift measures its distance to the Earth. The collection of (RA,DEC,Redshift) tuples thus forms a directional-linear dataset.

### 2. Mode and Ridge Estimation on Euclidean/directional product spaces with (Subspace Constrained) Mean Shift Algorithms

Our interested data consist of independent and identically distributed (i.i.d.) observations <img src="https://latex.codecogs.com/svg.latex?&space;\left\{\mathbf{Z}_i\right\}_{i=1}^n=\left\{(\mathbf{X}_i,\mathbf{Y}_i)\right\}_{i=1}^n"/> sampled from a distribution on <img src="https://latex.codecogs.com/svg.latex?&space;\mathcal{S}_1\times\mathcal{S}_2"/>, where 
<img src="https://latex.codecogs.com/svg.latex?&space;\mathcal{S}_j=\mathbb{R}^{D_j}\,\text{or}\,\Omega_{D_j}"/>
for <img src="https://latex.codecogs.com/svg.latex?&space;j=1,2"/>. While we only present the formulations of our proposed algorithms and related theory on (Cartesian) product spaces with two factors, our implementations (i.e., associated functions in **DirLinProdSCMS_fun.py** and **DirLinProdSCMS_Ray.py**) are adaptive to any product space with arbitrarily finte number of Euclidean/directional factor spaces.

#### 2.1 Kernel Density Estimator (KDE) on <img src="https://latex.codecogs.com/svg.latex?&space;\mathcal{S}_1\times\mathcal{S}_2"/>

It is natural to leverage a product kernel to construct a kernel density estimator (KDE) on <img src="https://latex.codecogs.com/svg.latex?&space;\mathbf{z}=(\mathbf{x},\mathbf{y})\in\mathcal{S}_1\times\mathcal{S}_2"/> as:

<img src="https://latex.codecogs.com/svg.latex?&space;\widehat{f}_{\mathbf{h}}(\mathbf{x},\mathbf{y})=\frac{1}{n}\sum_{i=1}^nK_1\left(\frac{\mathbf{x}-\mathbf{X}_i}{h_1}\right)K_2\left(\frac{\mathbf{y}-\mathbf{Y}_i}{h_2}\right)"/>,

where each element of <img src="https://latex.codecogs.com/svg.latex?&space;\mathbf{h}=(h_1,h_2)"/> is a bandwidth parameter and the kernel functions <img src="https://latex.codecogs.com/svg.latex?&space;K_j:\mathcal{S}_j\to\mathbb{R}"/> for <img src="https://latex.codecogs.com/svg.latex?&space;j=1,2"/> take the form as:

<img src="https://latex.codecogs.com/svg.latex?&space;K_j(\mathbf{u})&space;=&space;C_{k_j,D_j}(h_j)&space;\cdot&space;k_j\left(||\mathbf{u}||_2^2&space;\right)&space;=&space;\begin{cases}&space;\frac{C_{k,D_j}}{h_i^{D_j}}\cdot&space;k\left(\frac{||\mathbf{u}||_2^2}{2}&space;\right)&space;&&space;\text{&space;if&space;}&space;\mathcal{S}_j&space;=\mathbb{R}^{D_j},\\&space;C_{L,D_j}(h_j)&space;\cdot&space;L\left(\frac{||\mathbf{u}||_2^2}{2}&space;\right)&space;&&space;\text{&space;if&space;}&space;\mathcal{S}_j&space;=\Omega_{D_j},&space;\end{cases}"/>

with <img src="https://latex.codecogs.com/svg.latex?&space;k"/> and <img src="https://latex.codecogs.com/svg.latex?&space;L"/> being the profiles of linear and directional kernels, respectively. Under the Gaussian and/or von Mises kernels, i.e., <img src="https://latex.codecogs.com/svg.latex?&space;k(r)=L(r)=\exp(-r)"/>, the KDE reduces to the following concise form as:

<img src="https://latex.codecogs.com/svg.latex?&space;\widehat{f}_{\mathbf{h}}(\mathbf{z})&space;=&space;\frac{C(\mathbf{H})}{n}&space;\sum_{i=1}^n&space;\exp\left(-\frac{(\mathbf{z}-\mathbf{Z}_i)^T&space;\mathbf{H}^{-1}&space;(\mathbf{z}-\mathbf{Z}_i)}{2}&space;\right)"/>,

where <img src="https://latex.codecogs.com/svg.latex?&space;\mathbf{z}=(\mathbf{x},\mathbf{y})\in\mathcal{S}_1\times\mathcal{S}_2"/>,
<img src="https://latex.codecogs.com/svg.latex?\mathbf{H}&space;=\mathtt{Diag}\left(h_1^2\cdot\mathbf{I}_{D_1&plus;\mathbf{1}_{\{\mathcal{S}_1=\Omega_{D_1}\}}},&space;h_2^2\cdot\mathbf{I}_{D_2&space;&plus;&space;\mathbf{1}_{\{\mathcal{S}_2=\Omega_{D_2}\}}}&space;\right)" title="\mathbf{H} =\text{Diag}\left(h_1^2\cdot\mathbf{I}_{D_1+\mathbf{1}_{\{\mathcal{S}_1=\Omega_{D_1}\}}}, h_2^2\cdot\mathbf{I}_{D_2 + \mathbf{1}_{\{\mathcal{S}_2=\Omega_{D_2}\}}} \right)" />
is a (block) diagonal bandwidth matrix, <img src="https://latex.codecogs.com/svg.latex?&space;\mathbf{I}_D"/> is the identity matrix in <img src="https://latex.codecogs.com/svg.latex?&space;\mathbb{R}^{D\times&space;D}"/>, and <img src="https://latex.codecogs.com/svg.latex?\scriptsize&space;C(\mathbf{H}):=\prod_{j=1}^2C_{k_j,D_j}(h_j)"/> is the normalizing constant.

#### 2.2 Mean Shift Algorithm on <img src="https://latex.codecogs.com/svg.latex?&space;\mathcal{S}_1\times\mathcal{S}_2"/>

By taking the total gradient of KDE and equating each of its components to 0, we derive two different versions of the mean shift algorithm on <img src="https://latex.codecogs.com/svg.latex?&space;\mathcal{S}_1\times\mathcal{S}_2"/>; see more details in our paper.

* **Version A (Simultaneous Mean Shift).** This version updates all the components <img src="https://latex.codecogs.com/svg.latex?&space;\mathbf{z}^{(t)}=(\mathbf{x}^{(t)},\mathbf{y}^{(t)})=\mathcal{S}_1\times\mathcal{S}_2"/> simultaneously as:

<img src="https://latex.codecogs.com/svg.latex?\left(\mathbf{z}^{(t&plus;1)}&space;\right)^T&space;=\left(\mathbf{x}^{(t+1)},\mathbf{y}^{(t+1)}\right)^T\gets&space;\begin{pmatrix}&space;\frac{\sum\limits_{i=1}^n&space;\mathbf{X}_i&space;k_1'\left(\left|\left|\frac{\mathbf{x}^{(t)}&space;-\mathbf{X}_i}{h_1}\right|\right|_2^2&space;\right)&space;k_2\left(\left|\left|\frac{\mathbf{y}^{(t)}-\mathbf{Y}_i}{h_2}\right|\right|_2^2\right)&space;}{\sum\limits_{i=1}^n&space;k_1'\left(\left|\left|\frac{\mathbf{x}^{(t)}&space;-\mathbf{X}_i}{h_1}\right|\right|_2^2&space;\right)&space;k_2\left(\left|\left|\frac{\mathbf{y}^{(t)}-\mathbf{Y}_i}{h_2}\right|\right|_2^2\right)}\\&space;\frac{\sum\limits_{i=1}^n&space;\mathbf{Y}_i&space;k_1\left(\left|\left|\frac{\mathbf{x}^{(t)}-\mathbf{X}_i}{h_1}\right|\right|_2^2\right)&space;k_2'\left(\left|\left|\frac{\mathbf{y}^{(t)}-\mathbf{Y}_i}{h_2}\right|\right|_2^2&space;\right)&space;}{\sum\limits_{i=1}^n&space;k_1\left(\left|\left|\frac{\mathbf{x}^{(t)}&space;-\mathbf{X}_i}{h_1}\right|\right|_2^2\right)&space;k_2'\left(\left|\left|\frac{\mathbf{y}^{(t)}-\mathbf{Y}_i}{h_2}\right|\right|_2^2&space;\right)}&space;\end{pmatrix}"/>

for <img src="https://latex.codecogs.com/svg.latex?&space;t=0,1,..."/>, where we require extra standardizations <img src="https://latex.codecogs.com/svg.latex?&space;\mathbf{x}^{(t+1)}\gets\frac{\mathbf{x}^{(t+1)}}{||\mathbf{x}^{(t+1)}||_2}"/> and/or <img src="https://latex.codecogs.com/svg.latex?&space;\mathbf{y}^{(t+1)}\gets\frac{\mathbf{y}^{(t+1)}}{||\mathbf{y}^{(t+1)}||_2}"/> if <img src="https://latex.codecogs.com/svg.latex?&space;\mathcal{S}_1=\Omega_{D_1}"/> and/or <img src="https://latex.codecogs.com/svg.latex?&space;\mathcal{S}_2=\Omega_{D_2}"/>.

* **Version B (Componentwise Mean Shift).** This version updates the sequence <img src="https://latex.codecogs.com/svg.latex?&space;\left\{\mathbf{z}^{(t)}\right\}_{t=0}^{\infty}=\left\{(\mathbf{x}^{(t)},\mathbf{y}^{(t)})\right\}_{t=0}^{\infty}"/>in a two-step manner as:

<img src="https://latex.codecogs.com/svg.latex?\mathbf{x}^{(t&plus;1)}&space;\gets&space;\frac{\sum\limits_{i=1}^n&space;\mathbf{X}_i&space;k_1'\left(\left|\left|\frac{\mathbf{x}^{(t)}&space;-\mathbf{X}_i}{h_1}\right|\right|_2^2&space;\right)&space;k_2\left(\left|\left|\frac{\mathbf{y}^{(t)}-\mathbf{Y}_i}{h_2}\right|\right|_2^2\right)&space;}{\sum\limits_{i=1}^n&space;k_1'\left(\left|\left|\frac{\mathbf{x}^{(t)}&space;-\mathbf{X}_i}{h_1}\right|\right|_2^2&space;\right)&space;k_2\left(\left|\left|\frac{\mathbf{y}^{(t)}-\mathbf{Y}_i}{h_2}\right|\right|_2^2\right)}"/> with an additional standardization <img src="https://latex.codecogs.com/svg.latex?&space;\mathbf{x}^{(t+1)}\gets\frac{\mathbf{x}^{(t+1)}}{||\mathbf{x}^{(t+1)}||_2}"/> if <img src="https://latex.codecogs.com/svg.latex?&space;\mathcal{S}_1=\Omega_{D_1}"/>

and 

<img src="https://latex.codecogs.com/svg.latex?\mathbf{y}^{(t&plus;1)}&space;\gets\frac{\sum\limits_{i=1}^n&space;\mathbf{Y}_i&space;k_1\left(\left|\left|\frac{\mathbf{x}^{(t+1)}-\mathbf{X}_i}{h_1}\right|\right|_2^2\right)&space;k_2'\left(\left|\left|\frac{\mathbf{y}^{(t)}-\mathbf{Y}_i}{h_2}\right|\right|_2^2&space;\right)&space;}{\sum\limits_{i=1}^n&space;k_1\left(\left|\left|\frac{\mathbf{x}^{(t+1)}&space;-\mathbf{X}_i}{h_1}\right|\right|_2^2\right)&space;k_2'\left(\left|\left|\frac{\mathbf{y}^{(t)}-\mathbf{Y}_i}{h_2}\right|\right|_2^2&space;\right)}"/> with an additional standardization <img src="https://latex.codecogs.com/svg.latex?&space;\mathbf{y}^{(t+1)}\gets\frac{\mathbf{y}^{(t+1)}}{||\mathbf{y}^{(t+1)}||_2}"/> if <img src="https://latex.codecogs.com/svg.latex?&space;\mathcal{S}_2=\Omega_{D_2}"/>

for <img src="https://latex.codecogs.com/svg.latex?&space;t=0,1,..."/>. The formula updates the two components <img src="https://latex.codecogs.com/svg.latex?&space;\mathbf{x}^{(t)}"/> and <img src="https://latex.codecogs.com/svg.latex?&space;\mathbf{y}^{(t)}"/> alternatively by first holding <img src="https://latex.codecogs.com/svg.latex?&space;\mathbf{y}^{(t)}"/> , updating <img src="https://latex.codecogs.com/svg.latex?&space;\mathbf{x}^{(t)}"/>, and then switching their roles. Such updating procedures borrows the spirit of the well-known coordinate ascent/descent algorithm (Wright, 2015).

#### 2.3 SCMS Algorithm on <img src="https://latex.codecogs.com/svg.latex?&space;\mathcal{S}_1\times\mathcal{S}_2"/>

Naively, one may adopt the standard SCMS iterative formula in Ozertem and Erdogmus (2011) and update the SCMS sequence <img src="https://latex.codecogs.com/svg.latex?&space;\left\{\mathbf{z}^{(t)}\right\}_{t=0}^{\infty}=\left\{(\mathbf{x}^{(t)},\mathbf{y}^{(t)})\right\}_{t=0}^{\infty}\subset\mathcal{S}_1\times\mathcal{S}_2"/> as:

<img src="https://latex.codecogs.com/svg.latex?\mathbf{z}^{(t&plus;1)}&space;\gets&space;\mathbf{z}^{(t)}&space;&plus;&space;\widehat{V}_d(\mathbf{z}^{(t)})&space;\widehat{V}_d(\mathbf{z}^{(t)})^T\begin{pmatrix}&space;\frac{\sum\limits_{i=1}^n&space;\mathbf{X}_i&space;k_1'\left(\left|\left|\frac{\mathbf{x}^{(t)}&space;-\mathbf{X}_i}{h_1}\right|\right|_2^2&space;\right)&space;k_2\left(\left|\left|\frac{\mathbf{y}^{(t)}-\mathbf{Y}_i}{h_2}\right|\right|_2^2\right)&space;}{\sum\limits_{i=1}^n&space;k_1'\left(\left|\left|\frac{\mathbf{x}^{(t)}&space;-\mathbf{X}_i}{h_1}\right|\right|_2^2&space;\right)&space;k_2\left(\left|\left|\frac{\mathbf{y}^{(t)}-\mathbf{Y}_i}{h_2}\right|\right|_2^2\right)}-\mathbf{x}^{(t)}\\&space;\frac{\sum\limits_{i=1}^n&space;\mathbf{Y}_i&space;k_1\left(\left|\left|\frac{\mathbf{x}^{(t)}-\mathbf{X}_i}{h_1}\right|\right|_2^2\right)&space;k_2'\left(\left|\left|\frac{\mathbf{y}^{(t)}-\mathbf{Y}_i}{h_2}\right|\right|_2^2&space;\right)&space;}{\sum\limits_{i=1}^n&space;k_1\left(\left|\left|\frac{\mathbf{x}^{(t)}-\mathbf{X}_i}{h_1}\right|\right|_2^2\right)&space;k_2'\left(\left|\left|\frac{\mathbf{y}^{(t)}-\mathbf{Y}_i}{h_2}\right|\right|_2^2&space;\right)}-\mathbf{y}^{(t)}\end{pmatrix}"/>,

where <img src="https://latex.codecogs.com/svg.latex?&space;\widehat{V}_d(\mathbf{z})=\left[\widehat{\mathbf{v}}_{d+1}(\mathbf{z}),...,\widehat{\mathbf{v}}_{D_1+D_2}(\mathbf{z})\right]"/> has its columns as orthonormal eigenvectors of the (estimated) Riemannian Hessian <img src="https://latex.codecogs.com/svg.latex?&space;\mathcal{H}\widehat{f}_{\mathbf{h}}(\mathbf{z})"/> associated with the smallest <img src="https://latex.codecogs.com/svg.latex?&space;(D_1+D_2-d)"/> eigenvalues with the tangent space of <img src="https://latex.codecogs.com/svg.latex?&space;\mathcal{S}_1\times\mathcal{S}_2"/> at <img src="https://latex.codecogs.com/svg.latex?&space;\mathbf{z}"/>. This naive SCMS procedure, however, _does not_ converge to our interested ridges of KDE <img src="https://latex.codecogs.com/svg.latex?&space;\widehat{f}_{\mathbf{h}}"/>. What's worse, the incorrect ridges estimated by this naive SCMS procedure is also asymptotically invalid in estimating the ridges of the data-generating distribution.

Under the Gaussian and/or von Mises kernels, we formulate a valid SCMS iterative formula by rescaling each component of the mean shift vector with the bandwidth matrix <img src="https://latex.codecogs.com/svg.latex?&space;\mathbf{H}"/> as:

<img src="https://latex.codecogs.com/svg.latex?\mathbf{z}^{(t&plus;1)}&space;\gets\mathbf{z}^{(t)}&plus;\eta\cdot\hat{V}_d(\mathbf{z}^{(t)})&space;\hat{V}_d(\mathbf{z}^{(t)})^T\mathbf{H}^{-1}\begin{pmatrix}&space;\frac{\sum\limits_{i=1}^n&space;\mathbf{X}_i&space;k_1'\left(\left|\left|\frac{\mathbf{x}^{(t)}&space;-\mathbf{X}_i}{h_1}\right|\right|_2^2&space;\right)&space;k_2\left(\left|\left|\frac{\mathbf{y}^{(t)}-\mathbf{Y}_i}{h_2}\right|\right|_2^2&space;\right)&space;}{\sum\limits_{i=1}^n&space;k_1'\left(\left|\left|\frac{\mathbf{x}^{(t)}&space;-\mathbf{X}_i}{h_1}\right|\right|_2^2&space;\right)&space;k_2\left(\left|\left|\frac{\mathbf{y}^{(t)}-\mathbf{Y}_i}{h_2}\right|\right|&space;\right)}-\mathbf{x}^{(t)}&space;\\&space;\frac{\sum\limits_{i=1}^n&space;\mathbf{Y}_i&space;k_1\left(\left|\left|\frac{\mathbf{x}^{(t)}-\mathbf{X}_i}{h_1}\right|\right|_2^2&space;\right)k_2'\left(\left|\left|\frac{\mathbf{y}^{(t)}-\mathbf{Y}_i}{h_2}\right|\right|_2^2&space;\right)}{\sum\limits_{i=1}^n&space;k_1\left(\left|\left|\frac{\mathbf{x}^{(t)}-\mathbf{X}_i}{h_1}\right|\right|_2^2&space;\right)k_2'\left(\left|\left|\frac{\mathbf{y}^{(t)}-\mathbf{Y}_i}{h_2}\right|\right|_2^2&space;\right)}-\mathbf{y}^{(t)}&space;\end{pmatrix}"/>,

where <img src="https://latex.codecogs.com/svg.latex?&space;\eta"/> is the step size parameter managing the learning rate and convergence performance of our proposed SCMS algorithm. As a guideline, we suggest taking the step size to be adaptive to bandwidth parameters as:

<img src="https://latex.codecogs.com/svg.latex?&space;\eta=\min\{\max(\mathbf{h})\cdot\min(\bm{h}),1\}=\min\left\{h_1\cdot&space;h_2,1\right\}"/>

so that when <img src="https://latex.codecogs.com/svg.latex?&space;h_1,h_2\lesssim&space;h"/> are small, <img src="https://latex.codecogs.com/svg.latex?&space;\eta"/> mimics the asymptotic rate <img src="https://latex.codecogs.com/svg.latex?&space;O(h^2)"/> of adaptive step sizes in Euclidean/directional (subspace constrained) mean shift algorithms (Cheng, 1995; Arias-Castro et al., 2016; Zhang and Chen, 2021).

### 3. Example Code

The implementation of KDE in any Euclidean/directional product space is through the Python function called `DirLinProdKDE` in the script **DirLinProdSCMS_fun.py**.
Further, the implementations of simultaneous and componentwise mean shift algorithms are encapsulated into two Python functions called `DirLinProdMS` and `DirLinProdMSCompAsc` in the script **DirLinProdSCMS_fun.py**, respectively. The input arguments of `DirLinProdMS` and `DirLinProdMSCompAsc` are the same, and we notice that their outputs are identical, though the simultaneous version seems to be faster in the convergence speed. Finally, we implement our proposed SCMS algorithm in any Euclidean/directional product space on the Python functions `DirLinProdSCMS` and `DirLinProdSCMSLog` under log-density in the same script **DirLinProdSCMS_fun.py**. As the input arguments of `DirLinProdSCMSLog` subsume the ones of `DirLinProdKDE` and `DirLinProdMS`/`DirLinProdMSCompAsc`, we combine the descriptions of their arguments as follows:

`def DirLinProdKDE(x, data, h=[None,None], com_type=['Dir', 'Lin'], dim=[2,1]):`

`def DirLinProdMS(mesh_0, data, h=[None,None], com_type=['Dir','Lin'], dim=[2,1], eps=1e-7, max_iter=1000):`

`def DirLinProdMSCompAsc(mesh_0, data, h=[None,None], com_type=['Dir','Lin'], dim=[2,1], eps=1e-7, max_iter=1000):`

`def DirLinProdSCMSLog(mesh_0, data, d=1, h=[None,None], com_type=['Dir','Lin'], dim=[2,1], eps=1e-7, max_iter=1000, eta=None):`
    
- Parameters:
    - mesh_0: (m, sum(dim)+sum(com_type=='Dir'))-array
        ---- Eulidean coordinates of m query points in the product space, where 
        (dim\[0\]+1) / dim\[0\] is the Euclidean dimension of a directional/linear 
        component (first (dim\[0\]+1) columns), and so on.

    - data: (n, sum(dim)+sum(com_type=='Dir'))-array
        ---- Euclidean coordinates of n random sample points in the product space, 
        where (dim\[0\]+1) / dim\[0\] is the Euclidean dimension of a 
        directional/linear component (first (dim\[0\]+1) columns), and so on.
    
    - d: int
        ---- The order of the density ridge. (Default: d=1.)
   
    - h: list of floats
        ---- Bandwidth parameters for all the components. (Default: h=\[None\]*K, 
        where K is the number of components in the product space. Whenever
        h\[k\]=None for some k=1,...,K, then a rule of thumb for directional 
        KDE with the von Mises kernel in Garcia-Portugues (2013) is applied 
        to that directional component or the Silverman's rule of thumb is 
        applied to that linear component; see Chen et al.(2016) for details.)
        
    - com_type: list of strings
        ---- Indicators of the data type for all the components. If com_type\[k\]='Dir',
        then the corresponding component is directional. If com_type\[k\]='Lin', 
        then the corresponding component is linear.
        
    - dim: list of ints
        ---- Intrinsic data dimensions of all the directional/linear components.
   
    - eps: float
        ---- The precision parameter. (Default: eps=1e-7.)
   
    - max_iter: int
        ---- The maximum number of iterations for the SCMS algorithm on each 
        initial point. (Default: max_iter=1000.)
    
    - eta: float
        ---- The step size parameter for the SCMS algorithm. (Default: eta=None, 
        then eta=np.min(\[np.min(h) * np.max(h), 1\]).)
        
- Return:
    - SCMS_path: (m, sum(dim)+sum(com_type=='Dir'), T)-array
        ---- The entire iterative SCMS sequence for each initial point.

We also provide the corresponding implementations of the above functions under the [Ray](https://ray.io/) parallel programming environment as `DirLinProdKDE_Fast`, `DirLinProdMS_Fast`, `DirLinProdMSCompAsc_Fast`, and `DirLinProdSCMSLog_Fast` in the script **DirLinProdSCMS_Ray.py**.

Example code:
```bash
import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
from DirLinProdSCMS_fun import DirLinProdKDE, DirLinProdMS, DirLinProdSCMSLog
from Utility_fun import vMF_Gauss_mix

## Simulation 1: Mode-seeking on a directional-linear space $\Omega_1 \times \mathbb{R}$
np.random.seed(123)  ## Set an arbitrary seed for reproducibility
prob1 = [2/5, 1/5, 2/5]   ## Mixture probabilities
mu_N1 = np.array([[0], [1], [2]])  ## Means of the Gaussian component
cov1 = np.array([1/4, 1, 1]).reshape(1,1,3)   ## Variances of the Gaussian components
mu_vMF1 = np.array([[1, 0], [0, 1], [-1, 0]])   ## Means of the vMF components
kappa1 = [3, 10, 3]   ## Concentration parameters of the vMF components
# Sample 1000 points from the vMF-Gaussian mixture model
vMF_Gau_data = vMF_Gauss_mix(1000, q=1, D=1, mu_vMF=mu_vMF1, kappa=kappa1, 
                             mu_N=mu_N1, cov=cov1, prob=prob1)
# Convert the vMF components of the simulated data to their angular coordinates
Angs = np.arctan2(vMF_Gau_data[:,1], vMF_Gau_data[:,0])
vMF_Gau_Ang = np.concatenate([Angs.reshape(-1,1), vMF_Gau_data[:,2].reshape(-1,1)], axis=1)

# Bandwidth selection
data = vMF_Gau_data
n = vMF_Gau_data.shape[0]
q = 1
D = 1
data_Dir = data[:,:(q+1)]
data_Lin = data[:,(q+1):(q+1+D)]
## Rule-of-thumb bandwidth selector for the directional component
R_bar = np.sqrt(sum(np.mean(data_Dir, axis=0) ** 2))
kap_hat = R_bar * (q + 1 - R_bar ** 2) / (1 - R_bar ** 2)
h = ((4 * np.sqrt(np.pi) * sp.iv((q-1) / 2 , kap_hat)**2) / \
     (n * kap_hat ** ((q+1) / 2) * (2 * q * sp.iv((q+1)/2, 2*kap_hat) + \
     (q+2) * kap_hat * sp.iv((q+3)/2, 2*kap_hat)))) ** (1/(q + 4))
bw_Dir = h
print("The current bandwidth for directional component is " + str(h) + ".\n")
## Normal reference rule of bandwidth selector for the linear component
b = (4/(D+2))**(1/(D+4))*(n**(-1/(D+4)))*np.mean(np.std(data_Lin, axis=0))
bw_Lin = b
print("The current bandwidth for linear component is "+ str(b) + ".\n")

# Set up a set of mesh points and estimate the density values on it
nrows, ncols = (100, 100)
ang_qry = np.linspace(-np.pi-0.1, np.pi+0.1, nrows)
lin_qry = np.linspace(-2, 5.5, ncols)
ang_m1, lin_m1 = np.meshgrid(ang_qry, lin_qry)
X = np.cos(ang_m1.reshape(-1,1))
Y = np.sin(ang_m1.reshape(-1,1))
mesh1 = np.concatenate([X.reshape(-1,1), Y.reshape(-1,1), 
                        lin_m1.reshape(-1,1)], axis=1)
d_DirLin = DirLinProdKDE(mesh1, data=vMF_Gau_data, h=[bw_Dir, bw_Lin], 
                         com_type=['Dir','Lin'], dim=[1,1]).reshape(nrows, ncols)

# below 5% density quantile
d_DirLin_dat = DirLinProdKDE(vMF_Gau_data, vMF_Gau_data, h=[bw_Dir, bw_Lin], 
                             com_type=['Dir','Lin'], dim=[1,1])
vMF_Gau_data_thres = vMF_Gau_data[d_DirLin_dat > np.quantile(d_DirLin_dat, 0.05)]

# Mode-seeking on the denoised data with our proposed mean shift algorithm
DLMS_path = DirLinProdMS(vMF_Gau_data, vMF_Gau_data_thres, h=[bw_Dir, bw_Lin], com_type=['Dir','Lin'], 
                         dim=[1,1], eps=1e-7, max_iter=3000)

## Simulation 2: Ridge-finding on a directional-linear space $\Omega_1 \times \mathbb{R}$
N = 1000
sigma = 0.3
np.random.seed(123)  ## Set an arbitrary seed for reproducibility
# Simulated a curve with additive Gaussian noises on a cylinder (directional-linear case)
t = np.random.rand(N)*2*np.pi - np.pi
t_p = t + np.random.randn(1000) * sigma
X_p = np.cos(t_p)
Y_p = np.sin(t_p)
Z_p = t/2 + np.random.randn(1000) * sigma
cur_dat = np.concatenate([X_p.reshape(-1,1), Y_p.reshape(-1,1), 
                          Z_p.reshape(-1,1)], axis=1)
# Use the default bandwidths
bw_Dir = None
bw_Lin = None

# Create a set of mesh points and estimate the density value on it
nrows, ncols = (100, 100)
ang_qry = np.linspace(-np.pi, np.pi, nrows)
lin_qry = np.linspace(-2.5, 2.5, ncols)
ang_m2, lin_m2 = np.meshgrid(ang_qry, lin_qry)
X = np.cos(ang_m2.reshape(-1,1))
Y = np.sin(ang_m2.reshape(-1,1))
qry_pts = np.concatenate((X.reshape(-1,1), 
                          Y.reshape(-1,1), 
                          lin_m2.reshape(-1,1)), axis=1)
d_DirLinProd = DirLinProdKDE(qry_pts, cur_dat, h=[bw_Dir, bw_Lin], 
                             com_type=['Dir','Lin'], dim=[1,1]).reshape(ncols, nrows)

# Proposed SCMS algorithm with our rule-of-thumb step size eta=h1*h2
ProdSCMS_DL_p, lab_DL_p = DirLinProdSCMSLog(cur_dat, cur_dat, d=1, h=[bw_Dir,bw_Lin], 
                                            com_type=['Dir','Lin'], dim=[1,1], 
                                            eps=1e-7, max_iter=5000, eta=None)

## Plotting the results
fig = plt.figure(figsize=(16,10))
# Create a cylinder for the directional-linear space
theta = np.linspace(-np.pi, np.pi, 100)
z = np.linspace(-2, 5, 100)
th_m, Zc = np.meshgrid(theta, z)
Xc = np.cos(th_m)
Yc = np.sin(th_m)
# Plot the simulated data points and local modes on the cylinder
step = DLMS_path.shape[2] - 1
Modes_angs = np.arctan2(DLMS_path[:,1,step], DLMS_path[:,0,step])
ax = fig.add_subplot(221, projection='3d')
ax.view_init(30, 60)
ax.plot_surface(Xc, Yc, Zc, alpha=0.2, color='grey')
ax.scatter(vMF_Gau_data[:,0], vMF_Gau_data[:,1], vMF_Gau_data[:,2], 
           alpha=0.2, color='deepskyblue')
ax.scatter(DLMS_path[:,0,step], DLMS_path[:,1,step], DLMS_path[:,2,step], 
           color='red', s=40)
ax.axis('off')
plt.title('Simulated vMF-Gaussian mixture data and local modes \n estimated '\
          'by our mean shift algorithm on a cylinder')

# Plot the local modes on the contour plot of the estimated density
step = DLMS_path.shape[2] - 1
Modes_angs = np.arctan2(DLMS_path[:,1,step], DLMS_path[:,0,step])
plt.subplot(222)
plt.scatter(Angs, vMF_Gau_data[:,2], alpha=1)
plt.contourf(ang_m1, lin_m1, d_DirLin, 10, cmap='OrRd', alpha=0.7)
plt.colorbar()
plt.scatter(Modes_angs, DLMS_path[:,2,step], color='red', s=40)
plt.title('Estimated local modes on the contour plot of KDE')

# Plot the simulated data and estimated ridge on a cylinder
step_DL_p = ProdSCMS_DL_p.shape[2] - 1
ax = fig.add_subplot(223, projection='3d')
ax.view_init(30, 10)
## Mesh points on the cylinder
theta = np.linspace(-np.pi, np.pi, 100)
z = np.linspace(-2, 2, 100)
th_m, Zc = np.meshgrid(theta, z)
Xc = np.cos(th_m)
Yc = np.sin(th_m)
## True curve structure
t = np.linspace(-np.pi, np.pi, 200)
X_cur = np.cos(t)
Y_cur = np.sin(t)
Z_cur = t/2
ax.plot_surface(Xc, Yc, Zc, alpha=0.2)
ax.plot(X_cur, Y_cur, Z_cur, linewidth=5, color='green')
ax.scatter(ProdSCMS_DL_p[:,0,step_DL_p], ProdSCMS_DL_p[:,1,step_DL_p], 
           ProdSCMS_DL_p[:,2,step_DL_p], alpha=0.5, color='deepskyblue')
ax.axis('off')
plt.title('Simulated data and density ridges \n estimated '\
          'by our SCMS algorithm on a cylinder')

# Plot the estimated ridge on the contour plot of estimated density
plt.subplot(224)
plt.contourf(ang_m2, lin_m2, d_DirLinProd, 10, cmap='OrRd', alpha=0.5)
plt.colorbar()
Ridges_angs_p = np.arctan2(ProdSCMS_DL_p[:,1,step_DL_p], 
                           ProdSCMS_DL_p[:,0,step_DL_p])
plt.scatter(Ridges_angs_p, ProdSCMS_DL_p[:,2,step_DL_p], color='deepskyblue', alpha=0.6)
plt.xlabel('Directional Coordinate')
plt.ylabel('Linear Coordinate')
plt.title('Estimated density ridges on the contour plot of KDE')
fig.tight_layout()
fig.savefig('./Figures/DirLin_example.png')
```



### Additional References
 - R. Ahumada, C. A.Prieto, A. Almeida, F. Anders, S. F. Anderson, B. H. Andrews, B. Anguiano, R. Arcodia, E. Armengaud, M. Aubert, et al. The 16th data release of the sloan digital sky surveys: first release from the apogee-2 southern survey and full release of eboss spectra. _The Astrophysical Journal Supplement Series_, 249(1):3, 2020.
 - Y. Cheng. Mean shift, mode seeking, and clustering. _IEEE Transactions on Pattern Analysis and Machine Intelligence_, 17(8):790–799, 1995.
 - D. Comaniciu and P. Meer. Mean shift: a robust approach toward feature space analysis. _IEEE Transactions on Pattern Analysis and Machine Intelligence_, 24(5):603–619, 2002.
 - U. Ozertem and D. Erdogmus. Locally defined principal curves and surfaces. _Journal of Machine Learning Research_, 12(34):1249–1286, 2011.
 - S. J. Wright. Coordinate descent algorithms. _Mathematical Programming_, 151(1):3–34, 2015.
 - E. Arias-Castro, D. Mason, and B. Pelletier. On the estimation of the gradient lines of a density and the consistency of the mean-shift algorithm. _Journal of Machine Learning Research_, 17(43):1–28, 2016.
 - Y. Zhang and Y.-C. Chen. Linear convergence of the subspace constrained mean shift algorithm: From euclidean to directional data. arXiv preprint [arXiv:2104.14977](https://arxiv.org/abs/2104.14977), 2021.
 - E. Garcı́a-Portugués (2013). Exact risk improvement of bandwidth selectors for kernel density estimation with directional data. _Electronic Journal of Statistics_ **7** 1655–1685.
- Y.-C. Chen, C. Genovese, and L. Wasserman (2016). [A comprehensive approach to mode clustering](https://projecteuclid.org/euclid.ejs/1455715961). _Electronic Journal of Statistics_ **10**(1) 210-241.

