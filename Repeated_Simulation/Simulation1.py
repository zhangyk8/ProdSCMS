#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang

Last Editing: June 25, 2024

Description: This script contains code for mode-seeking simulation studies with
our proposed mean shift algorithm (Figure 3 in the arxiv version of the paper).
"""

import numpy as np
import pandas as pd
import scipy.special as sp
import sys
import pickle
from numpy import linalg as LA
from DirLinProdSCMS_fun import DirLinProdKDE, DirLinProdMS, DirLinProdMSCompAsc
from Utility_fun import vMF_Gauss_mix, Unique_Modes

from SCMS_fun import KDE, MS_KDE
from DirSCMS_fun import DirKDE, MS_DirKDE

job_id = int(sys.argv[1])
print(job_id)

#=======================================================================================#


def CompDist(x, ref_dat):
    return np.min(LA.norm(ref_dat - x.values, axis=1))


## Simulation 1: Mode-seeking on a directional-linear space $\Omega_1 \times \mathbb{R}$
np.random.seed(job_id)  ## Set an arbitrary seed for reproducibility
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
vMF_Gau_Ang = np.concatenate([Angs.reshape(-1,1), 
                              vMF_Gau_data[:,2].reshape(-1,1)], axis=1)

# True local mode
true_mode = np.concatenate([mu_vMF1, mu_N1], axis=1)

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
ang_m, lin_m = np.meshgrid(ang_qry, lin_qry)
X = np.cos(ang_m.reshape(-1,1))
Y = np.sin(ang_m.reshape(-1,1))
mesh1 = np.concatenate([X.reshape(-1,1), Y.reshape(-1,1), 
                        lin_m.reshape(-1,1)], axis=1)
d_DirLin = DirLinProdKDE(mesh1, data=vMF_Gau_data, h=[bw_Dir, bw_Lin], 
                         com_type=['Dir','Lin'], dim=[1,1]).reshape(nrows, ncols)
# Estimate the density values on the simulated data and remove data points
# below 5% density quantile
d_DirLin_dat = DirLinProdKDE(vMF_Gau_data, vMF_Gau_data, h=[bw_Dir, bw_Lin], 
                             com_type=['Dir','Lin'], dim=[1,1])
vMF_Gau_data_thres = vMF_Gau_data[d_DirLin_dat > np.quantile(d_DirLin_dat, 0.05)]

# Mode-seeking on the denoised data with our proposed mean shift algorithm
DLMS_path = DirLinProdMS(vMF_Gau_data, vMF_Gau_data_thres, h=[bw_Dir, bw_Lin], 
                         com_type=['Dir','Lin'], dim=[1,1], eps=1e-7, 
                         max_iter=3000)
DLMS_path2 = DirLinProdMSCompAsc(vMF_Gau_data, vMF_Gau_data_thres, 
                                 h=[bw_Dir, bw_Lin], com_type=['Dir','Lin'], 
                                 dim=[1,1], eps=1e-7, max_iter=3000)
DL_mode1, lab1 = Unique_Modes(DLMS_path[:,:,DLMS_path.shape[2]-1], tol=1e-3)
DL_mode2, lab2 = Unique_Modes(DLMS_path2[:,:,DLMS_path2.shape[2]-1], tol=1e-3)
print('\n The Euclidean norms between the local modes obtained by Version A '\
      'and B on the simulated directional-linear data are')
print(LA.norm(DL_mode1 - DL_mode2, axis=1))
print('\n The local modes obtained by our proposed mean shift algorithm are ')
print(DL_mode1)

DL1_to_true = pd.DataFrame(DL_mode1).apply(lambda x: CompDist(x, ref_dat=true_mode), axis=1)
true_to_DL1 = pd.DataFrame(true_mode).apply(lambda x: CompDist(x, ref_dat=DL_mode1), axis=1)
print('The Hausdorff distance between the estimated and true modes for the simulaneous MS algorithm is '\
      +str(max([max(DL1_to_true), max(true_to_DL1)])))
avg_DirLinMS = np.mean(DL1_to_true)


## Method (i): standard Euclidean mean shift on R^3
d_Eu_dat = KDE(vMF_Gau_data, vMF_Gau_data, h=None)
vMF_Gau_data_thres_Eu = vMF_Gau_data[d_Eu_dat > np.quantile(d_Eu_dat, 0.05)]
EuMS_path = MS_KDE(vMF_Gau_data_thres_Eu, data=vMF_Gau_data, h=None, eps=1e-7, max_iter=3000)

Eu_mode, lab_Eu = Unique_Modes(EuMS_path[:,:,EuMS_path.shape[2]-1], tol=1e-3)
Eu_mode[:,:2] = Eu_mode[:,:2]/LA.norm(Eu_mode[:,:2], axis=1).reshape(-1,1)
print(Eu_mode)

Eu_to_true = pd.DataFrame(Eu_mode).apply(lambda x: CompDist(x, ref_dat=true_mode), axis=1)
true_to_Eu = pd.DataFrame(true_mode).apply(lambda x: CompDist(x, ref_dat=Eu_mode), axis=1)
print('The Hausdorff distance between the estimated and true modes for the Euclidean MS algorithm is '\
      +str(max([max(Eu_to_true), max(true_to_Eu)])))
avg_EuMS = np.mean(Eu_to_true)


## Method (ii): directional mean shift algorithm on Omega_1 and the Euclidean mean shift algorithm on R independently
d_Dir_dat2 = DirKDE(vMF_Gau_data[:,:2], vMF_Gau_data[:,:2], h=None)
vMF_data_thres = vMF_Gau_data[d_Dir_dat2 > np.quantile(d_Dir_dat2, 0.05),:2]
d_Eu_dat2 = KDE(vMF_Gau_data[:,2].reshape(-1,1), vMF_Gau_data[:,2].reshape(-1,1), h=None)
Gau_data_thres = vMF_Gau_data[d_Eu_dat2 > np.quantile(d_Eu_dat2, 0.05),2]

DirMS_path2 = MS_DirKDE(vMF_data_thres, data=vMF_Gau_data[:,:2], h=None, eps=1e-9, max_iter=3000)
EuMS_path2 = MS_KDE(Gau_data_thres.reshape(-1,1), data=vMF_Gau_data[:,2].reshape(-1,1), h=None, 
                    eps=1e-7, max_iter=3000)
Dir_mode2, lab_Dir2 = Unique_Modes(DirMS_path2[:,:,DirMS_path2.shape[2]-1], tol=1e-3)
Eu_mode2, lab_Eu2 = Unique_Modes(EuMS_path2[:,:,EuMS_path2.shape[2]-1], tol=1e-3)
print(Eu_mode2)
DirEu_mode = np.concatenate([Dir_mode2, Eu_mode2*np.ones([3,1])], axis=1)
print(DirEu_mode)

DirEu_to_true = pd.DataFrame(DirEu_mode).apply(lambda x: CompDist(x, ref_dat=true_mode), axis=1)
true_to_DirEu = pd.DataFrame(true_mode).apply(lambda x: CompDist(x, ref_dat=DirEu_mode), axis=1)
print('The Hausdorff distance between the estimated and true modes by applying Euclidean'\
      ' and directional MS algorithms separately is '+str(max([max(DirEu_to_true), max(true_to_DirEu)])))
avg_EuDir = np.mean(DirEu_to_true)

with open('./Results/Simulation1_'+str(job_id)+'.dat', "wb") as file:
    pickle.dump([avg_EuMS, avg_EuDir, avg_DirLinMS], file)
