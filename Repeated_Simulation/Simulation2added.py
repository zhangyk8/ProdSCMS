#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang

Last Editing: January 26, 2025

Description: This script contains code for mode-seeking simulation studies with
our proposed mean shift algorithm (Simulation 2 with bivariate von Mises 
distribution in the paper).
"""

import numpy as np
import pandas as pd
import sys
import pickle
from numpy import linalg as LA
from DirLinProdSCMS_fun import DirLinProdKDE, DirLinProdMS, DirLinProdMSCompAsc
from Utility_fun import Unique_Modes, BiVonMisesSampling

from SCMS_fun import KDE, MS_KDE

job_id = int(sys.argv[1])
print(job_id)

#=======================================================================================#


def CompDist(x, ref_dat):
    return np.min(LA.norm(ref_dat - x.values, axis=1))


## Simulation 2: Mode-seeking on a directional-directional space $\Omega_1 \times \Omega_1$
## with bivariate von Mises distribution
for N in [500, 1000, 2000]:
    # Simulate points from an independent product of two vMF densities
    prob = np.array([1/2, 1/2])
    np.random.seed(job_id)  ## Set an arbitrary seed for reproducibility
    inds = np.random.choice(list(range(len(prob))), N, replace=True, p=prob)
    bivm_mix = np.zeros([N, 2])
    bivm_mix[inds == 0,:] = BiVonMisesSampling(sum(inds==0), mu1=0, mu2=0, kappa1=10, 
                                           kappa2=10, A=np.array([[-1,0.1], [0.1,1]]))
    bivm_mix[inds == 1,:] = BiVonMisesSampling(sum(inds==1), mu1=3*np.pi/4, mu2=np.pi/2, kappa1=5, 
                                           kappa2=5, A=np.array([[0,0], [0,1]]))
    sim_dat1 = np.copy(bivm_mix)
    sim_dat1_ang = np.copy(sim_dat1)
    sim_dat1_cart = np.concatenate([np.cos(sim_dat1[:,0]).reshape(-1, 1),
                                    np.sin(sim_dat1[:,0]).reshape(-1, 1),
                                    np.cos(sim_dat1[:,1]).reshape(-1, 1),
                                    np.sin(sim_dat1[:,1]).reshape(-1, 1)], axis=1)

    true_DM = np.array([[0, 0], [3*np.pi/4, np.pi/2]])

    # Create a set of mesh points and estimate the density on it
    nrows, ncols = (200, 100)
    th_m, phi_m = np.meshgrid(np.linspace(-np.pi, np.pi, ncols), 
                              np.linspace(-np.pi, np.pi, nrows))
    query_pts = np.concatenate((np.cos(th_m).reshape(nrows*ncols, 1),
                                np.sin(th_m).reshape(nrows*ncols, 1),
                                np.cos(phi_m).reshape(nrows*ncols, 1),
                                np.sin(phi_m).reshape(nrows*ncols, 1)), axis=1)
    query_pts_ang = np.concatenate((th_m.reshape(nrows*ncols, 1),
                                    phi_m.reshape(nrows*ncols, 1)), axis=1)
    # KDE in a circular-circular space
    d_Dir = DirLinProdKDE(query_pts, sim_dat1_cart, h=[None, None], 
                          com_type=['Dir','Dir'], dim=[1,1]).reshape(nrows, ncols)

    # Mode-seeking on the simulated data with our proposed mean shift algorithm
    DDMS_path = DirLinProdMS(sim_dat1_cart, sim_dat1_cart, h=[None,None], 
                             com_type=['Dir','Dir'], dim=[1,1], eps=1e-7, 
                             max_iter=3000)
    DDMS_path2 = DirLinProdMSCompAsc(sim_dat1_cart, sim_dat1_cart, h=[None,None], 
                                     com_type=['Dir','Dir'], dim=[1,1], 
                                     eps=1e-7, max_iter=3000)
    step = DDMS_path.shape[2] -1
    Modes_angs1 = np.arctan2(DDMS_path[:,1,step], DDMS_path[:,0,step])
    Modes_angs2 = np.arctan2(DDMS_path[:,3,step], DDMS_path[:,2,step])
    DDModes = np.concatenate([Modes_angs1.reshape(-1,1), 
                              Modes_angs2.reshape(-1,1)], axis=1)
    step = DDMS_path2.shape[2] -1
    Modes_angs1 = np.arctan2(DDMS_path2[:,1,step], DDMS_path2[:,0,step])
    Modes_angs2 = np.arctan2(DDMS_path2[:,3,step], DDMS_path2[:,2,step])
    DDModes2 = np.concatenate([Modes_angs1.reshape(-1,1), 
                               Modes_angs2.reshape(-1,1)], axis=1)
    DD_mode1, lab1 = Unique_Modes(DDModes, tol=1e-3)
    DD_mode2, lab2 = Unique_Modes(DDModes2, tol=1e-3)
    print('\n The Euclidean norms between the local modes obtained by Version A '\
          'and B on the simulated directional-linear data are')
    print(LA.norm(DD_mode1 - DD_mode2, axis=1))
    print('\n The local modes obtained by our proposed mean shift algorithm are ')
    print(DD_mode1)


    DD1_to_true = pd.DataFrame(DD_mode1).apply(lambda x: CompDist(x, ref_dat=true_DM), axis=1)
    true_to_DD1 = pd.DataFrame(true_DM).apply(lambda x: CompDist(x, ref_dat=DD_mode1), axis=1)
    print('The Hausdorff distance between the estimated and true modes for the simulaneous MS algorithm is '\
          +str(max([max(DD1_to_true), max(true_to_DD1)])))
    print(np.mean(true_to_DD1))

    DD2_to_true = pd.DataFrame(DD_mode2).apply(lambda x: CompDist(x, ref_dat=true_DM), axis=1)
    true_to_DD2 = pd.DataFrame(true_DM).apply(lambda x: CompDist(x, ref_dat=DD_mode2), axis=1)
    print('The Hausdorff distance between the estimated and true modes for the componentwise MS algorithm is '\
          +str(max([max(DD2_to_true), max(true_to_DD2)])))
    avg_DirLinMS = np.mean(true_to_DD2)


    EuMS_path = MS_KDE(sim_dat1_cart, sim_dat1_cart, h=None, eps=1e-7, max_iter=3000)
    step = EuMS_path.shape[2] - 1
    Modes_angs1 = np.arctan2(EuMS_path[:,1,step], EuMS_path[:,0,step])
    Modes_angs2 = np.arctan2(EuMS_path[:,3,step], EuMS_path[:,2,step])
    EuModes = np.concatenate([Modes_angs1.reshape(-1,1), Modes_angs2.reshape(-1,1)], axis=1)

    EuModes1, lab1 = Unique_Modes(EuModes, tol=1e-3)
    print(EuModes1)
    EuDD_to_true = pd.DataFrame(EuModes1).apply(lambda x: CompDist(x, ref_dat=true_DM), axis=1)
    true_to_EuDD = pd.DataFrame(true_DM).apply(lambda x: CompDist(x, ref_dat=EuModes1), axis=1)
    print('The Hausdorff distance between the estimated and true modes for the Euclidean MS algorithm is '\
          +str(max([max(EuDD_to_true), max(true_to_EuDD)])))
    avg_EuMS = np.mean(true_to_EuDD)


    with open('./Results/Simulation2_N_'+str(N)+'_'+str(job_id)+'_added.dat', "wb") as file:
        pickle.dump([avg_EuMS, avg_DirLinMS], file)