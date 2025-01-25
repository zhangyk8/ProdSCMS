#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang

Last Editing: January 23, 2025

Description: This script contains code for mode-seeking simulation studies with
our proposed mean shift algorithm (Simulation 2 in the paper).
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


## Simulation 2: Mode-seeking on a directional-directional space $\Omega_1 \times \Omega_1$
for N in [500, 1000, 2000, 5000]:
    # Simulate points from an independent product of two vMF densities
    p1,p2,p3,p4 = 1/2,1/2,1/2,1/2
    mu1,mu2,mu3,mu4 = 0, np.pi/2, 0, 3*np.pi/4
    kap1, kap2, kap3, kap4 = 5,5,7,7
    sim_dat1 = np.zeros((N, 2))
    np.random.seed(job_id)  ## Set an arbitrary seed for reproducibility
    mix_ind1 = np.random.choice(list(range(2)), N, replace=True, 
                                p=np.array([p1, p2])/sum([p1, p2]))
    mix_ind2 = np.random.choice(list(range(2)), N, replace=True, 
                                p=np.array([p3, p4])/sum([p3, p4]))
    # theta: Toroidal direction
    sim_dat1[mix_ind1 == 0, 0] = np.random.vonmises(mu1, kap1, 
                                                    size=sum(mix_ind1 == 0))
    sim_dat1[mix_ind1 == 1, 0] = np.random.vonmises(mu2, kap2, 
                                                    size=sum(mix_ind1 == 1))
    # phi: Poloidal direction
    sim_dat1[mix_ind2 == 0, 1] = np.random.vonmises(mu3, kap3, 
                                                    size=sum(mix_ind2 == 0))
    sim_dat1[mix_ind2 == 1, 1] = np.random.vonmises(mu4, kap4, 
                                                    size=sum(mix_ind2 == 1))
    sim_dat1_ang = np.copy(sim_dat1)
    sim_dat1_cart = np.concatenate([np.cos(sim_dat1[:,0]).reshape(-1, 1),
                                    np.sin(sim_dat1[:,0]).reshape(-1, 1),
                                    np.cos(sim_dat1[:,1]).reshape(-1, 1),
                                    np.sin(sim_dat1[:,1]).reshape(-1, 1)], axis=1)

    true_DM = np.array([[mu1, mu3], [mu1, mu4],
                        [mu2, mu3], [mu2, mu4]])

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


    with open('./Results/Simulation2_N_'+str(N)+'_'+str(job_id)+'.dat', "wb") as file:
        pickle.dump([avg_EuMS, avg_DirLinMS], file)