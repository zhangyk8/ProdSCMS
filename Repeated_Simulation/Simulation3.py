#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang

Last Editing: June 25, 2024

Description: This script contains code for comparing the results of the regular 
SCMS and our proposed SCMS algorithms on the simulated spiral curve data 
(Simulation 3 in the paper).
"""

import numpy as np
import pandas as pd
from numpy import linalg as LA
import sys
import pickle
import ray
from Utility_fun import cart2sph, sph2cart
from MS_SCMS_Ray import SCMS_Log_KDE_Fs
from DirLinProdSCMS_Ray import DirLinProdSCMSLog_Fast

# job_id = int(sys.argv[1])
# print(job_id)

#=======================================================================================#


def DistToCurve(x, true_cur):
    return min(LA.norm(x.values - true_cur, axis=1))

for job_id in range(1, 1001):
    ## Simulation 3: Ridge-Finding on $\Omega_2 \times R$
    N = 1000
    open_ang = np.pi/3
    sigma = 0.2   ## Variance of the additive Gaussian noises
    np.random.seed(job_id)  ## Set an arbitrary seed for reproducibility
    t = np.random.rand(N, 1)*4
    # Simulated points on the spiral curve data with angular-linear coordinates
    cur_dat_ang = np.concatenate([5*t, np.ones((N, 1))*(np.pi/2-open_ang), t], 
                                 axis=1)
    # Convert the first two radian coordinates to their degree measures
    cur_dat_ang[:,:2] = (cur_dat_ang[:,:2] % (2*np.pi))/np.pi * 180
    # Add some Gaussian noises
    cur_dat_ang[:,:2] = (cur_dat_ang[:,:2] + sigma*np.random.randn(N, 2)) % 360
    cur_dat_ang[:,2] = cur_dat_ang[:,2] + sigma*np.random.randn(N)
    # Convert the angular-linear coordinates of simulated data to their 
    # directional-linear coordinates
    X, Y, Z = sph2cart(*cur_dat_ang[:,:2].T)
    cur_dat = np.concatenate([X.reshape(-1,1), Y.reshape(-1,1), Z.reshape(-1,1), 
                              cur_dat_ang[:,2].reshape(-1,1)], axis=1)
    # Convert the angular-linear coordinates of simulated data to their 
    # Cartesian coordinates
    phi_pert = (cur_dat_ang[:,1]/180) * np.pi
    th_pert = (cur_dat_ang[:,0]/180) * np.pi
    R_pert = cur_dat_ang[:,2]
    Z_sim = R_pert * np.sin(phi_pert)
    X_sim = R_pert * np.cos(phi_pert) * np.cos(th_pert)
    Y_sim = R_pert * np.cos(phi_pert) * np.sin(th_pert)
    cur_dat_3D = np.concatenate([X_sim.reshape(-1,1), Y_sim.reshape(-1,1), 
                                 Z_sim.reshape(-1,1)], axis=1)

    # Regular SCMS on the simulated data with 3D Cartesian coordinates
    ray.init(include_dashboard=False)
    mesh_0 = cur_dat_3D
    dataset = cur_dat_3D
    bw_Eu1 = None
    chunksize = 10
    num_p = mesh_0.shape[0]
    result_ids = []
    for i in range(0, num_p, chunksize):
        result_ids.append(SCMS_Log_KDE_Fs.remote(mesh_0[i:(i+chunksize)], dataset, 
                                                 d=1, h=bw_Eu1, eps=1e-7, 
                                                 max_iter=5000))
    EuSCMS_pts1 = ray.get(result_ids)
    EuSCMS_pts1 = np.concatenate(EuSCMS_pts1, axis=0)
    ray.shutdown()


    # Regular SCMS on the simulated data with 3D angular-linear coordinates 
    # i.e., (longitudes, latitudes, linear covariates)
    ray.init(include_dashboard=False)
    mesh_0 = cur_dat_ang
    dataset = cur_dat_ang
    bw_Eu2 = None
    chunksize = 10
    num_p = mesh_0.shape[0]
    result_ids = []
    for i in range(0, num_p, chunksize):
        result_ids.append(SCMS_Log_KDE_Fs.remote(mesh_0[i:(i+chunksize)], dataset, 
                                                 d=1, h=bw_Eu2, eps=1e-7, 
                                                 max_iter=5000))
    EuSCMS_pts2 = ray.get(result_ids)
    EuSCMS_pts2 = np.concatenate(EuSCMS_pts2, axis=0)
    ray.shutdown()

    Phi = (EuSCMS_pts2[:,0]/180)*np.pi
    Eta = (EuSCMS_pts2[:,1]/180)*np.pi
    Eu_Ridges2 = np.concatenate([(EuSCMS_pts2[:,2]*np.cos(Phi)*np.cos(Eta)).reshape(-1,1), 
                                 (EuSCMS_pts2[:,2]*np.sin(Phi)*np.cos(Eta)).reshape(-1,1),
                                 (EuSCMS_pts2[:,2]*np.sin(Eta)).reshape(-1,1)], axis=1)

    # Our proposed DirLin SCMS on the simulated data in the directional-linear 
    # space $\Omega_2 \times \mathbb{R}$
    ray.init(include_dashboard=False)
    mesh_0 = cur_dat
    dataset = cur_dat
    chunksize = 10
    num_p = mesh_0.shape[0]
    result_ids = []
    for i in range(0, num_p, chunksize):
        result_ids.append(DirLinProdSCMSLog_Fast.remote(mesh_0[i:(i+chunksize)], 
                                                        dataset, d=1, h=[None,None], 
                                                        com_type=['Dir','Lin'], 
                                                        dim=[2,1], eps=1e-7, 
                                                        max_iter=5000))
    DLSCMS_pts = ray.get(result_ids)
    DLSCMS_pts = np.concatenate(DLSCMS_pts, axis=0)
    ray.shutdown()

    lon, lat, R = cart2sph(*DLSCMS_pts[:,:3].T)
    Phi = (lon/180)*np.pi
    Eta = (lat/180)*np.pi
    DL_Ridges = np.concatenate([(DLSCMS_pts[:,3]*np.cos(Phi)*np.cos(Eta)).reshape(-1,1), 
                                (DLSCMS_pts[:,3]*np.sin(Phi)*np.cos(Eta)).reshape(-1,1),
                                (DLSCMS_pts[:,3]*np.sin(Eta)).reshape(-1,1)], axis=1)


    # Compute the manifold recovering errors of the estimated ridges
    np.random.seed(123)  ## Set an arbitrary seed for reproducibility
    # Simulate 5000 points from the true spiral curve as its approximation
    N_t = 5000
    Z_t = np.random.rand(N_t, 1)*4
    # True spiral curve in its angular-linear coordinates
    cur_true_ang = np.concatenate([5*Z_t, np.ones((N_t, 1))*(np.pi/2-open_ang), 
                                   Z_t.reshape(N_t,1)], axis=1)
    th_true = np.pi/2 - cur_true_ang[:,1]
    phi_true = cur_true_ang[:,0]
    R_true = cur_true_ang[:,2]
    # True spiral curve in its Cartesian coordinates
    Z_true = R_true * np.cos(th_true)
    X_true = R_true * np.sin(th_true) * np.cos(phi_true)
    Y_true = R_true * np.sin(th_true) * np.sin(phi_true)
    cur_true_3D = np.concatenate([X_true.reshape(-1,1), Y_true.reshape(-1,1), 
                                  Z_true.reshape(-1,1)], axis=1)
    # 1. Euclidean distance errors from estimated ridges to the true curve
    DLRidge_Err = pd.DataFrame(DL_Ridges).apply(lambda x: DistToCurve(x, cur_true_3D), 
                                                axis=1)
    EuRidge1_Err = pd.DataFrame(EuSCMS_pts1).apply(lambda x: DistToCurve(x, cur_true_3D), 
                                                   axis=1)
    EuRidge2_Err = pd.DataFrame(Eu_Ridges2).apply(lambda x: DistToCurve(x, cur_true_3D), 
                                                  axis=1)
    DistErr_df = pd.DataFrame({'SCMS_3D': EuRidge1_Err, 
                               'SCMS_2Ang_1Lin': EuRidge2_Err, 
                               'DirLinSCMS_Omega2_Lin': DLRidge_Err})
    # 2. Euclidean distance errors from the true curve to estimated ridges
    DLRidge_CurRecErr = pd.DataFrame(cur_true_3D).apply(lambda x: DistToCurve(x, DL_Ridges), 
                                                        axis=1)
    EuRidge1_CurRecErr = pd.DataFrame(cur_true_3D).apply(lambda x: DistToCurve(x, EuSCMS_pts1), 
                                                         axis=1)
    EuRidge2_CurRecErr = pd.DataFrame(cur_true_3D).apply(lambda x: DistToCurve(x, Eu_Ridges2), 
                                                         axis=1)
    CurRecErr_df = pd.DataFrame({'SCMS_3D': EuRidge1_CurRecErr, 
                                  'SCMS_2Ang_1Lin': EuRidge2_CurRecErr, 
                                  'DirLinSCMS_Omega2_Lin': DLRidge_CurRecErr})
    # Manifold recovering errors of the estimated ridges
    print('The manifold recovering errors of the estimated ridges are \n')
    print((np.mean(DistErr_df, axis=0) + np.mean(CurRecErr_df, axis=0))/2)
    print('\n')


    with open('./Results/Simulation3_'+str(job_id)+'.dat', "wb") as file:
        pickle.dump([(np.mean(DistErr_df, axis=0) + np.mean(CurRecErr_df, axis=0))/2], file)
