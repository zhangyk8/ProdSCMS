#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang

Last Editing: Oct 10, 2021

Description: This script contains code for comparing the results of the regular 
SCMS and our proposed SCMS algorithms on the simulated spiral curve data. 
(Figure 2 in the arxiv version of the paper).
"""

import numpy as np
import pandas as pd
from numpy import linalg as LA
import matplotlib.pyplot as plt
import ray
from Utility_fun import cart2sph, sph2cart
from MS_SCMS_Ray import SCMS_Log_KDE_Fs
from DirLinProdSCMS_Ray import DirLinProdSCMSLog_Fast

def DistToCurve(x, true_cur):
    return min(LA.norm(x.values - true_cur, axis=1))

if __name__ == "__main__":
    N = 1000
    open_ang = np.pi/3
    sigma = 0.2   ## Variance of the additive Gaussian noises
    np.random.seed(123)  ## Set an arbitrary seed for reproducibility
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
    
    # Visualize the true spiral curve and simulated data in R^3
    Z_true = np.linspace(0, 4, 101)
    X = Z_true*np.sin(open_ang)*np.cos(Z_true*5)
    Y = Z_true*np.sin(open_ang)*np.sin(Z_true*5)
    Z = Z_true*np.cos(open_ang)
    
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(20, 30)
    ax.scatter(cur_dat_3D[:,0], cur_dat_3D[:,1], cur_dat_3D[:,2], 
               color='deepskyblue', alpha=0.4)
    ax.plot3D(X, Y, Z, 'red')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    fig.tight_layout()
    fig.savefig('./Figures/spiral_cur_samp.pdf')
    print("\n Save the plot as 'spiral_cur_samp.pdf' to the folder 'Figures'.\n\n")
    
    # Regular SCMS on the simulated data with 3D Cartesian coordinates
    ray.init()
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
    # Plot the true spiral curve and the estimated ridge obtained by the regular 
    # SCMS algorithm in the 3D Cartesian space
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(20, 30)
    ax.scatter(EuSCMS_pts1[:,0], EuSCMS_pts1[:,1], EuSCMS_pts1[:,2], 
               color='deepskyblue', alpha=0.3)
    ax.plot3D(X, Y, Z, 'red')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # plt.title('Regular SCMS in 3D Cartesian space $\mathbb{R}^3$')
    fig.tight_layout()
    fig.savefig('./Figures/spiral_cur_SCMS3D.pdf')
    print("\n Save the plot as 'spiral_cur_SCMS3D.pdf' to the folder 'Figures'.\n\n")
    
    # Regular SCMS on the simulated data with 3D angular-linear coordinates 
    # i.e., (longitudes, latitudes, linear covariates)
    ray.init()
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
    # Plot the true spiral curve and the estimated ridge obtained by the regular 
    # SCMS algorithm in the 3D angular-linear space
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(20, 30)
    Phi = (EuSCMS_pts2[:,0]/180)*np.pi
    Eta = (EuSCMS_pts2[:,1]/180)*np.pi
    Eu_Ridges2 = np.concatenate([(EuSCMS_pts2[:,2]*np.cos(Phi)*np.cos(Eta)).reshape(-1,1), 
                                 (EuSCMS_pts2[:,2]*np.sin(Phi)*np.cos(Eta)).reshape(-1,1),
                                 (EuSCMS_pts2[:,2]*np.sin(Eta)).reshape(-1,1)], axis=1)
    ax.scatter(Eu_Ridges2[:,0], Eu_Ridges2[:,1], Eu_Ridges2[:,2], 
               color='deepskyblue', alpha=0.3)
    ax.plot3D(X, Y, Z, 'red')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # plt.title('Regular SCMS in 3D Angular space $(\phi, \eta, z)$')
    fig.tight_layout()
    fig.savefig('./Figures/spiral_cur_SCMS3D_ang.pdf')
    print("\n Save the plot as 'spiral_cur_SCMS3D_ang.pdf' to the folder 'Figures'.\n\n")
    
    # Our proposed DirLin SCMS on the simulated data in the directional-linear 
    # space $\Omega_2 \times \mathbb{R}$
    ray.init()
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
    # Plot the true spiral curve and the estimated ridge obtained by our proposed 
    # SCMS algorithm in the directional-linear space $\Omega_2 \times \mathbb{R}$
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(20, 30)
    lon, lat, R = cart2sph(*DLSCMS_pts[:,:3].T)
    Phi = (lon/180)*np.pi
    Eta = (lat/180)*np.pi
    DL_Ridges = np.concatenate([(DLSCMS_pts[:,3]*np.cos(Phi)*np.cos(Eta)).reshape(-1,1), 
                                (DLSCMS_pts[:,3]*np.sin(Phi)*np.cos(Eta)).reshape(-1,1),
                                (DLSCMS_pts[:,3]*np.sin(Eta)).reshape(-1,1)], axis=1)
    ax.scatter(DL_Ridges[:,0], DL_Ridges[:,1], DL_Ridges[:,2], 
               color='deepskyblue', alpha=0.2)
    ax.plot3D(X, Y, Z, 'red')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # plt.title(r'DirLinSCMS on $\Omega_2\times\mathbb{R}$')
    fig.tight_layout()
    fig.savefig('./Figures/spiral_cur_DirLin.pdf')
    print("\n Save the plot as 'spiral_cur_DirLin.pdf' to the folder 'Figures'.\n\n")
    
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