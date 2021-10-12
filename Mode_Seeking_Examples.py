#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang

Last Editing: Oct 10, 2021

Description: This script contains code for mode-seeking simulation studies with
our proposed mean shift algorithm (Figure 3 in the arxiv version of the paper).
"""

import numpy as np
import pandas as pd
import scipy.special as sp
from numpy import linalg as LA
import matplotlib.pyplot as plt
from DirLinProdSCMS_fun import DirLinProdKDE, DirLinProdMS, DirLinProdMSCompAsc
from Utility_fun import vMF_Gauss_mix, Unique_Modes

if __name__ == "__main__":
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
    vMF_Gau_Ang = np.concatenate([Angs.reshape(-1,1), 
                                  vMF_Gau_data[:,2].reshape(-1,1)], axis=1)
    
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
    
    # Create a cylinder for the directional-linear space
    theta = np.linspace(-np.pi, np.pi, 100)
    z = np.linspace(-2, 5, 100)
    th_m, Zc = np.meshgrid(theta, z)
    Xc = np.cos(th_m)
    Yc = np.sin(th_m)
    # Plot the simulated data points and local modes on the cylinder
    step = DLMS_path.shape[2] - 1
    Modes_angs = np.arctan2(DLMS_path[:,1,step], DLMS_path[:,0,step])
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(30, 60)
    ax.plot_surface(Xc, Yc, Zc, alpha=0.2, color='grey')
    ax.scatter(vMF_Gau_data[:,0], vMF_Gau_data[:,1], vMF_Gau_data[:,2], 
               alpha=0.2, color='deepskyblue')
    ax.scatter(DLMS_path[:,0,step], DLMS_path[:,1,step], DLMS_path[:,2,step], 
               color='red', s=40)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig('./Figures/DirLin_modes_cyl.pdf')
    # Plot the local modes on the contour plot of the estimated density
    step = DLMS_path.shape[2] - 1
    Modes_angs = np.arctan2(DLMS_path[:,1,step], DLMS_path[:,0,step])
    plt.rcParams.update({'font.size': 10})  ## Change the font sizes of ouput figures
    fig = plt.figure(figsize=(6,4.5))
    plt.scatter(Angs, vMF_Gau_data[:,2], alpha=1)
    plt.contourf(ang_m, lin_m, d_DirLin, 10, cmap='OrRd', alpha=0.7)
    plt.colorbar()
    plt.scatter(Modes_angs, DLMS_path[:,2,step], color='red', s=30)
    fig.tight_layout()
    fig.savefig('./Figures/DirLin_modes_contour.pdf')
    print("\n Save the plots as 'DirLin_modes_cyl.pdf' and 'DirLin_modes_contour.pdf'"\
          " to the folder 'Figures'.\n\n")
    
    
    ## Simulation 2: Mode-seeking on a directional-directional space $\Omega_1 \times \Omega_1$
    N = 1000
    # Simulate points from an independent product of two vMF densities
    p1,p2,p3,p4 = 1/2,1/2,1/2,1/2
    mu1,mu2,mu3,mu4 = 0, np.pi/2, 0, 3*np.pi/4
    kap1, kap2, kap3, kap4 = 5,5,7,7
    sim_dat1 = np.zeros((N, 2))
    np.random.seed(123)  ## Set an arbitrary seed for reproducibility
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
    
    # Plot the simulated points and local modes on the contour plot of the 
    # estimated density on a torus
    c, a = 3, 1
    ## Mesh points in R^3
    x = (c + a*np.cos(th_m)) * np.cos(phi_m)
    y = (c + a*np.cos(th_m)) * np.sin(phi_m)
    z = a * np.sin(th_m)
    ## Simulated points in R^3
    x_p = (c + a*np.cos(sim_dat1[:,0])) * np.cos(sim_dat1[:,1])
    y_p = (c + a*np.cos(sim_dat1[:,0])) * np.sin(sim_dat1[:,1])
    z_p = a * np.sin(sim_dat1[:,0])
    ## Local modes in R^3
    x_m = (c + a*np.cos(DDModes[:,0])) * np.cos(DDModes[:,1])
    y_m = (c + a*np.cos(DDModes[:,0])) * np.sin(DDModes[:,1])
    z_m = a * np.sin(DDModes[:,0])
    
    plt.rcParams.update({'font.size': 10})  ## Change the font sizes of ouput figures
    fig = plt.figure(figsize=(8,8))
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.set_zlim(-3,3)
    ax1.view_init(40, 80)
    ax1.scatter(x, y, z, c=d_Dir, alpha=0.09, cmap='OrRd')
    ax1.scatter(x_p, y_p, z_p, color='dodgerblue', alpha=0.22)
    ax1.scatter(x_m, y_m, z_m, color='red', s=60)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    fig.tight_layout()
    fig.savefig('./Figures/DirDir_modes.pdf')
    print("\n Save the plot as 'DirDir_modes.pdf' to the folder 'Figures'.\n\n")