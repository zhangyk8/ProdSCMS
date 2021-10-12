#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang

Last Editing: Oct 10, 2021

Description: This script contains code for investigating the effects of varying
the stepsize parameter in our proposed SCMS algorithm in Euclidean/directional 
product spaces. (Figures 9 and 10 in the arxiv version of the paper). The script
takes more than 1.5 hours to execute due to the slow convergence of the proposed
SCMS algorithm with step size "eta=1". The SCMS algorithm with our suggested 
choice of the step size parameter, however, does converge very fast.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
from DirLinProdSCMS_fun import DirLinProdKDE, DirLinProdSCMSLog

if __name__ == "__main__":
    # Example 1: Directional-linear data case
    print('------Varying the step size of our proposed SCMS algorithm on '\
          'a simulated directional-linear dataset------')
    N = 1000
    sigma = 0.3
    np.random.seed(123)  ## Set an arbitrary seed for reproducibility
    # Simulated a curve with additive Gaussian noises on a cylindar 
    # (directional-linear case)
    t = np.random.rand(N)*2*np.pi - np.pi
    t_p = t + np.random.randn(1000) * sigma
    X_p = np.cos(t_p)
    Y_p = np.sin(t_p)
    Z_p = t/2 + np.random.randn(1000) * sigma
    cur_dat = np.concatenate([X_p.reshape(-1,1), Y_p.reshape(-1,1), 
                              Z_p.reshape(-1,1)], axis=1)
    # Plot the true curve structure and simulated points on the cylinder
    ## Mesh points on the cylindar
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
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(30, 10)
    ax.plot_surface(Xc, Yc, Zc, alpha=0.2)
    ax.plot(X_cur, Y_cur, Z_cur, linewidth=5, color='green')
    ax.scatter(cur_dat[:,0], cur_dat[:,1], cur_dat[:,2], alpha=0.5, 
               color='deepskyblue')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    fig.tight_layout()
    fig.savefig('./Figures/stepsize_DirLin_sam.pdf')
    print("\n Save the plot as 'stepsize_DirLin_sam.pdf' to the folder 'Figures'.\n\n")
    
    # Bandwidth selection
    data = cur_dat
    n = cur_dat.shape[0]
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
    
    # Create a set of mesh points and estimate the density value on it
    nrows, ncols = (100, 100)
    ang_qry = np.linspace(-np.pi, np.pi, nrows)
    lin_qry = np.linspace(-2.5, 2.5, ncols)
    ang_m, lin_m = np.meshgrid(ang_qry, lin_qry)
    X = np.cos(ang_m.reshape(-1,1))
    Y = np.sin(ang_m.reshape(-1,1))
    qry_pts = np.concatenate((X.reshape(-1,1), 
                              Y.reshape(-1,1), 
                              lin_m.reshape(-1,1)), axis=1)
    d_DirLinProd = DirLinProdKDE(qry_pts, cur_dat, h=[bw_Dir, bw_Lin], 
                                 com_type=['Dir','Lin'], dim=[1,1]).reshape(ncols, nrows)
    
    # Proposed SCMS algorithm with step size eta=0.1*h1*h2
    print('The current step size is 0.1*h1*h2='+str(bw_Dir*bw_Lin*0.1)+'.\n')
    ProdSCMS_DL1, lab_DL1 = DirLinProdSCMSLog(cur_dat, cur_dat, d=1, h=[bw_Dir,bw_Lin], 
                                              com_type=['Dir','Lin'], dim=[1,1], 
                                              eps=1e-7, max_iter=5000, 
                                              eta=bw_Dir*bw_Lin*0.1)
    # Plot the simulated data and estimated ridges on a cylindar
    step_DL1 = ProdSCMS_DL1.shape[2] - 1
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(30, 10)
    ax.plot_surface(Xc, Yc, Zc, alpha=0.2)
    ax.plot(X_cur, Y_cur, Z_cur, linewidth=5, color='green')
    ax.scatter(ProdSCMS_DL1[:,0,step_DL1], ProdSCMS_DL1[:,1,step_DL1], 
               ProdSCMS_DL1[:,2,step_DL1], alpha=0.5, color='deepskyblue')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    fig.tight_layout()
    fig.savefig('./Figures/stepsize_DirLin_0_1.pdf')
    print("\n Save the plot as 'stepsize_DirLin_0_1.pdf' to the folder 'Figures'.\n\n")
    
    # Proposed SCMS algorithm with step size eta=0.5*h1*h2
    print('The current step size is 0.5*h1*h2='+str(bw_Dir*bw_Lin*0.5)+'.\n')
    ProdSCMS_DL2, lab_DL2 = DirLinProdSCMSLog(cur_dat, cur_dat, d=1, 
                                              h=[bw_Dir,bw_Lin], com_type=['Dir','Lin'], 
                                              dim=[1,1], eps=1e-7, max_iter=5000, 
                                              eta=bw_Dir*bw_Lin*0.5)
    # Plot the simulated data and estimated ridges on a cylindar
    step_DL2 = ProdSCMS_DL2.shape[2] - 1
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(30, 10)
    ax.plot_surface(Xc, Yc, Zc, alpha=0.2)
    ax.plot(X_cur, Y_cur, Z_cur, linewidth=5, color='green')
    ax.scatter(ProdSCMS_DL2[:,0,step_DL2], ProdSCMS_DL2[:,1,step_DL2], 
               ProdSCMS_DL2[:,2,step_DL2], alpha=0.5, color='deepskyblue')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    fig.tight_layout()
    fig.savefig('./Figures/stepsize_DirLin_0_5.pdf')
    print("\n Save the plot as 'stepsize_DirLin_0_5.pdf' to the folder 'Figures'.\n\n")
    
    # Proposed SCMS algorithm with our rule-of-thumb step size eta=h1*h2
    print('The current step size is h1*h2='+str(bw_Dir*bw_Lin)+'.\n')
    ProdSCMS_DL_p, lab_DL_p = DirLinProdSCMSLog(cur_dat, cur_dat, d=1, 
                                                h=[bw_Dir,bw_Lin], 
                                                com_type=['Dir','Lin'], dim=[1,1], 
                                                eps=1e-7, max_iter=5000, 
                                                eta=bw_Dir*bw_Lin)
    # Plot the simulated data and estimated ridge on a cylindar
    step_DL_p = ProdSCMS_DL_p.shape[2] - 1
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(30, 10)
    ax.plot_surface(Xc, Yc, Zc, alpha=0.2)
    ax.plot(X_cur, Y_cur, Z_cur, linewidth=5, color='green')
    ax.scatter(ProdSCMS_DL_p[:,0,step_DL_p], ProdSCMS_DL_p[:,1,step_DL_p], 
               ProdSCMS_DL_p[:,2,step_DL_p], alpha=0.5, color='deepskyblue')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    fig.tight_layout()
    fig.savefig('./Figures/stepsize_DirLin_prop.pdf')
    print("\n Save the plot as 'stepsize_DirLin_prop.pdf' to the folder 'Figures'.\n\n")
    
    # Proposed SCMS algorithm with step size eta=h1*h2*2
    print('The current step size is 2*h1*h2='+str(bw_Dir*bw_Lin*2)+'.\n')
    ProdSCMS_DL3, lab_DL3 = DirLinProdSCMSLog(cur_dat, cur_dat, d=1, h=[bw_Dir,bw_Lin], 
                                              com_type=['Dir','Lin'], dim=[1,1], 
                                              eps=1e-7, max_iter=5000, 
                                              eta=bw_Dir*bw_Lin*2)
    # Plot the simulated data and estimated ridge on a cylindar 
    step_DL3 = ProdSCMS_DL3.shape[2] - 1
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(30, 10)
    ax.plot_surface(Xc, Yc, Zc, alpha=0.2)
    ax.plot(X_cur, Y_cur, Z_cur, linewidth=5, color='green')
    ax.scatter(ProdSCMS_DL3[:,0,step_DL3], ProdSCMS_DL3[:,1,step_DL3], 
               ProdSCMS_DL3[:,2,step_DL3], alpha=0.5, color='deepskyblue')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    fig.tight_layout()
    fig.savefig('./Figures/stepsize_DirLin_bw_2.pdf')
    print("\n Save the plot as 'stepsize_DirLin_bw_2.pdf' to the folder 'Figures'.\n\n")
    
    # Proposed SCMS algorithm with step size eta=1
    print('The current step size is '+str(1)+'.\n')
    ProdSCMS_DL4, lab_DL4 = DirLinProdSCMSLog(cur_dat, cur_dat, d=1, 
                                              h=[bw_Dir,bw_Lin], com_type=['Dir','Lin'], 
                                              dim=[1,1], eps=1e-7, max_iter=5000, eta=1)
    # Plot the simulated data and estimated ridge on a cylindar 
    step_DL4 = ProdSCMS_DL4.shape[2] - 1
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(30, 10)
    ax.plot_surface(Xc, Yc, Zc, alpha=0.2)
    ax.plot(X_cur, Y_cur, Z_cur, linewidth=5, color='green')
    ax.scatter(ProdSCMS_DL4[:,0,step_DL4], ProdSCMS_DL4[:,1,step_DL4], 
               ProdSCMS_DL4[:,2,step_DL4], alpha=0.5, color='deepskyblue')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    fig.tight_layout()
    fig.savefig('./Figures/stepsize_DirLin_1.pdf')
    print("\n Save the plot as 'stepsize_DirLin_1.pdf' to the folder 'Figures'.\n\n")
    
    # Proposed SCMS algorithm with step size eta=2
    print('The current step size is '+str(2)+'.\n')
    ProdSCMS_DL5, lab_DL5 = DirLinProdSCMSLog(cur_dat, cur_dat, d=1, h=[bw_Dir,bw_Lin], 
                                              com_type=['Dir','Lin'], dim=[1,1], 
                                              eps=1e-7, max_iter=5000, eta=2)
    # Plot the simulated data and estimated ridge on a cylindar
    step_DL5 = ProdSCMS_DL5.shape[2] - 1
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(30, 10)
    ax.plot_surface(Xc, Yc, Zc, alpha=0.2)
    ax.plot(X_cur, Y_cur, Z_cur, linewidth=5, color='green')
    ax.scatter(ProdSCMS_DL5[:,0,step_DL5], ProdSCMS_DL5[:,1,step_DL5], 
               ProdSCMS_DL5[:,2,step_DL5], alpha=0.5, color='deepskyblue')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    fig.tight_layout()
    fig.savefig('./Figures/stepsize_DirLin_2.pdf')
    print("\n Save the plot as 'stepsize_DirLin_2.pdf' to the folder 'Figures'.\n\n")
    
    # Plot the estimated ridge on the contour plot of estimated density
    plt.rcParams.update({'font.size': 15})  ## Change the font sizes of output figures
    fig = plt.figure(figsize=(8,6))
    plt.contourf(ang_m, lin_m, d_DirLinProd, 10, cmap='OrRd', alpha=0.5)
    plt.colorbar()
    Ridges_angs_p = np.arctan2(ProdSCMS_DL_p[:,1,step_DL_p], 
                               ProdSCMS_DL_p[:,0,step_DL_p])
    plt.scatter(Ridges_angs_p, ProdSCMS_DL_p[:,2,step_DL_p], color='deepskyblue', 
                alpha=0.6)
    plt.xlabel('Directional Coordinate')
    plt.ylabel('Linear Coordinate')
    fig.tight_layout()
    fig.savefig('./Figures/stepsize_DirLin_prop_density.pdf')
    print("\n Save the plot as 'stepsize_DirLin_prop_density.pdf' to the "\
          "folder 'Figures'.\n\n")
    
    
    print('------Varying the step size of our proposed SCMS algorithm on '\
          'a simulated directional-directional dataset------')
    # Example 2: Directional-directional data case
    N = 1000
    sigma = 0.3
    np.random.seed(123)  ## Set an arbitrary seed for reproducibility
    # Simulate two toroidal curves on a torus
    p1, p2 = 1/2, 1/2
    mix_ind = np.random.choice([0,1], size=N, replace=True, p=np.array([p1,p2]))
    th_pt = np.zeros((N, 1))
    th_pt[mix_ind == 0] = 0
    th_pt[mix_ind == 1] = 3*np.pi/4
    phi_p = np.random.rand(N, 1)*2*np.pi - np.pi
    samp_dat = np.concatenate([th_pt, phi_p], axis=1)
    # Add some Gaussian noises to the observations from the true curves
    samp_dat = samp_dat + np.random.multivariate_normal(mean=np.zeros(2,), 
                                                        cov=np.eye(2)*(sigma**2), 
                                                        size=N)
    samp_dat_cart = np.concatenate([np.cos(samp_dat[:,0]).reshape(-1, 1),
                                    np.sin(samp_dat[:,0]).reshape(-1, 1),
                                    np.cos(samp_dat[:,1]).reshape(-1, 1),
                                    np.sin(samp_dat[:,1]).reshape(-1, 1)], axis=1)
    
    # Bandwidth selection
    data = samp_dat_cart
    n = samp_dat_cart.shape[0]
    q = 1
    data_Dir1 = data[:,:(q+1)]
    data_Dir2 = data[:,(q+1):(2*q+2)]
    ## Rule-of-thumb bandwidth selector for the directional component
    R_bar = np.sqrt(sum(np.mean(data_Dir1, axis=0) ** 2))
    kap_hat = R_bar * (q + 1 - R_bar ** 2) / (1 - R_bar ** 2)
    h1 = ((4 * np.sqrt(np.pi) * sp.iv((q-1) / 2 , kap_hat)**2) / \
         (n * kap_hat ** ((q+1) / 2) * (2 * q * sp.iv((q+1)/2, 2*kap_hat) + \
         (q+2) * kap_hat * sp.iv((q+3)/2, 2*kap_hat)))) ** (1/(q + 4))
    bw_Dir1 = h1
    print("The current bandwidth for the 1st directional component is "+str(h1)+".\n")
    R_bar = np.sqrt(sum(np.mean(data_Dir2, axis=0) ** 2))
    kap_hat = R_bar * (q + 1 - R_bar ** 2) / (1 - R_bar ** 2)
    h2 = ((4 * np.sqrt(np.pi) * sp.iv((q-1) / 2 , kap_hat)**2) / \
         (n * kap_hat ** ((q+1) / 2) * (2 * q * sp.iv((q+1)/2, 2*kap_hat) + \
         (q+2) * kap_hat * sp.iv((q+3)/2, 2*kap_hat)))) ** (1/(q + 4))
    bw_Dir2 = h2
    print("The current bandwidth for the 2nd directional component is "+str(h2)+".\n")
    bw = [bw_Dir1, bw_Dir2]
    
    # Create a set of mesh points and estimate the density values on it
    nrows, ncols = (400, 200)
    th_m, phi_m = np.meshgrid(np.linspace(-np.pi, np.pi, ncols), 
                              np.linspace(-np.pi, np.pi, nrows))
    query_pts = np.concatenate((np.cos(th_m).reshape(nrows*ncols, 1),
                                np.sin(th_m).reshape(nrows*ncols, 1),
                                np.cos(phi_m).reshape(nrows*ncols, 1),
                                np.sin(phi_m).reshape(nrows*ncols, 1)), axis=1)
    query_pts_ang = np.concatenate((th_m.reshape(nrows*ncols, 1),
                                    phi_m.reshape(nrows*ncols, 1)), axis=1)
    # KDE in the circular-circular space
    d_Dir = DirLinProdKDE(query_pts, samp_dat_cart, h=bw, 
                          com_type=['Dir','Dir'], dim=[1,1]).reshape(ncols, nrows)
    
    # Plot the simulated points and true curves on a torus
    c, a = 3, 1
    x_p = (c + a*np.cos(samp_dat[:,0])) * np.cos(samp_dat[:,1])
    y_p = (c + a*np.cos(samp_dat[:,0])) * np.sin(samp_dat[:,1])
    z_p = a * np.sin(samp_dat[:,0])
    ## Torus
    x = (c + a*np.cos(th_m)) * np.cos(phi_m)
    y = (c + a*np.cos(th_m)) * np.sin(phi_m)
    z = a * np.sin(th_m)
    ## True curves
    phi = np.linspace(-np.pi, np.pi, 200)
    th1 = np.zeros((200,))
    th2 = np.ones((200,)) * 3*np.pi/4
    x_1 = (c + a*np.cos(th1)) * np.cos(phi)
    y_1 = (c + a*np.cos(th1)) * np.sin(phi)
    z_1 = a * np.sin(th1)
    x_2 = (c + a*np.cos(th2)) * np.cos(phi)
    y_2 = (c + a*np.cos(th2)) * np.sin(phi)
    z_2 = a * np.sin(th2)

    fig = plt.figure(figsize=(6,6))
    plt.rcParams.update({'font.size': 10})  ## Change the font sizes of output figures
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.set_zlim(-3,3)
    ax1.view_init(40, 120)
    ax1.plot_surface(x, y, z, rstride=5, cstride=5, color='grey', alpha=0.2)
    ax1.scatter(x_p, y_p, z_p, color='deepskyblue')
    ax1.plot(x_1, y_1, z_1, color='green', linewidth=5)
    ax1.plot(x_2, y_2, z_2, color='green', linewidth=5)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    fig.tight_layout()
    fig.savefig('./Figures/stepsize_DirDir_samp.pdf')
    print("\n Save the plot as 'stepsize_DirDir_samp.pdf' to the folder 'Figures'.\n\n")
    
    # Proposed SCMS algorithm with step size eta=0.1*h1*h2
    print('The current step size is 0.1*h1*h2='+str(np.prod(bw)*0.1)+'.\n')
    ProdSCMS_path1, conv_lab1 = DirLinProdSCMSLog(samp_dat_cart, samp_dat_cart, 
                                                  d=1, h=bw, com_type=['Dir','Dir'], 
                                                  dim=[1,1], eps=1e-7, max_iter=5000, 
                                                  eta=np.prod(bw)*0.1)
    conv_step = ProdSCMS_path1.shape[2] - 1
    th_r = np.arctan2(ProdSCMS_path1[:,1,conv_step], ProdSCMS_path1[:,0,conv_step])
    phi_r = np.arctan2(ProdSCMS_path1[:,3,conv_step], ProdSCMS_path1[:,2,conv_step])
    c, a = 3, 1
    x_r = (c + a*np.cos(th_r)) * np.cos(phi_r)
    y_r = (c + a*np.cos(th_r)) * np.sin(phi_r)
    z_r = a * np.sin(th_r)
    # Visualize the estimated ridge on the torus
    fig = plt.figure(figsize=(6,6))
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.set_zlim(-3,3)
    ax1.view_init(40, 120)
    ax1.plot_surface(x, y, z, rstride=5, cstride=5, color='grey', alpha=0.2)
    ax1.scatter(x_r, y_r, z_r, color='deepskyblue')
    ax1.plot(x_1, y_1, z_1, color='green', linewidth=5)
    ax1.plot(x_2, y_2, z_2, color='green', linewidth=5)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    fig.tight_layout()
    fig.savefig('./Figures/stepsize_DirDir0_1.pdf')
    print("\n Save the plot as 'stepsize_DirDir0_1.pdf' to the folder 'Figures'.\n\n")
    
    # Proposed SCMS algorithm with step size eta=0.5*h1*h2
    print('The current step size is 0.5*h1*h2='+str(np.prod(bw)*0.5)+'.\n')
    ProdSCMS_path2, conv_lab2 = DirLinProdSCMSLog(samp_dat_cart, samp_dat_cart, 
                                                  d=1, h=bw, com_type=['Dir','Dir'], 
                                                  dim=[1,1], eps=1e-7, max_iter=5000, 
                                                  eta=np.prod(bw)*0.5)
    conv_step = ProdSCMS_path2.shape[2] - 1
    th_r = np.arctan2(ProdSCMS_path2[:,1,conv_step], ProdSCMS_path2[:,0,conv_step])
    phi_r = np.arctan2(ProdSCMS_path2[:,3,conv_step], ProdSCMS_path2[:,2,conv_step])
    c, a = 3, 1
    x_r = (c + a*np.cos(th_r)) * np.cos(phi_r)
    y_r = (c + a*np.cos(th_r)) * np.sin(phi_r)
    z_r = a * np.sin(th_r)
    # Visualize the estimated ridge on the torus
    fig = plt.figure(figsize=(6,6))
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.set_zlim(-3,3)
    ax1.view_init(40, 120)
    ax1.plot_surface(x, y, z, rstride=5, cstride=5, color='grey', alpha=0.2)
    ax1.scatter(x_r, y_r, z_r, color='deepskyblue')
    ax1.plot(x_1, y_1, z_1, color='green', linewidth=5)
    ax1.plot(x_2, y_2, z_2, color='green', linewidth=5)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    fig.tight_layout()
    fig.savefig('./Figures/stepsize_DirDir0_5.pdf')
    print("\n Save the plot as 'stepsize_DirDir0_5.pdf' to the folder 'Figures'.\n\n")
    
    # Proposed SCMS algorithm with the rule-of-thumb step size eta=h1*h2
    print('The current step size is h1*h2='+str(np.prod(bw))+'.\n')
    ProdSCMS_path_p, conv_lab_p = DirLinProdSCMSLog(samp_dat_cart, samp_dat_cart, 
                                                    d=1, h=bw, com_type=['Dir','Dir'], 
                                                    dim=[1,1], eps=1e-7, max_iter=5000, 
                                                    eta=np.prod(bw))
    conv_step = ProdSCMS_path_p.shape[2] - 1
    th_r = np.arctan2(ProdSCMS_path_p[:,1,conv_step], ProdSCMS_path_p[:,0,conv_step])
    phi_r = np.arctan2(ProdSCMS_path_p[:,3,conv_step], ProdSCMS_path_p[:,2,conv_step])
    c, a = 3, 1
    x_r = (c + a*np.cos(th_r)) * np.cos(phi_r)
    y_r = (c + a*np.cos(th_r)) * np.sin(phi_r)
    z_r = a * np.sin(th_r)
    # Visualize the estimated ridge on the torus
    fig = plt.figure(figsize=(6,6))
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.set_zlim(-3,3)
    ax1.view_init(40, 120)
    ax1.plot_surface(x, y, z, rstride=5, cstride=5, color='grey', alpha=0.2)
    ax1.scatter(x_r, y_r, z_r, color='deepskyblue')
    ax1.plot(x_1, y_1, z_1, color='green', linewidth=5)
    ax1.plot(x_2, y_2, z_2, color='green', linewidth=5)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    fig.tight_layout()
    fig.savefig('./Figures/stepsize_DirDir_prop.pdf')
    print("\n Save the plot as 'stepsize_DirDir_prop.pdf' to the folder 'Figures'.\n\n")
    
    # Proposed SCMS algorithm with step size eta=1
    print('The current step size is '+str(1)+'.\n')
    ProdSCMS_path4, conv_lab4 = DirLinProdSCMSLog(samp_dat_cart, samp_dat_cart, 
                                                  d=1, h=bw, com_type=['Dir','Dir'], 
                                                  dim=[1,1], eps=1e-7, max_iter=5000, 
                                                  eta=1)
    conv_step = ProdSCMS_path4.shape[2] - 1
    th_r = np.arctan2(ProdSCMS_path4[:,1,conv_step], ProdSCMS_path4[:,0,conv_step])
    phi_r = np.arctan2(ProdSCMS_path4[:,3,conv_step], ProdSCMS_path4[:,2,conv_step])
    c, a = 3, 1
    x_r = (c + a*np.cos(th_r)) * np.cos(phi_r)
    y_r = (c + a*np.cos(th_r)) * np.sin(phi_r)
    z_r = a * np.sin(th_r)
    # Visualize the estimated ridge on the torus
    fig = plt.figure(figsize=(6,6))
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.set_zlim(-3,3)
    ax1.view_init(40, 120)
    ax1.plot_surface(x, y, z, rstride=5, cstride=5, color='grey', alpha=0.2)
    ax1.scatter(x_r, y_r, z_r, color='deepskyblue')
    ax1.plot(x_1, y_1, z_1, color='green', linewidth=5)
    ax1.plot(x_2, y_2, z_2, color='green', linewidth=5)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    fig.tight_layout()
    fig.savefig('./Figures/stepsize_DirDir_1.pdf')
    print("\n Save the plot as 'stepsize_DirDir_1.pdf' to the folder 'Figures'.\n\n")
    
    # Proposed SCMS algorithm with step size eta=2
    print('The current step size is '+str(2)+'.\n')
    ProdSCMS_path5, conv_lab5 = DirLinProdSCMSLog(samp_dat_cart, samp_dat_cart, 
                                                  d=1, h=bw, com_type=['Dir','Dir'], 
                                                  dim=[1,1], eps=1e-7, max_iter=5000, 
                                                  eta=2)
    conv_step = ProdSCMS_path5.shape[2] - 1
    th_r = np.arctan2(ProdSCMS_path5[:,1,conv_step], ProdSCMS_path5[:,0,conv_step])
    phi_r = np.arctan2(ProdSCMS_path5[:,3,conv_step], ProdSCMS_path5[:,2,conv_step])
    c, a = 3, 1
    x_r = (c + a*np.cos(th_r)) * np.cos(phi_r)
    y_r = (c + a*np.cos(th_r)) * np.sin(phi_r)
    z_r = a * np.sin(th_r)
    # Visualize the estimated ridge on the torus
    fig = plt.figure(figsize=(6,6))
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.set_zlim(-3,3)
    ax1.view_init(40, 120)
    ax1.plot_surface(x, y, z, rstride=5, cstride=5, color='grey', alpha=0.2)
    ax1.scatter(x_r, y_r, z_r, color='deepskyblue')
    ax1.plot(x_1, y_1, z_1, color='green', linewidth=5)
    ax1.plot(x_2, y_2, z_2, color='green', linewidth=5)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    fig.tight_layout()
    fig.savefig('./Figures/stepsize_DirDir_2.pdf')
    print("\n Save the plot as 'stepsize_DirDir_2.pdf' to the folder 'Figures'.\n\n")
    
    conv_step = ProdSCMS_path_p.shape[2] - 1
    th_r = np.arctan2(ProdSCMS_path_p[:,1,conv_step], ProdSCMS_path_p[:,0,conv_step])
    phi_r = np.arctan2(ProdSCMS_path_p[:,3,conv_step], ProdSCMS_path_p[:,2,conv_step])
    # Elongate the poloidal radius for a better visualization of estimated ridges
    c, a = 3, 1.03
    x_r = (c + a*np.cos(th_r)) * np.cos(phi_r)
    y_r = (c + a*np.cos(th_r)) * np.sin(phi_r)
    z_r = a * np.sin(th_r)
    # Visualize the estimated ridge on the contour of estimated density on the torus
    fig = plt.figure(figsize=(6,6))
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.set_zlim(-3,3)
    ax1.view_init(40, 120)
    ax1.scatter(x, y, z, c=d_Dir, alpha=0.02, cmap='OrRd')
    ax1.scatter(x_r, y_r, z_r, s=20, color='dodgerblue')
    # ax1.plot(x_1, y_1, z_1, color='green', linewidth=5)
    # ax1.plot(x_2, y_2, z_2, color='green', linewidth=5)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    fig.tight_layout()
    fig.savefig('./Figures/stepsize_DirDir_prop_density.pdf')
    print("\n Save the plot as 'stepsize_DirDir_prop_density.pdf' to the "\
          "folder 'Figures'.\n\n")
    
    