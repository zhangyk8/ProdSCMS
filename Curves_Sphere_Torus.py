#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang

Last Editing: Oct 9, 2021

Description: This script simulates a circular-circular dataset and plot its 
points on a unit sphere and torus, respectively. (Figure 2 in the arxiv version 
of the paper).
"""

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Simulate 1000 points uniformly from [2*p_1*np.pi, 2*(p_1+1)*np.pi) * {2*p_2*np.pi} 
    # for some integers p_1, p_2.
    N = 1000
    np.random.seed(123)  ## Set an arbitrary seed for reproducibility
    p_1 = np.random.choice(100, N) - 50
    p_2 = np.random.choice(100, N) - 50
    th_p = 2*np.pi*p_1 + np.random.rand(N) * 2*np.pi
    phi_p = 2*np.pi*p_2 + np.zeros((N,))
    cir_dat = np.concatenate([th_p.reshape(-1,1), phi_p.reshape(-1,1)], axis=1)
    
    # Create a set of mesh points on a torus
    n = 100
    theta = np.linspace(0, 2*np.pi, n)
    phi = np.linspace(0, 2*np.pi, n)
    th_m, phi_m = np.meshgrid(theta, phi)
    c, a = 3, 1
    x = (c + a*np.cos(phi_m)) * np.cos(th_m)
    y = (c + a*np.cos(phi_m)) * np.sin(th_m)
    z = a * np.sin(phi_m)
    # Project the simulated points onto the torus
    x_p = (c + a*np.cos(cir_dat[:,1])) * np.cos(cir_dat[:,0])
    y_p = (c + a*np.cos(cir_dat[:,1])) * np.sin(cir_dat[:,0])
    z_p = a * np.sin(cir_dat[:,1])
    # Plot the torus and curve
    fig = plt.figure(figsize=(6,6))
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.set_zlim(-2.5, 2.5)
    ax1.view_init(30, 120)
    ax1.plot_wireframe(x, y, z, color='grey', alpha=0.3)
    ax1.scatter(x_p, y_p, z_p, color='red')
    ax1.axis('off')
    fig.tight_layout()
    fig.savefig('./Figures/torus_curve.pdf')
    
    # Create a mesh point on the unit sphere
    u, v = np.mgrid[0:2 * np.pi:100j, 0:np.pi:50j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    # Project the simulated points on the unit sphere
    x_p = np.cos(cir_dat[:,1]) * np.cos(cir_dat[:,0])
    y_p = np.cos(cir_dat[:,1]) * np.sin(cir_dat[:,0])
    z_p = np.sin(cir_dat[:,1])
    # Plot the sphere and curve
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(x, y, z, color='grey', alpha=0.3)
    ax.scatter(x_p, y_p, z_p, color='red')
    ax.axis('off')
    fig.tight_layout()
    fig.savefig('./Figures/sphere_curve.pdf')
    
    print("Save the plots as 'torus_curve.pdf' and 'sphere_curve.pdf' "\
          "to the folder 'Figures'.\n\n")