#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang

Last Editing: Sep 6, 2021

Description: This script implements the functions of KDE, component-wise/simultaneous 
mean shift, and subspace constrained mean shift (SCMS) algorithms with the 
Gaussian/von Mises product kernels in a directional/linear (mixture) product space.
"""

import numpy as np
from numpy import linalg as LA
import scipy.special as sp
from scipy.linalg import block_diag

def DirLinProdKDE(x, data, h=[None,None], com_type=['Dir', 'Lin'], dim=[2,1]):
    '''
    Kernel density estimation with the von Mises product kernels in a 
    directional/linear (mixture) product space.
    
    Parameters:
        x: (m, sum(dim)+sum(com_type=='Dir'))-array
            Eulidean coordinates of m query points in the product space, where 
            (dim[0]+1) / dim[0] is the Euclidean dimension of a directional/linear 
            component (first (dim[0]+1) columns), and so on.
    
        data: (n, sum(dim)+sum(com_type=='Dir'))-array
            Euclidean coordinates of n random sample points in the product space, 
            where (dim[0]+1) / dim[0] is the Euclidean dimension of a 
            directional/linear component (first (dim[0]+1) columns), and so on.
       
        h: list of floats
            Bandwidth parameters for all the components. (Default: h=[None]*K, 
            where K is the number of components in the product space. Whenever
            h[k]=None for some k=1,...,K, then a rule of thumb for directional 
            KDE with the von Mises kernel in Garcia-Portugues (2013) is applied 
            to that directional component or the Silverman's rule of thumb is 
            applied to that linear component; see Chen et al.(2016) for details.)
            
        com_type: list of strings
            Indicators of the data type for all the components. If com_type[k]='Dir',
            then the corresponding component is directional. If com_type[k]='Lin', 
            then the corresponding component is linear.
            
        dim: list of ints
            Intrinsic data dimensions of all the directional/linear components.
    
    Return:
        f_hat: (m,)-array
            The corresponding density estimates at m query points in the 
            directional/linear (mixture) product space.
    '''
    n = data.shape[0]  ## Number of data points
    D_t = data.shape[1]  ## Total dimension of the data
    
    com_type = np.array(com_type)  ## Convert the list object "com_type" to a numpy array
    assert len(dim) == len(com_type), "The lengths of data type argument 'com_type'"\
    " and dimension argument 'dim' should be the same."
    
    assert sum(dim)+sum(com_type=='Dir') == D_t, "The dimension of the input data, "\
    +str(D_t)+", should be equal to the sum of the Euclidean dimension of the "\
    "directional components ("+str(sum(dim)+sum(com_type=='Dir'))+")."
    
    assert len(h) == len(dim), "The lengths of bandwidth argument 'h' and dimension"\
    " argument 'dim' should be the same."
    
    data_comp = []
    x_comp = []
    Eu_dim = []
    for k in range(len(dim)):
        if (k == 0) and (com_type[k] == 'Dir'):
            data_comp.append(data[:,:(dim[k]+1)])
            x_comp.append(x[:,:(dim[k]+1)])
            Eu_dim.append(dim[k]+1)
        elif (k == 0) and (com_type[k] == 'Lin'):
            data_comp.append(data[:,:dim[k]])
            x_comp.append(x[:,:dim[k]])
            Eu_dim.append(dim[k])
        elif com_type[k] == 'Dir':
            data_comp.append(data[:,sum(Eu_dim):(sum(Eu_dim)+dim[k]+1)])
            x_comp.append(x[:,sum(Eu_dim):(sum(Eu_dim)+dim[k]+1)])
            Eu_dim.append(dim[k]+1)
        else:
            data_comp.append(data[:,sum(Eu_dim):(sum(Eu_dim)+dim[k])])
            x_comp.append(x[:,sum(Eu_dim):(sum(Eu_dim)+dim[k])])
            Eu_dim.append(dim[k])
    
    # Rule of thumb applied to each component
    for k in range(len(h)):
        if (h[k] is None) and (com_type[k] == 'Dir'):
            R_bar = np.sqrt(sum(np.mean(data_comp[k], axis=0) ** 2))
            ## An approximation to kappa (Banerjee 2005 & Sra, 2011)
            kap_hat = R_bar * (dim[k] + 1 - R_bar ** 2) / (1 - R_bar ** 2)
            if dim[k] == 2:
                h[k] = (8*np.sinh(kap_hat)**2/(n*kap_hat * \
                 ((1+4*kap_hat**2)*np.sinh(2*kap_hat) - 2*kap_hat*np.cosh(2*kap_hat))))**(1/6)
            else:
                h[k] = ((4 * np.sqrt(np.pi) * sp.iv((dim[k]-1) / 2 , kap_hat)**2) / \
                 (n * kap_hat ** ((dim[k]+1) / 2) * (2 * dim[k] * sp.iv((dim[k]+1)/2, 2*kap_hat) + \
                    (dim[k]+2) * kap_hat * sp.iv((dim[k]+3)/2, 2*kap_hat)))) ** (1/(dim[k] + 4))
            print("The current bandwidth for the "+str(k)+"-th directional component is "\
                + str(h[k]) + ".\n")
        elif (h[k] is None) and (com_type[k] == 'Lin'):
            # Apply Silverman's rule of thumb to select the bandwidth parameter 
            # (Only works for Gaussian kernel)
            h[k] = (4/(dim[k]+2))**(1/(dim[k]+4))*(n**(-1/(dim[k]+4)))\
                *np.mean(np.std(data_comp[k], axis=0))
            print("The current bandwidth for the "+str(k)+"-th linear component is "\
                  + str(h[k]) + ".\n")
    
    # Compute the kernel weights contributed by each directional/linear component
    for k in range(len(dim)):
        if com_type[k] == 'Dir':
            # Compute the kernel weights contributed by directional components
            if dim[k] == 2:
                f_hat_comp = np.exp((np.dot(x_comp[k], data_comp[k].T)-1)/(h[k]**2)) / \
                    (2*np.pi*(1-np.exp(-2/h[k]**2))*h[k]**2)
            else:
                f_hat_comp = np.exp(np.dot(x_comp[k], data_comp[k].T)/(h[k]**2)) / \
                    ((2*np.pi)**((dim[k]+1)/2)*sp.iv((dim[k]-1)/2, 1/(h[k]**2))*h[k]**(dim[k]-1))
        elif com_type[k] == 'Lin':
            # Compute the kernel weights contributed by linear components
            f_hat_comp = np.zeros((x.shape[0], n))
            for i in range(x.shape[0]):
                f_hat_comp[i,:] = np.exp(np.sum(-((x_comp[k][i,:] - data_comp[k])/h[k])**2, axis=1)/2)/ \
                                ((2*np.pi)**(dim[k]/2)*h[k])
        if k == 0:
            f_hat = f_hat_comp
        else:
            f_hat = np.multiply(f_hat, f_hat_comp)
    
    f_hat = np.mean(f_hat, axis=1)
    return f_hat


def DirLinProdMS(mesh_0, data, h=[None,None], com_type=['Dir','Lin'], dim=[2,1], 
                 eps=1e-7, max_iter=1000):
    '''
    Mean Shift Algorithm with the von Mises/Gaussian product kernels in a 
    directional/linear (mixture) product space (Simultaneous version).
    
    Parameters:
        mesh_0: (m, sum(dim)+sum(com_type=='Dir'))-array
            Eulidean coordinates of m query points in the product space, where 
            (dim[0]+1) / dim[0] is the Euclidean dimension of a directional/linear 
            component (first (dim[0]+1) columns), and so on.
    
        data: (n, sum(dim)+sum(com_type=='Dir'))-array
            Euclidean coordinates of n random sample points in the product space, 
            where (dim[0]+1) / dim[0] is the Euclidean dimension of a 
            directional/linear component (first (dim[0]+1) columns), and so on.
       
        h: list of floats
            Bandwidth parameters for all the components. (Default: h=[None]*K, 
            where K is the number of components in the product space. Whenever
            h[k]=None for some k=1,...,K, then a rule of thumb for directional 
            KDE with the von Mises kernel in Garcia-Portugues (2013) is applied 
            to that directional component or the Silverman's rule of thumb is 
            applied to that linear component; see Chen et al.(2016) for details.)
            
        com_type: list of strings
            Indicators of the data type for all the components. If com_type[k]='Dir',
            then the corresponding component is directional. If com_type[k]='Lin', 
            then the corresponding component is linear.
            
        dim: list of ints
            Intrinsic data dimensions of all the directional/linear components.
       
        eps: float
            The precision parameter. (Default: eps=1e-7.)
       
        max_iter: int
            The maximum number of iterations for the MS algorithm on each 
            initial point. (Default: max_iter=1000.)
    
    Return:
        MS_path: (m, sum(dim)+sum(com_type=='Dir'), T)-array
            The entire iterative MS sequence for each initial point.
    '''
    
    n = data.shape[0]  ## Number of data points
    D_t = data.shape[1]  ## Total dimension of the data
    
    com_type = np.array(com_type)  ## Convert the list object "com_type" to a numpy array
    assert len(dim) == len(com_type), "The lengths of data type argument 'com_type'"\
    " and dimension argument 'dim' should be the same."
    
    assert sum(dim)+sum(com_type=='Dir') == D_t, "The dimension of the input data, "\
    +str(D_t)+", should be equal to the sum of the Euclidean dimension of the "\
    "directional components ("+str(sum(dim)+sum(com_type=='Dir'))+")."
    
    assert len(h) == len(dim), "The lengths of bandwidth argument 'h' and dimension"\
    " argument 'dim' should be the same."
    
    Eu_dim = [0]
    # Rule of thumb applied to each component
    for k in range(len(h)):
        if com_type[k] == 'Dir':
            if k == 0:
                data_comp = data[:,:(dim[k]+1)]
                Eu_dim.append(dim[k]+1)
            else:
                data_comp = data[:,sum(Eu_dim):(sum(Eu_dim)+dim[k]+1)]
                Eu_dim.append(dim[k]+1)
            if h[k] is None:
                R_bar = np.sqrt(sum(np.mean(data_comp, axis=0) ** 2))
                ## An approximation to kappa (Banerjee 2005 & Sra, 2011)
                kap_hat = R_bar * (dim[k] + 1 - R_bar ** 2) / (1 - R_bar ** 2)
                if dim[k] == 2:
                    h[k] = (8*np.sinh(kap_hat)**2/(n*kap_hat * \
                     ((1+4*kap_hat**2)*np.sinh(2*kap_hat) - 2*kap_hat*np.cosh(2*kap_hat))))**(1/6)
                else:
                    h[k] = ((4 * np.sqrt(np.pi) * sp.iv((dim[k]-1) / 2 , kap_hat)**2) / \
                     (n * kap_hat ** ((dim[k]+1) / 2) * (2 * dim[k] * sp.iv((dim[k]+1)/2, 2*kap_hat) + \
                        (dim[k]+2) * kap_hat * sp.iv((dim[k]+3)/2, 2*kap_hat)))) ** (1/(dim[k] + 4))
                print("The current bandwidth for the "+str(k)+"-th directional component is "\
                    + str(h[k]) + ".\n")
        elif com_type[k] == 'Lin':
            if k == 0:
                data_comp = data[:,:dim[k]]
                Eu_dim.append(dim[k])
            else:
                data_comp = data[:,sum(Eu_dim):(sum(Eu_dim)+dim[k])]
                Eu_dim.append(dim[k])
            if h[k] is None:
                # Apply Silverman's rule of thumb to select the bandwidth parameter 
                # (Only works for Gaussian kernel)
                h[k] = (4/(dim[k]+2))**(1/(dim[k]+4))*(n**(-1/(dim[k]+4)))\
                    *np.mean(np.std(data_comp, axis=0))
                print("The current bandwidth for the "+str(k)+"-th linear component is "\
                      + str(h[k]) + ".\n")
    
    MS_path = np.zeros((mesh_0.shape[0], D_t, max_iter))
    ## Create a vector indicating the convergent status of every mesh point
    conv_sign = np.zeros((mesh_0.shape[0], )) 
    MS_path[:,:,0] = mesh_0
    for t in range(1, max_iter):
        if all(conv_sign == 1):
            print('The MS algorithm in a directional/linear product space converges'\
                  ' in '+ str(t-1) + 'steps!')
            break
        for i in range(mesh_0.shape[0]):
            if conv_sign[i] == 0:
                for k in range(len(dim)):
                    data_comp = data[:,sum(Eu_dim[:(k+1)]):sum(Eu_dim[:(k+2)])]
                    x_comp = MS_path[i,sum(Eu_dim[:(k+1)]):sum(Eu_dim[:(k+2)]),t-1]
                    # Kernel weights
                    ker_w_comp = np.exp(-np.sum(((x_comp - data_comp)/h[k])**2, axis=1)/2)
                    if k == 0:
                        ker_w_mul = ker_w_comp.reshape(n,1)
                    else:
                        ker_w_mul = ker_w_mul * ker_w_comp.reshape(n,1)
                for k in range(len(dim)):
                    if com_type[k] == 'Dir':
                        data_comp = data[:,sum(Eu_dim[:(k+1)]):sum(Eu_dim[:(k+2)])]
                        # Mean shift updates for the directional component
                        x_Dir_new = np.sum(data_comp * ker_w_mul, axis=0)
                        x_Dir_new = x_Dir_new / LA.norm(x_Dir_new)
                        MS_path[i,sum(Eu_dim[:(k+1)]):sum(Eu_dim[:(k+2)]),t] = x_Dir_new
                    elif com_type[k] == 'Lin':
                        data_comp = data[:,sum(Eu_dim[:(k+1)]):sum(Eu_dim[:(k+2)])]
                        x_Lin_new = np.sum(data_comp * ker_w_mul, axis=0) / np.sum(ker_w_mul)
                        MS_path[i,sum(Eu_dim[:(k+1)]):sum(Eu_dim[:(k+2)]),t] = x_Lin_new
                # Check if the current point is converged
                if LA.norm(MS_path[i,:,t] - MS_path[i,:,t-1]) <= eps:
                    conv_sign[i] = 1
            else:
                MS_path[i,:,t] = MS_path[i,:,t-1]
        # print(t)
                
    if t >= max_iter-1:
        print('The MS algorithm reaches the maximum number of iterations,'\
               +str(max_iter)+', and has not yet converged.')
    return MS_path[:,:,:t]


def DirLinProdMSCompAsc(mesh_0, data, h=[None,None], com_type=['Dir','Lin'], 
                        dim=[2,1], eps=1e-7, max_iter=1000):
    '''
    Mean Shift Algorithm with the von Mises product kernels in a 
    directional/linear (mixture) product space (Componentwise ascending version).
    
    Parameters:
        mesh_0: (m, sum(dim)+sum(com_type=='Dir'))-array
            Eulidean coordinates of m query points in the product space, where 
            (dim[0]+1) / dim[0] is the Euclidean dimension of a directional/linear 
            component (first (dim[0]+1) columns), and so on.
    
        data: (n, sum(dim)+sum(com_type=='Dir'))-array
            Euclidean coordinates of n random sample points in the product space, 
            where (dim[0]+1) / dim[0] is the Euclidean dimension of a 
            directional/linear component (first (dim[0]+1) columns), and so on.
       
        h: list of floats
            Bandwidth parameters for all the components. (Default: h=[None]*K, 
            where K is the number of components in the product space. Whenever
            h[k]=None for some k=1,...,K, then a rule of thumb for directional 
            KDE with the von Mises kernel in Garcia-Portugues (2013) is applied 
            to that directional component or the Silverman's rule of thumb is 
            applied to that linear component; see Chen et al.(2016) for details.)
            
        com_type: list of strings
            Indicators of the data type for all the components. If com_type[k]='Dir',
            then the corresponding component is directional. If com_type[k]='Lin', 
            then the corresponding component is linear.
            
        dim: list of ints
            Intrinsic data dimensions of all the directional/linear components.
       
        eps: float
            The precision parameter. (Default: eps=1e-7.)
       
        max_iter: int
            The maximum number of iterations for the MS algorithm on each 
            initial point. (Default: max_iter=1000.)
    
    Return:
        MS_path: (m, sum(dim)+sum(com_type=='Dir'), T)-array
            The entire iterative MS sequence for each initial point.
    '''
    
    n = data.shape[0]  ## Number of data points
    D_t = data.shape[1]  ## Total dimension of the data
    
    com_type = np.array(com_type)  ## Convert the list object "com_type" to a numpy array
    assert len(dim) == len(com_type), "The lengths of data type argument 'com_type'"\
    " and dimension argument 'dim' should be the same."
    
    assert sum(dim)+sum(com_type=='Dir') == D_t, "The dimension of the input data, "\
    +str(D_t)+", should be equal to the sum of the Euclidean dimension of the "\
    "directional components ("+str(sum(dim)+sum(com_type=='Dir'))+")."
    
    assert len(h) == len(dim), "The lengths of bandwidth argument 'h' and dimension"\
    " argument 'dim' should be the same."
    
    Eu_dim = [0]
    # Rule of thumb applied to each component
    for k in range(len(h)):
        if com_type[k] == 'Dir':
            if k == 0:
                data_comp = data[:,:(dim[k]+1)]
                Eu_dim.append(dim[k]+1)
            else:
                data_comp = data[:,sum(Eu_dim):(sum(Eu_dim)+dim[k]+1)]
                Eu_dim.append(dim[k]+1)
            if h[k] is None:
                R_bar = np.sqrt(sum(np.mean(data_comp, axis=0) ** 2))
                ## An approximation to kappa (Banerjee 2005 & Sra, 2011)
                kap_hat = R_bar * (dim[k] + 1 - R_bar ** 2) / (1 - R_bar ** 2)
                if dim[k] == 2:
                    h[k] = (8*np.sinh(kap_hat)**2/(n*kap_hat * \
                     ((1+4*kap_hat**2)*np.sinh(2*kap_hat) - 2*kap_hat*np.cosh(2*kap_hat))))**(1/6)
                else:
                    h[k] = ((4 * np.sqrt(np.pi) * sp.iv((dim[k]-1) / 2 , kap_hat)**2) / \
                     (n * kap_hat ** ((dim[k]+1) / 2) * (2 * dim[k] * sp.iv((dim[k]+1)/2, 2*kap_hat) + \
                        (dim[k]+2) * kap_hat * sp.iv((dim[k]+3)/2, 2*kap_hat)))) ** (1/(dim[k] + 4))
                print("The current bandwidth for the "+str(k)+"-th directional component is "\
                    + str(h[k]) + ".\n")
        elif com_type[k] == 'Lin':
            if k == 0:
                data_comp = data[:,:dim[k]]
                Eu_dim.append(dim[k])
            else:
                data_comp = data[:,sum(Eu_dim):(sum(Eu_dim)+dim[k])]
                Eu_dim.append(dim[k])
            if h[k] is None:
                # Apply Silverman's rule of thumb to select the bandwidth parameter 
                # (Only works for Gaussian kernel)
                h[k] = (4/(dim[k]+2))**(1/(dim[k]+4))*(n**(-1/(dim[k]+4)))\
                    *np.mean(np.std(data_comp, axis=0))
                print("The current bandwidth for the "+str(k)+"-th linear component is "\
                      + str(h[k]) + ".\n")
    
    MS_path = np.zeros((mesh_0.shape[0], D_t, max_iter))
    ## Create a vector indicating the convergent status of every mesh point
    conv_sign = np.zeros((mesh_0.shape[0], )) 
    MS_path[:,:,0] = mesh_0
    ker_w = np.zeros((n, len(dim), mesh_0.shape[0]))
    for i in range(mesh_0.shape[0]):
        for k in range(len(dim)):
            data_comp = data[:,sum(Eu_dim[:(k+1)]):sum(Eu_dim[:(k+2)])]
            x_comp = MS_path[i,sum(Eu_dim[:(k+1)]):sum(Eu_dim[:(k+2)]),0]
            # Kernel weights
            ker_w[:,k,i] = np.exp(-np.sum(((x_comp - data_comp)/h[k])**2, axis=1)/2)
    for t in range(1, max_iter):
        if all(conv_sign == 1):
            print('The MS algorithm in a directional/linear product space converges'\
                  ' in '+ str(t-1) + 'steps!')
            break
        for i in range(mesh_0.shape[0]):
            if conv_sign[i] == 0:
                for k in range(len(dim)):
                    data_comp = data[:,sum(Eu_dim[:(k+1)]):sum(Eu_dim[:(k+2)])]
                    x_comp = MS_path[i,sum(Eu_dim[:(k+1)]):sum(Eu_dim[:(k+2)]),t-1]
                    # Kernel weights
                    ker_w[:,k,i] = np.exp(-np.sum(((x_comp - data_comp)/h[k])**2, axis=1)/2)
                    ker_w_mul = np.prod(ker_w[:,:,i], axis=1)
                    if com_type[k] == 'Dir':
                        x_Dir_new = np.sum(data_comp * ker_w_mul.reshape(n,1), axis=0)
                        x_Dir_new = x_Dir_new / LA.norm(x_Dir_new)
                        MS_path[i,sum(Eu_dim[:(k+1)]):sum(Eu_dim[:(k+2)]),t] = x_Dir_new
                    elif com_type[k] == 'Lin':
                        x_Lin_new = np.sum(data_comp * ker_w_mul.reshape(n,1), axis=0) \
                                   / np.sum(ker_w_mul)
                        MS_path[i,sum(Eu_dim[:(k+1)]):sum(Eu_dim[:(k+2)]),t] = x_Lin_new
                # Check if the current point is converged
                if LA.norm(MS_path[i,:,t] - MS_path[i,:,t-1]) <= eps:
                    conv_sign[i] = 1
            else:
                MS_path[i,:,t] = MS_path[i,:,t-1]
        # print(t)
                
    if t >= max_iter-1:
        print('The MS algorithm reaches the maximum number of iterations,'\
               +str(max_iter)+', and has not yet converged.')
    return MS_path[:,:,:t]


def DirLinProdSCMS(mesh_0, data, d=1, h=[None,None], com_type=['Dir','Lin'], 
                   dim=[2,1], eps=1e-7, max_iter=1000, eta=None):
    '''
    Subspace Constrained Mean Shift Algorithm with the Gaussian/von Mises kernels
    in a directional/linear (mixture) product space. (Our proposed version, 
    converging to density ridges under the correct (Riemannian) gradient of 
    directional/linear (mixture) KDE).
    
    Parameters:
        mesh_0: (m, sum(dim)+sum(com_type=='Dir'))-array
            Eulidean coordinates of m query points in the product space, where 
            (dim[0]+1) / dim[0] is the Euclidean dimension of a directional/linear 
            component (first (dim[0]+1) columns), and so on.
    
        data: (n, sum(dim)+sum(com_type=='Dir'))-array
            Euclidean coordinates of n random sample points in the product space, 
            where (dim[0]+1) / dim[0] is the Euclidean dimension of a 
            directional/linear component (first (dim[0]+1) columns), and so on.
        
        d: int
            The order of the density ridge. (Default: d=1.)
       
        h: list of floats
            Bandwidth parameters for all the components. (Default: h=[None]*K, 
            where K is the number of components in the product space. Whenever
            h[k]=None for some k=1,...,K, then a rule of thumb for directional 
            KDE with the von Mises kernel in Garcia-Portugues (2013) is applied 
            to that directional component or the Silverman's rule of thumb is 
            applied to that linear component; see Chen et al.(2016) for details.)
            
        com_type: list of strings
            Indicators of the data type for all the components. If com_type[k]='Dir',
            then the corresponding component is directional. If com_type[k]='Lin', 
            then the corresponding component is linear.
            
        dim: list of ints
            Intrinsic data dimensions of all the directional/linear components.
       
        eps: float
            The precision parameter. (Default: eps=1e-7.)
       
        max_iter: int
            The maximum number of iterations for the SCMS algorithm on each 
            initial point. (Default: max_iter=1000.)
            
        eta: float
            The step size parameter for the SCMS algorithm. (Default: eta=None, 
            then eta=np.min([np.min(h) * np.max(h), 1]).)
            
    Return:
        SCMS_path: (m, sum(dim)+sum(com_type=='Dir'), T)-array
            The entire iterative SCMS sequence for each initial point.
    '''
    
    n = data.shape[0]  ## Number of data points
    D_t = data.shape[1]  ## Total dimension of the data
    
    com_type = np.array(com_type)  ## Convert the list object "com_type" to a numpy array
    assert len(dim) == len(com_type), "The lengths of data type argument 'com_type'"\
    " and dimension argument 'dim' should be the same."
    
    assert sum(dim)+sum(com_type=='Dir') == D_t, "The dimension of the input data, "\
    +str(D_t)+", should be equal to the sum of the Euclidean dimension of the "\
    "directional components ("+str(sum(dim)+sum(com_type=='Dir'))+")."
    
    assert len(h) == len(dim), "The lengths of bandwidth argument 'h' and dimension"\
    " argument 'dim' should be the same."
    
    Eu_dim = [0]
    H = []   ## Bandwidth matrix
    # Rule of thumb applied to each component
    for k in range(len(h)):
        if com_type[k] == 'Dir':
            if k == 0:
                data_comp = data[:,:(dim[k]+1)]
                Eu_dim.append(dim[k]+1)
            else:
                data_comp = data[:,sum(Eu_dim):(sum(Eu_dim)+dim[k]+1)]
                Eu_dim.append(dim[k]+1)
            if h[k] is None:
                R_bar = np.sqrt(sum(np.mean(data_comp, axis=0) ** 2))
                ## An approximation to kappa (Banerjee 2005 & Sra, 2011)
                kap_hat = R_bar * (dim[k] + 1 - R_bar ** 2) / (1 - R_bar ** 2)
                if dim[k] == 2:
                    h[k] = (8*np.sinh(kap_hat)**2/(n*kap_hat * \
                     ((1+4*kap_hat**2)*np.sinh(2*kap_hat) - 2*kap_hat*np.cosh(2*kap_hat))))**(1/6)
                else:
                    h[k] = ((4 * np.sqrt(np.pi) * sp.iv((dim[k]-1) / 2 , kap_hat)**2) / \
                     (n * kap_hat ** ((dim[k]+1) / 2) * (2 * dim[k] * sp.iv((dim[k]+1)/2, 2*kap_hat) + \
                        (dim[k]+2) * kap_hat * sp.iv((dim[k]+3)/2, 2*kap_hat)))) ** (1/(dim[k] + 4))
                print("The current bandwidth for the "+str(k)+"-th directional component is "\
                    + str(h[k]) + ".\n")
            H.append((h[k]**2)*np.eye(dim[k]+1))
        elif com_type[k] == 'Lin':
            if k == 0:
                data_comp = data[:,:dim[k]]
                Eu_dim.append(dim[k])
            else:
                data_comp = data[:,sum(Eu_dim):(sum(Eu_dim)+dim[k])]
                Eu_dim.append(dim[k])
            if h[k] is None:
                # Apply Silverman's rule of thumb to select the bandwidth parameter 
                # (Only works for Gaussian kernel)
                h[k] = (4/(dim[k]+2))**(1/(dim[k]+4))*(n**(-1/(dim[k]+4)))\
                    *np.mean(np.std(data_comp, axis=0))
                print("The current bandwidth for the "+str(k)+"-th linear component is "\
                      + str(h[k]) + ".\n")
            H.append((h[k]**2)*np.eye(dim[k]))
    # Convert a list of diagonal matrices to the final block diagonal bandwidth matrix
    H = block_diag(*H)
    if eta is None:
        eta = np.min([np.min(h) * np.max(h), 1])  ## step size for SCMS iterations
    
    SCMS_path = np.zeros((mesh_0.shape[0], D_t, max_iter))
    SCMS_path[:,:,0] = mesh_0
    # Create a vector indicating the convergent status of every mesh point
    conv_sign = np.zeros((mesh_0.shape[0], ))
    for t in range(1, max_iter):
        if all(conv_sign > 0):
            print('The SCMS algorithm in a directional/linear product space converges'\
                  ' in ' + str(t-1) + ' steps!')
            break
        for i in range(mesh_0.shape[0]):
            if conv_sign[i] == 0:
                x_pts = SCMS_path[i,:,t-1]
                # tot_grad = np.sum(np.dot(x_pts - data, LA.inv(H)) \
                #     * np.diagonal(np.exp(-np.dot(np.dot(x_pts - data, LA.inv(H)), 
                #                                  (x_pts - data).T)/2)).reshape(n,1), axis=0)
                for k in range(len(dim)):
                    data_comp = data[:,sum(Eu_dim[:(k+1)]):sum(Eu_dim[:(k+2)])]
                    x_comp = SCMS_path[i,sum(Eu_dim[:(k+1)]):sum(Eu_dim[:(k+2)]),t-1]
                    # Kernel weights
                    ker_w_comp = np.exp(-np.sum(((x_comp - data_comp)/h[k])**2, axis=1)/2)
                    if k == 0:
                        ker_w_mul = ker_w_comp.reshape(n,1)
                    else:
                        ker_w_mul = ker_w_mul * ker_w_comp.reshape(n,1)
                # Compute the total gradient
                tot_grad = np.sum(np.dot(data - x_pts, LA.inv(H)) * ker_w_mul, axis=0)
                # Compute the projection matrix and radial gradient block matrix
                proj_mat = []
                rad_grad = []
                # Eigenvector of the Hessian in the normal direction
                x_eig = np.zeros((D_t, sum(com_type == 'Dir')))
                cnt_Dir = 0
                ms_v = []   ## (Modified) mean shift vector
                for k in range(len(dim)):
                    data_comp = data[:,sum(Eu_dim[:(k+1)]):sum(Eu_dim[:(k+2)])]
                    x_comp = SCMS_path[i,sum(Eu_dim[:(k+1)]):sum(Eu_dim[:(k+2)]),t-1]
                    if com_type[k] == 'Dir':
                        grad_comp = tot_grad[sum(Eu_dim[:(k+1)]):sum(Eu_dim[:(k+2)])]
                        proj_mat.append(np.eye(dim[k]+1) - np.dot(x_comp.reshape(-1,1), 
                                                                  x_comp.reshape(1,-1)))
                        rad_grad.append(np.eye(dim[k]+1)*np.dot(x_comp, grad_comp))
                        x_eig_can = np.zeros((D_t,))
                        x_eig_can[sum(Eu_dim[:(k+1)]):sum(Eu_dim[:(k+2)])] = x_comp
                        x_eig[:,cnt_Dir] = x_eig_can
                        # (Modified) mean shift vector in the directional component
                        ms_Dir = eta*np.sum(data_comp * ker_w_mul, axis=0) / \
                            np.sum(ker_w_mul) / (h[k]**2)
                        ms_v.extend(ms_Dir)
                        cnt_Dir += 1
                    elif com_type[k] == 'Lin':
                        proj_mat.append(np.eye(dim[k]))
                        rad_grad.append(0*np.eye(dim[k]))
                        # (Modified) mean shift vector in the linear component
                        ms_Lin = eta*(np.sum(data_comp * ker_w_mul, axis=0) / \
                                    np.sum(ker_w_mul) - x_comp) / (h[k]**2)
                        ms_v.extend(ms_Lin)
                proj_mat = block_diag(*proj_mat)
                rad_grad = block_diag(*rad_grad)
                # Compute the (Riemannian) Hessian matrix
                tot_Hess = np.dot(np.dot(x_pts - data, LA.inv(H)).T, 
                                  np.dot((x_pts - data) * ker_w_mul, LA.inv(H))) \
                           - LA.inv(H) * np.sum(ker_w_mul)
                Hess = np.dot(np.dot(proj_mat, tot_Hess - rad_grad), proj_mat)
                # Spectral decomposition
                w, v = LA.eig(Hess)
                # Obtain the eigenpairs within the tangent space
                if sum(com_type == 'Dir') > 0:
                    tang_eig_v = v[:, (abs(np.sum(np.dot(x_eig.T, v), axis=0)) < 1e-8)]
                    tang_eig_w = w[(abs(np.sum(np.dot(x_eig.T, v), axis=0)) < 1e-8)]
                else:
                    tang_eig_v = v
                    tang_eig_w = w
                V_d = tang_eig_v[:, np.argsort(tang_eig_w)[:(sum(dim)-d)]]
                # Subspace constrained gradient and mean shift vector
                SCMS_grad = np.dot(V_d, np.dot(V_d.T, tot_grad))
                SCMS_v = np.dot(V_d, np.dot(V_d.T, ms_v))
                # SCMS update
                x_new = SCMS_v + x_pts
                for k in range(len(dim)):
                    if com_type[k] == 'Dir':
                        x_norm = x_new[sum(Eu_dim[:(k+1)]):sum(Eu_dim[:(k+2)])]
                        x_new[sum(Eu_dim[:(k+1)]):sum(Eu_dim[:(k+2)])] = x_norm / LA.norm(x_norm)
                if LA.norm(SCMS_grad) < eps:
                    conv_sign[i] = 1
                SCMS_path[i,:,t] = x_new
            else:
                SCMS_path[i,:,t] = SCMS_path[i,:,t-1]
        # print(t)
    # print(conv_sign)
    if t >= max_iter-1:
        print('The SCMS algorithm in a directional/linear product space reaches '\
              'the maximum number of iterations,'+str(max_iter)+', and has not '\
              'yet converged.')
    # SCMS_path = SCMS_path[conv_sign != 0,:,:t]
    nan_cri = np.isnan(SCMS_path[:,0,t-1])
    SCMS_path = SCMS_path[~nan_cri,:,:t]
    return SCMS_path[:,:,:t], conv_sign


def DirLinProdSCMSLog(mesh_0, data, d=1, h=[None,None], com_type=['Dir','Lin'], 
                   dim=[2,1], eps=1e-7, max_iter=1000, eta=None):
    '''
    Subspace Constrained Mean Shift Algorithm with the Gaussian/von Mises kernels
    under the log-density in a directional/linear (mixture) product space. 
    (Our proposed version, converging to density ridges under the correct 
    (Riemannian) gradient of directional/linear (mixture) KDE).
    
    Parameters:
        mesh_0: (m, sum(dim)+sum(com_type=='Dir'))-array
            Eulidean coordinates of m query points in the product space, where 
            (dim[0]+1) / dim[0] is the Euclidean dimension of a directional/linear 
            component (first (dim[0]+1) columns), and so on.
    
        data: (n, sum(dim)+sum(com_type=='Dir'))-array
            Euclidean coordinates of n random sample points in the product space, 
            where (dim[0]+1) / dim[0] is the Euclidean dimension of a 
            directional/linear component (first (dim[0]+1) columns), and so on.
        
        d: int
            The order of the density ridge. (Default: d=1.)
       
        h: list of floats
            Bandwidth parameters for all the components. (Default: h=[None]*K, 
            where K is the number of components in the product space. Whenever
            h[k]=None for some k=1,...,K, then a rule of thumb for directional 
            KDE with the von Mises kernel in Garcia-Portugues (2013) is applied 
            to that directional component or the Silverman's rule of thumb is 
            applied to that linear component; see Chen et al.(2016) for details.)
            
        com_type: list of strings
            Indicators of the data type for all the components. If com_type[k]='Dir',
            then the corresponding component is directional. If com_type[k]='Lin', 
            then the corresponding component is linear.
            
        dim: list of ints
            Intrinsic data dimensions of all the directional/linear components.
       
        eps: float
            The precision parameter. (Default: eps=1e-7.)
       
        max_iter: int
            The maximum number of iterations for the SCMS algorithm on each 
            initial point. (Default: max_iter=1000.)
        
        eta: float
            The step size parameter for the SCMS algorithm. (Default: eta=None, 
            then eta=np.min([np.min(h) * np.max(h), 1]).)
            
    Return:
        SCMS_path: (m, sum(dim)+sum(com_type=='Dir'), T)-array
            The entire iterative SCMS sequence for each initial point.
    '''
    
    n = data.shape[0]  ## Number of data points
    D_t = data.shape[1]  ## Total dimension of the data
    
    com_type = np.array(com_type)  ## Convert the list object "com_type" to a numpy array
    assert len(dim) == len(com_type), "The lengths of data type argument 'com_type'"\
    " and dimension argument 'dim' should be the same."
    
    assert sum(dim)+sum(com_type=='Dir') == D_t, "The dimension of the input data, "\
    +str(D_t)+", should be equal to the sum of the Euclidean dimension of the "\
    "directional components ("+str(sum(dim)+sum(com_type=='Dir'))+")."
    
    assert len(h) == len(dim), "The lengths of bandwidth argument 'h' and dimension"\
    " argument 'dim' should be the same."
    
    Eu_dim = [0]
    H = []   ## Bandwidth matrix
    # Rule of thumb applied to each component
    for k in range(len(h)):
        if com_type[k] == 'Dir':
            if k == 0:
                data_comp = data[:,:(dim[k]+1)]
                Eu_dim.append(dim[k]+1)
            else:
                data_comp = data[:,sum(Eu_dim):(sum(Eu_dim)+dim[k]+1)]
                Eu_dim.append(dim[k]+1)
            if h[k] is None:
                R_bar = np.sqrt(sum(np.mean(data_comp, axis=0) ** 2))
                ## An approximation to kappa (Banerjee 2005 & Sra, 2011)
                kap_hat = R_bar * (dim[k] + 1 - R_bar ** 2) / (1 - R_bar ** 2)
                if dim[k] == 2:
                    h[k] = (8*np.sinh(kap_hat)**2/(n*kap_hat * \
                     ((1+4*kap_hat**2)*np.sinh(2*kap_hat) - 2*kap_hat*np.cosh(2*kap_hat))))**(1/6)
                else:
                    h[k] = ((4 * np.sqrt(np.pi) * sp.iv((dim[k]-1) / 2 , kap_hat)**2) / \
                     (n * kap_hat ** ((dim[k]+1) / 2) * (2 * dim[k] * sp.iv((dim[k]+1)/2, 2*kap_hat) + \
                        (dim[k]+2) * kap_hat * sp.iv((dim[k]+3)/2, 2*kap_hat)))) ** (1/(dim[k] + 4))
                print("The current bandwidth for the "+str(k)+"-th directional component is "\
                    + str(h[k]) + ".\n")
            H.append((h[k]**2)*np.eye(dim[k]+1))
        elif com_type[k] == 'Lin':
            if k == 0:
                data_comp = data[:,:dim[k]]
                Eu_dim.append(dim[k])
            else:
                data_comp = data[:,sum(Eu_dim):(sum(Eu_dim)+dim[k])]
                Eu_dim.append(dim[k])
            if h[k] is None:
                # Apply Silverman's rule of thumb to select the bandwidth parameter 
                # (Only works for Gaussian kernel)
                h[k] = (4/(dim[k]+2))**(1/(dim[k]+4))*(n**(-1/(dim[k]+4)))\
                    *np.mean(np.std(data_comp, axis=0))
                print("The current bandwidth for the "+str(k)+"-th linear component is "\
                      + str(h[k]) + ".\n")
            H.append((h[k]**2)*np.eye(dim[k]))
    # Convert a list of diagonal matrices to the final block diagonal bandwidth matrix
    H = block_diag(*H)
    if eta is None:
        eta = np.min([np.min(h) * np.max(h), 1])  ## step size for SCMS iterations
    
    SCMS_path = np.zeros((mesh_0.shape[0], D_t, max_iter))
    SCMS_path[:,:,0] = mesh_0
    # Create a vector indicating the convergent status of every mesh point
    conv_sign = np.zeros((mesh_0.shape[0], ))
    for t in range(1, max_iter):
        if all(conv_sign > 0):
            print('The SCMS algorithm in a directional/linear product space converges'\
                  ' in ' + str(t-1) + ' steps!')
            break
        for i in range(mesh_0.shape[0]):
            if conv_sign[i] == 0:
                x_pts = SCMS_path[i,:,t-1]
                for k in range(len(dim)):
                    data_comp = data[:,sum(Eu_dim[:(k+1)]):sum(Eu_dim[:(k+2)])]
                    x_comp = SCMS_path[i,sum(Eu_dim[:(k+1)]):sum(Eu_dim[:(k+2)]),t-1]
                    # Kernel weights
                    ker_w_comp = np.exp(-np.sum(((x_comp - data_comp)/h[k])**2, axis=1)/2)
                    if k == 0:
                        ker_w_mul = ker_w_comp.reshape(n,1)
                    else:
                        ker_w_mul = ker_w_mul * ker_w_comp.reshape(n,1)
                den_prop = np.sum(ker_w_mul)
                if den_prop == 0:
                    # Set those points with zero density values to NaN
                    nan_arr = np.zeros_like(x_pts)
                    nan_arr[:] = np.nan
                    conv_sign[i] = 1
                    x_new = nan_arr
                else:
                    # Compute the total gradient
                    tot_grad_Log = np.sum(np.dot(data - x_pts, LA.inv(H)) \
                                          * ker_w_mul, axis=0) / den_prop
                    # Compute the projection matrix and radial gradient block matrix
                    proj_mat = []
                    rad_grad = []
                    # Eigenvector of the Hessian in the normal direction
                    x_eig = np.zeros((D_t, sum(com_type == 'Dir')))
                    cnt_Dir = 0
                    ms_v = []   ## (Modified) mean shift vector
                    for k in range(len(dim)):
                        data_comp = data[:,sum(Eu_dim[:(k+1)]):sum(Eu_dim[:(k+2)])]
                        x_comp = SCMS_path[i,sum(Eu_dim[:(k+1)]):sum(Eu_dim[:(k+2)]),t-1]
                        if com_type[k] == 'Dir':
                            grad_comp = tot_grad_Log[sum(Eu_dim[:(k+1)]):sum(Eu_dim[:(k+2)])]
                            proj_mat.append(np.eye(dim[k]+1) - np.dot(x_comp.reshape(-1,1), 
                                                                      x_comp.reshape(1,-1)))
                            rad_grad.append(np.eye(dim[k]+1)*np.dot(x_comp, grad_comp))
                            x_eig_can = np.zeros((D_t,))
                            x_eig_can[sum(Eu_dim[:(k+1)]):sum(Eu_dim[:(k+2)])] = x_comp
                            x_eig[:,cnt_Dir] = x_eig_can
                            # (Modified) mean shift vector in the directional component
                            ms_Dir = eta*np.sum(data_comp * ker_w_mul, axis=0) / \
                                np.sum(ker_w_mul) / (h[k]**2)
                            ms_v.extend(ms_Dir)
                            cnt_Dir += 1
                        elif com_type[k] == 'Lin':
                            proj_mat.append(np.eye(dim[k]))
                            rad_grad.append(0*np.eye(dim[k]))
                            # (Modified) mean shift vector in the linear component
                            ms_Lin = eta*(np.sum(data_comp * ker_w_mul, axis=0) / \
                                        np.sum(ker_w_mul) - x_comp) / (h[k]**2)
                            ms_v.extend(ms_Lin)
                    proj_mat = block_diag(*proj_mat)
                    rad_grad = block_diag(*rad_grad)
                    # Compute the (Riemannian) Hessian matrix
                    tot_Hess = np.dot(np.dot(x_pts - data, LA.inv(H)).T, 
                                      np.dot((x_pts - data) * ker_w_mul, LA.inv(H)))/den_prop \
                               - LA.inv(H) * np.sum(ker_w_mul) / den_prop \
                               - np.dot(tot_grad_Log.reshape(D_t,1), tot_grad_Log.reshape(1,D_t))
                    Log_Hess = np.dot(np.dot(proj_mat, tot_Hess - rad_grad), proj_mat)
                    if (np.sum(np.isinf(Log_Hess)) > 0) or (np.sum(np.isnan(Log_Hess)) > 0):
                        # Set those points with zero density values to NaN
                        nan_arr = np.zeros_like(x_pts)
                        nan_arr[:] = np.nan
                        conv_sign[i] = 1
                        x_new = nan_arr
                    else:
                        # Spectral decomposition
                        w, v = LA.eig(Log_Hess)
                        # Obtain the eigenpairs within the tangent space
                        if sum(com_type == 'Dir') > 0:
                            tang_eig_v = v[:, (abs(np.sum(np.dot(x_eig.T, v), axis=0)) < 1e-8)]
                            tang_eig_w = w[(abs(np.sum(np.dot(x_eig.T, v), axis=0)) < 1e-8)]
                        else:
                            tang_eig_v = v
                            tang_eig_w = w
                        V_d = tang_eig_v[:, np.argsort(tang_eig_w)[:(sum(dim)-d)]]
                        # Subspace constrained gradient and mean shift vector
                        SCMS_grad = np.dot(V_d, np.dot(V_d.T, tot_grad_Log))
                        SCMS_v = np.dot(V_d, np.dot(V_d.T, ms_v))
                        # SCMS update
                        x_new = SCMS_v + x_pts
                        for k in range(len(dim)):
                            if com_type[k] == 'Dir':
                                x_norm = x_new[sum(Eu_dim[:(k+1)]):sum(Eu_dim[:(k+2)])]
                                x_new[sum(Eu_dim[:(k+1)]):sum(Eu_dim[:(k+2)])] = x_norm / LA.norm(x_norm)
                        if LA.norm(SCMS_grad) < eps:
                            conv_sign[i] = 1
                SCMS_path[i,:,t] = x_new
            else:
                SCMS_path[i,:,t] = SCMS_path[i,:,t-1]
        # print(t)
    # print(conv_sign)
    if t >= max_iter-1:
        print('The SCMS algorithm in a directional/linear product space reaches '\
              'the maximum number of iterations,'+str(max_iter)+', and has not '\
              'yet converged.')
    # SCMS_path = SCMS_path[conv_sign != 0,:,:t]
    nan_cri = np.isnan(SCMS_path[:,0,t-1])
    SCMS_path = SCMS_path[~nan_cri,:,:t]
    return SCMS_path[:,:,:t], conv_sign



########### The following codes are naive implementation of SCMS algorithm on 
########### directional/linear product spaces.

def DirLinProdSCMS_Naive(mesh_0, data, d=1, h=[None,None], com_type=['Dir','Lin'], 
                   dim=[2,1], eps=1e-7, max_iter=1000):
    '''
    Naive Subspace Constrained Mean Shift Algorithm with the Gaussian/von Mises 
    kernels in a directional/linear (mixture) product space. (Converging to the 
    transformed ridge, not necessarily the correct estimated ridge by KDE.)
    
    Parameters:
        mesh_0: (m, sum(dim)+sum(com_type=='Dir'))-array
            Eulidean coordinates of m query points in the product space, where 
            (dim[0]+1) / dim[0] is the Euclidean dimension of a directional/linear 
            component (first (dim[0]+1) columns), and so on.
    
        data: (n, sum(dim)+sum(com_type=='Dir'))-array
            Euclidean coordinates of n random sample points in the product space, 
            where (dim[0]+1) / dim[0] is the Euclidean dimension of a 
            directional/linear component (first (dim[0]+1) columns), and so on.
        
        d: int
            The order of the density ridge. (Default: d=1.)
       
        h: list of floats
            Bandwidth parameters for all the components. (Default: h=[None]*K, 
            where K is the number of components in the product space. Whenever
            h[k]=None for some k=1,...,K, then a rule of thumb for directional 
            KDE with the von Mises kernel in Garcia-Portugues (2013) is applied 
            to that directional component or the Silverman's rule of thumb is 
            applied to that linear component; see Chen et al.(2016) for details.)
            
        com_type: list of strings
            Indicators of the data type for all the components. If com_type[k]='Dir',
            then the corresponding component is directional. If com_type[k]='Lin', 
            then the corresponding component is linear.
            
        dim: list of ints
            Intrinsic data dimensions of all the directional/linear components.
       
        eps: float
            The precision parameter. (Default: eps=1e-7.)
       
        max_iter: int
            The maximum number of iterations for the SCMS algorithm on each 
            initial point. (Default: max_iter=1000.)
            
    Return:
        SCMS_path: (m, sum(dim)+sum(com_type=='Dir'), T)-array
            The entire iterative SCMS sequence for each initial point.
    '''
    
    n = data.shape[0]  ## Number of data points
    D_t = data.shape[1]  ## Total dimension of the data
    
    com_type = np.array(com_type)  ## Convert the list object "com_type" to a numpy array
    assert len(dim) == len(com_type), "The lengths of data type argument 'com_type'"\
    " and dimension argument 'dim' should be the same."
    
    assert sum(dim)+sum(com_type=='Dir') == D_t, "The dimension of the input data, "\
    +str(D_t)+", should be equal to the sum of the Euclidean dimension of the "\
    "directional components ("+str(sum(dim)+sum(com_type=='Dir'))+")."
    
    assert len(h) == len(dim), "The lengths of bandwidth argument 'h' and dimension"\
    " argument 'dim' should be the same."
    
    Eu_dim = [0]
    H = []   ## Bandwidth matrix
    # Rule of thumb applied to each component
    for k in range(len(h)):
        if com_type[k] == 'Dir':
            if k == 0:
                data_comp = data[:,:(dim[k]+1)]
                Eu_dim.append(dim[k]+1)
            else:
                data_comp = data[:,sum(Eu_dim):(sum(Eu_dim)+dim[k]+1)]
                Eu_dim.append(dim[k]+1)
            if h[k] is None:
                R_bar = np.sqrt(sum(np.mean(data_comp, axis=0) ** 2))
                ## An approximation to kappa (Banerjee 2005 & Sra, 2011)
                kap_hat = R_bar * (dim[k] + 1 - R_bar ** 2) / (1 - R_bar ** 2)
                if dim[k] == 2:
                    h[k] = (8*np.sinh(kap_hat)**2/(n*kap_hat * \
                     ((1+4*kap_hat**2)*np.sinh(2*kap_hat) - 2*kap_hat*np.cosh(2*kap_hat))))**(1/6)
                else:
                    h[k] = ((4 * np.sqrt(np.pi) * sp.iv((dim[k]-1) / 2 , kap_hat)**2) / \
                     (n * kap_hat ** ((dim[k]+1) / 2) * (2 * dim[k] * sp.iv((dim[k]+1)/2, 2*kap_hat) + \
                        (dim[k]+2) * kap_hat * sp.iv((dim[k]+3)/2, 2*kap_hat)))) ** (1/(dim[k] + 4))
                print("The current bandwidth for the "+str(k)+"-th directional component is "\
                    + str(h[k]) + ".\n")
            H.append((h[k]**2)*np.eye(dim[k]+1))
        elif com_type[k] == 'Lin':
            if k == 0:
                data_comp = data[:,:dim[k]]
                Eu_dim.append(dim[k])
            else:
                data_comp = data[:,sum(Eu_dim):(sum(Eu_dim)+dim[k])]
                Eu_dim.append(dim[k])
            if h[k] is None:
                # Apply Silverman's rule of thumb to select the bandwidth parameter 
                # (Only works for Gaussian kernel)
                h[k] = (4/(dim[k]+2))**(1/(dim[k]+4))*(n**(-1/(dim[k]+4)))\
                    *np.mean(np.std(data_comp, axis=0))
                print("The current bandwidth for the "+str(k)+"-th linear component is "\
                      + str(h[k]) + ".\n")
            H.append((h[k]**2)*np.eye(dim[k]))
    # Convert a list of diagonal matrices to the final block diagonal bandwidth matrix
    H = block_diag(*H)
    
    SCMS_path = np.zeros((mesh_0.shape[0], D_t, max_iter))
    SCMS_path[:,:,0] = mesh_0
    # Create a vector indicating the convergent status of every mesh point
    conv_sign = np.zeros((mesh_0.shape[0], ))
    for t in range(1, max_iter):
        if all(conv_sign > 0):
            print('The naive SCMS algorithm in a directional/linear product space '\
                  'converges in ' + str(t-1) + ' steps!')
            break
        for i in range(mesh_0.shape[0]):
            if conv_sign[i] == 0:
                x_pts = SCMS_path[i,:,t-1]
                for k in range(len(dim)):
                    data_comp = data[:,sum(Eu_dim[:(k+1)]):sum(Eu_dim[:(k+2)])]
                    x_comp = SCMS_path[i,sum(Eu_dim[:(k+1)]):sum(Eu_dim[:(k+2)]),t-1]
                    # Kernel weights
                    ker_w_comp = np.exp(-np.sum(((x_comp - data_comp)/h[k])**2, axis=1)/2)
                    if k == 0:
                        ker_w_mul = ker_w_comp.reshape(n,1)
                    else:
                        ker_w_mul = ker_w_mul * ker_w_comp.reshape(n,1)
                # Compute the total gradient
                tot_grad = np.sum(np.dot(data - x_pts, LA.inv(H)) * ker_w_mul, axis=0)
                tot_grad_trans = np.sum((data - x_pts) * ker_w_mul, axis=0)
                # Compute the projection matrix and radial gradient block matrix
                proj_mat = []
                rad_grad = []
                # Eigenvector of the Hessian in the normal direction
                x_eig = np.zeros((D_t, sum(com_type == 'Dir')))
                cnt_Dir = 0
                ms_v = []   ## (Modified) mean shift vector
                for k in range(len(dim)):
                    data_comp = data[:,sum(Eu_dim[:(k+1)]):sum(Eu_dim[:(k+2)])]
                    x_comp = SCMS_path[i,sum(Eu_dim[:(k+1)]):sum(Eu_dim[:(k+2)]),t-1]
                    if com_type[k] == 'Dir':
                        grad_comp = tot_grad[sum(Eu_dim[:(k+1)]):sum(Eu_dim[:(k+2)])]
                        proj_mat.append(np.eye(dim[k]+1) - np.dot(x_comp.reshape(-1,1), 
                                                                  x_comp.reshape(1,-1)))
                        rad_grad.append(np.eye(dim[k]+1)*np.dot(x_comp, grad_comp))
                        x_eig_can = np.zeros((D_t,))
                        x_eig_can[sum(Eu_dim[:(k+1)]):sum(Eu_dim[:(k+2)])] = x_comp
                        x_eig[:,cnt_Dir] = x_eig_can
                        # Mean shift vector in the directional component
                        ms_Dir = np.sum(data_comp * ker_w_mul, axis=0) / np.sum(ker_w_mul)
                        ms_v.extend(ms_Dir)
                        cnt_Dir += 1
                    elif com_type[k] == 'Lin':
                        proj_mat.append(np.eye(dim[k]))
                        rad_grad.append(0*np.eye(dim[k]))
                        # Mean shift vector in the linear component
                        ms_Lin = np.sum(data_comp * ker_w_mul, axis=0)/np.sum(ker_w_mul) - x_comp
                        ms_v.extend(ms_Lin)
                proj_mat = block_diag(*proj_mat)
                rad_grad = block_diag(*rad_grad)
                # Compute the (Riemannian) Hessian matrix
                tot_Hess = np.dot(np.dot(x_pts - data, LA.inv(H)).T, 
                                  np.dot((x_pts - data) * ker_w_mul, LA.inv(H))) \
                           - LA.inv(H) * np.sum(ker_w_mul)
                Hess = np.dot(np.dot(proj_mat, tot_Hess - rad_grad), proj_mat)
                # Spectral decomposition
                w, v = LA.eig(Hess)
                # Obtain the eigenpairs within the tangent space
                if sum(com_type == 'Dir') > 0:
                    tang_eig_v = v[:, (abs(np.sum(np.dot(x_eig.T, v), axis=0)) < 1e-8)]
                    tang_eig_w = w[(abs(np.sum(np.dot(x_eig.T, v), axis=0)) < 1e-8)]
                else:
                    tang_eig_v = v
                    tang_eig_w = w
                V_d = tang_eig_v[:, np.argsort(tang_eig_w)[:(sum(dim)-d)]]
                # Subspace constrained gradient and mean shift vector
                SCMS_grad_trans = np.dot(V_d, np.dot(V_d.T, tot_grad_trans))
                SCMS_v = np.dot(V_d, np.dot(V_d.T, ms_v))
                # SCMS update
                x_new = SCMS_v + x_pts
                for k in range(len(dim)):
                    if com_type[k] == 'Dir':
                        x_norm = x_new[sum(Eu_dim[:(k+1)]):sum(Eu_dim[:(k+2)])]
                        x_new[sum(Eu_dim[:(k+1)]):sum(Eu_dim[:(k+2)])] = x_norm / LA.norm(x_norm)
                if LA.norm(SCMS_grad_trans) < eps:
                    conv_sign[i] = 1
                SCMS_path[i,:,t] = x_new
            else:
                SCMS_path[i,:,t] = SCMS_path[i,:,t-1]
        # print(t)
    # print(conv_sign)
    if t >= max_iter-1:
        print('The naive SCMS algorithm in a directional/linear product space reaches '\
              'the maximum number of iterations,'+str(max_iter)+', and has not '\
              'yet converged.')
    # SCMS_path = SCMS_path[conv_sign != 0,:,:t]
    nan_cri = np.isnan(SCMS_path[:,0,t-1])
    SCMS_path = SCMS_path[~nan_cri,:,:t]
    return SCMS_path[:,:,:t], conv_sign


def DirLinProdSCMSLog_Naive(mesh_0, data, d=1, h=[None,None], com_type=['Dir','Lin'], 
                   dim=[2,1], eps=1e-7, max_iter=1000):
    '''
    Naive Subspace Constrained Mean Shift Algorithm with the Gaussian/von Mises 
    kernels under the log-density in a directional/linear (mixture) product space. 
    (Converging to the transformed ridge, not necessarily the correct estimated 
    ridge by KDE.)
    
    Parameters:
        mesh_0: (m, sum(dim)+sum(com_type=='Dir'))-array
            Eulidean coordinates of m query points in the product space, where 
            (dim[0]+1) / dim[0] is the Euclidean dimension of a directional/linear 
            component (first (dim[0]+1) columns), and so on.
    
        data: (n, sum(dim)+sum(com_type=='Dir'))-array
            Euclidean coordinates of n random sample points in the product space, 
            where (dim[0]+1) / dim[0] is the Euclidean dimension of a 
            directional/linear component (first (dim[0]+1) columns), and so on.
        
        d: int
            The order of the density ridge. (Default: d=1.)
       
        h: list of floats
            Bandwidth parameters for all the components. (Default: h=[None]*K, 
            where K is the number of components in the product space. Whenever
            h[k]=None for some k=1,...,K, then a rule of thumb for directional 
            KDE with the von Mises kernel in Garcia-Portugues (2013) is applied 
            to that directional component or the Silverman's rule of thumb is 
            applied to that linear component; see Chen et al.(2016) for details.)
            
        com_type: list of strings
            Indicators of the data type for all the components. If com_type[k]='Dir',
            then the corresponding component is directional. If com_type[k]='Lin', 
            then the corresponding component is linear.
            
        dim: list of ints
            Intrinsic data dimensions of all the directional/linear components.
       
        eps: float
            The precision parameter. (Default: eps=1e-7.)
       
        max_iter: int
            The maximum number of iterations for the SCMS algorithm on each 
            initial point. (Default: max_iter=1000.)
            
    Return:
        SCMS_path: (m, sum(dim)+sum(com_type=='Dir'), T)-array
            The entire iterative SCMS sequence for each initial point.
    '''
    
    n = data.shape[0]  ## Number of data points
    D_t = data.shape[1]  ## Total dimension of the data
    
    com_type = np.array(com_type)  ## Convert the list object "com_type" to a numpy array
    assert len(dim) == len(com_type), "The lengths of data type argument 'com_type'"\
    " and dimension argument 'dim' should be the same."
    
    assert sum(dim)+sum(com_type=='Dir') == D_t, "The dimension of the input data, "\
    +str(D_t)+", should be equal to the sum of the Euclidean dimension of the "\
    "directional components ("+str(sum(dim)+sum(com_type=='Dir'))+")."
    
    assert len(h) == len(dim), "The lengths of bandwidth argument 'h' and dimension"\
    " argument 'dim' should be the same."
    
    Eu_dim = [0]
    H = []   ## Bandwidth matrix
    # Rule of thumb applied to each component
    for k in range(len(h)):
        if com_type[k] == 'Dir':
            if k == 0:
                data_comp = data[:,:(dim[k]+1)]
                Eu_dim.append(dim[k]+1)
            else:
                data_comp = data[:,sum(Eu_dim):(sum(Eu_dim)+dim[k]+1)]
                Eu_dim.append(dim[k]+1)
            if h[k] is None:
                R_bar = np.sqrt(sum(np.mean(data_comp, axis=0) ** 2))
                ## An approximation to kappa (Banerjee 2005 & Sra, 2011)
                kap_hat = R_bar * (dim[k] + 1 - R_bar ** 2) / (1 - R_bar ** 2)
                if dim[k] == 2:
                    h[k] = (8*np.sinh(kap_hat)**2/(n*kap_hat * \
                     ((1+4*kap_hat**2)*np.sinh(2*kap_hat) - 2*kap_hat*np.cosh(2*kap_hat))))**(1/6)
                else:
                    h[k] = ((4 * np.sqrt(np.pi) * sp.iv((dim[k]-1) / 2 , kap_hat)**2) / \
                     (n * kap_hat ** ((dim[k]+1) / 2) * (2 * dim[k] * sp.iv((dim[k]+1)/2, 2*kap_hat) + \
                        (dim[k]+2) * kap_hat * sp.iv((dim[k]+3)/2, 2*kap_hat)))) ** (1/(dim[k] + 4))
                print("The current bandwidth for the "+str(k)+"-th directional component is "\
                    + str(h[k]) + ".\n")
            H.append((h[k]**2)*np.eye(dim[k]+1))
        elif com_type[k] == 'Lin':
            if k == 0:
                data_comp = data[:,:dim[k]]
                Eu_dim.append(dim[k])
            else:
                data_comp = data[:,sum(Eu_dim):(sum(Eu_dim)+dim[k])]
                Eu_dim.append(dim[k])
            if h[k] is None:
                # Apply Silverman's rule of thumb to select the bandwidth parameter 
                # (Only works for Gaussian kernel)
                h[k] = (4/(dim[k]+2))**(1/(dim[k]+4))*(n**(-1/(dim[k]+4)))\
                    *np.mean(np.std(data_comp, axis=0))
                print("The current bandwidth for the "+str(k)+"-th linear component is "\
                      + str(h[k]) + ".\n")
            H.append((h[k]**2)*np.eye(dim[k]))
    # Convert a list of diagonal matrices to the final block diagonal bandwidth matrix
    H = block_diag(*H)
    
    SCMS_path = np.zeros((mesh_0.shape[0], D_t, max_iter))
    SCMS_path[:,:,0] = mesh_0
    # Create a vector indicating the convergent status of every mesh point
    conv_sign = np.zeros((mesh_0.shape[0], ))
    for t in range(1, max_iter):
        if all(conv_sign > 0):
            print('The naive SCMS algorithm in a directional/linear product space'\
                  ' converges in ' + str(t-1) + ' steps!')
            break
        for i in range(mesh_0.shape[0]):
            if conv_sign[i] == 0:
                x_pts = SCMS_path[i,:,t-1]
                for k in range(len(dim)):
                    data_comp = data[:,sum(Eu_dim[:(k+1)]):sum(Eu_dim[:(k+2)])]
                    x_comp = SCMS_path[i,sum(Eu_dim[:(k+1)]):sum(Eu_dim[:(k+2)]),t-1]
                    # Kernel weights
                    ker_w_comp = np.exp(-np.sum(((x_comp - data_comp)/h[k])**2, axis=1)/2)
                    if k == 0:
                        ker_w_mul = ker_w_comp.reshape(n,1)
                    else:
                        ker_w_mul = ker_w_mul * ker_w_comp.reshape(n,1)
                den_prop = np.sum(ker_w_mul)
                if den_prop == 0:
                    # Set those points with zero density values to NaN
                    nan_arr = np.zeros_like(x_pts)
                    nan_arr[:] = np.nan
                    conv_sign[i] = 1
                    x_new = nan_arr
                else:
                    # Compute the total gradient
                    tot_grad_Log = np.sum(np.dot(data - x_pts, LA.inv(H)) \
                                          * ker_w_mul, axis=0) / den_prop
                    tot_grad_Log_trans = np.sum((data - x_pts)*ker_w_mul, axis=0) / den_prop
                    # Compute the projection matrix and radial gradient block matrix
                    proj_mat = []
                    rad_grad = []
                    # Eigenvector of the Hessian in the normal direction
                    x_eig = np.zeros((D_t, sum(com_type == 'Dir')))
                    cnt_Dir = 0
                    ms_v = []   ## (Modified) mean shift vector
                    for k in range(len(dim)):
                        data_comp = data[:,sum(Eu_dim[:(k+1)]):sum(Eu_dim[:(k+2)])]
                        x_comp = SCMS_path[i,sum(Eu_dim[:(k+1)]):sum(Eu_dim[:(k+2)]),t-1]
                        if com_type[k] == 'Dir':
                            grad_comp = tot_grad_Log[sum(Eu_dim[:(k+1)]):sum(Eu_dim[:(k+2)])]
                            proj_mat.append(np.eye(dim[k]+1) - np.dot(x_comp.reshape(-1,1), 
                                                                      x_comp.reshape(1,-1)))
                            rad_grad.append(np.eye(dim[k]+1)*np.dot(x_comp, grad_comp))
                            x_eig_can = np.zeros((D_t,))
                            x_eig_can[sum(Eu_dim[:(k+1)]):sum(Eu_dim[:(k+2)])] = x_comp
                            x_eig[:,cnt_Dir] = x_eig_can
                            # Mean shift vector in the directional component
                            ms_Dir = np.sum(data_comp * ker_w_mul, axis=0) / np.sum(ker_w_mul)
                            ms_v.extend(ms_Dir)
                            cnt_Dir += 1
                        elif com_type[k] == 'Lin':
                            proj_mat.append(np.eye(dim[k]))
                            rad_grad.append(0*np.eye(dim[k]))
                            # Mean shift vector in the linear component
                            ms_Lin = np.sum(data_comp * ker_w_mul, axis=0)/np.sum(ker_w_mul) - x_comp
                            ms_v.extend(ms_Lin)
                    proj_mat = block_diag(*proj_mat)
                    rad_grad = block_diag(*rad_grad)
                    # Compute the (Riemannian) Hessian matrix
                    tot_Hess = np.dot(np.dot(x_pts - data, LA.inv(H)).T, 
                                      np.dot((x_pts - data) * ker_w_mul, LA.inv(H)))/den_prop \
                               - LA.inv(H) * np.sum(ker_w_mul) / den_prop \
                               - np.dot(tot_grad_Log.reshape(D_t,1), tot_grad_Log.reshape(1,D_t))
                    Log_Hess = np.dot(np.dot(proj_mat, tot_Hess - rad_grad), proj_mat)
                    if (np.sum(np.isinf(Log_Hess)) > 0) or (np.sum(np.isnan(Log_Hess)) > 0):
                        # Set those points with zero density values to NaN
                        nan_arr = np.zeros_like(x_pts)
                        nan_arr[:] = np.nan
                        conv_sign[i] = 1
                        x_new = nan_arr
                    else:
                        # Spectral decomposition
                        w, v = LA.eig(Log_Hess)
                        # Obtain the eigenpairs within the tangent space
                        if sum(com_type == 'Dir') > 0:
                            tang_eig_v = v[:, (abs(np.sum(np.dot(x_eig.T, v), axis=0)) < 1e-8)]
                            tang_eig_w = w[(abs(np.sum(np.dot(x_eig.T, v), axis=0)) < 1e-8)]
                        else:
                            tang_eig_v = v
                            tang_eig_w = w
                        V_d = tang_eig_v[:, np.argsort(tang_eig_w)[:(sum(dim)-d)]]
                        # Subspace constrained gradient and mean shift vector
                        SCMS_grad_trans = np.dot(V_d, np.dot(V_d.T, tot_grad_Log_trans))
                        SCMS_v = np.dot(V_d, np.dot(V_d.T, ms_v))
                        # SCMS update
                        x_new = SCMS_v + x_pts
                        for k in range(len(dim)):
                            if com_type[k] == 'Dir':
                                x_norm = x_new[sum(Eu_dim[:(k+1)]):sum(Eu_dim[:(k+2)])]
                                x_new[sum(Eu_dim[:(k+1)]):sum(Eu_dim[:(k+2)])] = x_norm / LA.norm(x_norm)
                        if LA.norm(SCMS_grad_trans) < eps:
                            conv_sign[i] = 1
                SCMS_path[i,:,t] = x_new
            else:
                SCMS_path[i,:,t] = SCMS_path[i,:,t-1]
        # print(t)
    # print(conv_sign)
    if t >= max_iter-1:
        print('The naive SCMS algorithm in a directional/linear product space reaches '\
              'the maximum number of iterations,'+str(max_iter)+', and has not '\
              'yet converged.')
    # SCMS_path = SCMS_path[conv_sign != 0,:,:t]
    nan_cri = np.isnan(SCMS_path[:,0,t-1])
    SCMS_path = SCMS_path[~nan_cri,:,:t]
    return SCMS_path[:,:,:t], conv_sign
