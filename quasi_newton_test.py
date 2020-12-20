
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import copy
from mpl_toolkits.mplot3d import Axes3D
import broom_cupboard as bc
from importlib import reload 
import math
reload(bc)

def dodgy_cone(x, y, a):
    q = (x**4)/3 +x**2 + (y**4)/2 + a*y**2 - x
    return q

def dodgy_cone_grad(var_pars, deltas, a):
    p = var_pars[0]
    q = var_pars[1]
    
    dp = (dodgy_cone(p+deltas[0], q, a)-dodgy_cone(p, q, a))/deltas[0]
    dq = (dodgy_cone(p, q+deltas[1], a)-dodgy_cone(p, q, a))/deltas[1]
    return np.array([dp, dq])



def quasiNewton(input_params, grad_funct, funct, funct_params, alpha = 1e-4, deltas = None, max_iter = 1e5, convergence = 1e-5):    
    if not isinstance(input_params, np.ndarray):
        raise TypeError("input_params must be an array containing variable input parameters of grad_funct")
    
    if not input_params.dtype in [np.float32, np.float64, float]:
        input_params = input_params.astype(np.float64)
        # raise TypeError("input parameter values must be floats, not integers. Cannot cast different types later in function")
    
    if not isinstance(funct_params, dict):
        raise TypeError("funct_params must be a dictionary containing constant parameters of grad_funct")
    
    if not callable(grad_funct):
        raise TypeError("function must a function or bound method (i.e. callable)")
        
    x = input_params
    Ndim = len(x) # number of parameters to minimise with
    
    if deltas is None:
        deltas = np.full(Ndim, 1e-3)
    elif isinstance(deltas, (int, np.float64, np.int32, float)):
        deltas = np.full(Ndim, deltas)
    elif isinstance(deltas, np.ndarray) and len(deltas) == Ndim:
        pass
    elif type(deltas) == str: # for analytic functions
        pass
    else:
        raise TypeError("deltas must be either float/int, Ndim array, 'analytic' or None")
    
    
    us = np.zeros((Ndim, int(max_iter + 2))) # record of parameter iteration path
    us[:, 0] = input_params
    G = np.identity(Ndim) # G0
    
    if type(deltas) == str:
        grad = grad_funct(x, **funct_params) # if grad_funct is analytically calculated, no need for deltas
    else:
        grad = grad_funct(x, deltas, **funct_params)
    xold = copy.deepcopy(x) # xn-1
    
    x -= alpha*np.matmul(G, grad) # x1 #CAREFUL OF THE ORDER!! NP.DOT WILL MULTIPLY (2, 1)X(2, 2) MATRICES AND JUST DO IT WEIRDLY
    delta = x - xold # difference between current and previous parameters
    
    us[:, 1] = x
    
    epsilon = 2*convergence # start epsilon > convergence
    c = 0 # counter
    
    while c < max_iter and epsilon > convergence:
        if type(deltas) == str:
            gamma = grad_funct(x, **funct_params) - grad_funct(xold, **funct_params)
        else:
            gamma = grad_funct(x, deltas, **funct_params) - grad_funct(xold, deltas, **funct_params)
        Ggamma = np.matmul(G, gamma)
        G += (delta.reshape((Ndim,1))*delta)/(np.dot(gamma, delta)) - G@(delta.reshape((Ndim,1))*delta)@G/(np.dot(gamma, Ggamma))
        
        # Gdelta = np.matmul(np.outer(delta, delta), G)
        # G += np.outer(delta, delta)/np.dot(gamma, delta) - np.matmul(G, Gdelta)/np.matmul(gamma, Ggamma)
        if type(deltas) == str:
            grad = grad_funct(x, **funct_params) # if grad_funct is analytically calculated, no need for deltas
        else:
            grad = grad_funct(x, deltas, **funct_params)
        xold = copy.deepcopy(x) # x n becomes x n-1
        x -= alpha*np.matmul(G, grad) # new x n+1
        
        if funct(*x, **funct_params) > funct(*xold, **funct_params):
            x = xold
            break
        delta = x - xold # difference between current and previous parameters
        epsilon = np.linalg.norm(delta)/np.linalg.norm(xold)
        us[:, c+2] = x
        c += 1
    return us, epsilon, c
#%%

funct_params = {'a':0.5}
input_params = np.array([2, 3])
grad_funct =  dodgy_cone_grad
funct = dodgy_cone
deltas = np.full(2, 0.01)
max_iter = 1e5
convergence = 1e-6
alpha = 1e-4
us, eps, c = quasiNewton(input_params, grad_funct, funct, funct_params, alpha, deltas, max_iter, convergence)


us = np.delete(us, np.s_[c+2:], 1)

plot_num = 500
edge = 0.35
xx = np.linspace(0.1, .7, plot_num)
yy = np.linspace(-.2, .4, plot_num)

X, Y  = np.meshgrid(xx, yy)
Z = np.zeros_like(X)
M = len(X)
N = len(Y)
for i in range(M):
    for j in range(N):
        Z[i, j] = dodgy_cone(X[i, j], Y[i, j], funct_params['a'])


fig1, ax1 = plt.subplots()
ax1.plot(us[0], us[1], ls = 'None', marker = '.', mec = 'r', mfc = 'r', ms = 1.5, mew = 2, label = r'$x_{min}$')
cs = ax1.contour(X, Y, Z, cmap = cm.coolwarm)

ax1.plot(.2, .32, 'gd', mfc = 'None', ms = 14)
ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$y$')
labels = ['Iterations', 'Input Parameters']
handles = [mpl.lines.Line2D([0], [0], linestyle = 'None', marker = '.', mec = 'r', mfc = 'r', ms = 10, mew =2), mpl.lines.Line2D([0], [0], marker = 'd',mec = 'g', mfc = 'None', ms = 10)]# mpl.lines.Line2D([0], [0], marker = 'x', mec = 'k', mfc = 'None', ms = 10)]

ax1.text(0.31, -.1, fr'$\alpha$ = {alpha}, $\delta$ = {deltas[0]}, $z(x, y)$ = $x^4/3 + x^2 - x + y^4/2 + y^2/2$', fontsize = 18)
leg = ax1.legend(handles, labels, fontsize = 20, loc='best', markerfirst = True, frameon = True)
leg.get_frame().set_edgecolor('k')
leg.get_frame().set_facecolor('w')
bc.standard_axes_settings(ax1, bc.infile_figparams)
