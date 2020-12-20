# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 22:50:23 2020

@author: jnb19
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import copy
from mpl_toolkits.mplot3d import Axes3D
import broom_cupboard3 as bc
from importlib import reload

project_directory = 'C:\\Users\\jnb19\\Documents\\uni\\year4\\comp_phys\\Project'
fp = os.path.join(project_directory, 'jnb17_neutrino_data.txt')

theta23_0 = np.pi/4 # rad
dmsq23_0 = 0.0024 # ?
xi_0 = 1.35
L = 295 # km
# U_chi = '$\chi$ [emu mol$^{-1}$Oe$^{-1}$]'
U_E = r'$E$ [GeV]'
U_observed_muon_events = r'Observed Number of $\nu_{\mu}$ Events'
U_P_noosc_muon = r'$P(\nu_{\mu} \rightarrow \nu_{\mu})$'
U_theta23 = r'$\theta_{23}$'
U_deltamsq = r'$\Delta m^2_{{23}}$'
U_xi = r'$\xi$'
empty_handle = mpl.lines.Line2D([0], [0], linestyle = 'None', marker = 'None')
# plt.subplots_adjust(left=0.055, bottom=0.094, right=0.615, top=0.972, wspace=0.2, hspace=0.2)

bbox_props = dict(boxstyle="square,pad=0.2", fc="w", ec="k", lw=1)


print('REMEMBER: Theta and Delta msq are SCALED withing the P_noosc_muon function')

# n, bins, patches = plt.hist(x = event_data, bins=num_bins, range=(0, 10), density=False, weights=None, cumulative=False, bottom=None, histtype='bar', align='mid', orientation='vertical', rwidth=None, log=False, color=None, label=None, stacked=False)

x = bc.Dataset(fp)
plot_data = True
if plot_data:
    axx = x.flux_histogram(gridlines = True)
    # axy = x.event_histogram(gridlines = True)
    # axf = x.flux_event_histogram_line()
    plt.subplots_adjust(left=0.055, bottom=0.094, right=0.615, top=0.972, wspace=0.2, hspace=0.2)
scales  = {'thetas':theta23_0, 'deltamsqs':dmsq23_0, 'xis':xi_0}
colours = {'thetas':'firebrick', 'deltamsqs':'royalblue', 'xis':'limegreen'}
names = {'thetas':U_theta23, 'deltamsqs':U_deltamsq, 'xis':U_xi}


#%% #---------Plot oscilllation probability---------
plot_P_noosc_dependence = True

if plot_P_noosc_dependence:
    # Note: if theta23 = n * pi/2 where n is integer (unnormalised), P = 1 for all energy 
    Es = np.linspace(0., 10, 5000)
    fig, ax = plt.subplots()
    bc.standard_axes_settings(ax, bc.infile_figparams)
    
    num_params = 8
    
    #---Plot theta-P
    theta23 =  np.linspace(0, 1, num_params) # theta is scaled!!!!
    cs = [cm.inferno(i/num_params, 1) for i in range(num_params)][::-1]
    
    for i in range(num_params):
        to_plot = bc.P_noosc_muon(Es, L, theta23[i], 1)
        ax.plot(Es, to_plot, ls = '-', marker = 'None', color = cs[i], lw = 2, label = fr'$\theta_{{23}}$ = {theta23[i]*np.pi/4:.3f}')
    
    handles = [mpl.lines.Line2D([0], [0], ls = '-', marker = 'None', color = cs[0], lw = 2), mpl.lines.Line2D([0], [0], ls = '-', marker = 'None', color = cs[-1], lw = 2), empty_handle]
    labels = [r'$\theta_{23}$ = 0', r'$\theta_{23}$ = $\frac{\pi}{4}$', fr'$\Delta m^2_{{23}}$ = {dmsq23_0}']
    
    leg = ax.legend(handles, labels, fontsize = 20, loc='best', markerfirst = True, frameon = True)
    leg.get_frame().set_edgecolor('k')
    leg.get_frame().set_facecolor('w')
    ax.set_xlabel(U_E)
    ax.set_ylabel(U_P_noosc_muon)
    
    #---Plot delta_m-P
    deltamsq23 =np.linspace(0, 10, num_params) 
    fig2, ax2 = plt.subplots()
    bc.standard_axes_settings(ax2, bc.infile_figparams)
    
    for i in range(num_params):
        to_plot = bc.P_noosc_muon(Es, L, 1, deltamsq23[i])
        ax2.plot(Es, to_plot, ls = '-', marker = 'None', color = cs[i], lw = 2, label = fr'$\Delta m^2_{{23}}$ = {deltamsq23[i]*dmsq23_0:.3f}')
    
    handles = [mpl.lines.Line2D([0], [0], ls = '-', marker = 'None', color = cs[0], lw = 2), mpl.lines.Line2D([0], [0], ls = '-', marker = 'None', color = cs[-1], lw = 2), empty_handle]
    labels = [r'$\Delta m^2_{23}$ = 0', fr'$\Delta m^2_{{23}}$ ={deltamsq23[-1]*dmsq23_0}', r'$\theta_{23}$ = $\frac{\pi}{4}$']
    
    leg2 = ax2.legend(handles, labels, fontsize = 20, loc='best', markerfirst = True, frameon = True)
    leg2.get_frame().set_edgecolor('k')
    leg2.get_frame().set_facecolor('w')
    ax2.set_xlabel(U_E)
    ax2.set_ylabel(U_P_noosc_muon)
    plt.subplots_adjust(left=0.055, bottom=0.094, right=0.615, top=0.972, wspace=0.2, hspace=0.2)
#%% -------Apply probability to predicted flux
#flux, phi = simulated event number assuming no interactions i.e. number of muon neutrinos that arrive
# P = prob. that the muon neutrinos do NOT transform before arriving
# (1-P(Ei, deltam2, theta, L)) * phi_i = lamda_i (deltam2, theta) ; oscillated event rate prediction
# compare data and simulated oscillated flux

x = bc.Dataset(fp)
inputs = {'thetas':.83903/theta23_0, 'deltamsqs':.0026/dmsq23_0, 'xis': 1.39847/xi_0}
txt = ''
display_histograms = True

if display_histograms:
    # a = x.compare_sim_real_histogram(**inputs, stacked = True, gridlines = True, bin_energies = None)
    b = x.compare_sim_real(**inputs, stacked = True, gridlines = True, bin_energies = None)
    plt.subplots_adjust(left=0.055, bottom=0.094, right=0.615, top=0.972, wspace=0.2, hspace=0.2)
    for key in inputs:
        txt  = txt + fr'{names[key]} = {inputs[key]*scales[key]:.4f}'
        if key !='xis':
            txt = txt + '\n'
    b.text(8.2, 16, txt, fontsize = 18, bbox = bbox_props)

#%% --------Approx likelihood minimum with varying theta or deltamsq
plot_initial_approx = True
plot = 'xi'

if plot_initial_approx:
    fig, ax = plt.subplots()
    bc.standard_axes_settings(ax, bc.infile_figparams)

    
    N = 1000
    if plot == 'deltamsq': # NLL(deltam)
        xs = np.linspace(0, 2, N) #
        NLL = x.NLL_params([1]*N, xs, [1]*N) 
        scale = dmsq23_0
        ax.set_xlabel(U_deltamsq)
        txt = fr'{names["thetas"]} = {1*scales["thetas"]:.4f}'+'\n'+fr'{names["xis"]} = {1*scales["xis"]:.4f}'
        
    elif plot == 'theta':
        xs = np.linspace(0, 2, N) # NLL(theta)
        NLL = x.NLL_params(xs, [1]*N, [1]*N)
        scale = theta23_0
        ax.set_xlabel(U_theta23)
        txt = fr'{names["deltamsqs"]} = {1*scales["deltamsqs"]:.4f}'+'\n'+fr'{names["xis"]} = {1*scales["xis"]:.4f}'
        
    elif plot == 'xi':
        xs = np.linspace(0.001, 8, N) # NLL(xi)
        NLL = x.NLL_params([1]*N,[1]*N, xs )
        scale = xi_0
        ax.set_xlabel(U_xi)
        txt = fr'{names["thetas"]} = {1*scales["thetas"]:.4f}'+'\n'+fr'{names["deltamsqs"]} = {1*scales["deltamsqs"]:.4f}'
        
    else:
        pass
    
    ax.plot(xs*scale, NLL, marker= 'None', ls = ':', color = 'k', lw = 2)
    ax.set_ylabel(r'NLL')
    plt.subplots_adjust(left=0.055, bottom=0.094, right=0.74, top=0.972, wspace=0.2, hspace=0.2)
    ax.text(1.0, 550, txt, fontsize = 18, bbox = bbox_props)

#%%-------3.4 parabolic minimisation NLL--------
x = bc.Dataset(fp)
X = np.array([0.5, 1.6, .8]) # pick 3 initial points
dmsq_val = 1.
xi_val = 1./xi_0
theta_val = 1.
vary = 'deltamsq'

fig, ax = plt.subplots()
bc.standard_axes_settings(ax, bc.infile_figparams)

if vary == 'theta': 
    params = np.array([X, np.full(3, dmsq_val), np.full(3, xi_val)]) # (theta, deltamsq, xi)
    
    Y = x.NLL_params(*params)
    
    mintheta, eps, c, points = x.parabolic_minimisation_NLL_theta(X, deltamsq = dmsq_val, xi = xi_val, conv_condition = 1e-10, max_iter = 1e3)
    plot_num = 1000
    xs = np.linspace(0, 2, plot_num)
    ys = x.NLL_params(xs, np.full(plot_num, dmsq_val), np.full(plot_num, xi_val)) 
    
    ax.plot(X*theta23_0, Y, marker = 'x', ls = 'None', mec = 'r', ms = 13, mew = 3, label = r'Initial $\theta_{23}$ Choice')
    ax.plot(xs*theta23_0, ys, marker= 'None', ls = ':', color = 'k', lw = 2, label = 'NLL values')
    ax.plot(mintheta*theta23_0, x.NLL_params(thetas = [mintheta], deltamsqs = [dmsq_val], xis = [xi_val]), ls = 'None', marker = 'd', mec = 'g', mfc = 'None', ms = 14, mew = 3, label = r'$\theta^{min}_{23}$')
    ax.set_xlabel(U_theta23)
    ax.set_ylabel(fr'NLL($\theta_{{23}}$, $\Delta m^2_{{23}}$ = {params[1, 0]*dmsq23_0:.2e})')
    # need to check curvature!!
    
    NLL_min = x.NLL_params(thetas = mintheta, deltamsqs = dmsq_val, xis = xi_val)
    leg = ax.legend(fontsize = 20, loc='best', markerfirst = True, frameon = True)
    leg.get_frame().set_edgecolor('k')
    leg.get_frame().set_facecolor('w')
    #------3.5 Fit Accuracy------
    thp, thm = x.thetamin_stddev_NLL(mintheta, deltamsq = dmsq_val, xi = xi_val)
    
    ax.plot(thp*theta23_0, x.NLL_params([thp], [dmsq_val], [xi_val]), marker = 'o', mec = 'royalblue', ms = 13, mfc = 'None', label = r'$\sigma_{\theta}^+$', mew = 3, linestyle = 'None')
    ax.plot(thm*theta23_0, x.NLL_params([thm], [dmsq_val], [xi_val]), marker = 'o', ms =13, mec = 'darkmagenta', mfc ='None', label = r'$\sigma_{\theta}^-$', mew = 3, linestyle = 'None')
    ax.axhline(NLL_min+0.5, 0, np.pi/2)
    
    thp *= theta23_0
    thm *= theta23_0
    mintheta *= theta23_0
    stdp = thp - mintheta
    stdm = mintheta - thm
    
if vary == 'xi': 
    params = np.array([np.full(3, theta_val), np.full(3, dmsq_val), X]) # (theta, deltamsq, xi)
    
    Y = x.NLL_params(*params)
    
    minxi, eps, c, points = x.parabolic_minimisation_NLL_xi(X, deltamsq = dmsq_val, theta = theta_val, conv_condition = 1e-10, max_iter = 1e3)
    plot_num = 1000
    xs = np.linspace(0.01, 8, plot_num)
    ys = x.NLL_params(np.full(plot_num, theta_val), np.full(plot_num, dmsq_val), xs) 
    
    ax.plot(X*xi_0, Y, marker = 'x', ls = 'None', mec = 'r', ms = 13, mew = 3, label = r'Initial $\xi$ Choice')
    ax.plot(xs*xi_0, ys, marker= 'None', ls = ':', color = 'k', lw = 2, label = 'NLL values')
    ax.plot(minxi*xi_0, x.NLL_params(thetas = [theta_val], deltamsqs = [dmsq_val], xis = [minxi]), ls = 'None', marker = 'd', mec = 'g', mfc = 'None', ms = 18, mew = 3, label = r'$\xi_{min}$')
    ax.set_xlabel(U_xi)
    ax.set_ylabel(fr'NLL($\theta_{{23}}$ = {params[0,0]*theta23_0:.2f}, $\Delta m^2_{{23}}$ = {params[1, 0]*dmsq23_0:.2e}, $\xi$)')
    leg = ax.legend(fontsize = 20, loc='best', markerfirst = True, frameon = True)
    leg.get_frame().set_edgecolor('k')
    leg.get_frame().set_facecolor('w')
    
    NLL_min = x.NLL_params(thetas = theta_val, deltamsqs = dmsq_val, xis = minxi)

if vary == 'deltamsq': 
    params = np.array([np.full(3, theta_val), X, np.full(3, xi_val)]) # (theta, deltamsq, xi)
    
    Y = x.NLL_params(*params)
    
    mindm, eps, c, points = x.parabolic_minimisation_NLL_deltamsq(X, xi = xi_val, theta = theta_val, conv_condition = 1e-10, max_iter = 1e3)
    plot_num = 1000
    xs = np.linspace(0.01, 4, plot_num)
    ys = x.NLL_params(np.full(plot_num, theta_val), xs, np.full(plot_num, xi_val)) 
    
    ax.plot(X*dmsq23_0, Y, marker = 'x', ls = 'None', mec = 'r', ms = 13, mew = 3, label = r'Initial $\Delta m_{23}^2$ Choice')
    ax.plot(xs*dmsq23_0, ys, marker= 'None', ls = ':', color = 'k', lw = 2, label = 'NLL values')
    ax.plot(mindm*dmsq23_0, x.NLL_params(thetas = [theta_val], deltamsqs = [mindm], xis = [xi_val]), ls = 'None', marker = 'd', mec = 'g', mfc = 'None', ms = 18, mew = 3, label = r'$\xi_{min}$')
    ax.set_xlabel(U_deltamsq)
    ax.set_ylabel(fr'NLL($\theta_{{23}}$ = {params[0,0]*theta23_0:.2f}, $\Delta m^2_{{23}}$, $\xi$ = {params[2,0]*xi_0:.2f})')
    leg = ax.legend(fontsize = 20, loc='best', markerfirst = True, frameon = True)
    leg.get_frame().set_edgecolor('k')
    leg.get_frame().set_facecolor('w')
    
    NLL_min = x.NLL_params(thetas = theta_val, deltamsqs = mindm, xis = xi_val)
print("NEED TO DO STD DEV FOR DELTAMSQ AND XI TOO")

#%% -------- 4.1 Univariate Parabolic --------
x = bc.Dataset(fp)

input_thetas = np.array([0.5, 1.0, 1.1])
input_deltams = np.array([0.5, 1, 2])
input_xis = np.array([0.2, 3, 6])
Xs = {'thetas':input_thetas, 'deltamsqs':input_thetas}#, 'xis':input_xis}
bin_energies = None
Ndim = len(Xs)

max_iter = 1e3
acc = 1e-7
indiv_conv_cond = 1e-5
indiv_maxiter = 1e2
other_values = {'thetas':.5, 'deltamsqs':.8, 'xis':1./xi_0} # initial values for theta, dmsq, xi
us = [list(other_values.values())]
epsilon = 2*acc
i = 0

eps = {}
points = {}
c = {} 

while i < max_iter and epsilon > acc :
    u_old = np.array(list(other_values.values()))
    for key, value in Xs.items():
        NLL_args = {k: other_values[k] for k in set(list(other_values.keys())) - set([key])}
        NLL_args['bin_energies'] = bin_energies
        other_values[key], eps[key], c[key], points[key] = bc.parabolic_minimisation(funct = x.NLL_params, X = value, param_minimised = key, function_parameters = NLL_args, convergence_condition = indiv_conv_cond, max_iterations = max_iter)

    epsilon = np.linalg.norm(np.array(list(other_values.values())) - u_old)
    us.append(list(other_values.values()))
    i += 1

us = np.asarray(us).T

fig, ax = plt.subplots()
bc.standard_axes_settings(ax, bc.infile_figparams)
bbox_props = dict(boxstyle="square,pad=0.2", fc="w", ec="k", lw=1)

for k, key in enumerate(other_values):    
    other_values[key] *=  scales[key]
    ax.plot(us[k], marker= 'o', ls = ':', color = colours[key], lw = 2, ms = 8, mec = colours[key], label = names[key])
    ax.text(i-.4, us[k,-1]-.005, fr'{names[key]} = {us[k,-1]*scales[key]:.5f}', fontsize = 18, bbox = bbox_props)
            
ax.set_xlabel('Iteration')
ax.set_ylabel('Scaled Parameters')

leg = ax.legend(fontsize = 20, loc='best', markerfirst = True, frameon = True)
leg.get_frame().set_edgecolor('k')
leg.get_frame().set_facecolor('w')


# us[0] *= theta23_0
# us[1] *= dmsq23_0
# us[2] *= xi_0
#%%-------3D plot NLL-------
make3Dplot = False
notplot = 'xis'

if make3Dplot:
    N = 50
    ranges = {'thetas':np.linspace(0, 2, N), 'deltamsqs':np.linspace(0, 5, N), 'xis':np.linspace(0.01, 2, N)}
    const_values = {'thetas':1., 'deltamsqs':1., 'xis':1.} # initial values for theta, dmsq, xi
    plot_keys = list(set(list(const_values.keys())) - set([notplot]))
    
    points = {k: ranges[k] for k in set(list(const_values.keys())) - set([notplot])}
    
    
    X, Y  = np.meshgrid(points[plot_keys[0]], points[plot_keys[1]])
    plot_values = {notplot: const_values[notplot]}
    Z = np.zeros_like(X)
    
    for i in range(N):
        for j in range(N):
            plot_values[plot_keys[0]]  = X[i,j]
            plot_values[plot_keys[1]]  = Y[i,j]
            Z[i, j] = x.NLL_params(**plot_values)
    fig = plt.figure()
    
    ax = Axes3D(fig)
    ax.plot_surface(X*scales[plot_keys[0]], Y*scales[plot_keys[1]], Z, cmap = cm.coolwarm, linewidth=0, antialiased=False)
    # ax.scatter(us[0], us[1], x.NLL_params(us[0]/theta23_0, us[1]/dmsq23_0), c = 'k')
    
    ax.set_xlabel(names[plot_keys[0]])
    ax.set_ylabel(names[plot_keys[1]])
    ax.set_zlabel('NLL')
    
#%%-------4.2 Multivariate NLL : Gradient Method -------

alpha = 1e-5
params = np.array([1., 1., 1.]) # (theta, deltamsq)
deltas = 0.001
conv = 1e-6
eps = 2*conv
c = 0
us = [copy.copy(params)]

while c < 1e3 and eps > conv:
    us_old = us[-1]
    grad = x.grad_NLL(params, deltas)
    params -= alpha*grad
    c += 1
    # print(grad, params, '\n')
    eps = np.linalg.norm(params - us_old)/np.linalg.norm(us_old)
    us.append(copy.copy(params))
    
    
us = np.asarray(us).T

fig, ax = plt.subplots()
bc.standard_axes_settings(ax, bc.infile_figparams)
bbox_props = dict(boxstyle="square,pad=0.2", fc="w", ec="k", lw=1)

for k, key in enumerate(names):    
    ax.plot(us[k], marker= 'o', ls = ':', color = colours[key], lw = 2, ms = 8, mec = colours[key], label = names[key], mfc = 'None', markevery = int(c/15))
    ax.text(c*0.7, us[k,-1]-.005, fr'{names[key]} = {us[k,-1]*scales[key]:.5f}', fontsize = 18, bbox = bbox_props)
            
ax.set_xlabel('Iteration')
ax.set_ylabel('Scaled Parameters')

leg = ax.legend(fontsize = 20, loc='best', markerfirst = True, frameon = True)
leg.get_frame().set_edgecolor('k')
leg.get_frame().set_facecolor('w')


# us[0] *= theta23_0
# us[1] *= dmsq23_0
# us[2] *= xi_0

#%%--------4.2 Quasi Newton Method------

X = bc.Dataset(fp)

# inputs
alpha = 1e-4
input_params = np.array([1., 1., 1.]) # input parameters x0 = [theta0, deltamsq0, xi_0]
its = 1e6
conv = 1e-6
deltas = 1e-3#np.array([0.001, 0.001]) # size of step to do in gradient
grad_funct = X.grad_NLL
funct = X.NLL_params
NLL_funct_params = {'bin_energies': None}


us, eps, c, deltas = bc.quasiNewton(input_params, grad_funct, funct, funct_params = NLL_funct_params, alpha = alpha, deltas = deltas, max_iter = its, convergence = conv)

us = np.delete(us, np.s_[c+2:], 1)
input_params[0] *= theta23_0
input_params[1] *= dmsq23_0
input_params[2] *= xi_0

fig, ax = plt.subplots()
bc.standard_axes_settings(ax, bc.infile_figparams)
bbox_props = dict(boxstyle="square,pad=0.2", fc="w", ec="k", lw=1)

for k, key in enumerate(names):    
    ax.plot(us[k], marker= 'o', ls = ':', color = colours[key], lw = 2, ms = 8, mec = colours[key], label = names[key], mfc = 'None', markevery = int(c/15))
    ax.text(c*0.7, us[k,-1]-.005, fr'{names[key]} = {us[k,-1]*scales[key]:.5f}', fontsize = 18, bbox = bbox_props)
            
ax.set_xlabel('Iteration')
ax.set_ylabel('Scaled Parameters')

leg = ax.legend(fontsize = 20, loc='best', markerfirst = True, frameon = True)
leg.get_frame().set_edgecolor('k')
leg.get_frame().set_facecolor('w')


makecontourplot = True
notplot = 'deltamsqs'

if makecontourplot:
    N = 50
    # ranges = {'thetas':np.linspace(min(us[0])*0.9, max(us[0])*1.1, N), 'deltamsqs':np.linspace(min(us[1])*0.9, max(us[1])*1.1, N), 'xis':np.linspace(min(us[2])*0.9, max(us[2])*1.1, N)}
    ranges = {'thetas':np.linspace(0.5, 1.5, N), 'deltamsqs':np.linspace(0.5, 1.5, N), 'xis':np.linspace(0.5, 1.5, N)}
    const_values = {'thetas':us[0,-1], 'deltamsqs':us[1,-1], 'xis':us[2,-1]} # initial values for theta, dmsq, xi
    inputs = {'thetas':input_params[0], 'deltamsqs':input_params[1], 'xis':input_params[2]} # initial values for theta, dmsq, xi
    
    us[0] *= theta23_0
    us[1] *= dmsq23_0
    us[2] *= xi_0
    
    Us = {'thetas':us[0], 'deltamsqs':us[1], 'xis':us[2]} # iterations for theta, dmsq, xi
    plot_keys = list(set(list(const_values.keys())) - set([notplot]))
    
    points = {k: ranges[k] for k in set(list(const_values.keys())) - set([notplot])}
    
    X, Y  = np.meshgrid(points[plot_keys[0]], points[plot_keys[1]])
    plot_values = {notplot: const_values[notplot]}
    Z = np.zeros_like(X)
    
    makecontours = True
    if makecontours:
        for i in range(N):
            for j in range(N):
                plot_values[plot_keys[0]]  = X[i,j]
                plot_values[plot_keys[1]]  = Y[i,j]
                Z[i, j] = x.NLL_params(**plot_values)
                
    X *= scales[plot_keys[0]]
    Y *= scales[plot_keys[1]]
    fig1, ax1 = plt.subplots()
    
    
    ax1.plot(Us[plot_keys[0]], Us[plot_keys[1]], ls = 'None', marker = '.', mec = 'r', mfc = 'r', ms = 5, mew = 2, label = r'$x_{min}$')
    ax1.plot(Us[plot_keys[0]][-1], Us[plot_keys[1]][-1], ls = 'None', marker = 'o', mec = 'k', mfc = 'None', ms = 8, mew = 2, label = r'$x_{min}$')
    
    cs = ax1.contour(X, Y, Z, cmap = cm.coolwarm)
    
    ax1.plot(inputs.get(plot_keys[0]), inputs.get(plot_keys[1]), 'gd', mfc = 'None', ms = 15, mew = 2)
    ax1.set_xlabel(names[plot_keys[0]])
    ax1.set_ylabel(names[plot_keys[1]])
    
    labels = ['Iterations', 'Input Parameters', fr'{names[plot_keys[0]]} = {Us[plot_keys[0]][-1]:.4f}, {names[plot_keys[1]]} = {Us[plot_keys[1]][-1]:.4f}', fr'{names[notplot]} = {Us[notplot][-1]:.4f}']
    
    handles = [mpl.lines.Line2D([0], [0], linestyle = 'None', marker = '.', mec = 'r', mfc = 'r', ms = 10, mew =2), mpl.lines.Line2D([0], [0], linestyle = 'None', marker = 'd',mec = 'g', mfc = 'None', ms = 10, mew = 3), mpl.lines.Line2D([0], [0], linestyle = 'None', marker = 'o',mec = 'k', mfc = 'None', ms = 8, mew = 2), bc.empty_handle]# mpl.lines.Line2D([0], [0], marker = 'x', mec = 'k', mfc = 'None', ms = 10)]
    
    ax1.text(1.1, .00262, fr'$\alpha$ = {alpha}, $\delta$ = {deltas[0]}', fontsize = 18)
    leg = ax1.legend(handles, labels, fontsize = 20, loc='best', markerfirst = True, frameon = True)
    leg.get_frame().set_edgecolor('k')
    leg.get_frame().set_facecolor('w')
    bc.standard_axes_settings(ax1, bc.infile_figparams)


    
    
    
    
    
