# -*- coding: utf-8 -*-
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import copy
from mpl_toolkits.mplot3d import Axes3D
import broom_cupboard as bc
from importlib import reload

project_directory = 'C:\\Users\\jnb19\\Documents\\uni\\year4\\comp_phys\\Project'
fp = os.path.join(project_directory, 'jnb17_neutrino_data.txt')

theta23_0 = np.pi/4 # rad
dmsq23_0 = 0.0024 # ?
L = 295 # km
# U_chi = '$\chi$ [emu mol$^{-1}$Oe$^{-1}$]'
U_E = r'$E$ [GeV]'
U_observed_muon_events = r'Observed Number of $\nu_{\mu}$ Events'
U_P_noosc_muon = r'$P(\nu_{\mu} \rightarrow \nu_{\mu})$'
U_theta23 = r'$\theta_{23}$'
U_deltamsq = r'$\Delta m^2_{{23}}$'

empty_handle = mpl.lines.Line2D([0], [0], linestyle = 'None', marker = 'None')

print('REMEMBER: Theta and Delta msq are SCALED withing the P_noosc_muon function')

# n, bins, patches = plt.hist(x = event_data, bins=num_bins, range=(0, 10), density=False, weights=None, cumulative=False, bottom=None, histtype='bar', align='mid', orientation='vertical', rwidth=None, log=False, color=None, label=None, stacked=False)

x = bc.Dataset(fp)
plot_data = False
if plot_data:
    axx = x.flux_histogram(gridlines = False)
    axf = x.flux_event_histogram()
#
#%% #---------Plot oscilllation probability---------

# Note: if theta23 = n * pi/2 where n is integer, P = 1 for all energy 
Es = np.linspace(0., 10, 1000)
fig, ax = plt.subplots()
bc.standard_axes_settings(ax, bc.infile_figparams)
# params_vary_theta = np.array([[theta23_0, initial_delta_m23]])
# params_vary_theta = params_vary_theta[np.argsort(params_vary_theta[:, 0]), :]

num_params = 10

theta23 =  np.linspace(0, 2, num_params) # theta is scaled!!!!
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

deltamsq23 =np.linspace(0, 2, num_params) 
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


#%% -------apply probability to predicted flux
#flux, phi = simulated event number assuming no interaction s i.e. number of muon neutrinos that arrive
# P = prob. que les neutrinos muon ne se transforment PAS avant d'arriver
# (1-P(Ei, deltam2, theta, L)) * phi_i = lamda_i (deltam2, theta) ; oscillated event rate prediction

# compare data and simulated oscillated flux

x = bc.Dataset(fp)
a = x.compare_sim_real_histogram(1, 1, stacked = True)
b = x.compare_sim_real(1, 1, stacked = True)

# ------------ Compare simulation oscillated and real data
# p = x.event_data
# ll = x.lamda_u(theta23_0, dmsq23_0)
# a = x.compare_sim_real(theta23_0, dmsq23_0)
#%% --------Approx likelihood minimum with varying theta or deltamsq
N = 1000
plot_deltam = False
if plot_deltam: # NLL(deltam)
    xs = np.linspace(0.1, 2, N) #
    NLL = x.NLL_params([1]*N, xs) 
    scale = dmsq23_0
else:
    xs = np.linspace(0, 2, N) # NLL(theta)
    NLL = x.NLL_params(xs, [1]*N)
    scale = theta23_0

fig, ax = plt.subplots()
bc.standard_axes_settings(ax, bc.infile_figparams)
ax.plot(xs*scale, NLL, marker= 'None', ls = ':', color = 'k', lw = 2)
if plot_deltam:
    ax.set_xlabel(U_deltamsq)
else: 
    ax.set_xlabel(U_theta23)
ax.set_ylabel(r'NLL')
# leg = ax.legend(fontsize = 20, loc='best', markerfirst = True, frameon = True)
# leg.get_frame().set_edgecolor('k')
# leg.get_frame().set_facecolor('w')

# N = params.shape[1]
# NLL = np.empty(N)
# for i in range(N):
#     lamdas = x.lamda_u(params[0,i], params[1,i], mean_bin_energies = x.bin_means)
#     NLL[i] = NLL_poisson(x.event_data, lamdas)
        
#%%-------testing parabolic minimisation----------

convergence_condition = 1e-5
max_iterations = 1e5

E = 5 

X = np.array([0.001,.2,1.3]) # pick 3 initial points
X = X[np.argsort(X)] # x's must be in order for lagrange
params = np.array([np.full(3, E), np.full(3, L), X, np.full(3, dmsq23_0)])
Y = bc.P_noosc_muon(*params)

#array method
points = np.empty((2, 4))
points[0, 0:3] = X # 3 x points (theta), 0
points[1, 0:3] = Y # 3 y points (NLL), 0
x3_old = min(X)
c = 0 
epsilon = convergence_condition + 1
while epsilon > convergence_condition and c < max_iterations:
    x3 = bc.find_x3(points[0, 0:3], points[1, 0:3]) # find minimum of parabola formed from X, Y points
    points[0,-1] = x3 # set last element of x points to 0
    points[1, -1] = bc.P_noosc_muon(E, L, x3, dmsq23_0)
    largest_f = np.argmax(points[1]) # find index of largest y point
    mask = np.ones(4, dtype = bool) # create True array of same shape as points
    mask[largest_f] = False # set the column containing largest y point to False in mask array
    points[:, 0:3] = points[:, mask] # reset columns 1-3 of points to the 3 smallest x-y points - i.e. delete the largest  y point
    points[:, 0:3] = points[:, 0:3][:, np.argsort(points[0, 0:3])] # reorder the 3 smallest x-y points in order of ascending x
    epsilon = np.abs(x3-x3_old)/x3_old
    x3_old = x3 # enter new iteration - x3 is now the old x3
    c += 1

plot_num = 100
xs = np.linspace(min(X), max(X), plot_num)
ys = bc.P_noosc_muon(E, L, xs, dmsq23_0)

fig, ax = plt.subplots()
bc.standard_axes_settings(ax, bc.infile_figparams)
ax.plot(X, Y, marker = 'x', ls = 'None', mec = 'r', ms = 13, mew = 3, label = r'Initial $\theta_{23}$ Choice')
ax.plot(xs, ys, marker= 'None', ls = ':', color = 'k', lw = 2, label = 'P values')
ax.plot(x3, bc.P_noosc_muon(E, L, x3, dmsq23_0), ls = 'None', marker = 'd', mec = 'g', mfc = 'None', ms = 18, mew = 3, label = r'$\theta^{min}_{23}$')
ax.set_xlabel(U_theta23)
ax.set_ylabel(r'P')
leg = ax.legend(fontsize = 20, loc='best', markerfirst = True, frameon = True)
leg.get_frame().set_edgecolor('k')
leg.get_frame().set_facecolor('w')




#%%-------3.4 parabolic minimisation NLL--------
x = bc.Dataset(fp)
X = np.array([0.6, 0.8, 0.9]) # pick 3 initial points
params = np.array([X, np.full(3, 1)]) # (theta, deltamsq)
Y = x.NLL_params(*params)

mintheta, eps, c, points = x.parabolic_minimisation_NLL_theta(X, deltamsq = 1, conv_condition = 1e-10, max_iter = 1e3)
plot_num = 1000
xs = np.linspace(0, 2, plot_num)
ys = x.NLL_params(xs, np.full(plot_num, 1)) 

fig, ax = plt.subplots()
bc.standard_axes_settings(ax, bc.infile_figparams)
ax.plot(X*theta23_0, Y, marker = 'x', ls = 'None', mec = 'r', ms = 13, mew = 3, label = r'Initial $\theta_{23}$ Choice')
ax.plot(xs*theta23_0, ys, marker= 'None', ls = ':', color = 'k', lw = 2, label = 'NLL values')
ax.plot(mintheta*theta23_0, x.NLL_params(thetas = [mintheta], deltamsqs = [1]), ls = 'None', marker = 'd', mec = 'g', mfc = 'None', ms = 18, mew = 3, label = r'$\theta^{min}_{23}$')
ax.set_xlabel(U_theta23)
ax.set_ylabel(fr'NLL($\theta_{{23}}$, $\Delta m^2_{{23}}$ = {params[1, 0]*dmsq23_0:.2e})')
leg = ax.legend(fontsize = 20, loc='best', markerfirst = True, frameon = True)
leg.get_frame().set_edgecolor('k')
leg.get_frame().set_facecolor('w')
# need to check curvature!!

NLL_min = x.NLL_params(thetas = mintheta, deltamsqs = 1)
#------3.5 Fit Accuracy------
thp, thm = x.thetamin_stddev_NLL(mintheta)

ax.plot(thp*theta23_0, x.NLL_params([thp], [1]), marker = 'o', ms = 15)
ax.plot(thm*theta23_0, x.NLL_params([thm], [1]), marker = 'o', ms =15)
ax.axhline(NLL_min+0.5, 0, np.pi/2)


#%% -------- 4.1 Univariate Parabolic --------
x = bc.Dataset(fp)

input_thetas = np.array([0.6, 1.5, 1.8])
input_deltams = np.array([0.7, 1.0, 1.8])
X = input_thetas
Y = input_deltams
max_iter = 2e2
acc = 1e-7
indiv_conv_cond = 1e-5
indiv_maxiter = 1e2
other_values = np.array([1.7, 0.2])
us = [copy.copy(other_values)]
epsilon = 2
i = 0
while i < max_iter and epsilon > acc :
    u_old = copy.copy(other_values)

    th, epst, c, points = x.parabolic_minimisation_NLL_theta(X = X, conv_condition = indiv_conv_cond, max_iter = indiv_maxiter, deltamsq = other_values[1])
    # thp, thm = x.thetamin_stddev_NLL(th, other_values[1])
    other_values[0] = th
    dm, epsm, c_m, pointsm = x.parabolic_minimisation_NLL_deltamsq(X = Y, conv_condition = indiv_conv_cond, max_iter = indiv_maxiter, theta = other_values[0])
    # dmp, dmm = x.deltammin_stddev_NLL(dm, other_values[0])
    other_values[1] = dm

    epsilon = np.linalg.norm(other_values-u_old)
    us.append(copy.copy(other_values))
    i += 1
    
us = np.asarray(us).T
us[0] *= theta23_0
us[1] *= dmsq23_0
# fig, ax = plt.subplots()
# ax.plot(us[0], us[1], marker= '+', ls = ':', color = 'k', lw = 2, ms = 8, mec = 'k')
# ax.set_xlabel(U_theta23)
# ax.set_ylabel(U_deltamsq)
# leg = ax.legend(fontsize = 20, loc='best', markerfirst = True, frameon = True)
# leg.get_frame().set_edgecolor('k')
# leg.get_frame().set_facecolor('w')


plot_num = 150
xx = np.linspace(.1, 1.9, plot_num)
yy = np.linspace(0.1, 1.5, plot_num)

make_contours = False

if make_contours:
    XX, YY  = np.meshgrid(xx, yy)
    Z = np.zeros_like(XX)
    M = len(XX)
    N = len(YY)
    for i in range(M):
        for j in range(N):
            Z[i, j] = x.NLL_params(XX[i, j], YY[i, j])
            
    # Unnormalise
    XX *= theta23_0
    YY *= dmsq23_0


fig, ax = plt.subplots()
bc.standard_axes_settings(ax, bc.infile_figparams)
ax.plot(us[0], us[1], ls = ':', marker = '.', mec = 'r', mfc = 'r', ms = 5, mew = 2, label = r'$x_{min}$', lw = 2)
ax.plot(us[0,-1], us[1, -1], ls = 'None', marker = 'o', mec = 'k', mfc = 'None', ms = 8, mew = 2, label = r'$x_{min}$')

cs = ax.contour(XX, YY, Z, cmap = cm.coolwarm)

ax.plot(*us[:, 0], 'gd', mfc = 'None', ms = 15, mew = 2)
ax.set_xlabel(U_theta23)
ax.set_ylabel(U_deltamsq)
labels = ['Iterations', 'Input Parameters', fr'$\theta_{{23}}$ = {us[0,-1]:.4f}, $\Delta m^2_{{23}}$ = {us[1,-1]:.4e}']
handles = [mpl.lines.Line2D([0], [0], linestyle = 'None', marker = '.', mec = 'r', mfc = 'r', ms = 10, mew =2), mpl.lines.Line2D([0], [0], linestyle = 'None', marker = 'd',mec = 'g', mfc = 'None', ms = 10, mew = 3), mpl.lines.Line2D([0], [0], linestyle = 'None', marker = 'o',mec = 'k', mfc = 'None', ms = 8, mew = 2)]# mpl.lines.Line2D([0], [0], marker = 'x', mec = 'k', mfc = 'None', ms = 10)]
plt.subplots_adjust(left=0.055, bottom=0.094, right=0.555, top=0.972, wspace=0.2, hspace=0.2)



#Iterations
scales  = {'thetas':theta23_0, 'deltamsqs':dmsq23_0}
colours = {'thetas':'firebrick', 'deltamsqs':'royalblue'}
names = {'thetas':U_theta23, 'deltamsqs':U_deltamsq}
fig1, ax1 = plt.subplots()
bc.standard_axes_settings(ax1, bc.infile_figparams)
bbox_props = dict(boxstyle="square,pad=0.2", fc="w", ec="k", lw=1)
us[0] /= theta23_0
us[1] /= dmsq23_0
for k, key in enumerate(scales):    
    ax1.plot(us[k], marker= 'o', ls = ':', color = colours[key], lw = 2, ms = 8, mec = colours[key], label = names[key])

ax1.text(1.5, .6, fr'{U_theta23} = {us[0,-1]*theta23_0:.4f}'+'\n'+fr'{U_deltamsq} = {us[1,-1]*dmsq23_0:.4e}', fontsize = 18, bbox = bbox_props)
            
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Scaled Parameters')

leg = ax1.legend(fontsize = 20, loc='best', markerfirst = True, frameon = True)
leg.get_frame().set_edgecolor('k')
leg.get_frame().set_facecolor('w')
plt.subplots_adjust(left=0.055, bottom=0.094, right=0.555, top=0.972, wspace=0.2, hspace=0.2)

us[0] *= theta23_0
us[1] *= dmsq23_0

#error
u = us[:,-1]
fn = x.NLL_params
H = bc.HessianCDA(u, fn, deltas = 0.01)
std_devth = np.sqrt(np.linalg.inv(H)[0, 0])
std_devdm = np.sqrt(np.linalg.inv(H)[1,1])

#%%-------3D plot NLL-------
# from mpl_toolkits.mplot3d import Axes3D

N = 400
M = 400
thetas = np.linspace(0, 2, N)
deltams = np.linspace(0.5,10, M)
X, Y  = np.meshgrid(thetas, deltams)
Z = np.zeros_like(X)
for i in range(M):
    for j in range(N):
        Z[i, j] = x.NLL_params(X[i, j], Y[i, j])
# ax = Axes3D.plot_surface(X, Y, Z, *args, **kwargs)

#%%
fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.contour3D(X, Y, Z, 50, cmap='binary')
ax = Axes3D(fig)
ax.plot_surface(X*theta23_0, Y*dmsq23_0, Z, cmap = cm.coolwarm, linewidth=0, antialiased=False)
ax.scatter(us[0], us[1], x.NLL_params(us[0]/theta23_0, us[1]/dmsq23_0), c = 'k')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
#%%-------4.2 Multivariate NLL-------

#-----Gradient Method-----
alpha = 1e-5
params = np.array([1.1, 0.8]) # (theta, deltamsq)
eps = 1e-8
c = 0
us = [copy.copy(params)]

while c < 1e3:
    diff_th = (x.NLL_params(params[0]+eps, params[1]) - x.NLL_params(params[0], params[1]))/(eps)
    diff_dm = (x.NLL_params(params[0], params[1]+eps) - x.NLL_params(params[0], params[1]))/(eps)
    grad = np.array([diff_th, diff_dm])
    params -= alpha*grad
    c += 1
    print(diff_th, diff_dm, grad, params, '\n')
    us.append(copy.copy(params))
    
us = np.asarray(us).T
us[0] *= theta23_0
us[1] *= dmsq23_0



fig, ax = plt.subplots()
bc.standard_axes_settings(ax, bc.infile_figparams)
ax.plot(us[0], us[1], marker= '+', ls = ':', color = 'k', lw = 2, ms = 8, mec = 'k')
ax.set_xlabel(U_theta23)
ax.set_ylabel(U_deltamsq)
    
#%%--------Quasi Newton Method------

X = bc.Dataset(fp)

# inputs
alpha = 1e-2
input_params = np.array([1.7, 0.2]) # input parameters x0 = [theta0, deltamsq0]
its = 1e6
deltas = 1e-3#np.array([0.001, 0.001]) # size of step to do in gradient
grad_funct = X.grad_NLL
funct = X.NLL_params
NLL_funct_params = {}


us, eps, c, deltas = bc.quasiNewton(input_params, grad_funct, funct, funct_params = NLL_funct_params, alpha = alpha, deltas = deltas, max_iter = its, convergence = 1e-7)

us = np.delete(us, np.s_[c+2:], 1)
us[0] *= theta23_0
us[1] *= dmsq23_0
input_params[0] *= theta23_0
input_params[1] *= dmsq23_0

plot_num = 100
xx = np.linspace(-3.5, 1.8, plot_num)
yy = np.linspace(0.1, 3, plot_num)


make_contours = False

if make_contours:
    XX, YY  = np.meshgrid(xx, yy)
    Z = np.zeros_like(XX)
    M = len(XX)
    N = len(YY)
    for i in range(M):
        for j in range(N):
            Z[i, j] = X.NLL_params(XX[i, j], YY[i, j], **NLL_funct_params)
            
    # Unnormalise
    XX *= theta23_0
    YY *= dmsq23_0


fig1, ax1 = plt.subplots()
ax1.plot(us[0], us[1], ls = 'None', marker = '.', mec = 'r', mfc = 'r', ms = 5, mew = 2, label = r'$x_{min}$')
ax1.plot(us[0,-1], us[1, -1], ls = 'None', marker = 'o', mec = 'k', mfc = 'None', ms = 8, mew = 2, label = r'$x_{min}$')

cs = ax1.contour(XX, YY, Z, cmap = cm.coolwarm)

ax1.plot(*input_params, 'gd', mfc = 'None', ms = 15, mew = 2)
ax1.set_xlabel(U_theta23)
ax1.set_ylabel(U_deltamsq)
labels = ['Iterations', 'Input Parameters', fr'$\theta_{{23}}$ = {us[0,-1]:.4f}, $\Delta m^2_{{23}}$ = {us[1,-1]:.4e}']
handles = [mpl.lines.Line2D([0], [0], linestyle = 'None', marker = '.', mec = 'r', mfc = 'r', ms = 10, mew =2), mpl.lines.Line2D([0], [0], linestyle = 'None', marker = 'd',mec = 'g', mfc = 'None', ms = 10, mew = 3), mpl.lines.Line2D([0], [0], linestyle = 'None', marker = 'o',mec = 'k', mfc = 'None', ms = 8, mew = 2)]# mpl.lines.Line2D([0], [0], marker = 'x', mec = 'k', mfc = 'None', ms = 10)]

ax1.text(-1.6, .0055, fr'$\alpha$ = {alpha}, $\delta$ = {deltas[0]}', fontsize = 18)
leg = ax1.legend(handles, labels, fontsize = 20, loc='best', markerfirst = True, frameon = True)
leg.get_frame().set_edgecolor('k')
leg.get_frame().set_facecolor('w')
bc.standard_axes_settings(ax1, bc.infile_figparams)

#Iterations
scales  = {'thetas':theta23_0, 'deltamsqs':dmsq23_0}
colours = {'thetas':'firebrick', 'deltamsqs':'royalblue'}
names = {'thetas':U_theta23, 'deltamsqs':U_deltamsq}
fig, ax = plt.subplots()
bc.standard_axes_settings(ax, bc.infile_figparams)
bbox_props = dict(boxstyle="square,pad=0.2", fc="w", ec="k", lw=1)
us[0] /= theta23_0
us[1] /= dmsq23_0
for k, key in enumerate(scales):    
    ax.plot(us[k], marker= 'o', ls = ':', color = colours[key], lw = 2, ms = 8, mec = colours[key], label = names[key])

ax.text(1, .6, fr'{U_theta23} = {us[0,-1]*theta23_0:.4f}'+'\n'+fr'{U_deltamsq} = {us[1,-1]*dmsq23_0:.4e}', fontsize = 18, bbox = bbox_props)
            
ax.set_xlabel('Iteration')
ax.set_ylabel('Scaled Parameters')

leg = ax.legend(fontsize = 20, loc='best', markerfirst = True, frameon = True)
leg.get_frame().set_edgecolor('k')
leg.get_frame().set_facecolor('w')
# plt.subplots_adjust(left=0.055, bottom=0.094, right=0.555, top=0.972, wspace=0.2, hspace=0.2)



us[0] *= theta23_0
us[1] *= dmsq23_0
#error
u = us[:,-1]
fn = x.NLL_params
H = bc.HessianCDA(u, fn, deltas = 1)
std_devth = np.sqrt(np.linalg.inv(H)[0, 0])
std_devdm = np.sqrt(np.linalg.inv(H)[1,1])












