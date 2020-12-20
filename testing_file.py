
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
#%%#------ Testing Parabolic Minimisation Function--------

def sq(a):
    return a**2 -3*a - 8

# expect error thrown as X has 5 initial points rather than 3
plot_num = 60
x = np.linspace(-4, 6, plot_num)
y = sq(x)

conv_cond = 1e-5
max_iterations = 1e5
X = np.array([-3.5, -2, 0, 2, 1]) # pick initial points
Y = sq(X)

x3, epsilon, c, parabola = bc.parabolic_minimisation(sq, X, param_minimised = None, function_parameters = {}, convergence_condition = conv_cond, max_iterations = max_iterations)


#%%------test on 1D simple quadratic

plot_num = 60
x = np.linspace(-4, 6, plot_num)
y = sq(x)

conv_cond = 1e-5
max_iterations = 1e5
X = np.array([-3.5, -2, 0]) # pick 3 initial points
Y = sq(X)

x3, epsilon, c, parabola = bc.parabolic_minimisation(sq, X, param_minimised = None, function_parameters = {}, convergence_condition = conv_cond, max_iterations = max_iterations)

fig, ax = plt.subplots()
bc.standard_axes_settings(ax, bc.infile_figparams)

ax.plot(X, Y, marker = 'x', ls = 'None', mec = 'r', ms = 13, mew = 3, label = r'Initial $x$ Choices')
ax.plot(x, y, marker= 'None', ls = ':', color = 'k', lw = 2, label = 'Function')
ax.plot(x3, sq(x3), ls = 'None', marker = 'd', mec = 'g', mfc = 'None', ms = 18, mew = 3, label = r'$x_{min}$')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')

leg = ax.legend(fontsize = 20, loc='best', markerfirst = True, frameon = True)
leg.get_frame().set_edgecolor('k')
leg.get_frame().set_facecolor('w')

#%%-------test on 1D function with more than one local minimum and multiple parameters 
def cube(z, a, b, c, d):
    return a*z**3 + b*z**2 + c*z + d


plot_num = 60
x = np.linspace(-3, 8, plot_num)
func_params = {'a':0.5,
               'b':-3,
               'c': 0,
               'd':-9,
               }
y = cube(x, **func_params)

conv_cond = 1e-5
max_iterations = 1e5
X = np.array([2.5, 2, 7]) # pick 3 initial points
Y = cube(X, **func_params)

x3, epsilon, c, parabola = bc.parabolic_minimisation(cube, X, param_minimised = 'z', function_parameters = func_params, convergence_condition = conv_cond, max_iterations = max_iterations)

fig, ax = plt.subplots()
bc.standard_axes_settings(ax, bc.infile_figparams)

ax.plot(X, Y, marker = 'x', ls = 'None', mec = 'r', ms = 13, mew = 3, label = r'Initial $x$ Choices')
ax.plot(x, y, marker= 'None', ls = ':', color = 'k', lw = 2, label = 'Function')
ax.plot(x3, cube(x3, **func_params), ls = 'None', marker = 'd', mec = 'g', mfc = 'None', ms = 18, mew = 3, label = r'$x_{min}$')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')

leg = ax.legend(fontsize = 20, loc='best', markerfirst = True, frameon = True)
leg.get_frame().set_edgecolor('k')
leg.get_frame().set_facecolor('w')
#%%------test on 2D smooth function--------
def dodgy_cone(x, y, a):
    q = (x**4)/3 +x**2 + (y**4)/2 + a*y**2 - x*y**2
    return q


N = 60
M = 50
x = np.linspace(-2, 2, N)
y = np.linspace(-2, 2, M)
func_params = {'a':1,
               }


X, Y  = np.meshgrid(x, y)
Z = np.zeros_like(X)
for i in range(M):
    for j in range(N):
        Z[i, j] = dodgy_cone(X[i, j], Y[i, j], **func_params)


#----------minimisation----------
conv_cond = 1e-6
indiv_max_iterations = 1e3
max_iters = 1e3
acc = 1e-14
Xs = np.array([-.9, 0.2, 0.6]) # pick 3 initial points
Ys = np.array([0.8, -.002, 0.1])

func_params['x'] = 0.75
func_params['y'] = -.1
us = [np.array([func_params['x'], func_params['y']])] # record of iterations

epsilon = 2
i=0

while i < max_iters and epsilon > acc :
    u_old = np.array([func_params['x'], func_params['y']]) # old parameters
    x3, epsilonx, c, parabola = bc.parabolic_minimisation(dodgy_cone, Xs, param_minimised = 'x', function_parameters = func_params, convergence_condition = conv_cond, max_iterations = indiv_max_iterations)
    func_params['x'] = x3
    y3, epsilony, cy, parabolay = bc.parabolic_minimisation(dodgy_cone, Ys, param_minimised = 'y', function_parameters = func_params, convergence_condition = conv_cond, max_iterations = indiv_max_iterations)
    func_params['y'] = y3
    epsilon = np.linalg.norm(np.array([func_params['x'], func_params['y']])-u_old)
    us.append(np.array([func_params['x'], func_params['y']]))
    i += 1
    
us = np.asarray(us)

fig1, ax1 = plt.subplots()
bc.standard_axes_settings(ax1, bc.infile_figparams)
ax1.plot(us[:, 0], us[:, 1], marker= '+', ls = ':', color = 'k', lw = 2, ms = 8, mec = 'k')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
# leg = ax.legend(fontsize = 20, loc='best', markerfirst = True, frameon = True)
# leg.get_frame().set_edgecolor('k')
# leg.get_frame().set_facecolor('w')

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(X, Y, Z, cmap = cm.coolwarm, linewidth=0, antialiased=False)
# ax.scatter(us[:, 0], us[:, 1], dodgy_cone(us[:, 0], us[:, 1], func_params['a']), c = 'k')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

print('Final Answer:', us[-1])
#%%--------------QuasiNewton ------------

plot_num = 500
funct_params = {'a':0.5}

# def dodgy_cone_grad(var_pars, deltas, a):
#     p = var_pars[0]
#     q = var_pars[1]
#     return np.array([4*p**3/3 + 2*p, 2*(q**3) + 2*q*a])
    
def dodgy_cone_grad(var_pars, deltas, a):
    p = var_pars[0]
    q = var_pars[1]
    
    dp = (dodgy_cone(p+deltas[0], q, a)-dodgy_cone(p, q, a))/deltas[0]
    dq = (dodgy_cone(p, q+deltas[1], a)-dodgy_cone(p, q, a))/deltas[1]
    return np.array([dp, dq])


def cube_grad(z, coeffs):
    return coeffs['a']*z**3 - coeffs['b']*z**2 + coeffs['c']*z - coeffs['d']


# us, c = bc.quasiNewton(np.array([-8]), sq_grad, grad_funct_params)

input_params = np.array([.2, .32])
grad_funct =  dodgy_cone_grad
funct = dodgy_cone
deltas = 1e-2
max_iter = 1e4
convergence = 1e-9
alpha = 1e-2

us, eps, c, deltas = bc.quasiNewton(input_params, dodgy_cone_grad, dodgy_cone, funct_params, alpha, deltas, max_iter, convergence)
                                    
us = np.delete(us, np.s_[c+2:], 1)


edge = .5
xx = np.linspace(-edge, edge, plot_num)
yy = np.linspace(-edge, edge, plot_num)

X, Y  = np.meshgrid(xx, yy)
Z = np.zeros_like(X)
M = len(X)
N = len(Y)
for i in range(M):
    for j in range(N):
        Z[i, j] = dodgy_cone(X[i, j], Y[i, j], **funct_params)


fig1, ax1 = plt.subplots()

ax1.plot(us[0], us[1], ls = 'None', marker = '.', mec = 'r', mfc = 'r', ms = 1.5, mew = 2, label = r'$x_{min}$')
cs = ax1.contour(X, Y, Z, cmap = cm.coolwarm)
ax1.plot(0, 0, ls = 'None', marker = 'x', mec = 'k', ms = 12)
ax1.text(0.31, -.1, fr'$\alpha$ = {alpha}, $\delta$ = {deltas[0]}, $z(x, y)$ = $x^4/3 + x^2 - x + y^4/2 + y^2/2$', fontsize = 18)

ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$y$')

# ax = Axes3D(fig)
# ax.plot_surface(X, Y, Z, cmap = cm.coolwarm, linewidth=0, antialiased=False)

#%%-------------Gradient---------------

def dodgy_cone_grad(var_pars, deltas, a):
    p = var_pars[0]
    q = var_pars[1]
    
    dp = (dodgy_cone(p+deltas[0], q, a)-dodgy_cone(p, q, a))/deltas[0]
    dq = (dodgy_cone(p, q+deltas[1], a)-dodgy_cone(p, q, a))/deltas[1]
    return np.array([dp, dq])

x  = np.array([.2, .32])
deltas = np.full(2, 0.01)
alpha = 0.0001
fig1, ax1 = plt.subplots()


for i in range(10000):
    # xold = copy.deepcopy(x)
    u = dodgy_cone_grad(x, deltas, **funct_params)
    x -= alpha*u
    # dd = x-xold
    # print('grad ', u, 'x ', x)
    ax1.plot(x[0], x[1], ls = 'None', marker = '.', mec = 'r', mfc = 'r', ms = 1.7, mew = 2)

# plt.plot(0,0, 'x')
edge = 0.35
xx = np.linspace(0.1, .7, plot_num)
yy = np.linspace(-.2, .4, plot_num)

X, Y  = np.meshgrid(xx, yy)
Z = np.zeros_like(X)
M = len(X)
N = len(Y)
for i in range(M):
    for j in range(N):
        Z[i, j] = dodgy_cone(X[i, j], Y[i, j], **funct_params)


cs = ax1.contour(X, Y, Z, cmap = cm.coolwarm)
# ax1.plot(0.38546, 0, ls = 'None', marker = 'x', mec = 'k', ms = 12)
ax1.plot(.2, .32, 'gd', mfc = 'None', ms = 14)
ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$y$')
labels = ['Iterations', 'Input Parameters']
handles = [mpl.lines.Line2D([0], [0], linestyle = 'None', marker = '.', mec = 'r', mfc = 'r', ms = 10, mew =2), mpl.lines.Line2D([0], [0], marker = 'd',mec = 'g', mfc = 'None', ms = 10)]# mpl.lines.Line2D([0], [0], marker = 'x', mec = 'k', mfc = 'None', ms = 10)]

ax1.text(0.31, -.1, fr'$\alpha$ = {alpha}, $\delta$ = {deltas[0]}, $z(x, y)$ = $x^4/3 + x^2 - xy^2 + y^4/2 + y^2/2$', fontsize = 18)
leg = ax1.legend(handles, labels, fontsize = 20, loc='best', markerfirst = True, frameon = True)
leg.get_frame().set_edgecolor('k')
leg.get_frame().set_facecolor('w')
bc.standard_axes_settings(ax1, bc.infile_figparams)
#%%---------Hessian------------

fun = dodgy_cone
funct_params = {'a':0.5}
deltas = np.full(2, 1)
u = us[:,-1]
act_H = np.array([[18, -2], [-2, 3]])

H_cda = bc.HessianCDA(u, dodgy_cone, {'a':0.5}, 0.1)
std_devx = np.sqrt(np.linalg.inv(H_cda)[0, 0])
std_devy = np.sqrt(np.linalg.inv(H_cda)[1,1])


print(H_cda)

