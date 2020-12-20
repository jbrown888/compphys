import numpy as np
import copy
import os
import math
import matplotlib.pyplot as plt
import matplotlib as mpl

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

def parabolic_minimisation_old(funct, X, param_minimised, function_parameters, convergence_condition, max_iterations = 1e6):
    
    
    if not callable(funct):
        raise TypeError("function must a function or bound method (i.e. callable)")
    
    if not isinstance(param_minimised, str) and param_minimised is not None:
        raise TypeError("param_minimised should be a string or None")
    
    if not isinstance(function_parameters, dict):
        raise TypeError("function_parameters must be a dictionary containing parameters of funct")
    if len(X) != 3:
        raise TypeError("X array must be of length 3")
        
    X = X[np.argsort(X)] # x's must be in order for lagrange
    
    #function parameters: dictionary of values to take
    func_params = copy.deepcopy(function_parameters)
    
    Y = np.empty(3)
    
    if param_minimised is not None:
        for i in range(3):
            func_params[param_minimised] = X[i]
            Y[i] = funct(**func_params)
    else:
        for i in range(3):
            Y[i] = funct(X[i], **func_params)
        
    #array method
    points = np.empty((2, 4))
    points[0, 0:3] = X # 3 x points (theta), 0
    points[1, 0:3] = Y # 3 y points (NLL), 0
    x3_old = min(X)
    c = 0 
    epsilon = convergence_condition + 1
    
    while epsilon >= convergence_condition and c < max_iterations:
        x3 = find_x3(points[0, 0:3], points[1, 0:3]) # find minimum of parabola formed from X, Y points
        points[0,-1] = x3 # set last element of x points to 0
        if param_minimised is not None:
            func_params[param_minimised] = x3
            points[1, -1] = funct(**func_params) # set last element of y points to y(x3)
        else:
            points[1,-1] = funct(x3, **func_params)
        largest_f = np.argmax(points[1]) # find index of largest y point
        mask = np.ones(4, dtype = bool) 
        mask[largest_f] = False # set the column containing largest y point to False in mask array
        points[:, 0:3] = points[:, mask] # reset columns 1-3 of points to the 3 smallest x-y points - i.e. delete the largest  y point
        points[:, 0:3] = points[:, 0:3][:, np.argsort(points[0, 0:3])] # reorder the 3 smallest x-y points in order of ascending x
        # poins[:, -1] is ignored in the calculation, and only used to store x3
        if x3_old != 0:
            epsilon = np.abs(x3-x3_old)/x3_old
        elif x3 != 0:
            epsilon = np.abs(x3-x3_old)/x3
        else:
            epsilon = 0
        x3_old = x3 # enter new iteration - x3 is now the old x3
        parabola = points[:, 0:3] # points used in current parabola
        c += 1
    return x3, epsilon, c, parabola

def quasiNewton(input_params, grad_funct, funct, funct_params = {}, alpha = 1e-4, deltas = None, max_iter = 1e5, convergence = 1e-6):
    """
    

    Parameters
    ----------
    input_params : TYPE
        DESCRIPTION.
    grad_funct : TYPE
        DESCRIPTION.
    funct : TYPE
        DESCRIPTION.
    funct_params : TYPE, optional
        DESCRIPTION. The default is {}.
    alpha : TYPE, optional
        DESCRIPTION. The default is 1e-4.
    deltas : TYPE, optional
        DESCRIPTION. The default is None.
    max_iter : TYPE, optional
        DESCRIPTION. The default is 1e5.
    convergence : TYPE, optional
        DESCRIPTION. The default is 1e-6.

    Raises
    ------
    TypeError
        DESCRIPTION.

    Returns
    -------
    us : TYPE
        DESCRIPTION.
    epsilon : TYPE
        DESCRIPTION.
    c : TYPE
        DESCRIPTION.
    deltas : TYPE
        DESCRIPTION.

    """
    if not isinstance(input_params, np.ndarray):
        raise TypeError("input_params must be an array containing variable input parameters of grad_funct")
    
    if not input_params.dtype in [np.float32, np.float64, float]:
        input_params = input_params.astype(np.float64)
        # raise TypeError("input parameter values must be floats, not integers. Cannot cast different types later in function")
    
    if not isinstance(funct_params, dict):
        raise TypeError("funct_params must be a dictionary containing constant parameters of grad_funct")
    
    if not callable(grad_funct):
        raise TypeError("function must a function or bound method (i.e. callable)")
        
    x = copy.deepcopy(input_params)
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
    return us, epsilon, c, deltas


def parabolic_minimisation(funct, X, param_minimised, function_parameters, convergence_condition, max_iterations = 1e6):
    
    
    if not callable(funct):
        raise TypeError("function must a function or bound method (i.e. callable)")
    
    if not isinstance(param_minimised, str) and param_minimised is not None:
        raise TypeError("param_minimised should be a string or None")
    
    if not isinstance(function_parameters, dict):
        raise TypeError("function_parameters must be a dictionary containing parameters of funct")
    if len(X) != 3:
        raise TypeError("X array must be of length 3")
        
    X = X[np.argsort(X)] # x's must be in order for lagrange
    
    #function parameters: dictionary of values to take
    func_params = copy.deepcopy(function_parameters)
    
    Y = np.empty(3)
    
    if param_minimised is not None:
        for i in range(3):
            func_params[param_minimised] = X[i]
            Y[i] = funct(**func_params)
    else:
        for i in range(3):
            Y[i] = funct(X[i], **func_params)
        
    #array method
    points = np.empty((2, 4))
    points[0, 0:3] = X # 3 x points (theta), 0
    points[1, 0:3] = Y # 3 y points (NLL), 0
    x3_old = min(X)
    c = 0 
    epsilon = convergence_condition + 1

    while epsilon >= convergence_condition and c < max_iterations:
        x3 = find_x3(points[0, 0:3], points[1, 0:3]) # find minimum of parabola formed from X, Y points
        if x3 in points[0]: #would cause NaN value
            # print(points)
            # print(x3)
            x3 = x3_old
            parabola = points[:, 0:3] # points used in current parabola
            print("Minimisation stopped because x3 value already exists in points. Values of previous loop returned")
        if math.isnan(x3): # stop it when two points are same
            x3 = x3_old
            parabola = points[:, 0:3] # points used in current parabola
            print("Minimisation stopped because x3 = nan. Values of previous loop returned")
            break
        points[0,-1] = x3 # set last element of x points to 0
        if param_minimised is not None:
            func_params[param_minimised] = x3
            points[1, -1] = funct(**func_params) # set last element of y points to y(x3)
        else:
            points[1,-1] = funct(x3, **func_params)
        largest_f = np.argmax(points[1]) # find index of largest y point
        mask = np.ones(4, dtype = bool) 
        mask[largest_f] = False # set the column containing largest y point to False in mask array
        points[:, 0:3] = points[:, mask] # reset columns 1-3 of points to the 3 smallest x-y points - i.e. delete the largest  y point
        points[:, 0:3] = points[:, 0:3][:, np.argsort(points[0, 0:3])] # reorder the 3 smallest x-y points in order of ascending x
        # points[:, -1] is ignored in the calculation, and only used to store x3
        if x3_old != 0:
            epsilon = np.abs((x3-x3_old)) # now is absolute, not fractional error
        elif x3 != 0:
            epsilon = np.abs((x3-x3_old))
        else:
            epsilon = 0
        x3_old = x3 # enter new iteration - x3 is now the old x3
        parabola = points[:, 0:3] # points used in current parabola
        c += 1
    return x3, epsilon, c, parabola

def HessianCDA(u, fun, funct_params ={}, deltas = None):
    if not isinstance(u, np.ndarray):
        raise TypeError("u must be an array containing variable input parameters of grad_funct")
    
    if not u.dtype in [np.float32, np.float64, float]:
        u = u.astype(np.float64)
        # raise TypeError("input parameter values must be floats, not integers. Cannot cast different types later in function")
    Ndim = len(u)
    
    if not isinstance(funct_params, dict):
        raise TypeError("funct_params must be a dictionary containing constant parameters of grad_funct")
    
    if not callable(fun):
        raise TypeError("function must a function or bound method (i.e. callable)")
        
    if deltas is None:
        deltas = np.full(Ndim, 0.01)
    elif isinstance(deltas, (int, np.float64, np.int32, float)):
        deltas = np.full(Ndim, deltas)
    elif isinstance(deltas, np.ndarray) and len(deltas) == Ndim:
        pass
    elif type(deltas) == str: # for analytic functions
        pass
    else:
        raise TypeError("deltas must be either float/int, Ndim array, 'analytic' or None")
        
    fxp = fun(u[0]+deltas[0], u[1], **funct_params)
    f = fun(*u, **funct_params)
    fxm = fun(u[0] - deltas[0], u[1], **funct_params)
    fyp = fun(u[0], u[1] + deltas[1], **funct_params)
    fym = fun(u[0], u[1] - deltas[1], **funct_params)
    fxyp = fun(u[0] + deltas[0], u[1] + deltas[1], **funct_params)
    fxym = fun(u[0] - deltas[0], u[1] - deltas[1], **funct_params)
    fxpym = fun(u[0] + deltas[0], u[1] - deltas[1], **funct_params)
    fxmyp = fun(u[0] - deltas[0], u[1] + deltas[1], **funct_params)
    #FWD_diff
    # dxx = (fxp - 2*f + fxm)/(deltas[0]**2)
    # dyy = (fyp - 2*f + fym)/(deltas[1]**2)
    # dxy = (fxyp -fxp - fyp + 2*f -fxm - fym + fxym)/(2*deltas[0]*deltas[1])
    # H = np.array([[dxx, dxy], [dxy, dyy]])

    dxx = (- fun(u[0]+2*deltas[0], u[1], **funct_params) + 16*fxp - 30*f + 16*fxm - fun(u[0]-2*deltas[0], u[1], **funct_params))/(12*deltas[0]**2)
    dyy = (- fun(u[0], u[1]+2*deltas[1], **funct_params) + 16*fyp - 30*f + 16*fym - fun(u[0], u[1]-2*deltas[1], **funct_params))/(12*deltas[1]**2)
    
    dxy = (fxyp - fxpym - fxmyp +fxym)/(4*deltas[0]*deltas[1])
    
    H_cda = np.array([[dxx, dxy], [dxy, dyy]])
    return H_cda

def standard_axes_settings(ax, figparams):
        """
        Apply standard settings to a graph for label sizes, gridlines, frame etc
    
        Parameters
        ----------
        ax : matplotlib axes object
            axes for graph you want to edit
        figparams : dictionary
            has values for sizes on graph wanted
    
        Returns
        -------
        None.
    
        """
        ax.set_frame_on
        ax.grid(b=True, which='major', axis='both', c='grey', ls='--')
#       ax.grid(b=True, which='minor', axis='both', c='darkgrey', ls = '--', linewidth =2)
        ax.tick_params(axis ='both', which = 'major', direction ='in', labelsize = 22)
        ax.tick_params(axis ='both', which = 'minor', direction ='in')
        ax.xaxis.label.set_size(figparams['labelsize'])
        ax.yaxis.label.set_size(figparams['labelsize'])
        ax.minorticks_on()
        ax.yaxis.get_offset_text().set_fontsize(figparams['offsetsize'])
        ax.xaxis.get_offset_text().set_fontsize(figparams['offsetsize'])
        ax.ticklabel_format(axis = 'x', style = 'sci', scilimits = (-3,3), useOffset = True)
        ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (-3,3), useOffset=True)

infile_figparams = {'offsetsize':20, 
    'labelsize':26, 
    'ec' :'r',
    'fc' :'r',
    'linewidth':2,
    }

def get_file_name(path):
    # taken from stack exchange
    return os.path.basename(path).split(".")[0].strip().lower() 

def find_line_number(phrase: str, filename):
    with open(filename,'r') as f:
        for (i, line) in enumerate(f):
            if phrase in line:
                return i+1
    raise Exception(f"Phrase '{phrase}' not found in file")
    return 0

def find_line_numbers_repeated(phrase:str, filename):
    lines = []
    with open(filename,'r') as f:
        for (i, line) in enumerate(f):
            if phrase in line:
                lines.append(i+1)
    return lines

def P_noosc_muon(E, L, theta23, delta_msq23):
    """
    Probability that a muon neutrino has NOT oscillated in to a tau neutrino 

    Parameters
    ----------
    E :  np.float64 or np.array
        energy of neutrino
    L : np.float64 or np.array
        length travelled for
    theta23 : np.float64 or np.array
        mixing angle
    delta_msq23 : np.float64 or np.array
        difference in mass of muon and tau neutrino squared

    Returns
    -------
    np.float64
        probability not oscillated

    """
    #calculates prob. with non-scaled parameters
    #theta23 and delta_msq23 are scaled when given to program.
    return  1 - (np.sin(2*theta23*theta23_0)**2) * (np.sin(1.267*delta_msq23*dmsq23_0*L/E)**2)

def poisson(m, lamda):
    """
    Poisson distribution - probability of event occuring m times wiht a mean of lamda

    Parameters
    ----------
    m : np.ndarray or float
        number of times event occurs/number of events
    lamda : np.ndarray or float
        distriubtion mean

    Returns
    -------
    np.float64
        Probability

    """
    return (lamda**m)*np.exp(-m)/math.factorial(m)

def NLL_poisson(m_i, lamda_i):
    """
    Evaluates the negative log likelihood(NLL) for given dataset m_i with means lambda_i.
    In this context, m_i is the number of muon events recorded per energy bin 

    Parameters
    ----------
    m_i : np.ndarray
        event data/measurements we assume to have come from a poisson distribution of mean lamda_i
    lamda_i : np.ndarray, same length as m_i. 
        mean of poisson - average expected number for the bin. A function of theta and deltamsq parameters.

    Returns
    -------
    NLL : np.float64
        NLL for a dataset, as a function of parameters. The parameters set the means lamda

    """
    if m_i.shape != lamda_i.shape:
        raise ValueError(f"lamda and m arrays could not be broadcast together with shapes {m_i.shape} {lamda_i.shape}")
    NLL = 0
    for i in range(len(m_i)):
        if m_i[i] == 0:
            q = 0
        else: 
            q = m_i[i]*np.log(m_i[i]/lamda_i[i])
        z = lamda_i[i] - m_i[i] + q
        NLL +=z
    return NLL

fitting_str = "Data to fit:"
flux_str = "Unoscillated flux:"

def find_x3(X, Y):
    return 0.5*((X[2]**2 - X[1]**2)*Y[0] + (X[0]**2 - X[2]**2)*Y[1] + (X[1]**2 - X[0]**2)*Y[2])/((X[2]-X[1])*Y[0] + (X[0] - X[2])*Y[1] + (X[1] - X[0])*Y[2])


class Dataset:
    def __init__(self, fp, energy_range = None, L = 295):
        event_rows = find_line_number(flux_str, fp) # returns row number of line 'Unoscillated flux:'
        # Num rows = row number 'unoscillated flux' - 4 as blank lines are skipped
        self.event_data = np.loadtxt(fp, dtype=np.float64, delimiter='\t', skiprows=1, max_rows=event_rows-4) # observed number muon neutrino events 0-10GeV
        # self.event_data[0] = number events with 0<E<0.05 GeV ; this is already histogram data!!
        self.flux_data = np.loadtxt(fp, dtype = np.float64, delimiter = '\t', skiprows = event_rows) # unoscillated event rate prediction i.e. simulated event # prediction assuming no osc.
        self.num_bins = self.event_data.size
        if self.num_bins != 200:
                raise Exception(f"I excected there to be 200 energy bins, not {self.num_bins}. Has a row been added/missed out?")
        self.filename = fp
        if energy_range is None:
            self.energy_range = [0, 10] #GeV
        else:
            self.energy_range = energy_range
        self.bin_size = (self.energy_range[1]-self.energy_range[0])/self.num_bins # assumes bins are of equal size!!!
        self.bins = np.arange(self.energy_range[0], self.energy_range[1], self.bin_size)
        self.bin_means = self.bins + self.bin_size/2
        self.L = L
        
    
    def event_histogram(self, gridlines = True):
        """
        Generates histogram of observed event data

        Parameters
        ----------
        gridlines : TYPE, optional
            The default is True.

        Returns
        -------
        ax : TYPE
            DESCRIPTION.

        """
        fig, ax = plt.subplots()
        standard_axes_settings(ax, infile_figparams)
        ax.bar(self.bins, height = self.event_data, width = self.bin_size, bottom = 0, align = 'center', color = 'slateblue', edgecolor ='k', linewidth = 0) 
        ax.grid(b=True, which='minor', axis='x', c='grey', ls = '--', linewidth =1)
        if not gridlines:
            ax.grid(b=False, which='both', axis='both', c='None', ls = '--', linewidth =1)
        ax.set_xlabel(U_E)
        ax.set_ylabel(U_observed_muon_events)
        return ax
    
    def flux_histogram(self, gridlines = True):
        """
        Generates histogram of predicted unoscillated flux/numbers of muon neutrionos

        Parameters
        ----------
        gridlines : TYPE, optional
            The default is True.

        Returns
        -------
        ax : TYPE
            DESCRIPTION.

        """
        fig, ax = plt.subplots()
        standard_axes_settings(ax, infile_figparams)
        ax.bar(self.bins, height = self.flux_data, width = self.bin_size, bottom = 0, align = 'center', color = 'slateblue', edgecolor ='k', linewidth = 0) 
        ax.grid(b=True, which='minor', axis='x', c='grey', ls = '--', linewidth =1)
        if not gridlines:
            ax.grid(b=False, which='both', axis='both', c='None', ls = '--', linewidth =1)
        ax.set_xlabel(U_E)
        ax.set_ylabel(r'Unoscillated Event Rate prediction')
        return ax
    
    def flux_event_histogram(self, gridlines = True):
        """
        Generate histogram showing both observed data and predicted unoscillated flux

        Parameters
        ----------
        gridlines : TYPE, optional
            The default is True.

        Returns
        -------
        list
            DESCRIPTION.

        """
        fig, ax = plt.subplots()
        standard_axes_settings(ax, infile_figparams)
        ax.bar(np.arange(self.energy_range[0], self.energy_range[1], self.bin_size), height = self.event_data, width = self.bin_size, bottom = 0, align = 'center', color = 'slateblue', edgecolor ='k', linewidth = 0) 
        ax2 = ax.twinx()
        standard_axes_settings(ax2, infile_figparams)
        cc = 'r'
        ax2.plot(np.arange(self.energy_range[0], self.energy_range[1], self.bin_size), self.flux_data, ls = '--', color = cc, marker = None, lw = 2)
        ax2.tick_params(axis='y', labelcolor=cc)
        ax.tick_params(axis='y', labelcolor='k')
        ax.set_xlabel(U_E)
        ax2.set_ylabel(r'Unoscillated Event Rate prediction', color = cc)
        ax.set_ylabel(U_observed_muon_events) 
        if not gridlines:
            ax.grid(b=False, which='both', axis='both', c='None', ls = '--', linewidth =1)
            
        return [ax, ax2]
    
    def lamda_u(self, theta23, delta_msq23, mean_bin_energies = None):
        """
        Applies oscillation probability to the simulated event rate. Returns oscillated event rate prediction lamda_i - event rate prediction of MUON neutrino events

        Parameters
        ----------
        theta23 : np.float64
            mixing angle
        delta_msq23 : np.float64
            deltamsq parameter
        mean_bin_energies : np.ndarray, optional
            The default is None. Mean energies to use for bins. If none, uses calculated mean bin energies for dataset. 

        Returns
        -------
        lamdas : TYPE
            DESCRIPTION.

        """
        #returns oscillated flux rate
        if mean_bin_energies is None:
            mean_bin_energies = self.bin_means
        lamdas = np.multiply(self.flux_data, P_noosc_muon(mean_bin_energies, self.L, theta23, delta_msq23)) 
        return lamdas
    
    
    def compare_sim_real_histogram(self, theta23, deltamsq23, stacked = True, gridlines = True):
        """
        Generates histogram comparing simulated oscillated event rate and observed data. Simulated event rate calculated with theta and deltamsq parameters

        Parameters
        ----------
        theta23 : np.float64
            DESCRIPTION.
        deltamsq23 :  np.float64
            DESCRIPTION.
        stacked : bool, optional
            The default is True. If true, bars are stacked on top of each other. If false, displayed in front/behind 
        gridlines : bool, optional
            The default is True.

        Returns
        -------
        axl : TYPE
            DESCRIPTION.

        """
        ll = self.lamda_u(theta23, deltamsq23)
        fig, axl = plt.subplots()
        standard_axes_settings(axl, infile_figparams)
        axl.bar(self.bins, height = self.event_data, width = self.bin_size, bottom = 0, align = 'center', zorder = 10, color = 'slateblue', edgecolor ='k', linewidth = 0, label = 'Observed Data') 
        if stacked:
            bottom = self.event_data
        else: 
            bottom = 0
        axl.bar(self.bins, height = ll, width = self.bin_size, bottom = bottom, align = 'center', color = 'firebrick', edgecolor ='k', linewidth = 0, label = 'Simulated Oscillated Rate') 
        
        axl.grid(b=True, which='minor', axis='x', c='grey', ls = '--', linewidth =1)
        if not gridlines:
            axl.grid(b=False, which='both', axis='both', c='None', ls = '--', linewidth =1)
        axl.set_xlabel(U_E)
        axl.set_ylabel(r'Number of $\nu_{\mu}$ Events')
        leg = axl.legend(fontsize = 20, loc='best', markerfirst = True, frameon = True)
        leg.get_frame().set_edgecolor('k')
        leg.get_frame().set_facecolor('w')
        
        return axl
    
    def compare_sim_real(self, theta23, deltamsq23, stacked = True, gridlines = True):
        """
        Generates histogram comparing simulated oscillated event rate and observed data. Simulated event rate calculated with theta and deltamsq parameters. Observed data displayed as bar graph, predicted rate as line.

        Parameters
        ----------
        theta23 : TYPE
            DESCRIPTION.
        deltamsq23 : TYPE
            DESCRIPTION.
        stacked : TYPE, optional
            The default is True.
        gridlines : TYPE, optional
            The default is True.

        Returns
        -------
        list
            DESCRIPTION.

        """
        ll = self.lamda_u(theta23, deltamsq23)
        fig, ax = plt.subplots()
        standard_axes_settings(ax, infile_figparams)
        ax.bar(self.bins, height = self.event_data, width = self.bin_size, bottom = 0, align = 'center', color = 'slateblue', edgecolor ='k', linewidth = 0, label = 'Observed Data') 
        cc = 'r'
        ax.plot(self.bins, ll, ls = '--', color = cc, marker = None, lw = 2, label = 'Simulated Oscillated Rate')
        ax.set_xlabel(U_E)
        ax.set_ylabel(r'Number of $\nu_{\mu}$ Events') 
        if not gridlines:
            ax.grid(b=False, which='both', axis='both', c='None', ls = '--', linewidth =1)
        leg = ax.legend(fontsize = 20, loc='best', markerfirst = True, frameon = True)
        leg.get_frame().set_edgecolor('k')
        leg.get_frame().set_facecolor('w')
        return ax
    
    def NLL_params(self, thetas, deltamsqs):
        """
        Returns NLL values for a series of parameters u_i = (theta_i, deltamsq_i) with which to calculate lamdas(predicted event rate/poisson means). 

        Parameters
        ----------
        thetas : (N, ) array or float/int
            theta values
        deltamsqs : (N, ) array or float/int
            deltasqm values 

        Returns
        -------
        NLL : (N, array) or np.float64
            NLL values s.t. NLL[i] = NLL(params[:, i])

        """
        if isinstance(thetas, (float, np.float64, int, np.int32)):
            thetas = [thetas]
        if isinstance(deltamsqs, (float, np.float64, int, np.int32)):
            deltamsqs = [deltamsqs]
        if len(thetas) != len(deltamsqs):
            raise TypeError(f'theta and deltamsq arrays must be of same length, of length {len(thetas)}, {len(deltamsqs)}')
            
        params = np.array([thetas,deltamsqs])    
        N = params.shape[1] # number of different parameter values to evalute
        NLL = np.empty(N)
        for i in range(N):
            lamdas = self.lamda_u(params[0,i], params[1,i], mean_bin_energies = self.bin_means)
            NLL[i] = NLL_poisson(self.event_data, lamdas) # evaluates NLL for event data and lamdas given
        if N == 1:
            NLL = NLL[0] # if only one evaluation, returns float rather than array
        return NLL
    
    def thetamin_stddev_NLL(self, mintheta, deltamsq = 1):
        c = 0
        z = 1
        NLL_min = self.NLL_params([mintheta], [deltamsq])
        evals = 30
        
        #sigma theta +
        if mintheta < 1: # NLL symmetric about np.pi/4 ( scaled to np.pi/4 in P_noosc_muon function)
            a = np.linspace(mintheta, 1, evals)
        else:
            a = np.linspace(mintheta, mintheta+0.2, evals)
           
        ys = self.NLL_params(a, np.full(evals, deltamsq))
        
        while c < 4:
            q = np.abs(ys - (NLL_min+0.5)).argmin()
            thp = a[q]
            # NLL_05p = self.NLL_params([thp], [deltamsq])
            a = np.linspace(thp-0.01*z, thp+0.01*z, evals)
            ys = self.NLL_params(a, np.full(evals, deltamsq))
            c += 1
            z /= 10
        
        # sigma theta -
        if mintheta < 1:
            a = np.linspace(mintheta - 0.2, mintheta, evals)
        else:
            a = np.linspace(1., mintheta, evals)
        c=0
        z=1
        ys = self.NLL_params(a, np.full(evals, deltamsq))
        while c < 4:
            q = np.abs(ys - (NLL_min+0.5)).argmin()
            thm = a[q]
            # NLL_05m = self.NLL_params([thm], [deltamsq])
            a = np.linspace(thm-0.01*z, thm+0.01*z, evals)
            ys = self.NLL_params(a, np.full(evals, deltamsq))
            c += 1
            z /= 10
            
        # sigma_p = thp - mintheta
        # sigma_m = mintheta - thm
        
        # assert sigma_p > 0 and sigma_m > 0 , "std dev should be positive"
        assert thp > mintheta and mintheta > thm , "std dev should be positive"

        return thp, thm

    def grad_NLL(self, params, deltas):
        """
        Calculates gradient of NLL function at (theta, dmsq) with different deltas using forward finite different approximation.

        Parameters
        ----------
        params : (2,) array
            (theta, dmsq) at which to calculate grad
        deltas : (2,) array or np.float64
            steps to make in theta, dmsq.

        Returns
        -------
        grad : (2,) array
            gradient of NLL.

        """
        diff_th = (self.NLL_params(params[0]+deltas[0], params[1]) - self.NLL_params(*params))/deltas[0]
        diff_dm = (self.NLL_params(params[0], params[1]+deltas[1]) - self.NLL_params(*params))/deltas[1]
        grad = np.array([diff_th, diff_dm])
        return grad
    
    def parabolic_minimisation_NLL_deltamsq(self, X, theta, conv_condition, max_iter = None):
        """
        Finds minimum value of NLL function using Parabolic minimisation in the deltamsq direction

        Parameters
        ----------
        X :  (3,) array
            3 theta points with which to start minimisation
        theta : float or int
            Value of theta at which to evaluate NLL
        conv_condition : float or int
            value for epsilon at which convergence is accepted
        max_iter : int, optional
            Max mumber of iterations. The default is 1e6.

        Returns
        -------
        x3 : np.float64
            Minimum value of parabola approximated by three deltamsq points given.
        epsilon : np.float64
            Final value of fractional difference in x3.
        c : int
            Number of iterations.
        parabola : np.array shape (2, 3)
            (deltamsq, NLL points) for final + 1th iteration .

        """
        NLL_args = {'thetas':theta}
        if max_iter is None:
            max_iter = 1e6
        x3, epsilon, c, parabola  = parabolic_minimisation(self.NLL_params, X, 'deltamsqs', NLL_args, conv_condition, max_iterations = max_iter)
        return x3, epsilon, c, parabola
    
    def parabolic_minimisation_NLL_theta(self, X, deltamsq, conv_condition, max_iter = None):
        """
        Finds minimum value of NLL function using Parabolic minimisation in the theta direction

        Parameters
        ----------
        X :  (3,) array
            3 theta points with which to start minimisation
        deltamsq : float or int
            Value of deltamsq at which to evaluate NLL
        conv_condition : float or int
            value for epsilon at which convergence is accepted
        max_iter : int, optional
            Max mumber of iterations. The default is 1e6.

        Returns
        -------
        x3 : np.float64
            Minimum value of parabola approximated by three theta points given.
        epsilon : np.float64
            Final value of fractional difference in x3.
        c : int
            Number of iterations.
        parabola : np.array shape (2, 3)
            (theta, NLL points) for final + 1th iteration .

        """

        NLL_args = {'deltamsqs':deltamsq}
        if max_iter is None:
            max_iter = 1e6
        x3, epsilon, c, parabola  = parabolic_minimisation(self.NLL_params, X, 'thetas', NLL_args, conv_condition, max_iterations = max_iter)
        return x3, epsilon, c, parabola


def lagrange_interp(x_eval, x, f):
    """
    Performs lagrange interpolation

    Parameters
    ----------
    x_eval : np.array - points at which to evaluate the function f(x)
    x : np.array - given points x_i
    f : np.array - given points f_i


    Returns
    -------
    P : np.array, length = len(x_eval)
        interpolated points

    """
    if len(x) != len(f):
        raise ValueError(f"Cannot broadcast arrays of different shapes x {len(x)}, f {len(f)}")
        
    N = len(x)
    if isinstance(x_eval, (float, np.float64, int)):
        x_eval = [x_eval]
    M = len(x_eval)
    P = np.zeros(M)
    for m in range(M):
        for i in range(N):
            S = 1
            for j in range(N):
                if i == j:
                    continue
                S *= (x_eval[m] - x[j])/(x[i] - x[j])
            P[m] += S*f[i]
    return P


