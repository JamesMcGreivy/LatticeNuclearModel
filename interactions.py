import torch
import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.interpolate import interp1d

# Need 5 distinct interaction terms:
# - f_coulomb
# - f_nuclear_mm
# - f_nuclear_pm
# - f_nuclear_mp
# - f_nuclear_pp

lattice_distances = [0, 1, np.sqrt(2), 2, np.sqrt(5), np.sqrt(8), 3, np.sqrt(10)] #can add more...
def get_nuclear_interaction(a_list):
    x_list = torch.tensor([lattice_distances[i] for i in range(len(a_list))])
    y_list = torch.tensor(a_list)
    
    def f(r):
        if r > x_list[-1]:
            return y_list[-1] / (r - x_list[-1] + 1)**2.0
        diff = np.abs(x_list - r)
        return y_list[diff.argmin()]
    return f

def get_coulomb_interaction(a_list):
    a_1, a_2 = a_list
    
    def f(r):
        if r < 0.999:
            return a_1
        else:
            return a_2 / r**2.0
    return f

def get_full_interaction(a_list_mm, a_list_mp, a_list_pm, a_list_pp, a_list_coulomb):
    f_nuclear_mm = get_nuclear_interaction(a_list_mm)
    f_nuclear_mp = get_nuclear_interaction(a_list_mp)
    f_nuclear_pm = get_nuclear_interaction(a_list_pm)
    f_nuclear_pp = get_nuclear_interaction(a_list_pp)
    f_coulomb = get_coulomb_interaction(a_list_coulomb)
    
    def f_interaction(r1, spin1, isospin1, r2, spin2, isospin2):
        r = torch.linalg.norm(r1 - r2)
        
        s_dot_s = spin1 * spin2
        i_dot_i = isospin1 * isospin2
        
        E = 0.0
        if s_dot_s < 0.0 and i_dot_i < 0.0:
            E += f_nuclear_mm(r)
        if s_dot_s < 0.0 and i_dot_i > 0.0:
            E += f_nuclear_mp(r)
        if s_dot_s > 0.0 and i_dot_i < 0.0:
            E += f_nuclear_pm(r)
        if s_dot_s > 0.0 and i_dot_i > 0.0:
            E += f_nuclear_pp(r)
        if isospin1 > 0.0 and isospin2 > 0.0:
            E += f_coulomb(r)
        return E
    
    return f_interaction
            
            