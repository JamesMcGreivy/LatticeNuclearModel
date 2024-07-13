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
    x_boundaries = torch.tensor([-torch.inf] + [(lattice_distances[i] + lattice_distances[i + 1])/2.0 for i in range(len(a_list))])
    y_list = torch.tensor(a_list)
    def V(r):
        near = torch.sum(torch.stack([y * torch.logical_and(r > x_boundaries[i], r <= x_boundaries[i + 1]) for i, y in enumerate(y_list)]), axis = 0)
        #far = torch.nan_to_num(x_boundaries[-1]**2.0 * y_list[-1] / r**2.0) * (r > x_boundaries[-1])
        return near# + far
    return V

def get_coulomb_interaction(a_list):
    a_1, a_2 = a_list
    def V(r):
        distant = (r > 0.999)
        return (a_1 * torch.logical_not(distant)) + (torch.nan_to_num(a_2 / r**2.0) * distant)
    return V

def build_full_interaction(a_list_mm, a_list_mp, a_list_pm, a_list_pp, a_list_coulomb):
    V_nuclear_mm = get_nuclear_interaction(a_list_mm)
    V_nuclear_mp = get_nuclear_interaction(a_list_mp)
    V_nuclear_pm = get_nuclear_interaction(a_list_pm)
    V_nuclear_pp = get_nuclear_interaction(a_list_pp)
    V_coulomb = get_coulomb_interaction(a_list_coulomb)
    def V_interaction(state, lattice):
        pos, spin, isospin = state[0:-2], state[-2], state[-1]
        lattice_pos, lattice_spin, lattice_isospin = lattice[0:-2], lattice[-2], lattice[-1]
        # Computes the pythagorean distance between the state space position and every point on the lattice
        r = torch.sqrt(torch.sum(torch.stack([(pos[i] - lattice_pos[i])**2.0 for i in range(len(pos))]), axis = 0))
        s_dot_s = spin * lattice_spin
        i_dot_i = isospin * lattice_isospin
        V = (
            V_nuclear_mm(r) * torch.logical_and(s_dot_s < 0.0, i_dot_i < 0.0) 
            + V_nuclear_mp(r) * torch.logical_and(s_dot_s < 0.0, i_dot_i > 0.0) 
            + V_nuclear_pm(r) * torch.logical_and(s_dot_s > 0.0, i_dot_i < 0.0) 
            + V_nuclear_pp(r) * torch.logical_and(s_dot_s > 0.0, i_dot_i > 0.0)
        )
        if isospin > 0:
            V += V_coulomb(r) * (lattice_isospin > 0.0)
        return V
    return V_interaction
            
            