import torch
import numpy as np
import matplotlib.pyplot as plt
import itertools
from torch.multiprocessing import Pool

from lattice import NuclearLattice
from interactions import build_full_interaction
from data.data import get_nuclear_data

from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

def get_equilibrium_binding_mse(args):
    N, Z, a_list_mm, a_list_mp, a_list_pm, a_list_pp, a_list_coulomb, device = args

    V_interaction = build_full_interaction(a_list_mm, a_list_mp, a_list_pm, a_list_pp, a_list_coulomb)

    # ~~ ! Lattice Parameters ! ~~ #
    bounds = [[-4, 4], [-4, 4], [-4, 4]]
    lattice = NuclearLattice(Z, N, V_interaction, bounds, device=device)

    E_tot = torch.inf
    while True:
        lattice.step()
        E_tot_new = lattice.E_tot()
        if E_tot_new < E_tot:
            E_tot = E_tot_new
        elif E_tot_new > E_tot + 1e-6:
            print("ERROR")
        else:
            break

    dat = get_nuclear_data()
    E_tot_actual = -float(dat[(dat['z'] == Z) & (dat['n'] == N)]['binding'].iloc[0])
    return (E_tot - E_tot_actual)**2.0

def optimize_interactions(args):
    a_mm_0, a_mm_1, a_mp_0, a_mp_1, a_pm_0, a_pm_1, a_pp_1, a_c_0, a_c_1 = args
    
    # The value of a_pp_0 is arbitrary since two nucleons with the same spin and isospin
    # cannot exist at zero distance
    a_pp_0 = a_pp_1
    
    # Package into lists
    a_list_mm = [a_mm_0, a_mm_1]
    a_list_mp = [a_mp_0, a_mp_1]
    a_list_pm = [a_pm_0, a_pm_1]
    a_list_pp = [a_pp_0, a_pp_1]
    a_list_coulomb = [a_c_0, a_c_1]
    
    # ~~ ! Parameters ! ~~ #
    z_min = 12
    z_max = 14
    device = "cpu"
    processes = 70
    
    dat = get_nuclear_data()
    dat = dat[(dat['z'] >= z_min) & (dat['z'] <= z_max)]
    
    if device in "cpu":
        with Pool(processes) as p:
            result = p.map(
                get_equilibrium_binding_mse, 
                ((N, Z, a_list_mm, a_list_mp, a_list_pm, a_list_pp, a_list_coulomb, device) for Z, N in zip(dat['z'], dat['n']))
            )
    elif device in "cuda":
        result = []
        for Z, N in zip(dat['z'], dat['n']):
            result.append(get_equilibrium_binding_mse((N, Z, a_list_mm, a_list_mp, a_list_pm, a_list_pp, a_list_coulomb, device)))
        result = np.array(result)
    
    print(f"MSE: {np.mean(result):.3f}")
    return np.mean(result)

from scipy.optimize import minimize
    
if __name__ == "__main__":
    # a_mm_0, a_mm_1, a_mp_0, a_mp_1, a_pm_0, a_pm_1, a_pp_1, a_c_0, a_c_1
    initial_guess = [-2, -1, -2, -1, -2, -1, -1, 1, 0.5]
    bounds = ((-10, 0), 
              (-10, 0), 
              (-10, 0), 
              (-10, 0), 
              (-10, 0),
              (-10, 0),
              (-10, 0),
              (0, 10),
              (0, 10))
    # Bounded region of parameter space
    result = minimize(optimize_interactions, initial_guess, bounds=bounds, options={'eps':0.1})
    print(result.x)