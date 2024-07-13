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
    bounds = [[-8, 8], [-8, 8], [-8, 8]]
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

    #print(f"Z : {Z}, N : {N}, E_tot (pred) : {E_tot:.3f}, E_tot (actual) : {E_tot_actual:.3f}")
    return ((E_tot - E_tot_actual)/(N + Z))**2.0

def optimize_interactions(
    a_mm_0, a_mm_1, a_mm_2, a_mm_3,
    a_mp_0, a_mp_1, a_mp_2, a_mp_3,
    a_pm_0, a_pm_1, a_pm_2, a_pm_3,
            a_pp_1, a_pp_2, a_pp_3,
    a_c_1, a_c_2
):
    # The value of a_pp_0 is arbitrary since two nucleons with the same spin and isospin
    # cannot exist at zero distance
    a_pp_0 = a_pp_1
    
    # Package into lists
    a_list_mm = [a_mm_0, a_mm_1, a_mm_2, a_mm_3]
    a_list_mp = [a_mp_0, a_mp_1, a_mp_2, a_mp_3]
    a_list_pm = [a_pm_0, a_pm_1, a_pm_2, a_pm_3]
    a_list_pp = [a_pp_0, a_pp_1, a_pp_2, a_pp_3]
    a_list_coulomb = [a_c_1, a_c_2]
    
    # ~~ ! Parameters ! ~~ #
    z_min = 8
    z_max = 26
    device = "cpu"
    processes = 90
    
    dat = get_nuclear_data()
    dat = dat[(dat['z'] >= z_min) & (dat['z'] <= z_max)]
    
    if device in "cpu":
        with Pool(processes) as p:
            result = p.map(
                get_equilibrium_binding_mse, 
                ((N, Z, a_list_mm, a_list_mp, a_list_pm, a_list_pp, a_list_coulomb, device) for Z, N in zip(dat['z'], dat['n']))
            )
        
        print(f"MSE: {np.mean(result):.3f}")
        return -np.mean(result)
    
    if device in "cuda":
        result = []
        for Z, N in zip(dat['z'], dat['n']):
            result.append(get_equilibrium_binding_mse((N, Z, a_list_mm, a_list_mp, a_list_pm, a_list_pp, a_list_coulomb, device)))
        result = np.array(result)
        print(f"MSE: {np.mean(result):.3f}")
        return -np.mean(result)

if __name__ == "__main__":
    # Bounded region of parameter space
    pbounds = {
        'a_mm_0': (-40, 20), 
        'a_mm_1': (-40, 20),
        'a_mm_2': (-40, 20),
        'a_mm_3': (-40, 20),
        
        'a_mp_0': (-40, 20),
        'a_mp_1': (-40, 20),
        'a_mp_2': (-40, 20),
        'a_mp_3': (-40, 20),
        
        'a_pm_0': (-40, 20),
        'a_pm_1': (-40, 20),
        'a_pm_2': (-40, 20),
        'a_pm_3': (-40, 20),
        
        'a_pp_1': (-40, 20),
        'a_pp_2': (-40, 20),
        'a_pp_3': (-40, 20),
        
        'a_c_1' : (0, 20),
        'a_c_2' : (0, 10),
    }

    bounds_transformer = SequentialDomainReductionTransformer(eta=0.95, minimum_window=1e-4)
    optimizer = BayesianOptimization(
        f=optimize_interactions, 
        pbounds=pbounds,
        random_state=5,
        bounds_transformer=bounds_transformer,
        verbose=1,
    )
    
    load_logs(optimizer, logs=["./logs.log.json"])
    logger = JSONLogger(path="./logs5.log", reset=False)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    optimizer.maximize(init_points=0, n_iter=800)