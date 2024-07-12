import torch
import torch.nn as nn
import numpy as np
import itertools

class NuclearLattice(nn.Module):
    
    """
    Class representing a nuclear lattice simulation.
    -> Z : number of protons
    -> N : number of neutrons
    -> f_interaction(r_i, spin_i, isospin_i, r_j, spin_j, isospin_j) : interaction term
    """
    def __init__(self, Z, N, f_interaction, lattice_width=10, state_space_dim=2):
        super().__init__()
        self.N = N
        self.Z = Z 
        self.A = N + Z
        self.f_interaction = self._wrap_f_interaction(f_interaction)
        
        self.lattice_width = lattice_width
        self.state_space_dim = state_space_dim
        self.boundaries = torch.tensor(state_space_dim * [[-lattice_width//2, lattice_width//2]] + 2 * [[-0.5, 0.5]])
        self.shape = torch.tensor([int(bound[1] - bound[0] + 1) for bound in self.boundaries])
        
        self.states = self._dense_initialization(self.N, self.Z, self.state_space_dim) # a list of every particle's position / spin / isospin
        self.mean_field = torch.tensor(torch.prod(self.shape) * [float('nan')]).reshape(*self.shape) # contains an index for every possible position / spin / isospin, initially all nans
    
    """
    Pauli blocking prevents any two nucleons with identical quantum numbers
    """
    def _pauli_blocking(self, r1, spin1, isospin1, r2, spin2, isospin2):
        state_dist = torch.linalg.norm(r1 - r2) + torch.abs(spin1 - spin2) + torch.abs(isospin1 - isospin2)
        if state_dist < 1e-3:
            return 1e6
        else:
            return 0.0

    """
    """
    def _wrap_f_interaction(self, f_interaction):
        def new_f_interaction(r1, spin1, isospin1, r2, spin2, isospin2):
            return self._pauli_blocking(r1, spin1, isospin1, r2, spin2, isospin2) + f_interaction(r1, spin1, isospin1, r2, spin2, isospin2)
        return new_f_interaction
    
    """
    Initializes every neutron and proton directly at the origin with spin = isospin
    """
    def _dense_initialization(self, N, Z, state_space_dim):      
        neutron_positions = torch.zeros((N, state_space_dim))
        neutron_spins = torch.zeros((N, 1)) - 0.5
        neutron_isospins = torch.zeros((N, 1)) - 0.5
        neutron_states = torch.hstack((neutron_positions, neutron_spins, neutron_isospins))

        proton_positions = torch.zeros((Z, state_space_dim))
        proton_spins = torch.zeros((Z, 1)) + 0.5
        proton_isospins = torch.zeros((Z, 1)) + 0.5
        proton_states = torch.hstack((proton_positions, proton_spins, proton_isospins))
        
        return torch.vstack((neutron_states, proton_states))

    """
    """
    def _calc_field_at(self, site):
        r_site, spin_site, isospin_site = site[0:self.state_space_dim], site[-2], site[-1]
        field = 0.0
        for state in self.states:
            r_i, spin_i, isospin_i = state[0:self.state_space_dim], state[-2], state[-1]
            field += self.f_interaction(r_i, spin_i, isospin_i, r_site, spin_site, isospin_site)
        return field

    """
    """
    def get_mean_field_at(self, site):
        idx = (site - self.boundaries[:,0]).int()
        if torch.isnan(self.mean_field[*idx]):
            self.mean_field[*idx] = self._calc_field_at(site)
        return self.mean_field[*idx]

    """
    """
    def set_mean_field_at(self, site, value):
        idx = (site - self.boundaries[:,0]).int()
        self.mean_field[*idx] = value

    
    """
    Prevents any nucleon from escaping the lattice boundaries
    """
    def _outside_boundaries(self, state):
        if torch.any(state < self.boundaries[:,0]) or torch.any(state > self.boundaries[:,1]):
            return True
        else:
            return False
    
    """
    """
    def step(self, dist=1):
        for i, state in enumerate(self.states):
            r, spin, isospin = state[0:self.state_space_dim], state[-2], state[-1]
            
            # iterate over every lattice site within dist in state space, any spin, same isospin, finding the lowest energy lattice site
            f_new = np.inf
            site_new = state
            bounds = self.state_space_dim * [range(2*dist + 1)] + [range(2)]
            for idx in itertools.product(*bounds):
                site = torch.tensor(idx + (isospin,))
                site[0:self.state_space_dim] += (r - dist)
                site[-2] -= 0.5
                
                f = self.get_mean_field_at(site)
                if f < f_new and not self._outside_boundaries(site):
                    f_new = f
                    site_new = site

            r_new, spin_new, isospin_new = site_new[0:self.state_space_dim], site_new[-2], site_new[-1]
            
            active_indexes = torch.vstack(torch.where(torch.logical_not(torch.isnan(self.mean_field)))).T
            for idx in active_indexes:
                site = self.boundaries[:,0] + idx
                r_site, spin_site, isospin_site = site[0:self.state_space_dim], site[-2], site[-1]
                self.mean_field[*idx] -= self.f_interaction(r, spin, isospin, r_site, spin_site, isospin_site)
                self.mean_field[*idx] += self.f_interaction(r_new, spin_new, isospin_new, r_site, spin_site, isospin_site)

            self.states[i] = site_new

    """
    Computes the total energy of the system
    """
    def E_tot(self):
        E_tot = 0.0
        for i in range(self.A):
            for j in range(i, self.A):
                if i == j:
                    continue
                r_i, spin_i, isospin_i = self.states[i,0:self.state_space_dim], self.states[i,-2], self.states[i,-1]
                r_j, spin_j, isospin_j = self.states[j,0:self.state_space_dim], self.states[j,-2], self.states[j,-1]
                E_tot += self.f_interaction(r_i, spin_i, isospin_i, r_j, spin_j, isospin_j)
        return E_tot
