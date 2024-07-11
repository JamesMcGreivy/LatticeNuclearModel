import torch
import torch.nn as nn
import numpy as np
import itertools

class NuclearLattice(nn.Module):
    
    """
    Class representing a nuclear lattice simulation
    """
    def __init__(self, N, Z, lattice_width=10, state_space_dim=2):
        super().__init__()
        self.N = N
        self.Z = Z 
        self.A = N + Z
        
        self.lattice_width = lattice_width
        self.state_space_dim = state_space_dim
        self.boundaries = torch.tensor(state_space_dim * [[-lattice_width//2, lattice_width//2]] + [[-0.5, 0.5]])
        self.lattice_shape = torch.tensor([int(bound[1] - bound[0] + 1) for bound in self.boundaries])
        
        self.states = self._dense_initialization(N, Z, state_space_dim)
    
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
    Pauli blocking prevents any two nucleons with identical quantum numbers
    """
    def _pauli_blocking(self, state1, state2):
        state_dist = torch.linalg.norm(state1 - state2)
        if state_dist < 1e-3:
            return True
        else:
            return False
        
    """
    Boundary blocking prevents any nucleon from escaping the lattice boundaries
    """
    def _boundary_blocking(self, state):
        if torch.any(state[0:-1] < self.boundaries[:,0]) or torch.any(state[0:-1] > self.boundaries[:,1]):
            return True
        else:
            return False

    """
    Takes an interaction term f_interaction(r1, s1, i1, r2, s2, i2) -> energy and computes E_i = sum_j f_interaction(r_i, s_i, i_i, r_j, s_j, i_j)
    """
    def _get_E_i(self, i, state_i, f_interaction):
        # Out of bounds has an infinite potential barrier
        if self._boundary_blocking(state_i):
            return torch.inf
        
        r_i = state_i[0:self.state_space_dim]
        spin_i = state_i[-2]
        isospin_i = state_i[-1]

        E_i = 0.0
        for j in range(self.A):
            if i == j:
                continue
            
            # Pauli exclusion principle gives each occupied state a delta function potential
            if self._pauli_blocking(state_i, self.states[j]):
                return torch.inf
            
            r_j = self.states[j,0:self.state_space_dim]
            spin_j = self.states[j,-2]
            isospin_j = self.states[j,-1]
            E_i += f_interaction(r_i, spin_i, isospin_i, r_j, spin_j, isospin_j)
        return E_i

    """
    For every point in the lattice, 
    temporarily moves nucleon i to that point and computes E_i from that point.
    """
    def _get_E_i_lattice(self, i, f_interaction):
        state = self.states[i]
        
        E_i_lattice = torch.zeros(*self.lattice_shape)
        # iterates over every possible state space position and spin within bounds
        for lattice_idx in itertools.product(*[range(shape) for shape in self.lattice_shape]):
            state[0:-1] = self.boundaries[:,0] + torch.tensor(lattice_idx)
            E_i_lattice[lattice_idx] = self._get_E_i(i, state, f_interaction)

        return E_i_lattice        
    
    """
    For every adjacent point, 
    temporarily moves nucleon i to that point and computes E_i from that point.
    """
    def _get_E_i_adjacent(self, i, f_interaction, num_adjacent=1):
        state = self.states[i]
        # Make spin down to start so that offset can add 0 (down) or 1 (up)
        state[-2] = -0.5
        
        search_dim = 1 + 2*num_adjacent
        adjacency_shape = [search_dim]*self.state_space_dim + [2]
        E_i_adjacent = torch.zeros(adjacency_shape)
        
        # iterates over every possible state space position and spin within bounds
        for idx in itertools.product(*[range(shape) for shape in adjacency_shape]):
            offset = torch.tensor(idx + (0,))
            # Make sure spin offset stays either -1 or 0
            offset[0:self.state_space_dim] -= num_adjacent
            offset_state = state + offset
            E_i_adjacent[idx] = self._get_E_i(i, offset_state, f_interaction)

        return E_i_adjacent
    
    """
    
    """
    def step(self, f_interaction, num_adjacent=1):
        search_dim = 1 + 2*num_adjacent
        for i in range(self.A):
            E_i_adjacent = self._get_E_i_adjacent(
                i, 
                f_interaction, 
                num_adjacent
            )
            best_idx = torch.unravel_index(
                torch.argmin(E_i_adjacent), 
                E_i_adjacent.shape
            )
            best_offset = torch.tensor(best_idx + (0,))
            best_offset[0:self.state_space_dim] -= num_adjacent
            self.states[i] += best_offset
        return self.E_tot(f_interaction)

    """
    Computes the total energy of the system
    """
    def E_tot(self, f_interaction):
        E_tot = 0.0
        for i in range(self.A):
            if self._boundary_blocking(self.states[i]):
                return torch.inf
            
            for j in range(i, self.A):
                if i == j:
                    continue
                if self._pauli_blocking(self.states[i], self.states[j]):
                    return torch.inf
                
                r_i, spin_i, isospin_i = self.states[i,0:self.state_space_dim], self.states[i,-2], self.states[i,-1]
                r_j, spin_j, isospin_j = self.states[j,0:self.state_space_dim], self.states[j,-2], self.states[j,-1]
                E_tot += f_interaction(r_i, spin_i, isospin_i, r_j, spin_j, isospin_j)
        return E_tot
