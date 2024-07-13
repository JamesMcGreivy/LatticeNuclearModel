import torch
import torch.nn as nn
import numpy as np
import itertools

class NuclearLattice(nn.Module):
    """
    Class representing a nuclear lattice simulation.
    -> Z : number of protons
    -> N : number of neutrons
    -> V_interaction(state, lattice) : given interaction between state and the lattice of all possible states
    """
    def __init__(self, Z, N, V_interaction, boundaries=[[-10, 10], [-10, 10], [-10, 10]], device="cpu"):
        super().__init__()
        self.device = device
        
        self.N = N
        self.Z = Z
        self.A = N + Z
        self.V_interaction = V_interaction
        
        self.states = self._dense_init(self.N, self.Z, len(boundaries)) # a list of every particle's position / spin / isospin
        
        boundaries = torch.tensor(boundaries, device=device)
        self.boundaries = torch.concat(
            (boundaries, torch.tensor([[-0.5, 0.5], [-0.5, 0.5]], device=device))
        ) # need to add on the boundary conditions for spin and isospin
        self.shape = torch.tensor([int(bound[1] - bound[0] + 1) for bound in self.boundaries], device=device)
        
        # a tuple of meshes which track the position (lattice[0:-2]), spin (lattice[-2]), and isospin (lattice[-1]) of each point in the lattice
        self.lattice = torch.meshgrid(
            *[torch.arange(bound[0], bound[1] + 1, device=device) for bound in self.boundaries], indexing='ij'
        )

        # For every possible lattice site:
        # - mean_field stores the potential energy at that lattice site
        # - population stores the number of particles at that lattice site
        self.mean_field, self.population = self._init_mean_field(self.states, self.V_interaction)

    """
    Initializes every neutron and proton directly at the origin with spin = isospin
    """
    def _dense_init(self, N, Z, state_space_dim):      
        neutron_positions = torch.zeros((N, state_space_dim), device=self.device)
        neutron_spins = torch.zeros((N, 1), device=self.device) - 0.5
        neutron_isospins = torch.zeros((N, 1), device=self.device) - 0.5
        neutron_states = torch.hstack((neutron_positions, neutron_spins, neutron_isospins))

        proton_positions = torch.zeros((Z, state_space_dim), device=self.device)
        proton_spins = torch.zeros((Z, 1), device=self.device) + 0.5
        proton_isospins = torch.zeros((Z, 1), device=self.device) + 0.5
        proton_states = torch.hstack((proton_positions, proton_spins, proton_isospins))
        
        return torch.vstack((neutron_states, proton_states))

    """
    """
    def _init_mean_field(self, states, V_interaction):
        mean_field = torch.zeros(*self.shape, device=self.device)
        population = torch.zeros(*self.shape, device=self.device)
        for state in states:
            mean_field += V_interaction(state, self.lattice)
            idx = (state - self.boundaries[:,0]).int()
            population[*idx] += 1
        return mean_field, population

    """
    """
    def get_mean_field_at(self, state):
        return 2
    
    """
    """
    def step(self):
        # Helper tensor
        zero_lattice = torch.zeros(*self.shape, device=self.device)
        
        E_tot = 0.0
        for i, state in enumerate(self.states):
            idx = (state - self.boundaries[:,0]).int()
            
            zero_lattice[*idx] = 1
            pauli_blocking = 1e12 * (self.population - zero_lattice)
            zero_lattice[*idx] = 0

            # Remove the contribution of this state to the mean field
            new_mean_field = self.mean_field - self.V_interaction(state, self.lattice)
            # the potential that state sees is the new mean field + pauli blocking
            V_state = new_mean_field + pauli_blocking
            
            isospin_idx = (state[-1] - self.boundaries[-1,0]).int()
            new_idx = torch.tensor(
                torch.unravel_index(V_state[...,isospin_idx].argmin(), V_state.shape[0:-1]) + (isospin_idx,), device=self.device
            )

            # If the new index is equal to the old index, don't do anything
            if torch.sum(torch.abs(new_idx - idx)) < 0.5:
                continue

            new_state = self.boundaries[:,0] + new_idx

            self.population[*idx] -= 1
            self.population[*new_idx] += 1
            self.mean_field = new_mean_field + self.V_interaction(new_state, self.lattice)
            self.states[i] = new_state

    """
    Computes the total energy of the system
    """
    def E_tot(self):
        E_tot = 0.0
        for state in self.states:
            idx = (state - self.boundaries[:,0]).int()
            slices = [slice(i, i+1) for i in idx]
            
            pauli_blocking = 1e12 * (self.population[*idx] - 1)
            E_tot += self.mean_field[*idx] - self.V_interaction(state, [lat[*slices] for lat in self.lattice]) + pauli_blocking
        return E_tot.item()

            


        