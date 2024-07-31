import torch
import torch.nn as nn

class MeanFieldGenerator(nn.Module):
    
    def __init__(self, rho_max, z_max, max_range, device='cpu'):
        super().__init__()
        self.max_range = max_range
        self.device = device
        
        RHO = torch.arange(1, rho_max + 1, 1, device=device)
        Z = torch.arange(-z_max, z_max + 1, 1, device=device)
        self.RHO, self.Z = torch.meshgrid(RHO.float(), Z.float(), indexing='ij')
        
        self.r_in_range = self.RHO**2.0 + self.Z**2.0 <= self.max_range**2.0
        self.mask = torch.logical_and(torch.abs(self.RHO) <= self.max_range, torch.abs(self.Z) <= self.max_range)
        
        self.lin1 = nn.Linear(
            2,
            4*(self.max_range)*(2*self.max_range + 1), 
            device=device
        )
        self.lin2 = nn.Linear(
            4*(self.max_range)*(2*self.max_range + 1), 
            2*(self.max_range)*(2*self.max_range + 1), 
            device=device
        )
        self.output = nn.Linear(
            2*(self.max_range)*(2*self.max_range + 1),
            (self.max_range)*(2*self.max_range + 1),
            device=device
        )
        
        self.relu = nn.ReLU()
        
    def forward(self, num_nucleons):
        field = self.relu(self.lin1(num_nucleons.float()))
        field = self.relu(self.lin2(field))
        field = self.output(field)
        field = field.flatten(start_dim=1)
        field_extended = torch.zeros(len(num_nucleons), *self.mask.shape, device=field.device)
        field_extended[:,self.mask.to(field.device)] = field
        return field_extended * self.r_in_range.to(field_extended.device)
        
        
        
                              