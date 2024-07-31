import torch
import torch.nn as nn

class CylindricalSchrodingerSolver(nn.Module):
    
    def __init__(self, rho_max, z_max, device='cpu'):
        super().__init__()
        self.device = device
        self.rho_max = rho_max
        self.z_max = z_max
        
        RHO = torch.arange(1, rho_max + 1, 1, device=device)
        Z = torch.arange(-z_max, z_max + 1, 1, device=device)
        self.RHO, self.Z = torch.meshgrid(RHO.float(), Z.float(), indexing='ij')
        self.dd_rho, self.inv_rho_d_rho, self.dd_z = self._init_finite_differences(self.RHO, self.Z, device)
        self.laplacian = -(self.dd_rho + self.inv_rho_d_rho + self.dd_z)
        import torch.linalg as linalg
        self.linalg = linalg
        
    def _init_finite_differences(self, RHO, Z, device='cpu'):
        dd_rho = torch.zeros((RHO.shape[0], RHO.shape[1], RHO.shape[0], RHO.shape[1]), device=device)
        inv_rho_d_rho = torch.zeros((RHO.shape[0], RHO.shape[1], RHO.shape[0], RHO.shape[1]), device=device)
        dd_z = torch.zeros((RHO.shape[0], RHO.shape[1], RHO.shape[0], RHO.shape[1]), device=device)

        for i in range(RHO.shape[0]):
            for j in range(RHO.shape[1]):
                # Central difference for dd_rho
                if i-1 >= 0:
                    dd_rho[i,j,i-1,j] = 1.0
                dd_rho[i,j,i,j] = -2.0
                if i+1 <= RHO.shape[0] - 1:
                    dd_rho[i,j,i+1,j] = 1.0

                # Forward difference for d_rho
                inv_rho_d_rho[i,j,i,j] = (1.0/RHO[i,j]) * -1.0
                if i+1 <= RHO.shape[0] - 1:
                    inv_rho_d_rho[i,j,i+1,j] = (1.0/RHO[i,j]) * 1.0

                # Central difference for dd_z
                if j-1 >= 0:
                    dd_z[i,j,i,j-1] = 1.0
                dd_z[i,j,i,j] = -2.0
                if j+1 <= RHO.shape[1] - 1:
                    dd_z[i,j,i,j+1] = 1.0

        dd_rho = dd_rho.flatten(start_dim=2, end_dim=3).flatten(start_dim=0, end_dim=1)
        inv_rho_d_rho = inv_rho_d_rho.flatten(start_dim=2, end_dim=3).flatten(start_dim=0, end_dim=1)
        dd_z = dd_z.flatten(start_dim=2, end_dim=3).flatten(start_dim=0, end_dim=1)
        return dd_rho, inv_rho_d_rho, dd_z
    
    # mean_fields - [batch_size x rho_max x 2*z_max]
    def _solve_energy_eigenstates(self, mean_fields, return_eigstates=False, max_alpha=3):
        
        batches = len(mean_fields)
        
        V_hat_batched = torch.diag_embed(mean_fields.flatten(start_dim=1)).to(mean_fields.device)
        inv_rho_batched = torch.stack(batches*[torch.diag(1.0/self.RHO.flatten())]).to(mean_fields.device)
        laplacian_batched = torch.stack(batches*[self.laplacian]).to(mean_fields.device)
        E_hat_batched = torch.zeros(0, *self.laplacian.shape, device=mean_fields.device)
        
        alphas = list(range(max_alpha+1))
        for alpha in alphas:
            if alpha == 0:
                V_eff_batched = V_hat_batched
            else:
                V_eff_batched = V_hat_batched + (alpha * inv_rho_batched)**2.0
            E_hat_batched = torch.concat((E_hat_batched, laplacian_batched + V_eff_batched))
            
        if return_eigstates:
            eigvals, eigvecs = self.linalg.eig(E_hat_batched)
            eigvals, eigvecs = eigvals.real, (eigvecs.abs()**2.0).transpose(dim0=-1, dim1=-2)
        else:
            eigvals = self.linalg.eigvals(E_hat_batched)
            eigvals = eigvals.real
        
        eigvals = torch.hstack([eigvals[i*batches:(i+1)*batches] for i in range(max_alpha+1)])
        if return_eigstates:
            eigvecs = torch.hstack([eigvecs[i*batches:(i+1)*batches] for i in range(max_alpha+1)])
        
        sort = eigvals.sort()
        sorted_eigvals = sort.values
        if return_eigstates:
            return sort.values, sort.indices, eigvecs.reshape(batches, sorted_eigvals.shape[1], mean_fields.shape[1], mean_fields.shape[2])
        else:
            return sort.values
        
    def _num_nucleons_to_mask(self, num_nucleons, mask_shape):
        mask = torch.zeros(mask_shape, device=num_nucleons.device)
        zero = torch.tensor([0], device=num_nucleons.device)
        two = torch.tensor([2], device=num_nucleons.device)
        four = torch.tensor([4], device=num_nucleons.device)
        for i, col in enumerate(range(mask.shape[1])):
            # Two-fold degeneracy for alpha == 0
            if i == 0:
                mask[:,col] = torch.minimum(torch.maximum(num_nucleons[:,0] - 2*i, zero), two)
            # Four-fold degeneracy for alpha > 0
        return mask
    
    # num_nucleons - [batch_size x 1]
    def forward(self, num_nucleons, mean_fields, max_alpha=4):
        eigvals = self._solve_energy_eigenstates(mean_fields, return_eigstates=False, max_alpha=max_alpha)
        eigval_mask = self._num_nucleons_to_mask(num_nucleons, eigvals.shape)
        binding_energies = torch.sum(eigvals * eigval_mask, dim=1).unsqueeze(dim=-1)
        return binding_energies