import torch
import matplotlib.pyplot as plt
device = 'cuda:0'

import schrodinger_solvers
import generator
from data import data

dat_ = data.get_nuclear_data()[12:74]
num_nucleons = torch.tensor(dat_[['z','n']].to_numpy(), device=device).float()
Z = num_nucleons[:,0].unsqueeze(1)
N = num_nucleons[:,1].unsqueeze(1)
binding = -torch.tensor(dat_['binding'].to_numpy(), device=device).unsqueeze(1).float()

rho_max = 12
z_max = 12
max_range = 7

solver = schrodinger_solvers.CylindricalSchrodingerSolver(rho_max, z_max, device=device)
coulomb_field_generator = generator.MeanFieldGenerator(rho_max, z_max, max_range, device=device)
strong_field_generator = generator.MeanFieldGenerator(rho_max, z_max, max_range, device=device)

#coulomb_field_generator.load_state_dict(torch.load(f"saves/coulomb_field_generator-max_range={max_range}"))
#strong_field_generator.load_state_dict(torch.load(f"saves/strong_field_generator-max_range={max_range}"))

# Parallelize data across multiple GPUs
solver_parallel = torch.nn.DataParallel(solver, [0,1,2,3])

params = list(coulomb_field_generator.parameters()) + list(strong_field_generator.parameters())
optimizer = torch.optim.AdamW(params, lr=0.001, weight_decay=0.01)
loss = torch.nn.MSELoss()

while True:
    try:
        coulomb_field = -torch.abs(coulomb_field_generator(num_nucleons))
        strong_field = strong_field_generator(num_nucleons)

        proton_field = coulomb_field + strong_field
        neutron_field = strong_field

        binding_proton = solver_parallel(Z, proton_field, max_alpha=4)
        binding_neutron = solver_parallel(N, neutron_field, max_alpha=4)
        binding_pred = binding_proton + binding_neutron

        l = loss(binding, binding_pred)

        optimizer.zero_grad()
        l.backward()

        optimizer.step()
        print(l)

        torch.save(coulomb_field_generator.state_dict(), f"saves/coulomb_field_generator-max_range={max_range}")
        torch.save(strong_field_generator.state_dict(), f"saves/strong_field_generator-max_range={max_range}")
    except Exception as e:
        print(e)
        continue
