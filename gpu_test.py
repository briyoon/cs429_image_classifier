import torch
print(torch.backends.mps.is_built())

mps_device = torch.device("mps")
z = torch.ones(5, device=mps_device)