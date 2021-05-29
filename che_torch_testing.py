from tools import che_torch
import torch

p = che_torch.Props(['Ethanol','Isopropanol', 'Water'])
x= torch.tensor([1/3,1/3,1/3],requires_grad=True)
T=torch.tensor([298])
print(p.NRTL_gamma(x, T))
print(p.NRTL_gamma2(x,T))

