import torch
import numpy as np

# exact solution u
def u_Exact_Square2D(x, y):
    
    return torch.sin(2*np.pi*x) * (torch.cos(2*np.pi*y) - 1)

# exact solution dudx
def dudx_Exact_Square2D(x, y):

	return 2*np.pi * torch.cos(2*np.pi*x) * (torch.cos(2*np.pi*y) - 1)  

# exact solution dudy
def dudy_Exact_Square2D(x, y):

	return - 2*np.pi * torch.sin(2*np.pi*x) * torch.sin(2*np.pi*y) 




# right hand side
def f_Exact_Square2D(x, y):
    
    return 4 * np.pi * np.pi * torch.sin(2*np.pi*x) * (2 * torch.cos(2*np.pi*y) - 1)  

# Dirichlet boundary condition on boundaries except for Gamma
def g_Exact_Square2D(x, y):
    
    return torch.sin(2*np.pi*x) * (torch.cos(2*np.pi*y) - 1) 

# Boundary condition on Gamma
def h_Exact_Square2D(prob_type, x, y):

	if prob_type == 1:
		# Dirichlet boundary condition
		return torch.sin(2*np.pi*x) * (torch.cos(2*np.pi*y) - 1) 

	if prob_type == 2:
		# Neumann boundary condition
		return -2*np.pi * torch.cos(2*np.pi*x) * (torch.cos(2*np.pi*y) - 1)        

    