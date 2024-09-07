# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 10:27:16 2018

@author: nsde
"""

#%%
import torch
from torch.utils.cpp_extension import load
from .findcellidx import findcellidx
from .expm import expm
from ..core.utility import get_dir
import torch.cuda.amp as amp

#%%
class _notcompiled:
    # Small class, with structure similar to the compiled modules we can default
    # to. The class will never be called but the program can compile at run time
    def __init__(self):
        def f(*args):
            return None
        self.forward = f
        self.backward = f

#%%
_dir = get_dir(__file__)
_verbose = False # TODO: set this flag in the main class, maybe
# Jit compile cpu source
try:
    cpab_cpu = load(name = 'cpab_cpu',
                    sources = [_dir + '/transformer.cpp',
                               _dir + '/../core/cpab_ops.cpp'],
                    verbose=_verbose)
    _cpu_succes = True
    if _verbose:
        print(70*'=')
        print('succesfully compiled cpu source')
        print(70*'=')
except Exception as e:
    cpab_cpu = _notcompiled()
    _cpu_succes = False
    if _verbose:
        print(70*'=')
        print('Unsuccesfully compiled cpu source')
        print('Error was: ')
        print(e)
        print(70*'=')

# Jit compile gpu source
try:
    cpab_gpu = load(name = 'cpab_gpu',
                    sources = [_dir + '/transformer_cuda.cpp',
                               _dir + '/transformer_cuda.cu',
                               _dir + '/../core/cpab_ops.cu'],
                    verbose=_verbose,
                    with_cuda=True)
    if _verbose:
        print(70*'=')
        print('Successfully compiled gpu source, but using slow implementation')
        print(70*'=')
except Exception as e:
    cpab_gpu = _notcompiled()
    if _verbose:
        print(70*'=')
        print('Unsuccessfully compiled gpu source')
        print('Error was: ')
        print(e)
        print('Using slow implementation')
        print(70*'=')

# Always use slow implementation
_gpu_succes = False

#%%
def CPAB_transformer(points, theta, params):
    if points.is_cuda and theta.is_cuda:
        if not params.use_slow and _gpu_succes:
            if _verbose: print('using fast gpu implementation')
            return CPAB_transformer_fast(points, theta, params)
        else:
            if _verbose: print('using slow gpu implementation')
            return CPAB_transformer_slow(points, theta, params)
    else:
        if not params.use_slow and _cpu_succes:
            if _verbose: print('using fast cpu implementation')
            return CPAB_transformer_fast(points, theta, params)
        else:
            if _verbose: print('using slow cpu implementation')
            return CPAB_transformer_slow(points, theta, params)
        
#%%
def CPAB_transformer_slow(points, theta, params):
    # Problem parameters
    n_theta = theta.shape[0]
    n_points = points.shape[-1]
    
    # Create homogenous coordinates
    ones = torch.ones((n_theta, 1, n_points), dtype=torch.float32, device=points.device)
    # Normalize initial points to be between -1 and 1
    points_min = points.min()
    points_max = points.max()
    points_normalized = 2 * (points - points_min) / (points_max - points_min) - 1
    if len(points.shape) == 2:
        newpoints = points[None].repeat(n_theta, 1, 1) # [n_theta, ndim, n_points]
    else:
        newpoints = points
    newpoints = torch.cat((newpoints, ones), dim=1) # [n_theta, ndim+1, n_points]
    newpoints = newpoints.permute(0, 2, 1) # [n_theta, n_points, ndim+1]
    newpoints = torch.reshape(newpoints, (-1, params.ndim+1)) #[n_theta*n_points, ndim+1]]
    newpoints = newpoints[:,:,None] # [n_theta*n_points, ndim+1, 1]
    
    # Get velocity fields
    B = torch.tensor(params.basis, dtype=torch.float32, device=theta.device)
    with amp.autocast(enabled=True):
        Avees = torch.matmul(B, theta.t())
    As = Avees.t().reshape(n_theta*params.nC, *params.Ashape)
    zero_row = torch.zeros(n_theta*params.nC, 1, params.ndim+1, dtype=torch.float32, device=As.device)
    AsSquare = torch.cat([As, zero_row], dim=1)
    
    # Take matrix exponential
    dT = 1.0 / params.nstepsolver
    Trels = expm(dT*AsSquare)
    
    # Take care of batch effect
    batch_idx = params.nC*(torch.ones(n_points, n_theta, dtype=torch.int64) * torch.arange(n_theta))
    batch_idx = batch_idx.t().flatten().to(theta.device)
    
    # Do integration
    # print("nstepsolver", params.nstepsolver)
    for i in range(params.nstepsolver):
        idx = findcellidx(params.ndim, newpoints[:,:,0].t(), params.nc) + batch_idx
        Tidx = Trels[idx.long()]
        
        # Ensure Tidx is in float32
        Tidx = Tidx.float()
        
        # Normalize Tidx
        Tidx_norm = torch.norm(Tidx, dim=(1,2), keepdim=True)
        Tidx_normalized = Tidx / Tidx_norm.clamp(min=1e-8)

        with amp.autocast(enabled=False):
            newpoints_before = newpoints.float()
            
            # Apply transformation
            newpoints = torch.matmul(Tidx_normalized, newpoints_before)
            
            # Renormalize newpoints
            newpoints_norm = torch.norm(newpoints, dim=1, keepdim=True)
            newpoints = newpoints / newpoints_norm.clamp(min=1e-8)

        # Check for NaN or Inf
        if torch.isnan(newpoints).any() or torch.isinf(newpoints).any():
            print(f"NaN or Inf detected at step {i}")
            print(f"Tidx min: {Tidx.min()}, max: {Tidx.max()}")
            print(f"newpoints_before min: {newpoints_before.min()}, max: {newpoints_before.max()}")
            return None

    newpoints = newpoints.squeeze()[:,:params.ndim].t()
    newpoints = newpoints.reshape(params.ndim, n_theta, n_points).permute(1,0,2)
    newpoints = 0.5 * (newpoints + 1) * (points_max - points_min) + points_min
    return newpoints

#%%
def CPAB_transformer_fast(points, theta, params):
    if params.numeric_grad: return _CPABFunction_NumericGrad.apply(points, theta, params)
    else: return _CPABFunction_AnalyticGrad.apply(points, theta, params)

#%%
class _CPABFunction_AnalyticGrad(torch.autograd.Function):
    # Function that connects the forward pass to the backward pass
    @staticmethod
    def forward(ctx, points, theta, params):
        device = points.device
        
        # Problem size
        n_theta = theta.shape[0]
        
        # Get velocity fields
        B = torch.Tensor(params.basis).to(device)
        Avees = torch.matmul(B, theta.t())
        As = Avees.t().reshape(n_theta*params.nC, *params.Ashape)
        zero_row = torch.zeros(n_theta*params.nC, 1, params.ndim+1).to(device)
        AsSquare = torch.cat([As, zero_row], dim=1)
        
        # Take matrix exponential
        dT = 1.0 / params.nstepsolver
        Trels = expm(dT*AsSquare)
        Trels = Trels[:,:params.ndim,:].view(n_theta, params.nC, *params.Ashape)
        
        # Convert to tensor
        nstepsolver = torch.tensor(params.nstepsolver, dtype=torch.int32, device=device)
        nc = torch.tensor(params.nc, dtype=torch.int32, device=device)

        # Call integrator
        if points.is_cuda:
            newpoints = cpab_gpu.forward(points.contiguous(), 
                                         Trels.contiguous(), 
                                         nstepsolver.contiguous(), 
                                         nc.contiguous())
        else:            
            newpoints = cpab_cpu.forward(points.contiguous(), 
                                         Trels.contiguous(), 
                                         nstepsolver.contiguous(), 
                                         nc.contiguous())

        # Save of backward
        Bs = B.t().view(-1, params.nC, *params.Ashape)
        As = As.view(n_theta, params.nC, *params.Ashape)
        ctx.save_for_backward(points, theta, As, Bs, nstepsolver, nc)
        # Output result
        return newpoints

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad): # grad [n_theta, ndim, n]
        # Grap input
        points, theta, As, Bs, nstepsolver, nc = ctx.saved_tensors

        # Call integrator, gradient: [d, n_theta, ndim, n]
        if points.is_cuda:
            gradient = cpab_gpu.backward(points.contiguous(), 
                                         As.contiguous(), 
                                         Bs.contiguous(), 
                                         nstepsolver.contiguous(), 
                                         nc.contiguous())
        else:
            gradient = cpab_cpu.backward(points.contiguous(), 
                                         As.contiguous(), 
                                         Bs.contiguous(), 
                                         nstepsolver.contiguous(), 
                                         nc.contiguous())
            
        # Backpropagate and reduce to [d, n_theta] matrix
        g = gradient.mul_(grad).sum(dim=(2,3))
        return None, g.t(), None # transpose, since pytorch expects a [n_theta, d] matrix

#%%
class _CPABFunction_NumericGrad(torch.autograd.Function):
    # Function that connects the forward pass to the backward pass
    @staticmethod
    def forward(ctx, points, theta, params):
        device = points.device
        
        # Problem size
        n_theta = theta.shape[0]
        
        # Get velocity fields
        B = torch.Tensor(params.basis).to(device)
        Avees = torch.matmul(B, theta.t())
        As = Avees.t().reshape(n_theta*params.nC, *params.Ashape)
        zero_row = torch.zeros(n_theta*params.nC, 1, params.ndim+1).to(device)
        AsSquare = torch.cat([As, zero_row], dim=1)
        
        # Take matrix exponential
        dT = 1.0 / params.nstepsolver
        Trels = expm(dT*AsSquare)
        Trels = Trels[:,:params.ndim,:].view(n_theta, params.nC, *params.Ashape)
        
        # Convert to tensor
        nstepsolver = torch.tensor([params.nstepsolver], dtype=torch.int32).to(device)
        nc = torch.tensor(params.nc, dtype=torch.int32).to(device)

        # Call integrator
        if points.is_cuda:
            newpoints = cpab_gpu.forward(points.contiguous(), 
                                         Trels.contiguous(), 
                                         nstepsolver.contiguous(), 
                                         nc.contiguous())
        else:            
            newpoints = cpab_cpu.forward(points.contiguous(), 
                                         Trels.contiguous(), 
                                         nstepsolver.contiguous(), 
                                         nc.contiguous())
            
        # Save of backward
        ctx.save_for_backward(points, theta, newpoints, nstepsolver, nc, params)
        # Output result
        return newpoints
        
    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad): # grad [n_theta, ndim, n]
        # Grap input
        points, theta, newpoints, nstepsolver, nc, params = ctx.saved_tensors
        device = points.device
        h = 0.01
        gradient = [ ]
        
        # Problem size
        n_theta, d = theta.shape
        
        for i in range(d):
            # Permute theta
            temp = theta.clone()
            temp[:,i] += h
            
            # Get velocity fields
            B = torch.Tensor(params.basis).to(device)
            Avees = torch.matmul(B, temp.t())
            As = Avees.t().reshape(n_theta*params.nC, *params.Ashape)
            zero_row = torch.zeros(n_theta*params.nC, 1, params.ndim+1).to(device)
            AsSquare = torch.cat([As, zero_row], dim=1)
            
            # Take matrix exponential
            dT = 1.0 / params.nstepsolver
            Trels = expm(dT*AsSquare)
            Trels = Trels[:,:params.ndim,:].view(n_theta, params.nC, *params.Ashape)
            
            # Call integrator
            if points.is_cuda:
                temp_points = cpab_gpu.forward(points.contiguous(), 
                                               Trels.contiguous(), 
                                               nstepsolver.contiguous(), 
                                               nc.contiguous())
            else:            
                temp_points = cpab_cpu.forward(points.contiguous(), 
                                               Trels.contiguous(), 
                                               nstepsolver.contiguous(), 
                                               nc.contiguous())
            
            diff = (temp_points - newpoints) / h
            
            # Do finite gradient
            gradient.append((grad * diff).sum(dim=[1,2])) # gradient [n_theta, ]
        
        # Reshaping
        gradient = torch.stack(gradient, dim = 1) # [n_theta, d]
        return None, gradient

