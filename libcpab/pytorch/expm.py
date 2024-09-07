import torch
import torch.cuda.amp as amp

def expm(A):
    n_A = A.shape[0]
    A_fro = torch.sqrt(A.abs().pow(2).sum(dim=(1,2), keepdim=True))
    
    # Scaling step
    maxnorm = torch.tensor(5.371920351148152, dtype=torch.float32, device=A.device)
    zero = torch.tensor(0.0, dtype=torch.float32, device=A.device)
    n_squarings = torch.max(zero, torch.ceil(log2(A_fro.float() / maxnorm)))
    Ascaled = A / (2.0**n_squarings.to(A.dtype))
    n_squarings = n_squarings.flatten().type(torch.int64)
    
    # Pade 13 approximation
    with amp.autocast(enabled=False):
        U, V = pade13(Ascaled.float())  # Upcasting to float32
    
    # Use log-sum-exp trick for P and Q
    P = log_sum_exp(U, V)
    Q = log_sum_exp(-U, V)
    
    # Solve log(P) = log(Q) + log(R)
    log_R = P - Q
    
    # Unsquaring step    
    n = n_squarings.max()
    log_res = [log_R]
    for i in range(n):
        log_res.append(log_matrix_multiply(log_res[-1], log_res[-1]))
    if n > 0:
        log_R = torch.stack(log_res)
        log_expmA = log_R[n_squarings, torch.arange(n_A)]
    else:
        log_expmA = log_R
    
    # Convert back from log space, keeping the result in float32
    with amp.autocast(enabled=False):
        expmA = torch.exp(log_expmA)
    
    return safe_to_fp16(expmA)

def log2(x):
    return torch.log(x) / torch.log(torch.tensor(2.0, dtype=x.dtype, device=x.device))

def pade13(A):
    b = torch.tensor([
        64764752532480000., 32382376266240000., 7771770303897600.,
        1187353796428800., 129060195264000., 10559470521600.,
        670442572800., 33522128640., 1323241920., 40840800.,
        960960., 16380., 182., 1.
    ], dtype=A.dtype, device=A.device)
    
    ident = torch.eye(A.shape[1], dtype=A.dtype, device=A.device)
    A2 = torch.matmul(A, A)
    A4 = torch.matmul(A2, A2)
    A6 = torch.matmul(A4, A2)
    U = torch.matmul(A, torch.matmul(A6, b[13]*A6 + b[11]*A4 + b[9]*A2) + b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident)
    V = torch.matmul(A6, b[12]*A6 + b[10]*A4 + b[8]*A2) + b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*ident
    return torch.log(torch.abs(U) + 1e-20), torch.log(torch.abs(V) + 1e-20)

def log_sum_exp(log_a, log_b):
    max_val = torch.max(log_a, log_b)
    return max_val + torch.log(torch.exp(log_a - max_val) + torch.exp(log_b - max_val))

def log_matrix_multiply(log_A, log_B):
    m, n, p = log_A.shape[1], log_A.shape[2], log_B.shape[2]
    log_C = torch.empty(log_A.shape[0], m, p, dtype=log_A.dtype, device=log_A.device)
    
    for i in range(m):
        for j in range(p):
            log_sum = log_A[:, i, 0] + log_B[:, 0, j]
            for k in range(1, n):
                log_sum = log_sum_exp(log_sum, log_A[:, i, k] + log_B[:, k, j])
            log_C[:, i, j] = log_sum
    
    return log_C

def safe_to_fp16(x):
    fp16_max = torch.finfo(torch.float16).max
    return torch.clamp(x, min=-fp16_max, max=fp16_max).to(torch.float16)