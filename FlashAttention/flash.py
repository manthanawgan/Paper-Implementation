import cupy as cp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.Functional as F


#O=softmax(QK^T)V

def FlashAttention(Q, K, V, block_size):
    N, d = Q.shape  #N = number of tokens, d = dimension of each token's query vector
    
    O = cp.zeroes((N, d), dtype = Q.dtype)
    L = cp.zeroes(N, dtype = Q.dtype)
    M = cp.full(N, -cp.inf, dtype= Q.dtype)

    for j in range(0, N, block_size):

        K_j = K[j: j + block_size]
        V_j = V[j: j + block_size]

        for i in range(0, N, block_size):

            Q_i = Q[i: i + block_size]

            O_i = O[i: i + block_size]
            L_i = L[i: i + block_size]
            M_i = M[i: i + block_size]

            S_ij = Q_i @ K_j.T

            M_tilde = cp.max(S, axis= 1)
            P_tilde = cp.exp(
                    S_ij - M_tilde[:, None]
            )
            L_tilde = cp.sum(P_tilde, axis= 1)

            m_new = cp.max(M_i, M_tilde)
            l_new = cp.exp(M_i - m_new) * L_i + cp.exp(M_tilde - m_new) * L_tilde

            o_new = (
                (L_i[:, None] * cp.exp(M_i - m_new) * O_i)
                +
                (cp.exp(M_tilde - m_new) * P_tilde * V_j)
            ) / l_new[:, None]


            O[i: i + block_size] = o_new
            L[i: i + block_size] = l_new
            M[i: i + block_size] = m_new
return O
