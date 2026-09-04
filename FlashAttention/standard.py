import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class StandardSelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x, mask=None):
        #x shape: [Batch Size, Sequence Length, Embedding Dimension]
        b, s, d = x.shape
        
        Q = self.q_proj(x)  #[B, S, D]
        K = self.k_proj(x)  
        V = self.v_proj(x)  
        
        #(dot product Q and K^T)
        scores = torch.matmul(Q, K.transpose(-2, -1))  #[B, S, S]
        
        scores = scores / math.sqrt(d)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attention_weights = F.softmax(scores, dim=-1)  #[B, S, S]
        
        output = torch.matmul(attention_weights, V)  #[B, S, D]
        
        return output, attention_weights


B, S, D = 2, 8, 64  #Batch size 2, Sequence length 8, Embedding size 64
x = torch.randn(B, S, D)

attn_layer = StandardSelfAttention(embed_dim=D)
output, weights = attn_layer(x)

print("Output shape:", output.shape)    #[2, 8, 64]
print("Weights shape:", weights.shape)  #[2, 8, 8]
