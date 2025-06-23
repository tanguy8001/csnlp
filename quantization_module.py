import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size, dim, tau=2.0, commitment_beta=1.0):
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        self.tau = tau
        self.commitment_beta = commitment_beta # For encoder commitment loss

        # Codebook: K x D
        self.codebook = nn.Parameter(torch.randn(codebook_size, dim))

    def forward(self, x, hard_gumbel_softmax_indices=True):
        """
        x: (B, T, D) – chunk embeddings from encoder
        hard_gumbel_softmax_indices: If True, indices are from hard Gumbel-Softmax.
        returns:
            quantized_soft: (B, T, D) - soft, differentiable quantized embeddings
            indices: (B, T) - hard indices
            codebook_loss: scalar - loss to make codebook vectors match encoder outputs
            enc_commitment_loss: scalar - loss to make encoder outputs commit to codebook
            entropy: scalar - mean entropy of the soft probabilities over the codebook
        """
        B, T, D = x.shape
        x_flat = x.view(-1, D)  # (B*T, D)

        # Compute distances to codebook for logits (negative distances)
        x_sq = x_flat.pow(2).sum(dim=1, keepdim=True)  # (B*T, 1)
        c_sq = self.codebook.pow(2).sum(dim=1)         # (K,)
        xc = x_flat @ self.codebook.T                  # (B*T, K)
        
        logits = -(x_sq + c_sq - 2 * xc)  # (B*T, K)

        probabilities_soft = F.gumbel_softmax(logits, tau=self.tau, hard=False, dim=-1) # (B*T, K)
        
        quantized_soft = probabilities_soft @ self.codebook  # (B*T, D)
        quantized_soft = quantized_soft.view(B, T, D)

        if hard_gumbel_softmax_indices:
            probabilities_hard = F.gumbel_softmax(logits, tau=self.tau, hard=True, dim=-1)
            indices = torch.argmax(probabilities_hard, dim=1) # (B*T,)
        else:
            indices = torch.argmax(logits, dim=1) # (B*T,)

        # VQ-VAE style losses
        codebook_loss = F.mse_loss(quantized_soft, x.detach()) 
        enc_commitment_loss = F.mse_loss(x, quantized_soft.detach())

        # Calculate entropy of the soft probabilities
        # H(p) = -sum(p_i * log(p_i))
        entropy = -torch.sum(probabilities_soft * torch.log(probabilities_soft + 1e-12), dim=1)
        mean_entropy = entropy.mean()

        return quantized_soft, indices.view(B, T), codebook_loss, enc_commitment_loss, mean_entropy