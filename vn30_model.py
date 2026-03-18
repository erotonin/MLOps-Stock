import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class SparseMoE(nn.Module):
    def __init__(self, d_model, num_experts=4, k=2):
        super(SparseMoE, self).__init__()
        self.num_experts = num_experts
        self.k = k
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.ReLU(),
                nn.Linear(d_model * 2, d_model)
            ) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(d_model, num_experts)

    def forward(self, x):
        # x shape: [batch, seq_len, d_model]
        batch, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model) # [batch * seq_len, d_model]
        
        # Gating
        gate_logits = self.gate(x_flat)
        weights, indices = torch.topk(F.softmax(gate_logits, dim=-1), self.k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True) # Normalize

        # Expert output
        final_output = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            # Mask for current expert
            mask = (indices == i).any(dim=-1)
            if mask.any():
                # Apply weight
                # Extract indices where this expert is selected
                idx_in_batch = torch.where(indices == i)[0]
                # Note: This is a simplified MoE for demonstration
                # In real FinCast, weights are applied per expert
                expert_out = expert(x_flat[mask])
                # We need to distribute weights. This loop-based approach is simple.
                for j in range(self.k):
                    match_mask = (indices[:, j] == i)
                    if match_mask.any():
                        final_output[match_mask] += weights[match_mask, j].unsqueeze(1) * expert(x_flat[match_mask])
        
        return final_output.view(batch, seq_len, d_model)

class FinCastMini(nn.Module):
    def __init__(self, input_size=8, d_model=128, nhead=4, num_layers=3, dropout=0.1):
        super(FinCastMini, self).__init__()
        self.d_model = d_model
        
        # 1. Input Embedding & Instance Norm
        self.input_projection = nn.Linear(input_size, d_model)
        self.norm = nn.LayerNorm(d_model)
        
        # 2. Learnable Frequency Embedding (0: Daily, 1: 1H, 2: 1m)
        self.freq_emb = nn.Embedding(5, d_model)
        
        # 3. Transformer Decoder Backbone
        self.pos_encoder = PositionalEncoding(d_model)
        
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*4, 
            dropout=dropout, 
            batch_first=True
        )
        # Replacing the MLP in Transformer with MoE is complex in standard nn.Transformer
        # So we build a custom stack for simplicity
        self.transformer_blocks = nn.ModuleList([
            nn.ModuleDict({
                'attention': nn.MultiheadAttention(d_model, nhead, batch_first=True),
                'norm1': nn.LayerNorm(d_model),
                'moe': SparseMoE(d_model),
                'norm2': nn.LayerNorm(d_model)
            }) for _ in range(num_layers)
        ])
        
        # 4. Output: Point projection + Quantile projections (0.1, 0.5, 0.9)
        self.output_head = nn.Linear(d_model, 3) # Output 3 values for quantiles

    def forward(self, x, freq_idx=0):
        # x: [batch, seq_len, input_size]
        batch, seq, feat = x.shape
        
        # Instance Normalization (simplified)
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-5
        x = (x - mean) / std
        
        # Projection & Frequency Embedding
        x = self.input_projection(x)
        freq = self.freq_emb(torch.tensor([freq_idx]).to(x.device)).expand(batch, seq, -1)
        x = x + freq
        
        x = self.pos_encoder(x)
        
        # Transformer blocks with MoE
        for block in self.transformer_blocks:
            # Self-Attention
            attn_out, _ = block['attention'](x, x, x)
            x = block['norm1'](x + attn_out)
            
            # MoE Layer
            moe_out = block['moe'](x)
            x = block['norm2'](x + moe_out)
            
        # Global average pooling or take last token
        out = x[:, -1, :] # [batch, d_model]
        
        # Get quantiles
        quantiles = self.output_head(out) # [batch, 3]
        
        # Denormalization (using Close price stats - index 3)
        # close_mean = mean[:, :, 3]
        # close_std = std[:, :, 3]
        # quantiles = quantiles * close_std + close_mean
        
        return quantiles

if __name__ == "__main__":
    # Test model
    model = FinCastMini()
    dummy_input = torch.randn(2, 60, 8)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}") # [2, 3] -> 10th, 50th, 90th quantiles
    print(f"Quantiles: {output[0]}")
