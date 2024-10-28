"""
Baseline 3: Implicit No CoT (not knowing #steps)
"""
import math

import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, d_head, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_head = d_head

        self.qkv_proj = nn.Linear(d_model, 3 * n_heads * d_head)
        self.out_proj = nn.Linear(n_heads * d_head, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        batch_size, seq_len, d_model = x.size()
        qkv = self.qkv_proj(x)  # (batch_size, seq_len, 3 * n_heads * d_head)
        qkv = qkv.view(batch_size, seq_len, self.n_heads, 3 * self.d_head)
        qkv = qkv.permute(2, 0, 1, 3).contiguous()  # (n_heads, batch_size, seq_len, 3 * d_head)
        q, k, v = torch.chunk(qkv, 3, dim=-1)  # Each is (n_heads, batch_size, seq_len, d_head)

        # Compute scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)  # (n_heads, batch_size, seq_len, seq_len)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)  # Expand for n_heads
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)  # (n_heads, batch_size, seq_len, d_head)
        context = context.permute(1, 2, 0, 3).contiguous()  # (batch_size, seq_len, n_heads, d_head)
        context = context.view(batch_size, seq_len, self.n_heads * self.d_head)  # (batch_size, seq_len, d_model)
        output = self.out_proj(context)
        return output

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(self.act(self.fc1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, n_heads, d_model, d_head, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.ln_1 = nn.LayerNorm(d_model, eps=1e-5)
        self.attn = MultiHeadAttention(n_heads, d_model, d_head, dropout)
        self.ln_2 = nn.LayerNorm(d_model, eps=1e-5)
        self.mlp = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        attn_output = self.attn(self.ln_1(x), attn_mask)
        x = x + self.dropout(attn_output)
        mlp_output = self.mlp(self.ln_2(x))
        x = x + self.dropout(mlp_output)
        return x

class GPT2CustomModel(nn.Module):
    def __init__(self, vocab_size, max_seq_len, n_layers, n_heads, d_model, d_ff, dropout=0.1, tol=1e-5):
        super(GPT2CustomModel, self).__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(n_heads, d_model, d_model // n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model, eps=1e-5)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        self.tol = tol

    def forward(self, input_ids, attention_mask=None,num_iter=20):
        """
        Inference mode, no loss calculation, returns only logits.
        """
        batch_size, seq_len = input_ids.size()
        positions = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)

        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.dropout(x)

        # Inference mode: Run until equilibrium
        for iter in range(num_iter):  # Maximum 5 iterations as in your initial request
            x_prev = x.clone()  # Save the previous state of x

            # Pass through the transformer blocks
            for block in self.blocks:
                x = block(x, attn_mask=attention_mask)

            # Check if x has changed significantly (early stopping condition)
            change = torch.abs(x - x_prev).mean().item()  # Measure the mean change
            if change < self.tol:
                break  # Stop if the change is smaller than the tolerance

        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    def compute_loss(self, input_ids, attention_mask=None, labels=None, criterion=None, chunk_size=5, max_chunks=5, predict_start_idx=0):
        """
        Computes the loss during training with fixed-size chunks.
        Starts predicting from `predict_start_idx` and chunks the remaining tokens.
        
        Parameters:
        - chunk_size: Number of tokens per chunk.
        - max_chunks: Maximum number of chunks to process.
        - predict_start_idx: Index in the sequence where prediction starts (input tokens before this index are ignored).
        """
        batch_size, seq_len = input_ids.size()
         ###################### Baseline 3:  #################
        logits = self.forward(input_ids, attention_mask=attention_mask)       

        total_loss = 0.0  # Accumulate loss here

        end_idx = seq_len        

        # Compare only the tokens in the current chunk
        logits_chunk = logits[:, predict_start_idx:end_idx, :]  # (batch_size, chunk_size, vocab_size)
        target_chunk = labels[:, predict_start_idx:end_idx]      # (batch_size, chunk_size)

        # print(logits_chunk.shape, logits.size(-1), target_chunk.shape)
        # Flatten logits and targets for loss computation
        logits_flat = logits_chunk.reshape(-1, logits.size(-1))  # (batch_size * chunk_size, vocab_size)
        target_flat = target_chunk.reshape(-1)                   # (batch_size * chunk_size)
        #########################################################
        
        # Calculate the loss for this chunk
        # term 1 & 2 encouraging matching
        chunk_loss = criterion(logits_flat, target_flat)
        total_loss += chunk_loss

        return logits, total_loss
    