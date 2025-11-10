#/usr/bin/env python3

# https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.transformer.Transformer.html

import torch
# install torch
# pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# py -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
from torch import nn


### SCORE = CHANGEseq_reads = Activty

class CrossSeqTransformer(nn.Module):
    def __init__(self, vocab_size=5, d_model=128, nhead=8,
                 num_encoder_layers=3, num_decoder_layers=3,
                 max_len=32, dropout=0.1):
        super().__init__()
        
        # Token embeddings
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model) # positional encoding
        
        # Transformer (endocer-decoder)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead, # multihead attention
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=4*d_model,
            dropout=dropout,
            batch_first=True
        )

        # Feature projection layers
        self.score_proj = nn.Linear(1, d_model // 4)
        self.mismatch_proj = nn.Linear(1, d_model // 4)

        # Output regression head
        self.regressor = nn.Sequential(
            nn.Linear(d_model + d_model // 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
            nn.Sigmoid() # activity either 0 or 1
        )

    def forward(self, on_seq, off_seq, score, mismatches):
        B, L = on_seq.shape
        device = on_seq.device

        # Positional encoding
        pos = torch.arange(L, device=device).unsqueeze(0).expand(B,-1) # position indices

        # Embed sequences
        src = self.token_embed(on_seq) + self.pos_embed(pos)
        tgt = self.token_embed(off_seq) + self.pos_embed(pos)

        # Transformer expects (batch, seq, dim)
        out = self.transformer(src, tgt)  # shape [B, L, d_model]
        
        # Pooling
        pooled = out.mean(dim=1)
        
        # Project features
        score_emb = self.regressor(pooled)
        mismatch_emb = self.mistmatch_proj(mismatches)

        # Concatenate all features
        combined = torch.cat([pooled, score_emb, mismatch_emb], dim=1)

        # Predict activity
        activity = self.regressor(combined)

        return activity.squeeze(-1)
    

model = CrossSeqTransformer()
batch_size = 32
seq_len = 23
on_seq = torch.randint(0, 5, (batch_size, seq_len))
off_seq = torch.randint(0, 5, (batch_size, seq_len))
score = torch.rand(batch_size, 1)
mismatches = torch.rand(batch_size, 1)

out = model(on_seq, off_seq, score, mismatches)
print(f"Output shape: {out.shape}")
print(f"Output range: [{out.min():.3f}, {out.max():.3f}]")

## -------- Training --------- ##

batch_size = 4 # weights updated per batch
seq_len = 24
num_batches = ?

model = CrossSeqTransformer()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

## NOTE: 10 epochs for now
for epoch in range(10):
    epoch_loss = 0.0
    for _ in range(num_batches):
        on_seq = torch.randint(0, 5, (batch_size, seq_len))
        off_seq = torch.randint(0, 5, (batch_size, seq_len))
        activity_scores = torch.rand(batch_size)

        # Forward pass
        out = model(on_seq, off_seq)
        loss = criterion(out, activity_scores)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}: avg loss = {epoch_loss / num_batches:.4f}")

print("Training complete")