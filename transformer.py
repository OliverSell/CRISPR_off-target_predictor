#/usr/bin/env python3

# https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.transformer.Transformer.html

import torch
# install torch
# pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# py -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
from torch import nn

class CrossSeqTransformer(nn.Module):
    def __init__(self, vocab_size=5, d_model=128, nhead=8,
                 num_encoder_layers=3, num_decoder_layers=3,
                 max_len=32, dropout=0.1):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model) # positional encoding
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead, # multihead attention
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=4*d_model,
            dropout=dropout,
            batch_first=True
        )
        self.regressor = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, on_seq, off_seq):
        B, L = on_seq.shape
        pos = torch.arange(L, device=on_seq.device).unsqueeze(0) # position indices
        src = self.token_embed(on_seq) + self.pos_embed(pos)
        tgt = self.token_embed(off_seq) + self.pos_embed(pos)
        # Transformer expects (batch, seq, dim)
        out = self.transformer(src, tgt)  # shape [B, L, d_model]
        pooled = out.mean(dim=1)
        score = self.regressor(pooled)
        return score.squeeze(-1)
    

model = CrossSeqTransformer()
batch_size = 4
seq_len = 24
on_seq = torch.randint(0, 5, (batch_size, seq_len)) # target sequence
off_seq = torch.randint(0, 5, (batch_size, seq_len)) # off-target sequence
activity_scores = torch.rand(batch_size) # labels

out = model(on_seq, off_seq) # predicted activity scores
print(out.shape, out)

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