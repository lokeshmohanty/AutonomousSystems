import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

# Building Blocks of a Transformer
#
# Positional Encoding
# Attention (Masked, Multi-Head, Cross, Self)
# Feed-Forward Networks
# Normalization (Layer)
# Residual Connections (Skip)
# Dropout

class Attention(nn.Module):
    """
    - Self Attention
    - Cross Attention (if y is not None)
    - Multi-Head Attention (if n_heads > 1)
    - Masked Attention (if mask is not None)
    """
    def __init__(self, d_model, n_heads = 1):
        super(Attention, self).__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads

        # this matrix need not be split
        self.Q = nn.Linear(d_model, d_model)
        self.K = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
    
    def split_attn(self, q, k, v):
        a = q @ k.T / math.sqrt(self.d_head)
        return torch.softmax(a, dim=-1) @ v

    def forward(self, x, y=None, mask=None):
        """
        Args:
            x: query tokens
            y: key and value tokens
         mask: query mask
        """
        y = y if y else x

        if mask:
            x = x.masked_fill(mask, 1e-9)

        q = self.Q(x)
        k = self.K(y)
        v = self.V(y)

        attn = [self.split_attn(
            q[:, i:i + self.d_head],
            k[:, i:i + self.d_head],
            v[:, i:i + self.d_head],
        ) for i in range(n_heads)]

        return self.fc(attn)

class FFN(nn.Module):
    def __init__(self, d_model, d_hidden):
        super(FFN, self).__init__()

        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_tokens):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_tokens, d_model)
        position = torch.arange(0, max_tokens, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # add as model parameter while being set as non trainable
        self.register_buffer('pe', pe.unsqueze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_hidden, dropout):
        super(EncoderLayer, self).__init__()

        self.attn = Attention(d_model, n_heads)
        self.ffn = FFN(d_model, d_hidden)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm1(x + self.dropout(self.attn(x)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_hidden, dropout):
        super(DecoderLayer, self).__init__()

        self.self_attn = Attention(d_model, n_heads)
        self.cross_attn = Attention(d_model, n_heads)
        self.ffn = FFN(d_model, d_hidden)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None, src_mask=None, tgt_mask=None):
        x = self.norm1(x + self.dropout(self.self_attn(x, mask=src_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, y, mask=tgt_mask))) if y else x
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x

class Seq2SeqTransformer(nn.Module):
    def __init__(
        self, 
        src_vocab_size, tgt_vocab_size, 
        context_size, d_model, n_layers,
        n_heads, d_hidden, dropout,
    ):
        super(Transformer, self).__init__()

        # Embedding takes in an index and gives the embedding
        self.encoder_emb = nn.Embedding(src_vocab_size, d_model)
        self.decoder_emb = nn.Embedding(tgt_vocab_size, d_model)

        self.pe = PositionalEncoding(d_model, context_size)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_hidden, dropout) for _ in n_layers])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_hidden, dropout) for _ in n_layers])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src = self.dropout(self.pe(self.encoder_emb(src)))
        tgt = self.dropout(self.pe(self.decoder_emb(tgt)))

        for layer in self.encoder_layers:
            src = layer(src, src_mask)

        for layer in self.decoder_layers:
            tgt = layer(tgt, src, src_mask, tgt_mask)

        return self.fc(tgt)


## Translation Data
src_vocab_size = 500
tgt_vocab_size = 500
context_length = 100
d_model = 200

src_data = torch.randint(1, src_vocab_size, (64, context_length))
tgt_data = torch.randint(1, tgt_vocab_size, (64, context_length))

model = Seq2SeqTransformer(src_vocab_size, tgt_vocab_size, context_length, d_model, 8, 1, 1024, 0.1)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(.9, .98), eps=1e-9)

# Train
model.train()

src_mask = torch.ones_like(src_data[0], dtype=torch.bool)
tgt_mask = torch.zeros_like(tgt_data[0], dtype=torch.bool)

for epoch in range(100):
    for i in range(tgt_data.shape[-1]):
        if i != 0:
            tgt_mask[i-1] = True
        optimizer.zero_grad()

        out = model(src_data, tgt_data, src_mask, tgt_mask)
        loss = criterion(out, tgt_data[i])

        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch+1}\tloss: {loss.item():.3f}")

    with torch.no_grad():
        pred = torch.zeros_like(tgt_data)
        for i in range(tgt_data.shape[-1]):
            out = model(src_data, pred)
            pred[i] = out.argmax(dim=-1)
        similarity = torch.cosine_similarity(model.decoder_emb(pred), model.decoder_emb(tgt_data), dim=1)
        print(f"Prediction similarity score: {similarity.sum()}")

model.eval()

src_data = torch.randint(1, src_vocab_size, (64, context_length))
tgt_data = torch.randint(1, tgt_vocab_size, (64, context_length))

tgt_mask = torch.zeros_like(tgt_data[0], dtype=torch.bool)

with torch.no_grad():
    for i in range(tgt_data.shape[-1]):
        if i != 0:
            tgt_mask[i-1] = True
        out = model(src_data, tgt_data, src_mask, tgt_mask)
        loss = criterion(out, tgt_data[i])

        print(f"Validation - \tloss: {loss.item():.3f}")

    pred = torch.zeros_like(tgt_data)
    for i in range(tgt_data.shape[-1]):
        out = model(src_data, pred)
        pred[i] = out.argmax(dim=-1)
    similarity = torch.cosine_similarity(model.decoder_emb(pred), model.decoder_emb(tgt_data), dim=1)
    print(f"Prediction similarity score: {similarity}")



        
        

## Data
# inputs: x, y, output: z; 0 < x,y,z < 9
# relation: z = (x + y) mod 10

## Training


