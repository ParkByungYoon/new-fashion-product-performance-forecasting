import torch.nn as nn
from MVTSF.layer.Transformer import *

class InvertedEndogenousEncoder(nn.Module):
    def __init__(self, embedding_dim, input_len, num_heads=4, dropout=0.2):
        super().__init__()
        self.input_linear = TimeDistributed(nn.Linear(input_len, embedding_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, inputs, fusion_emb):
        inputs = inputs.permute(0,2,1)
        if inputs.dim() <= 2: inputs = inputs.unsqueeze(dim=-1)
        emb = self.input_linear(inputs)
        emb = torch.cat([emb, fusion_emb], dim=1)
        emb = self.encoder(emb)
        return emb[:,-1,:]


class EndogenousEncoder(nn.Module):
    def __init__(self, embedding_dim, input_len, num_vars):
        super().__init__()
        self.input_linear = TimeDistributed(nn.Linear(num_vars, embedding_dim))
        self.pos_embedding = PositionalEncoding(embedding_dim, max_len=input_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=4, dropout=0.2)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, inputs, fusion_emb):
        if inputs.dim() <= 2: inputs = inputs.unsqueeze(dim=-1)
        emb = self.input_linear(inputs)
        emb = torch.cat([emb, fusion_emb], dim=1)
        emb = self.pos_embedding(emb)
        emb = self.encoder(emb)
        return emb[:,-1,:]


class ExogenousEncoder(nn.Module):
    def __init__(self, embedding_dim, input_len):
        super().__init__()
        self.input_linear = TimeDistributed(nn.Linear(input_len, embedding_dim))

    def forward(self, inputs):
        inputs = inputs.permute(0,2,1)
        if inputs.dim() <= 2: inputs = inputs.unsqueeze(dim=-1)
        emb = self.input_linear(inputs)
        return emb