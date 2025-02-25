import torch.nn as nn
from MVTSF.layer.Transformer import *


class ExogenousInvertedEncoder(nn.Module):
    def __init__(self, embedding_dim, input_len):
        super().__init__()
        self.input_linear = TimeDistributed(nn.Linear(input_len, embedding_dim))

    def forward(self, inputs):
        inputs = inputs.permute(0,2,1)
        if inputs.dim() <= 2: inputs = inputs.unsqueeze(dim=-1)
        emb = self.input_linear(inputs)
        return emb

class ExogenousEncoder(nn.Module):
    def __init__(self, embedding_dim, input_len, num_vars):
        super().__init__()
        self.input_linear = TimeDistributed(nn.Linear(num_vars, embedding_dim))
        self.pos_embedding = PositionalEncoding(embedding_dim, max_len=input_len)

    def forward(self, inputs):
        emb = self.input_linear(inputs)
        emb = self.pos_embedding(emb)
        return emb


class EndogenousEncoder(nn.Module):
    def __init__(self, embedding_dim, input_len, num_vars):
        super().__init__()
        self.input_linear = TimeDistributed(nn.Linear(num_vars, embedding_dim))
        self.pos_embedding = PositionalEncoding(embedding_dim, max_len=input_len+1)

    def forward(self, inputs, fusion_embedding):
        emb = self.input_linear(inputs)
        emb = torch.cat([fusion_embedding.unsqueeze(1), emb], dim=1)
        emb = self.pos_embedding(emb)
        return emb