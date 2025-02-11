import torch.nn as nn
from MVTSF.layer.Transformer import *

class InversedTransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, input_len, num_heads=4, dropout=0.2):
        super().__init__()
        self.input_linear = TimeDistributed(nn.Linear(input_len, embedding_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, inputs):
        inputs = inputs.permute(0,2,1)
        if inputs.dim() <= 2: inputs = inputs.unsqueeze(dim=-1)
        emb = self.input_linear(inputs)
        emb = self.encoder(emb)
        return emb