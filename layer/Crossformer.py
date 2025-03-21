import torch.nn as nn
from einops import rearrange
from MVTSF.layer.Transformer import *

class CrossedTransformerEncoder(nn.Module):
    def __init__(self, output_dim, input_len, segment_len, num_heads=4, dropout=0.2):
        super().__init__()
        self.input_linear = SegmentEmbedding(output_dim, segment_len)
        self.num_segments = input_len//segment_len
        self.pos_embedding = PositionalEncoding(output_dim, max_len=self.num_segments)
        time_encoder_layer = nn.TransformerEncoderLayer(d_model=output_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.time_encoder = nn.TransformerEncoder(time_encoder_layer, num_layers=1)
        var_encoder_layer = nn.TransformerEncoderLayer(d_model=output_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.var_encoder = nn.TransformerEncoder(var_encoder_layer, num_layers=1)

    def forward(self, inputs):
        batch, num_vars, input_len = inputs.shape # (64, 3, 52)
        emb = self.input_linear(inputs) # (64, 3, 13, 512)
        # time --> domain
        emb = rearrange(emb, 'b d num_segments embedding_dim -> (b d) num_segments embedding_dim') # (64*3, 13, 512)
        emb = self.pos_embedding(emb)
        emb = self.time_encoder(emb)
        emb = rearrange(emb, '(b d) num_segments embedding_dim -> (b num_segments) d embedding_dim', b = batch, d = num_vars) # (64*13, 3, 512)
        emb = self.var_encoder(emb)
        emb = rearrange(emb, '(b num_segments) d embedding_dim-> b (d num_segments) embedding_dim', b = batch, num_segments=self.num_segments) # (64, 13*3, 512)
        return emb