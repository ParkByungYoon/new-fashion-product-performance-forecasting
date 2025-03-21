import torch.nn as nn
from MVTSF.layer.Transformer import *


class ExogenousEncoder(nn.Module):
    def __init__(self, output_dim, input_len):
        super().__init__()
        self.input_linear = TimeDistributed(nn.Linear(input_len, output_dim))

    def forward(self, inputs):
        inputs = inputs.permute(0,2,1)
        if inputs.dim() <= 2: inputs = inputs.unsqueeze(dim=-1)
        emb = self.input_linear(inputs)
        return emb
    

class EndogenousEncoder(nn.Module):
    def __init__(self, output_dim, input_len, segment_len):
        super().__init__()
        self.input_linear = SegmentEmbedding(output_dim, segment_len)
        self.num_segments = input_len//segment_len
        self.pos_embedding = PositionalEncoding(output_dim, max_len=self.num_segments)

    def forward(self, inputs, fusion_embedding):
        if inputs.dim() <= 2: inputs = inputs.unsqueeze(1)
        emb = self.input_linear(inputs).squeeze()
        emb = self.pos_embedding(emb)
        emb = torch.cat([fusion_embedding.unsqueeze(1), emb], dim=1)
        return emb