from MVTSF.model.Crossformer import *
from MVTSF.layer.Fullformer import FullAttentionTransformerEncoder
import torch

class Fullformer(Crossformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transformer_encoder = FullAttentionTransformerEncoder(self.output_dim, self.input_len, self.segment_len)