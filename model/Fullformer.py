from model.Crossformer import *
from layer.Fullformer import FullAttentionTransformerEncoder

class Fullformer(Crossformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transformer_encoder = FullAttentionTransformerEncoder(self.output_dim, self.input_len, self.segment_len)