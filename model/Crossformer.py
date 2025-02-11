from MVTSF.model.Transformer import Transformer
from MVTSF.layer.Crossformer import CrossedTransformerEncoder
import torch
from einops import rearrange

class Crossformer(Transformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transformer_encoder = CrossedTransformerEncoder(self.hidden_dim, self.input_len, self.segment_len)