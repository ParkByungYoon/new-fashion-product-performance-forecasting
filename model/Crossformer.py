from MVTSF.model.Transformer import Transformer
from MVTSF.layer.Crossformer import CrossedTransformerEncoder
import torch
from einops import rearrange

class Crossformer(Transformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transformer_encoder = CrossedTransformerEncoder(self.output_dim, self.exo_input_len, self.segment_len)