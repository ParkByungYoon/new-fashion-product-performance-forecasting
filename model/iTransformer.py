from MVTSF.model.Transformer import Transformer
from MVTSF.layer.iTransformer import InversedTransformerEncoder
import torch

class iTransformer(Transformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transformer_encoder = InversedTransformerEncoder(self.output_dim, self.exo_input_len)