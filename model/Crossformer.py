from model.Transformer import Transformer
from layer.Crossformer import CrossedTransformerEncoder

class Crossformer(Transformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transformer_encoder = CrossedTransformerEncoder(self.output_dim, self.exo_input_len, self.segment_len)