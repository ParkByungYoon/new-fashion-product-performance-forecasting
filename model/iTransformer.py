from model.Transformer import Transformer
from layer.iTransformer import InversedTransformerEncoder

class iTransformer(Transformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transformer_encoder = InversedTransformerEncoder(self.output_dim, self.exo_input_len)