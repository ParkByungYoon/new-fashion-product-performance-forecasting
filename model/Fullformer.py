from MVTSF.model.Crossformer import *
from MVTSF.layer.Fullformer import FullAttentionTransformerEncoder
import torch

class GTMFullformer(Crossformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transformer_encoder = FullAttentionTransformerEncoder(self.hidden_dim, self.input_len, self.segment_len)


class GTMFullformerV2(CrossformerV2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transformer_encoder = FullAttentionTransformerEncoder(self.hidden_dim, self.input_len, self.segment_len)