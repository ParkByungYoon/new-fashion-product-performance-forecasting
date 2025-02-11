from MVTSF.model.Crossformer import *
from MVTSF.layer.Timer import TimerEncoder

class Timer(Crossformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transformer_encoder = TimerEncoder(self.hidden_dim, self.input_len, self.segment_len)