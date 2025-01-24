from MVTSF.model.Transformer import Transformer
from MVTSF.layer.TimeXer import *

class TimeXer(Transformer):
    def __init__(self, args):
        super().__init__()
        self.num_endo_vars = args.num_endo_vars
        self.num_exo_vars = args.num_exo_vars
        self.exogenous_encoder = ExogenousEncoder(self.hidden_dim, self.input_len)
        self.endogenous_encoder = EndogenousEncoder(self.hidden_dim, self.input_len+1, self.num_endo_vars)
        # self.endogenous_encoder = InvertedEndogenousEncoder(self.hidden_dim, self.input_len)

    def forward(self, inputs, release_dates, image_embedding, text_embedding, meta_data):
        temporal_embedding = self.temporal_feature_encoder(release_dates)
        fusion_embedding = self.feature_fusion_network(image_embedding, text_embedding, temporal_embedding, meta_data)

        endo_inputs, exo_inputs = self.split_inputs(inputs, meta_data)
        endogenous_embedding = self.endogenous_encoder(endo_inputs, fusion_embedding.unsqueeze(1))
        exogenous_embedding = self.exogenous_encoder(exo_inputs)

        tgt = endogenous_embedding.unsqueeze(0)
        memory = exogenous_embedding
        decoder_out, attn_weights = self.decoder(tgt, memory)
        forecast = self.decoder_fc(decoder_out)

        return forecast.view(-1, self.output_len), attn_weights