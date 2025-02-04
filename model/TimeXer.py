from MVTSF.model.Transformer import Transformer
from MVTSF.layer.TimeXer import *
from model.Lightning import PytorchLightningBase

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

        inputs = (inputs - inputs.mean()) / inputs.std()
        endo_inputs, exo_inputs = self.split_inputs(inputs, meta_data)
        endogenous_embedding = self.endogenous_encoder(endo_inputs, fusion_embedding.unsqueeze(1))
        exogenous_embedding = self.exogenous_encoder(exo_inputs)

        tgt = endogenous_embedding.unsqueeze(0)
        memory = exogenous_embedding
        decoder_out, attn_weights = self.decoder(tgt, memory)
        forecast = self.decoder_fc(decoder_out)

        return forecast.view(-1, self.output_len), attn_weights


class TimeXerV2(PytorchLightningBase):
    def __init__(self, args):
        super().__init__()
        self.num_endo_vars = args.num_endo_vars
        self.embedding_dim = args.embedding_dim
        self.input_len = args.input_len
        self.output_len = args.output_len
        self.lr = args.learning_rate
        self.num_heads = args.num_heads
        self.num_layers = args.num_layers
        self.save_hyperparameters()


        self.input_linear = TimeDistributed(nn.Linear(self.num_endo_vars, self.embedding_dim))
        self.exo_encoder = ExogenousEncoder(self.embedding_dim, self.input_len)
        self.pos_embedding = PositionalEncoding(self.embedding_dim, max_len=self.input_len)
        self.encoders = nn.ModuleList(
            [
                nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=self.embedding_dim, 
                        nhead=4, 
                        dropout=0.2
                    ), 
                    num_layers=1
                )
            ] * self.num_layers
        )
        self.decoders = nn.ModuleList(
            [   
                nn.TransformerDecoder(
                    nn.TransformerDecoderLayer(
                        d_model=self.embedding_dim, 
                        nhead=4, 
                        dim_feedforward=self.embedding_dim * 4, 
                        dropout=0.2
                    ),
                    num_layers=1
                )
            ] * self.num_layers
        )
        self.decoder_fc = nn.Sequential(
            nn.Linear(self.hidden_dim, self.output_len),
            nn.Dropout(0.2)
        )
    
    def forward(self, endo_inputs, exo_inputs, fusion_emb):
        endo_emb = self.input_linear(endo_inputs)
        endo_emb = self.pos_embedding(endo_emb)
        endo_emb = torch.cat([endo_emb, fusion_emb], dim=1)
        exo_emb = self.exo_encoder(exo_inputs)

        for l in range(self.num_layers):
            endo_emb = self.encoders[l](endo_emb)
            cross_emb, attn_weights = self.decoders[l](endo_emb[:,-1,:], exo_emb)

        forecast = self.decoder_fc(cross_emb)
        return forecast.view(-1, self.output_len), attn_weights