from MVTSF.layer.Transformer import *
from MVTSF.layer.TimeXer import *
from model.Transformer import Transformer


class TimeXer(Transformer):
    def __init__(self, args):
        super().__init__(args)
        self.save_hyperparameters()
        self.num_endo_vars = args.num_endo_vars
        self.num_exo_vars = args.num_exo_vars

        self.temporal_feature_encoder = TemporalFeatureEncoder(self.embedding_dim)
        self.feature_fusion_network = FeatureFusionNetwork(self.embedding_dim, self.num_meta)

        self.exo_encoder = ExogenousInvertedEncoder(self.embedding_dim, self.exo_input_len)
        self.endo_encoder = EndogenousEncoder(self.embedding_dim, self.endo_input_len, self.num_endo_vars)

        self.encoders = nn.ModuleList(
            [
                nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=self.embedding_dim, 
                        nhead=4, 
                        dropout=0.2,
                        batch_first=True,
                    ), 
                    num_layers=1
                )
            ] * self.num_layers
        )
        self.decoders = nn.ModuleList(
            [   
                nn.TransformerDecoder(
                    TransformerDecoderLayer(
                        d_model=self.hidden_dim, 
                        nhead=4, 
                        dim_feedforward=self.hidden_dim * 4, 
                        dropout=0.2,
                    ),
                    num_layers=1,
                )
            ] * self.num_layers
        )
        self.decoder_fc = nn.Sequential(
            nn.Linear(self.embedding_dim, self.output_len),
            nn.Dropout(0.2)
        )


    def forward(self, endo_inputs, exo_inputs, release_dates, image_embedding, text_embedding, meta_data):
        temporal_embedding = self.temporal_feature_encoder(release_dates)
        fusion_embedding = self.feature_fusion_network(image_embedding, text_embedding, temporal_embedding, meta_data)
        
        exo_emb = self.exo_encoder(exo_inputs)
        endo_emb = self.endo_encoder(endo_inputs, fusion_embedding)
        
        for l in range(self.num_layers):
            endo_emb = self.encoders[l](endo_emb)
            tgt = endo_emb[:,:1,:].permute(1,0,2)
            memory = exo_emb
            cross_emb, attn_weights = self.decoders[l](tgt, memory)
            endo_emb = torch.cat([cross_emb.permute(1,0,2), endo_emb[:, 1:, :]], dim=1)

        forecast = self.decoder_fc(cross_emb)
        return forecast.view(-1, self.output_len), attn_weights