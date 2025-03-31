from MVTSF.layer.TimeXer import *
from MVTSF.model.Transformer import Transformer
from MVTSF.util.metric import get_score

    
class TimeXer(Transformer):
    def __init__(self, args):
        super().__init__(args)
        self.save_hyperparameters()
        self.num_endo_vars = 1
        self.num_exo_vars = args.num_exo_vars

        self.exo_encoder = ExogenousEncoder(self.output_dim, self.exo_input_len)
        self.endo_encoder = EndogenousEncoder(self.output_dim, self.endo_input_len, self.num_endo_vars)

        self.revin_layer = RevIN(num_features=self.num_endo_vars)
        self.encoders = nn.ModuleList(
            [
                nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=self.output_dim, 
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
                        d_model=self.output_dim, 
                        nhead=4, 
                        dim_feedforward=self.output_dim * 4, 
                        dropout=0.2,
                    ),
                    num_layers=1,
                )
            ] * self.num_layers
        )


    def forward(self, endo_inputs, exo_inputs, release_dates, image_embedding, text_embedding, meta_data):
        if self.use_revin:
            endo_inputs = self.revin_layer(endo_inputs.unsqueeze(2), 'norm').squeeze()
        else:
            endo_inputs = self.normalize(endo_inputs)

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
        if self.use_revin:
            forecast = self.revin_layer(forecast.view(-1, self.output_len, 1), 'denorm')
        else:
            forecast = self.denormalize(forecast)

        return forecast.view(-1, self.output_len), attn_weights