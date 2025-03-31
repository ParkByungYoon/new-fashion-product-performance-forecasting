import torch.nn as nn
import torch.nn.functional as F

from layer.Transformer import *
from model.Lightning import PytorchLightningBase

from util.metric import get_score

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class Transformer(PytorchLightningBase):
    def __init__(self, args):
        super().__init__()
        self.endo_input_len = args.endo_input_len
        self.exo_input_len = args.exo_input_len
        self.output_len = args.output_len
        self.output_dim = args.output_dim
        self.input_dim = args.input_dim
        self.lr = args.learning_rate
        self.num_heads = args.num_heads
        self.num_exo_vars = args.num_exo_vars
        self.num_meta = args.num_meta
        self.num_layers = args.num_layers
        self.segment_len = args.segment_len
        self.use_endo = args.use_endo
        self.use_revin = args.use_revin
        if self.use_revin:
            self.revin_layer = RevIN(num_features=1)
        else:
            self.center = args.center
            self.scale = args.scale
        self.save_hyperparameters()

        self.transformer_encoder = TransformerEncoder(self.output_dim, self.exo_input_len, self.num_exo_vars+1 if args.use_endo else self.num_exo_vars)
        self.temporal_feature_encoder = TemporalFeatureEncoder(self.input_dim)
        self.feature_fusion_network = FeatureFusionNetwork(self.input_dim, self.output_dim, self.num_meta)

        decoder_layer = TransformerDecoderLayer(d_model=self.output_dim, nhead=self.num_heads, dim_feedforward=self.output_dim * 4, dropout=0.1)
        self.decoder = nn.TransformerDecoder(decoder_layer, self.num_layers)
        self.decoder_fc = nn.Sequential(
            nn.Linear(self.output_dim, self.output_len),
            nn.Dropout(0.2)
        )
    
    def forward(self, endo_inputs, exo_inputs, release_dates, image_embeddings, text_embeddings, meta_data):
        if self.use_endo:
            if self.use_revin:
                endo_inputs = self.revin_layer(endo_inputs.unsqueeze(2), 'norm').squeeze()
            else:
                endo_inputs = self.normalize(endo_inputs)

            if self.num_exo_vars == 0:
                exo_inputs = endo_inputs.unsqueeze(1)
            else:
                exo_inputs = torch.cat([exo_inputs, endo_inputs.unsqueeze(1)], dim=1) 

        encoder_embedding = self.transformer_encoder(exo_inputs)
        
        temporal_embedding = self.temporal_feature_encoder(release_dates)
        fusion_embedding = self.feature_fusion_network(image_embeddings, text_embeddings, temporal_embedding, meta_data)
        
        tgt = fusion_embedding.unsqueeze(0)
        memory = encoder_embedding
        decoder_out, attn_weights = self.decoder(tgt, memory)
        forecast = self.decoder_fc(decoder_out)

        if self.use_revin:
            forecast = self.revin_layer(forecast.view(-1, self.output_len, 1), 'denorm')
        else:
            forecast = self.denormalize(forecast)
            
        return forecast.view(-1, self.output_len), attn_weights

    def phase_step(self, batch, phase):
        item_sales, endo_inputs, exo_inputs, release_dates, image_embeddings, text_embeddings, meta_data = batch
        forecasted_sales, _ = self.forward(endo_inputs, exo_inputs, release_dates, image_embeddings, text_embeddings, meta_data)
        
        loss = F.mse_loss(item_sales, forecasted_sales)
        forecasted_sales = torch.clamp(forecasted_sales, min=0)
        score = get_score(item_sales, forecasted_sales)

        self.log_dict({f"{phase}_rescaled_{k}":v for k,v in score.items()}, on_step=False, on_epoch=True)
        return loss if phase != 'predict' else forecasted_sales
    
    def normalize(self, x):
        return (x - self.center) / self.scale
    
    def denormalize(self, x):
        return (x * self.scale) + self.center
    