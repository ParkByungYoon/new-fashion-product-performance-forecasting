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
        self.hidden_dim = args.hidden_dim
        self.embedding_dim = args.embedding_dim
        self.lr = args.learning_rate
        self.num_heads = args.num_heads
        self.num_exo_vars = args.num_exo_vars
        self.num_meta = args.num_meta
        self.num_layers = args.num_layers
        self.segment_len = args.segment_len
        self.center = args.center
        self.scale = args.scale
        self.save_hyperparameters()

        self.transformer_encoder = TransformerEncoder(self.hidden_dim, self.exo_input_len, self.num_exo_vars)
        self.temporal_feature_encoder = TemporalFeatureEncoder(self.embedding_dim)
        self.feature_fusion_network = FeatureFusionNetwork(self.embedding_dim, self.num_meta)

        decoder_layer = TransformerDecoderLayer(d_model=self.hidden_dim, nhead=self.num_heads, dim_feedforward=self.hidden_dim * 4, dropout=0.1)
        self.decoder = nn.TransformerDecoder(decoder_layer, self.num_layers)
        self.decoder_fc = nn.Sequential(
            nn.Linear(self.hidden_dim, self.output_len),
            nn.Dropout(0.2)
        )
    
    def forward(self, endo_inputs, exo_inputs, release_dates, image_embeddings, text_embeddings, meta_data):
        encoder_embedding = self.transformer_encoder(exo_inputs)
        
        temporal_embedding = self.temporal_feature_encoder(release_dates)
        fusion_embedding = self.feature_fusion_network(image_embeddings, text_embeddings, temporal_embedding, meta_data)
            
        tgt = fusion_embedding.unsqueeze(0)
        memory = encoder_embedding
        decoder_out, attn_weights = self.decoder(tgt, memory)
        forecast = self.decoder_fc(decoder_out)

        return forecast.view(-1, self.output_len), attn_weights

    def phase_step(self, batch, phase):
        item_sales, endo_inputs, exo_inputs, release_dates, image_embeddings, text_embeddings, meta_data = batch
        # if phase in ['train', 'valid']:
        #     center, scale = 39.9502, 72.5971
        #     # center, scale = 22.9259, 37.9783
        #     # center, scale = 0, 875
        # elif phase in ['test', 'predict']:
        #     center, scale = 40.8935, 74.3419
        #     # center, scale = 22.7742, 37.4761
        #     # center, scale = 0, 875
        center, scale = self.center, self.scale
        item_sales = self.normalize(item_sales, center, scale)
        
        forecasted_sales, _ = self.forward(endo_inputs, exo_inputs, release_dates, image_embeddings, text_embeddings, meta_data)
        if phase != 'predict':
            score = get_score(item_sales, forecasted_sales)
            score['loss'] = F.mse_loss(item_sales, forecasted_sales.squeeze())
            self.log_dict({f"{phase}_{k}":v for k,v in score.items()}, on_step=False, on_epoch=True)

            rescaled_sales = self.denormalize(item_sales, center, scale)
            rescaled_forecasted_sales = self.denormalize(forecasted_sales, center, scale)
            rescaled_score = get_score(rescaled_sales, rescaled_forecasted_sales)
            self.log_dict({f"{phase}_rescaled_{k}":v for k,v in rescaled_score.items()}, on_step=False, on_epoch=True)
            return score['loss']
        else:
            return self.denormalize(forecasted_sales, center, scale)
    
    def denormalize(self, x, center, scale):
        return (x * scale) + center

    def normalize(self, x, center, scale):
        return (x - center) / scale