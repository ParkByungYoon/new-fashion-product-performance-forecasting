import torch.nn as nn
import torch.nn.functional as F

from layer.Transformer import *
from model.Lightning import PytorchLightningBase

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class Transformer(PytorchLightningBase):
    def __init__(self, args):
        super().__init__()
        self.input_len = args.input_len
        self.output_len = args.output_len
        self.hidden_dim = args.hidden_dim
        self.embedding_dim = args.embedding_dim
        self.lr = args.learning_rate
        self.num_heads = args.num_heads
        self.num_vars = args.num_vars
        self.num_layers = args.num_layers
        self.segment_len = args.segment_len
        self.save_hyperparameters()

        self.transformer_encoder = TransformerEncoder(self.hidden_dim, self.input_len, self.num_vars)
        self.temporal_feature_encoder = TemporalFeatureEncoder(self.embedding_dim)
        self.feature_fusion_network = FeatureFusionNetwork(self.embedding_dim, self.hidden_dim)

        decoder_layer = TransformerDecoderLayer(d_model=self.hidden_dim, nhead=self.num_heads, dim_feedforward=self.hidden_dim * 4, dropout=0.1)
        self.decoder = nn.TransformerDecoder(decoder_layer, self.num_layers)
        self.decoder_fc = nn.Sequential(
            nn.Linear(self.hidden_dim, self.output_len),
            nn.Dropout(0.2)
        )
    
    def forward(self, inputs, release_dates, image_embedding, text_embedding, meta_data):
        inputs, _ = self.split_inputs(inputs, meta_data)
        encoder_embedding = self.transformer_encoder(inputs)
        temporal_embedding = self.temporal_feature_encoder(release_dates)
        fusion_embedding = self.feature_fusion_network(image_embedding, text_embedding, temporal_embedding, meta_data)
            
        tgt = fusion_embedding.unsqueeze(0)
        memory = encoder_embedding
        decoder_out, attn_weights = self.decoder(tgt, memory)
        forecast = self.decoder_fc(decoder_out)

        return forecast.view(-1, self.output_len), attn_weights

    def phase_step(self, batch, phase):
        item_sales, inputs, release_dates, image_embeddings, text_embeddings, meta_data = batch
        forecasted_sales, _ = self.forward(inputs, release_dates, image_embeddings, text_embeddings, meta_data)
        if phase != 'predict':
            score = self.get_score(item_sales, forecasted_sales)
            score['loss'] = F.mse_loss(item_sales, forecasted_sales.squeeze())
            self.log_dict({f"{phase}_{k}":v for k,v in score.items()}, on_step=False, on_epoch=True)

            rescaled_score = self.get_score(item_sales * 1820, forecasted_sales * 1820)
            self.log_dict({f"{phase}_rescaled_{k}":v for k,v in rescaled_score.items()}, on_step=False, on_epoch=True)
            return score['loss']
        else:
            return forecasted_sales * 1820