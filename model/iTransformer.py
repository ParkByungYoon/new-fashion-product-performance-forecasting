from MVTSF.model.Transformer import Transformer
from MVTSF.layer.iTransformer import InversedTransformerEncoder
import torch

class iTransformer(Transformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transformer_encoder = InversedTransformerEncoder(self.hidden_dim, self.input_len)


class iTransformerV2(iTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, inputs, release_dates, image_embedding, text_embedding, meta_data):
        encoder_embedding = self.transformer_encoder(inputs)
        meta_idx = torch.stack([meta_data[:,:27].argmax(dim=1),\
                                meta_data[:,27:34].argmax(dim=1)+27, \
                                meta_data[:,34:].argmax(dim=1)+34],axis=-1).unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        encoder_embedding = encoder_embedding.gather(dim=1, index=meta_idx)

        temporal_embedding = self.temporal_feature_encoder(release_dates)
        fusion_embedding = self.feature_fusion_network(image_embedding, text_embedding, temporal_embedding, meta_data)
            
        tgt = fusion_embedding.unsqueeze(0)
        memory = encoder_embedding
        decoder_out, attn_weights = self.decoder(tgt, memory)
        forecast = self.decoder_fc(decoder_out)

        return forecast.view(-1, self.output_len), attn_weights