from MVTSF.layer.Transformer import *

class FullAttentionTransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, input_len, segment_len):
        super().__init__()
        self.input_linear = SegmentEmbedding(embedding_dim, segment_len)
        self.num_segments = input_len//segment_len
        self.pos_embedding = PositionalEncoding(embedding_dim, max_len=self.num_segments)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=4, dropout=0.2)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, inputs):
        batch, num_vars, input_len = inputs.shape # (64, 3, 52)
        emb = self.input_linear(inputs) # (64, 3, 13, 512)

        emb = rearrange(emb, 'b d num_segments embedding_dim -> (b d) num_segments embedding_dim') # (64*3, 13, 512)
        emb = self.pos_embedding(emb)
        emb = rearrange(emb, '(b d) num_segments embedding_dim-> b (num_segments d) embedding_dim', b = batch) # (64, 13*3, 512)
        emb = self.encoder(emb)
        
        return emb