from MVTSF.layer.Transformer import *

class TimerEncoder(nn.Module):
    def __init__(self, output_dim, input_len, segment_len, num_heads=4, dropout=0.2):
        super().__init__()
        self.input_linear = SegmentEmbedding(output_dim, segment_len)
        self.num_segments = input_len//segment_len
        self.pos_embedding = PositionalEncoding(output_dim, max_len=self.num_segments)
        encoder_layer = nn.TransformerEncoderLayer(d_model=output_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, inputs):
        batch, num_vars, input_len = inputs.shape # (64, 50, 52)
        emb = self.input_linear(inputs) # (64, 50, 13, 512)
        emb = rearrange(emb, 'b d num_segments embedding_dim -> (b d) num_segments embedding_dim') # (64*50, 13, 512)
        emb = self.pos_embedding(emb)
        emb = rearrange(emb, '(b d) num_segments embedding_dim-> b (num_segments d) embedding_dim', b = batch) # (64, 13*50, 512)
        
        dependency_mask = torch.ones(num_vars, num_vars)
        # dependency_mask = torch.eye(num_vars)
        # dependency_mask[:n,:n] = 1
        time_mask = (torch.triu(torch.ones(self.num_segments, self.num_segments)) == 1).transpose(0, 1)
        time_mask = time_mask.float().contiguous()
        mask = torch.kron(dependency_mask, time_mask)
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to(inputs.device)
        emb = self.encoder(emb, mask)
        # emb = rearrange(emb, 'b (num_segments d) embedding_dim-> b d num_segments embedding_dim', d = num_vars) # (64, 52, 12, 512)
        # emb = emb[:,:n]
        # emb = rearrange(emb, 'b d num_segments embedding_dim-> b (d num_segments) embedding_dim') # (64, 4*12, 512)
        return emb