from MVTSF.layer.TimeXer import *
from MVTSF.model.Transformer import Transformer
from MVTSF.util.metric import get_score

    
class TimeXer(Transformer):
    def __init__(self, args):
        super().__init__(args)
        self.save_hyperparameters()
        self.num_endo_vars = args.num_endo_vars
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
        # means = endo_inputs.mean(1, keepdim=True)
        # endo_inputs = endo_inputs - means
        # stdev = torch.sqrt(torch.var(endo_inputs, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # endo_inputs /= stdev
        endo_inputs = self.revin_layer(endo_inputs.unsqueeze(2), 'norm').squeeze()

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
        # forecast = forecast.view(-1, self.output_len) * stdev + means
        forecast = self.revin_layer(forecast.view(-1, self.output_len, 1), 'denorm').squeeze()

        return forecast, attn_weights


    def phase_step(self, batch, phase):
        item_sales, endo_inputs, exo_inputs, release_dates, image_embeddings, text_embeddings, meta_data = batch
        forecasted_sales, _ = self.forward(endo_inputs, exo_inputs, release_dates, image_embeddings, text_embeddings, meta_data)
        
        rescaled_forecasted_sales = torch.clamp(forecasted_sales, min=0)
        rescaled_score = get_score(item_sales, forecasted_sales)
        rescaled_score['loss'] = F.mse_loss(item_sales, forecasted_sales)

        if phase == 'predict': 
            return rescaled_forecasted_sales

        self.log_dict({f"{phase}_rescaled_{k}":v for k,v in rescaled_score.items()}, on_step=False, on_epoch=True)
        return rescaled_score['loss']


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x