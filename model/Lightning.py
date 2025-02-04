import torch
import pytorch_lightning as pl
from torchmetrics.regression import R2Score, SymmetricMeanAbsolutePercentageError, MeanSquaredError, MeanAbsoluteError

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class PytorchLightningBase(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, train_batch, batch_idx):
        output = self.phase_step(train_batch, phase='train')
        return output

    def validation_step(self, valid_batch, batch_idx):
        self.eval()
        with torch.no_grad():
            self.phase_step(valid_batch, phase='valid')

    def test_step(self, test_batch, batch_idx):
        self.eval()
        with torch.no_grad():
            self.phase_step(test_batch, phase='test')

    def predict_step(self, predict_batch, batch_idx):
        self.eval()
        with torch.no_grad():
            predictions = self.phase_step(predict_batch, phase='predict')
        return predictions

    def get_score(self, gt, pred):
        adjusted_smape = SymmetricMeanAbsolutePercentageError()
        r2_score = R2Score()
        mean_squared_error = MeanSquaredError()
        mean_absolute_error = MeanAbsoluteError()
        score = {}

        pred = pred.detach().cpu()
        gt = gt.detach().cpu()

        if gt.dim() == 1:
            adj_smape = adjusted_smape(pred,gt)*0.5
            r2 = r2_score(pred,gt)
            mse = mean_squared_error(pred,gt)
            mae = mean_absolute_error(pred,gt)
        else:
            adj_smape, r2, mse, mae = [[] for i in range(4)]
            for i in range(len(gt)):
                adj_smape.append(adjusted_smape(pred[i], gt[i]) * 0.5)
                r2.append(r2_score(pred[i], gt[i]))
                mse.append(mean_squared_error(pred[i], gt[i]))
                mae.append(mean_absolute_error(pred[i], gt[i]))

            score['adjusted_smape'] = torch.mean(torch.stack(adj_smape))
            score['r2_score'] = torch.mean(torch.stack(r2))
            score['mse'] = torch.mean(torch.stack(mse))
            score['mae'] = torch.mean(torch.stack(mae))
            # score['accum_adjusted_smape'] = adjusted_smape(pred.sum(axis=1), gt.sum(axis=1)) * 0.5

        return score

    def split_inputs(self, inputs, meta_data):
        batch_size, num_vars = meta_data.shape
        endo_idx = torch.stack([meta_data[:,:27].argmax(dim=1),\
                                meta_data[:,27:34].argmax(dim=1)+27, \
                                meta_data[:,34:].argmax(dim=1)+34],axis=-1)
        gather_idx = endo_idx.unsqueeze(-1).expand(-1,-1, 52)
        endo_inputs = inputs.gather(dim=1, index=gather_idx)

        mask = torch.ones((batch_size, num_vars), dtype=torch.bool)
        rows = torch.arange(batch_size).unsqueeze(-1).expand(-1, 3)
        mask[rows, endo_idx] = False
        exo_inputs = inputs[mask].view(batch_size, -1, 52)
        return endo_inputs, exo_inputs