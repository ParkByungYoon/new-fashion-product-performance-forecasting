import torch
from torchmetrics.regression import R2Score, SymmetricMeanAbsolutePercentageError, WeightedMeanAbsolutePercentageError, MeanSquaredError, MeanAbsoluteError

def get_score(gt, pred):
    adjusted_smape = SymmetricMeanAbsolutePercentageError()
    weighted_mape = WeightedMeanAbsolutePercentageError()
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
        adj_smape, wmape, r2, mse, mae = [[] for i in range(5)]
        for i in range(len(gt)):
            adj_smape.append(adjusted_smape(pred[i], gt[i]) * 0.5)
            wmape.append(weighted_mape(pred[i], gt[i]))
            r2.append(r2_score(pred[i], gt[i]))
            mse.append(mean_squared_error(pred[i], gt[i]))
            mae.append(mean_absolute_error(pred[i], gt[i]))

        score['adjusted_smape'] = torch.mean(torch.stack(adj_smape))
        score['wape'] = torch.mean(torch.stack(wmape))
        score['r2_score'] = torch.mean(torch.stack(r2))
        score['mse'] = torch.mean(torch.stack(mse))
        score['mae'] = torch.mean(torch.stack(mae))

    return score