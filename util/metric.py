from torchmetrics.regression import SymmetricMeanAbsolutePercentageError, WeightedMeanAbsolutePercentageError, MeanSquaredError, MeanAbsoluteError
import torch

def weighted_mean_absolute_percentage_error(gt: torch.Tensor, pred: torch.Tensor) -> float:
    abs_error = torch.abs(gt - pred)
    sum_abs_error = torch.sum(torch.sum(abs_error, dim=-1))
    sum_gt = torch.sum(gt)
    return 100 * (sum_abs_error / sum_gt).item()

def get_score(gt, pred):
    adjusted_smape = SymmetricMeanAbsolutePercentageError()
    weighted_mape = WeightedMeanAbsolutePercentageError()
    mean_squared_error = MeanSquaredError()
    mean_absolute_error = MeanAbsoluteError()
    
    score = {}
    score['adjusted_smape'] = adjusted_smape(pred, gt) * 0.5
    score['flattened_wape'] = weighted_mean_absolute_percentage_error(pred, gt)
    score['wape'] = weighted_mape(pred, gt)
    score['mse'] = mean_squared_error(pred, gt)
    score['mae'] = mean_absolute_error(pred, gt)
    return score