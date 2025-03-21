from torchmetrics.regression import SymmetricMeanAbsolutePercentageError, WeightedMeanAbsolutePercentageError, MeanSquaredError, MeanAbsoluteError
    
def get_score(gt, pred):
    pred = pred.detach().cpu()
    gt = gt.detach().cpu()
    adjusted_smape = SymmetricMeanAbsolutePercentageError()
    weighted_mape = WeightedMeanAbsolutePercentageError()
    mean_squared_error = MeanSquaredError()
    mean_absolute_error = MeanAbsoluteError()
    
    score = {}
    score['adjusted_smape'] = adjusted_smape(pred, gt) * 0.5
    score['wape'] = weighted_mape(pred, gt)
    score['mse'] = mean_squared_error(pred, gt)
    score['mae'] = mean_absolute_error(pred, gt)

    return score