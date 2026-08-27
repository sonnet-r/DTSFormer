import numpy as np


EPSILON = 1e-8


def RSE(pred, true):
    numerator = np.sqrt(np.sum((true - pred) ** 2))
    denominator = np.sqrt(np.sum((true - true.mean()) ** 2))
    return numerator / max(denominator, EPSILON)


def CORR(pred, true):
    pred_centered = pred - pred.mean(axis=0)
    true_centered = true - true.mean(axis=0)
    numerator = np.sum(true_centered * pred_centered, axis=0)
    denominator = np.sqrt(
        np.sum(true_centered**2, axis=0) * np.sum(pred_centered**2, axis=0)
    )
    correlation = numerator / np.maximum(denominator, EPSILON)
    return np.mean(correlation)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    denominator = np.maximum(np.abs(true), EPSILON)
    return np.mean(np.abs((pred - true) / denominator))


def MSPE(pred, true):
    denominator = np.maximum(np.abs(true), EPSILON)
    return np.mean(np.square((pred - true) / denominator))


def metric(pred, true):
    return (
        MAE(pred, true),
        MSE(pred, true),
        RMSE(pred, true),
        MAPE(pred, true),
        MSPE(pred, true),
        RSE(pred, true),
        CORR(pred, true),
    )
