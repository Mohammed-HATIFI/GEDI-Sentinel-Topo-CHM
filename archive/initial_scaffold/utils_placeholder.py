"""
Utility functions for GEDI-Sentinel canopy height estimation.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(y_true, y_pred, mask=None):
    """
    Calculate evaluation metrics.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        mask: Optional mask for valid pixels

    Returns:
        Dictionary of metrics
    """
    if mask is not None:
        y_true = y_true[mask]
        y_pred = y_pred[mask]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # Regression slope
    slope = np.polyfit(y_true, y_pred, 1)[0]

    # Bias
    bias = np.mean(y_pred - y_true)

    # Spread (predicted std / observed std)
    spread = np.std(y_pred) / (np.std(y_true) + 1e-8)

    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'slope': slope,
        'bias': bias,
        'spread': spread
    }


def height_class_metrics(y_true, y_pred, height_bins=None, mask=None):
    """
    Calculate metrics stratified by canopy height classes.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        height_bins: Bin edges for height classes
        mask: Optional mask for valid pixels

    Returns:
        Dictionary of metrics per height class
    """
    if mask is not None:
        y_true = y_true[mask]
        y_pred = y_pred[mask]

    if height_bins is None:
        height_bins = [0, 5, 10, 15, 20, 100]

    results = {}
    for i in range(len(height_bins) - 1):
        class_mask = (y_true >= height_bins[i]) & (y_true < height_bins[i + 1])
        if class_mask.sum() > 0:
            class_true = y_true[class_mask]
            class_pred = y_pred[class_mask]
            results[f'{height_bins[i]}-{height_bins[i+1]}m'] = calculate_metrics(
                class_true, class_pred
            )

    return results


def mse_decomposition(y_true, y_pred, mask=None):
    """
    Decompose MSE into variance and covariance components.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        mask: Optional mask for valid pixels

    Returns:
        Dictionary with MSE decomposition
    """
    if mask is not None:
        y_true = y_true[mask]
        y_pred = y_pred[mask]

    mse = mean_squared_error(y_true, y_pred)

    # Bias component
    bias_component = (np.mean(y_pred) - np.mean(y_true)) ** 2

    # Variance component
    var_obs = np.var(y_true)
    var_pred = np.var(y_pred)
    variance_component = (var_pred - var_obs) ** 2 + 2 * var_obs * var_pred * (1 - np.corrcoef(y_true, y_pred)[0, 1])

    return {
        'MSE': mse,
        'bias_component': bias_component,
        'variance_component': variance_component
    }
