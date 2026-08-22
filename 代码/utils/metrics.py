"""评估指标与统计检验。"""
import numpy as np
from scipy.stats import wilcoxon


def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > 1e-9
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def eval_metrics(y_true, y_pred):
    return {"MAPE": mape(y_true, y_pred), "R2": r2(y_true, y_pred), "RMSE": rmse(y_true, y_pred)}


def wilcoxon_p(ae_a, ae_b):
    """两组逐样本绝对误差差异是否显著（双尾）。ae 越小越好。"""
    try:
        diff = np.asarray(ae_a, dtype=float) - np.asarray(ae_b, dtype=float)
        if np.all(diff == 0):
            return 1.0
        stat, p = wilcoxon(np.asarray(ae_a), np.asarray(ae_b))
        return float(p)
    except Exception:
        return float("nan")


def summarize(per_seed_metrics):
    """per_seed_metrics: list[dict] -> {metric: 'mean±std'}."""
    out = {}
    keys = per_seed_metrics[0].keys()
    for k in keys:
        arr = np.array([m[k] for m in per_seed_metrics], dtype=float)
        out[k] = f"{arr.mean():.3f}±{arr.std():.3f}"
        out[k + "_mean"] = float(arr.mean())
        out[k + "_std"] = float(arr.std())
    return out
