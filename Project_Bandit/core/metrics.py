# core/metrics.py
import numpy as np
import pandas as pd

def equity_curve(r: pd.Series, start=1.0) -> pd.Series:
    r = pd.Series(r).fillna(0.0)
    return start * (1.0 + r).cumprod()

def ann_ret_vol_sharpe(r: pd.Series, rf: pd.Series | None = None):
    """
    r  : strategy daily returns (decimal)
    rf : optional risk-free daily returns (decimal). If given, metrics use (r - rf).
    """
    r = pd.Series(r).dropna()

    if rf is not None:
        rf = pd.Series(rf).reindex(r.index).ffill()
        ex = r - rf
    else:
        ex = r

    ann_ret = ex.mean() * 252
    ann_vol = ex.std(ddof=0) * np.sqrt(252)
    sharpe  = np.nan if ann_vol == 0 else (ann_ret / ann_vol)
    return ann_ret, ann_vol, sharpe
