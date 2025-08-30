import numpy as np
import pandas as pd

def equity_curve(r: pd.Series) -> pd.Series:
    return (1 + r.fillna(0)).cumprod()

def ann_ret_vol_sharpe(r: pd.Series, periods=252):
    r = r.dropna()
    mu = r.mean(); sd = r.std()
    ann_ret = mu * periods
    ann_vol = sd * (periods ** 0.5)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan
    return ann_ret, ann_vol, sharpe
