import pandas as pd

def load_yields(path_csv: str) -> pd.DataFrame:
    df = pd.read_csv(path_csv, parse_dates=["Date"])
    return df.sort_values("Date").reset_index(drop=True)

def build_rewards(df, k91=0.0, k364=0.25, k10=0.50):
    out = df.copy()
    for col in ["Yield_91d","Yield_364d","Yield_10y"]:
        out[f"d_{col}"] = out[col].diff()
    def r(y_col, d_col, k):
        gross = (out[y_col] / 100.0) / 365.0
        pen   = k * (out[d_col].abs() / 100.0)  # unit match
        return gross - pen
    out["r_91d"]  = r("Yield_91d","d_Yield_91d",k91)
    out["r_364d"] = r("Yield_364d","d_Yield_364d",k364)
    out["r_10y"]  = r("Yield_10y","d_Yield_10y",k10)
    return out

def build_features(df, use_slope=True, use_dslope=True, use_mom=True):
    feat = pd.DataFrame(index=df.index)
    if use_slope:
        feat["slope"] = df["Yield_10y"] - df["Yield_91d"]
    if use_dslope:
        s = df["Yield_10y"] - df["Yield_91d"]
        feat["dslope"] = s.diff()
    if use_mom:
        feat["dm10"] = df["Yield_10y"].diff().rolling(5).mean()
    return feat.shift(1)  # lag 1 day
