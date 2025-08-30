import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

def plot_equity(dates, curves_dict):
    """
    dates: pandas Series (master date index)
    curves_dict: {name: pd.Series or array-like}, any length; will be aligned to dates.index
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, s in curves_dict.items():
        s = pd.Series(s)  # ensure Series
        # align to the master index; matplotlib will ignore NaNs (gap at start)
        s_aligned = s.reindex(dates.index)
        ax.plot(dates, s_aligned, label=name)
    ax.set_title("Equity Curves (₹1 start)")
    ax.set_xlabel("Date"); ax.set_ylabel("Wealth")
    ax.legend()
    return fig

def plot_allocation(dates, picks, labels):
    map_idx = {lab:i for i,lab in enumerate(labels)}
    y = [map_idx[a] for a in picks]
    fig, ax = plt.subplots(figsize=(8,2.2))
    ax.scatter(dates, y, s=3)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_title("Chosen Arm Over Time"); ax.set_xlabel("Date")
    return fig


# ===== Plotly helpers for cleaner allocation views =====


def _rle_segments(dates: pd.Series, picks_labels: list[str]) -> pd.DataFrame:
    """
    Run-length encode daily labels -> segments with [start, end, arm].
    Assumes `dates` and `picks_labels` are aligned and contiguous.
    """
    dates = dates.reset_index(drop=True)
    if len(picks_labels) == 0:
        return pd.DataFrame(columns=["start", "end", "arm"])
    segs = []
    start_i = 0
    cur = picks_labels[0]
    for i in range(1, len(picks_labels)):
        if picks_labels[i] != cur:
            segs.append((dates.iloc[start_i], dates.iloc[i-1], cur))
            start_i = i
            cur = picks_labels[i]
    segs.append((dates.iloc[start_i], dates.iloc[len(picks_labels)-1], cur))
    return pd.DataFrame(segs, columns=["start", "end", "arm"])

def allocation_timeline_plotly(dates: pd.Series, picks_labels: list[str], labels_order: list[str]):
    """
    Timeline plot that compresses consecutive identical choices into bars.
    - dates: pd.Series of datetimes (already filtered to the window you want)
    - picks_labels: list[str] like ["91d","364d","91d",...], same length as dates
    - labels_order: y-axis order, e.g. ["91d","364d","10y"]
    """
    segs = _rle_segments(dates, picks_labels)
    if segs.empty:
        return px.line(title="No data in window")
    fig = px.timeline(
        segs, x_start="start", x_end="end", y="arm", color="arm",
        category_orders={"arm": labels_order},
        title="Allocation timeline (compressed runs)"
    )
    fig.update_yaxes(autorange="reversed")  # put first label at bottom
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
    return fig

def rolling_choice_share(dates: pd.Series, picks_labels: list[str], labels_order: list[str], window: int = 60):
    """
    Area chart of rolling fraction of days spent in each arm over a trailing window.
    """
    d = pd.DataFrame({"Date": dates.reset_index(drop=True), "arm": picks_labels})
    parts = []
    for lab in labels_order:
        parts.append(d["arm"].eq(lab).rolling(window, min_periods=1).mean().rename(lab))
    shares = pd.concat(parts, axis=1)
    shares["Date"] = d["Date"]
    melt = shares.melt("Date", var_name="Arm", value_name="Share")
    fig = px.area(melt, x="Date", y="Share", color="Arm",
                  title=f"Rolling {window}-day allocation share")
    fig.update_layout(height=220, margin=dict(l=10, r=10, t=40, b=10))
    fig.update_yaxes(tickformat=".0%")
    return fig
