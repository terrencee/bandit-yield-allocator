# --- path shim so we can import ../core ---
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime  # <-- NEW

from core.data_loader import load_yields, build_rewards, build_features
from core.bandits import epsilon_greedy, linucb_disjoint
from core.metrics import equity_curve, ann_ret_vol_sharpe
from core.plots import plot_equity, plot_allocation

# Try to import the nicer timeline plots (optional).
try:
    from core.plots import allocation_timeline_plotly, rolling_choice_share
    HAS_TIMELINE = True
except Exception:
    HAS_TIMELINE = False

# Persist "ready" state across reruns
if "ready" not in st.session_state:
    st.session_state["ready"] = False



# ---------- helpers ----------
def resolve_path(p: str) -> str:
    """Return absolute path for relative strings like ../data/... from app/ folder."""
    if os.path.isabs(p):
        return p
    base = os.path.abspath(os.path.dirname(__file__))  # .../Project_Bandit/app
    return os.path.abspath(os.path.join(base, p))


@st.cache_data(show_spinner=False)
def _load_from_disk(abs_path: str) -> pd.DataFrame:
    df = pd.read_csv(abs_path, parse_dates=["Date"])
    return df.sort_values("Date").reset_index(drop=True)
# -----------------------------


st.set_page_config(page_title="Bandit Yield Allocator", layout="wide")
st.title("Bandit Allocation in Indian Yield Curve")

with st.sidebar:
    st.header("Inputs")

    # Either a relative path or an uploaded CSV
    csv_path = st.text_input(
        "Cleaned CSV (relative to app/)",
        "../data/yields_daily_91d_364d_10y.csv",
        help="Path from the app/ folder to your pre-cleaned daily yields CSV."
    )
    uploaded = st.file_uploader("…or upload CSV", type=["csv"])

    st.subheader("Reward Penalties (k × |Δyield|)")
    k91  = st.number_input("k (91d)",  0.0, 2.0, 0.00, 0.05,
                           help="Penalty for 91-day bill when yields move. Set near 0 for very low duration risk."
                           " Higher k means more reluctance to switch to this arm."
                           " (In practice, 91d has the lowest duration/roll risk.)"
                           "having the lowest duration/roll risk means it should have the lowest k.)")
    k364 = st.number_input("k (364d)", 0.0, 2.0, 0.25, 0.05,
                           help="Penalty per unit of daily yield change for 364-day bill (duration/roll risk)."
                           " Should be between k91 and k10."
                           " (In practice, 364d has medium duration/roll risk.)")
    k10  = st.number_input("k (10y)",  0.0, 2.0, 0.50, 0.05,
                           help="Penalty for 10-year G-Sec (highest duration risk among the three)."
                           " Should be the highest among the three ks.")
    switch_bps = st.number_input("Switch cost (bps)", 0.0, 10.0, 1.0, 0.5,
                                 help="Transaction/slippage cost when switching arms. 1 bps = 0.01%."
                                 " Discourages frequent switching in ε-greedy."
                                 " (LinUCB is less sensitive to this parameter.)"
                                 " Set to 0 for no cost."
                                 " Typical real-world costs may be around 1-2 bps."
                                 " Higher values discourage switching more strongly."
                                 " bps means 'basis points', where 1 bps = 0.01% = 0.0001 in decimal.")

    st.subheader("ε-Greedy",
                 help="Simple non-contextual bandit: with probability ε, explore a random arm; "
                      "otherwise exploit the best estimated arm."
                      " Includes a small switch cost to avoid excessive trading."
                      " Requires no features; just the reward history."
                      " Good baseline to compare against."
                      " Works well if one arm is clearly best most of the time."
                      " More exploration (higher ε) helps if the best arm changes often."
                      " e_greedy is fast and simple, but doesn't adapt to context like LinUCB.")
    eps  = st.slider("epsilon", 0.0, 0.2, 0.05, 0.01,
                     help="Exploration rate. With probability ε, try a random arm to keep learning."
                     " Higher ε means more exploration, which can help if the best arm changes often."
                     " But too high ε means too much random exploration, hurting returns.")
    seed = st.number_input("seed", 0, 9999, 0, 1,
                           help="Random seed for reproducible ε-greedy. (LinUCB is deterministic here.)"
                           " Change to get a different random exploration sequence."
                           " Only matters if ε > 0."
                           " Different seeds can lead to different results due to randomness."
                           " Try a few seeds to see variability."
                           " In practice, averaging over multiple seeds gives a more robust estimate of performance."
                           " But for simplicity, we just use one seed here."
                           " In real-world use, consider running multiple seeds and averaging results."
                           " reproducible e-greedy means the same random choices each run with the same seed.")

    st.subheader("LinUCB",
                 help="Contextual bandit with linear models & an optimism bonus (α). "
                      "Uses features like slope, Δslope, and momentum. Higher α = more exploration."
                      " Adapts to changing conditions by exploring uncertain arms."
                      " Works well if the best arm depends on context (features)."
                        " More exploration (higher α) helps when uncertainty is high."
                        " But too high α means too much exploration, hurting returns."
                        " LinUCB is more complex and computationally intensive than ε-greedy."
                        " But it can adapt to changing conditions better by using context."
                        " In practice, LinUCB often outperforms simple methods like ε-greedy when good features are available."
                        " However, it requires careful feature engineering and parameter tuning."
                        " ε-greedy is simpler and faster, but may miss opportunities that LinUCB can exploit."
                        " LinUCB is more robust to changing environments due to its contextual nature."
                        " But it can be sensitive to feature quality and parameter choices.")
    alpha = st.slider("alpha", 0.0, 1.0, 0.5, 0.05,
                      help="Optimism bonus. Larger α = more exploration when uncertainty is high."
                      " Helps LinUCB explore arms with uncertain rewards."
                      " But too high α means too much exploration, hurting returns."
                      " If α = 0, LinUCB becomes a greedy algorithm, always exploiting the best estimated arm."
                      " Typical values are between 0.1 and 1.0."
                      " In practice, tuning α based on validation performance can help."
                      " α controls the trade-off between exploration and exploitation."
                      " Higher α encourages trying arms with uncertain rewards more often."
                      " But too high α can lead to excessive exploration, hurting overall returns.")
    f_slope  = st.checkbox("slope (10y − 91d)", True,
                           help="Term-structure slope feature (lagged 1 day). "
                                "Steepening favors short; flattening favors long."
                                " Captures overall curve shape."
                                " A positive slope (10y > 91d) often indicates economic growth expectations."
                                " A negative slope (inversion) can signal economic slowdown."
                                " Including slope helps LinUCB adapt to changing yield curve conditions.")
    f_dslope = st.checkbox("Δslope", True,
                           help="Daily change in slope (lagged 1 day). Captures steepening/flattening."
                           " Helps detect shifts in market sentiment."
                           " Positive Δslope (steepening) often favors short-term bills."
                            " Negative Δslope (flattening) often favors long-term bonds."
                            " Including Δslope helps LinUCB respond to recent changes in the yield curve."
                            " Δslope captures momentum in yield curve movements.")
    f_mom    = st.checkbox("10y momentum (5d)", True,
                           help="Short-term 10y yield momentum (lagged 1 day). Captures recent trend."
                           " Positive momentum often favors long-term bonds."
                           " Negative momentum often favors short-term bills."
                           " Including momentum helps LinUCB adapt to recent yield trends."
                           " Momentum captures short-term trends that slope alone may miss."
                           " Helps LinUCB respond to recent market movements."
                           " Momentum can indicate investor sentiment and risk appetite."
                           " Including momentum can improve LinUCB's adaptability to changing conditions.")

    
    # ----------------------------------------------------------------------

    if st.button("Run"):
        st.session_state["ready"] = True


if st.session_state["ready"]:
    # 0) Load data -------------------------------------------------------------
    if uploaded is not None:
        df = pd.read_csv(uploaded, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
        source_note = "Using uploaded CSV"
    else:
        abs_csv = resolve_path(csv_path)
        if not os.path.exists(abs_csv):
            st.error(f"CSV not found:\n{abs_csv}")
            st.info(
                "Tips: (1) check the exact file name in /data, "
                "(2) run `streamlit run app/app.py` from the project root, "
                "(3) or use the uploader above."
            )
            st.stop()
        df = _load_from_disk(abs_csv)
        source_note = f"Loaded from: {abs_csv}"
    # -------------------------------------------------------------------------

    # Re-render the date slider with true bounds and keep current selection if possible
    st.sidebar.caption(f"Data: {df['Date'].min().date()} → {df['Date'].max().date()}")

    lo_default = df["Date"].min().to_pydatetime()
    hi_default = df["Date"].max().to_pydatetime()

    # Start from whatever is in session_state (if present), otherwise full range
    cur_lo, cur_hi = st.session_state.get("date_window", (lo_default, hi_default))
    # Clip previous selection into new bounds
    cur_lo = max(lo_default, cur_lo)
    cur_hi = min(hi_default, cur_hi)

    st.sidebar.slider(
        "Date range to display",
        min_value=lo_default,
        max_value=hi_default,
        value=(cur_lo, cur_hi),
        key="date_window",
    )
    view_lo, view_hi = st.session_state["date_window"]
    view_mask = (df["Date"] >= view_lo) & (df["Date"] <= view_hi)

    # after computing view_mask
    df_view = df.loc[view_mask].copy()


    # 1) Build rewards
    df = build_rewards(df, k91=k91, k364=k364, k10=k10)

    # 2) Baselines
    r91, r364, r10 = df["r_91d"], df["r_364d"], df["r_10y"]
    eq_base = {
        "Always_91d":  equity_curve(r91),
        "Always_364d": equity_curve(r364),
        "Always_10y":  equity_curve(r10),
        "Oracle":      equity_curve(pd.concat([r91, r364, r10], axis=1).max(axis=1)),
    }

    # 3) ε-greedy
    R = pd.concat([r91, r364, r10], axis=1).to_numpy()
    switch_cost_decimal = switch_bps / 10000.0  # bps → decimal
    picks_e, rew_e = epsilon_greedy(R, eps=eps, switch_cost=switch_cost_decimal, seed=seed)
    eq_e = equity_curve(pd.Series(rew_e, index=df.index))

    # 4) LinUCB
    X = build_features(df, use_slope=f_slope, use_dslope=f_dslope, use_mom=f_mom)
    mask = ~X.isna().any(axis=1)
    picks_l, rew_l, eq_l = None, None, None
    if mask.sum() > 10:
        picks_l, rew_l = linucb_disjoint(X[mask].to_numpy(), R[mask], alpha=alpha)
        eq_l = equity_curve(pd.Series(rew_l, index=df.index[mask]))

    # 5) UI tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Equity", "Allocations", "Metrics"])

    # ------------ Overview ------------
    with tab1:
        with st.expander("What is this app doing? (30-second explainer)", expanded=True):
            st.markdown("""
            **Goal.** Allocate daily among **91-day T-Bill**, **364-day T-Bill**, and **10-year G-Sec** using simple *bandits*.
            
            **Reward proxy.** Daily accrual = (annual % / 100) / 365, minus a penalty **k × |Δyield|** for duration/roll risk.

            **Bandits.**
            - **ε-Greedy** (*non-contextual*): most days exploit best estimated arm; with probability ε explore randomly.
            - **LinUCB** (*contextual*): uses features (slope, Δslope, momentum) and an optimism bonus (α) to explore where uncertain.

            **Why bandits?** They adapt to the curve: short wins when the curve steepens; long wins when it flattens.
            """)
        st.caption(source_note)
        st.dataframe(
            df_view[["Date","Yield_91d","Yield_364d","Yield_10y"]].head(25),
            use_container_width=True,
            height=360,
        )

    # ------------ Equity ------------
    with tab2:
        curves = {**eq_base, "EpsGreedy": eq_e.reindex(df.index)}
        if eq_l is not None:
            curves["LinUCB"] = eq_l.reindex(df.index)   # align lengths
        st.pyplot(plot_equity(df_view["Date"], {k: v.reindex(df_view.index) for k,v in curves.items()}))

    # some footnotes
        st.markdown( "At each day, the Oracle always picks the instrument with the highest reward that day."
                    "In reality, you cannot know this ahead of time (it’s “future information”" \
                    "It’s an upper bound: the best possible path if you had perfect foresight."
                    "Think of Oracle as your theoretical ceiling — no bandit or strategy can ever do better, "
                    "but if you approach it, you’re doing very well.")

    # ------------ Allocations ------------
    with tab3:
        labels = ["91d", "364d", "10y"]

        # ε-greedy allocation (filter to the selected window)
        picks_e_labels = [labels[p] for p in picks_e]
        picks_e_labels_view = [lab for lab, keep in zip(picks_e_labels, view_mask) if keep]
        dates_view = df["Date"][view_mask]

        if HAS_TIMELINE:
            st.plotly_chart(
                allocation_timeline_plotly(dates_view, picks_e_labels_view, labels),
                use_container_width=True, config={"displaylogo": False}
            )
            st.plotly_chart(
                rolling_choice_share(dates_view, picks_e_labels_view, labels),
                use_container_width=True, config={"displaylogo": False}
            )
        else:
            st.pyplot(plot_allocation(dates_view, picks_e_labels_view, labels))

        # LinUCB allocation (if available)
        if picks_l is not None:
            picks_l_labels = [labels[p] for p in picks_l]
            view_mask_l = view_mask & mask
            dates_view_l = df["Date"][view_mask_l]
            picks_l_labels_view = [lab for lab, keep in zip(picks_l_labels, view_mask[mask]) if keep]

            if HAS_TIMELINE:
                st.plotly_chart(
                    allocation_timeline_plotly(dates_view_l, picks_l_labels_view, labels),
                    use_container_width=True, config={"displaylogo": False}
                )
                st.plotly_chart(
                    rolling_choice_share(dates_view_l, picks_l_labels_view, labels),
                    use_container_width=True, config={"displaylogo": False}
                )
            else:
                st.pyplot(plot_allocation(dates_view_l, picks_l_labels_view, labels))

    # ------------ Metrics ------------
    with tab4:
        rows = []
        for name, r in [
            ("Always_91d", r91),
            ("Always_364d", r364),
            ("Always_10y", r10),
            ("EpsGreedy", pd.Series(rew_e, index=df.index)),
            ]:
            r_view = r.reindex(df_view.index).dropna()
            ar, av, sh = ann_ret_vol_sharpe(r_view)
            rows.append((name, ar, av, sh))

        if rew_l is not None:
            # series of LinUCB rewards, aligned to full df (only valid where mask=True)
            rew_l_series = pd.Series(rew_l, index=df.index[mask])
            # then clip it to the view window
            rew_l_series = rew_l_series.reindex(df_view.index)
            ar, av, sh = ann_ret_vol_sharpe(rew_l_series.dropna())
            rows.append(("LinUCB", ar, av, sh))

        metr = pd.DataFrame(rows, columns=["Strategy", "AnnRet", "AnnVol", "Sharpe"]).set_index("Strategy")
        # Help popover above the table
        with st.popover("ℹ️ Metrics help"):
            st.markdown(
        """
            **Annualized return (AnnRet), volatility (AnnVol), and Sharpe ratio** (risk-adjusted return).

            - **AnnRet** = average daily return × **252**  
            - **AnnVol** = std. dev. of daily returns × **√252**  
            - **Sharpe** = AnnRet / AnnVol → **higher is better**

                Notes: Negative Sharpe indicates underperformance vs a risk-free asset.  
                These are historical, ignore many frictions, and don’t predict the future.
        """
    )
        st.dataframe(
            metr.style.format({"AnnRet":"{:.2%}",
                               "AnnVol":"{:.2%}",
                               "Sharpe":"{:.2f}"}),
            use_container_width=True,
            height=340,
        )
        st.caption(
    "AnnRet/AnnVol annualized with 252 trading days; Sharpe = AnnRet / AnnVol. "
    "Use alongside other metrics and qualitative context."
)
        st.info(
    " ! These are historical metrics that ignore many real-world frictions. "
    "They do not predict future performance. "
    "Do not use this app for real trading without further validation and risk controls."
)
        # adding an important note on why epsGreedy has performed better than LinUCB
        st.markdown(
        "Why is ε-Greedy Sharpe sometimes better than LinUCB? " 
        "A few reasons:"

       "Stability vs. complexity"

        "ε-Greedy only looks at past average rewards. It exploits the best so far, with a little random exploration."
        "It doesn’t try to use features (slope, Δslope, momentum), so it won’t “overfit” to noisy signals."
        "That simplicity can actually yield smoother allocations and lower volatility → better Sharpe."

        "Feature quality"

        "LinUCB’s edge comes from good context features. If slope, Δslope, momentum aren’t very predictive in your dataset," 
         "then LinUCB won’t beat a simpler rule.Worse, if they are noisy, LinUCB may over-explore, leading to choppier allocations" 
         "(higher vol, same or lower return)."

        "Parameter tuning"

        "LinUCB has an α (optimism bonus) hyperparameter. Too high → too much exploration (hurts Sharpe)."
         " Too low → too conservative (acts almost like greedy, adds no benefit)."
         "ε-Greedy with ε≈0.05 is often “just right” for balancing exploration/exploitation, so it can look cleaner."

        "Regret horizon"

        "Bandits shine in very long horizons. If you only have a finite dataset (~25 years)," 
         "sometimes the simpler policy ends up winning just by avoiding noise."

        "So: ε-Greedy ≠ worse by definition — it can look better if your features don’t strongly explain yield shifts." 
         " LinUCB should outperform only if slope/Δslope/momentum contain reliable signals."
        )