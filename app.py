import io
import os
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
from scipy.spatial.distance import squareform
from scipy.optimize import minimize

warnings.filterwarnings("ignore")
ANN_FACTOR = 52  # uploaded workbook contains predominantly weekly observations

st.set_page_config(page_title="Portfolio Optimisation Lab", page_icon="📊", layout="wide")
st.title("📊 Portfolio Optimisation & Backtesting Lab")
st.caption("MVP • HRP • Eigen Portfolio • HERC • Black–Litterman | Close-price-only analysis")
st.info("The included Beta calculation.xlsx loads automatically. To update your data later, replace that Excel file in this folder (keep the same filename), then restart the app. You can also temporarily choose another workbook in the sidebar.")


def load_workbook(file):
    xls = pd.ExcelFile(file)
    prices, benchmark = {}, None
    for sheet in xls.sheet_names:
        try:
            df = pd.read_excel(xls, sheet_name=sheet)
        except Exception:
            continue
        cols = {str(c).strip().lower(): c for c in df.columns}
        if "date" not in cols or "close" not in cols:
            continue
        tmp = df[[cols["date"], cols["close"]]].copy()
        tmp.columns = ["Date", "Close"]
        tmp["Date"] = pd.to_datetime(tmp["Date"], errors="coerce")
        tmp["Close"] = pd.to_numeric(tmp["Close"], errors="coerce")
        tmp = tmp.dropna().sort_values("Date").drop_duplicates("Date", keep="last")
        tmp = tmp[tmp["Close"] > 0]
        if tmp.empty:
            continue
        if sheet.strip().upper() == "NIFTY":
            benchmark = tmp.set_index("Date")["Close"].rename("NIFTY")
        else:
            prices[sheet.strip()] = tmp.set_index("Date")["Close"]
    if not prices:
        raise ValueError("No stock sheets with valid Date and Close columns were found.")
    return pd.concat(prices, axis=1).sort_index(), benchmark


def clean_prices(raw, min_coverage=0.80):
    # Keep stocks with sufficient observations across the combined history.
    coverage = raw.notna().mean()
    keep = coverage[coverage >= min_coverage].index.tolist()
    pxs = raw[keep].copy()
    # Forward-fill only short internal gaps; never back-fill unavailable history.
    pxs = pxs.ffill(limit=2)
    # Require a common complete panel for unbiased covariance estimation.
    pxs = pxs.dropna(how="any")
    return pxs


def cov_shrink(returns):
    # Stable covariance estimate: shrink sample covariance toward its diagonal.
    s = returns.cov() * ANN_FACTOR
    diag = np.diag(np.diag(s.values))
    cov = 0.80 * s.values + 0.20 * diag
    return pd.DataFrame(cov, index=s.index, columns=s.columns)


def portfolio_stats(weights, mu, cov, rf):
    w = np.asarray(weights)
    ret = float(w @ mu)
    vol = float(np.sqrt(max(w @ cov @ w, 0)))
    sharpe = (ret - rf) / vol if vol > 1e-12 else np.nan
    return ret, vol, sharpe


def min_variance(cov):
    n = len(cov)
    obj = lambda w: float(w @ cov @ w)
    res = minimize(obj, np.ones(n)/n, method="SLSQP",
                   bounds=[(0, 1)]*n, constraints=[{"type":"eq","fun":lambda w: np.sum(w)-1}],
                   options={"maxiter":2000, "ftol":1e-12})
    return res.x / res.x.sum() if res.success else np.ones(n)/n


def max_sharpe(mu, cov, rf):
    n = len(mu)
    def obj(w):
        v = np.sqrt(max(w @ cov @ w, 1e-14))
        return -((w @ mu) - rf) / v
    res = minimize(obj, np.ones(n)/n, method="SLSQP",
                   bounds=[(0,1)]*n, constraints=[{"type":"eq","fun":lambda w: np.sum(w)-1}],
                   options={"maxiter":2000, "ftol":1e-10})
    return res.x/res.x.sum() if res.success else min_variance(cov)


def corr_distance(returns):
    corr = returns.corr().clip(-1,1).fillna(0)
    dist = np.sqrt(np.maximum((1-corr.values)/2, 0))
    np.fill_diagonal(dist, 0)
    return squareform(dist, checks=False)


def hrp_weights(returns, cov):
    names = list(returns.columns)
    if len(names) == 1:
        return pd.Series([1.0], index=names)
    tree = linkage(corr_distance(returns), method="single")
    order = list(leaves_list(tree))
    covm = cov.loc[names, names]
    w = pd.Series(1.0, index=[names[i] for i in order])
    clusters = [order]
    while clusters:
        next_clusters = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            mid = len(cluster)//2
            left, right = cluster[:mid], cluster[mid:]
            def cluster_var(ids):
                labels = [names[i] for i in ids]
                sub = covm.loc[labels, labels].values
                inv = 1/np.maximum(np.diag(sub), 1e-12)
                ivp = inv/inv.sum()
                return float(ivp @ sub @ ivp)
            vl, vr = cluster_var(left), cluster_var(right)
            alpha = 1 - vl/(vl+vr)
            w.loc[[names[i] for i in left]] *= alpha
            w.loc[[names[i] for i in right]] *= (1-alpha)
            next_clusters.extend([left, right])
        clusters = next_clusters
    return w.reindex(names)/w.sum()


def eigen_weights(returns):
    cov = returns.cov().values
    vals, vecs = np.linalg.eigh(cov)
    v = vecs[:, np.argmax(vals)]
    # Long-only implementation: use absolute loadings, then normalise.
    w = np.abs(v)
    if w.sum() == 0:
        w = np.ones(len(v))
    return pd.Series(w/w.sum(), index=returns.columns)


def herc_weights(returns, cov):
    names = list(returns.columns)
    n = len(names)
    if n == 1:
        return pd.Series([1.0], index=names)
    tree = linkage(corr_distance(returns), method="ward")
    k = max(2, min(n, int(round(np.sqrt(n)))))
    labels = fcluster(tree, t=k, criterion="maxclust")
    clusters = [np.where(labels == lab)[0].tolist() for lab in sorted(set(labels))]
    cluster_vols = []
    within = []
    for ids in clusters:
        cols = [names[i] for i in ids]
        sub = cov.loc[cols, cols].values
        inv = 1/np.maximum(np.diag(sub), 1e-12)
        w_in = inv/inv.sum()
        within.append((cols, w_in))
        cluster_vols.append(np.sqrt(max(w_in @ sub @ w_in, 1e-12)))
    # Equal risk budget across clusters, inverse-vol approximation between clusters.
    cw = 1/np.asarray(cluster_vols)
    cw = cw/cw.sum()
    out = pd.Series(0.0, index=names)
    for (cols, win), weight in zip(within, cw):
        out.loc[cols] = win*weight
    return out/out.sum()


def black_litterman(returns, cov, rf, delta=2.5, tau=0.05):
    # No external market-cap data is supplied: equal-weight market proxy.
    n = returns.shape[1]
    w_mkt = np.ones(n)/n
    sigma = cov.values
    pi = delta * sigma @ w_mkt
    # Default is a neutral/no-view BL posterior. User can optionally add relative views below.
    P = np.empty((0,n))
    Q = np.empty((0,))
    omega = np.empty((0,0))
    if len(Q):
        middle = np.linalg.inv(P @ (tau*sigma) @ P.T + omega)
        posterior = np.linalg.inv(np.linalg.inv(tau*sigma) + P.T @ middle @ P) @ (
            np.linalg.inv(tau*sigma) @ pi + P.T @ middle @ Q)
    else:
        posterior = pi
    return max_sharpe(posterior, cov, rf), posterior


def performance(weights, test_returns, rf):
    w = weights.reindex(test_returns.columns).fillna(0).values
    port = test_returns @ w
    wealth = (1+port).cumprod()
    n = len(port)
    years = n/ANN_FACTOR
    total = float(wealth.iloc[-1]-1) if n else np.nan
    cagr = float(wealth.iloc[-1]**(1/years)-1) if n and years > 0 and wealth.iloc[-1] > 0 else np.nan
    vol = float(port.std(ddof=1)*np.sqrt(ANN_FACTOR)) if n > 1 else np.nan
    sharpe = float((port.mean()*ANN_FACTOR-rf)/vol) if vol and vol > 0 else np.nan
    peak = wealth.cummax()
    maxdd = float((wealth/peak-1).min())
    return {"Total Return": total, "Annualised Return (CAGR)": cagr,
            "Annualised Risk": vol, "Sharpe Ratio": sharpe, "Max Drawdown": maxdd}, port, wealth


st.sidebar.header("Model settings")
# Default workbook lives beside app.py. Replace this file to update the dataset.
default_workbook = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Beta calculation.xlsx")
replacement_workbook = st.sidebar.file_uploader(
    "Optional: select a different Excel workbook", type=["xlsx", "xls"]
)
min_cov = st.sidebar.slider("Minimum stock data coverage", 0.40, 1.00, 0.45, 0.05)
rf = st.sidebar.number_input("Annual risk-free rate (%)", min_value=0.0, max_value=30.0, value=6.0, step=0.25)/100
capital = st.sidebar.number_input("Portfolio capital (₹)", min_value=1000, value=200000, step=10000)

if replacement_workbook is not None:
    selected_workbook = replacement_workbook
    st.sidebar.caption(f"Using replacement workbook: {replacement_workbook.name}")
elif os.path.exists(default_workbook):
    selected_workbook = default_workbook
    st.sidebar.caption("Using bundled workbook: Beta calculation.xlsx")
else:
    st.error("The default workbook is missing. Place 'Beta calculation.xlsx' in the same folder as app.py, or choose another workbook in the sidebar.")
    st.stop()

try:
    raw, nifty = load_workbook(selected_workbook)
    prices = clean_prices(raw, min_cov)
except Exception as e:
    st.error(str(e))
    st.stop()

if prices.shape[1] < 2 or len(prices) < 20:
    st.error("Not enough overlapping history. At least 2 stocks and 20 common weekly observations are required.")
    st.stop()

last_date = prices.index.max()
test_start = last_date - pd.DateOffset(months=6)
train_prices = prices.loc[prices.index < test_start]
test_prices = prices.loc[prices.index >= test_start]
if len(train_prices) < 15 or len(test_prices) < 5:
    st.error(f"Insufficient observations after the six-month split. Common data: {prices.index.min().date()} to {last_date.date()}. Training: {len(train_prices)}, testing: {len(test_prices)}. Consider lowering the coverage threshold or adding more history.")
    st.stop()

train_ret = train_prices.pct_change().dropna()
test_ret = test_prices.pct_change().dropna()
# First test return is calculated from the final training close, so the test period includes the boundary return.
boundary = prices.loc[[train_prices.index[-1]] + list(test_prices.index)].pct_change().dropna()
test_ret = boundary
mu = train_ret.mean()*ANN_FACTOR
cov = cov_shrink(train_ret)
names = list(train_ret.columns)

weights = {}
weights["MVP"] = pd.Series(min_variance(cov), index=names)
weights["HRP"] = hrp_weights(train_ret, cov)
weights["Eigen Portfolio"] = eigen_weights(train_ret)
weights["HERC"] = herc_weights(train_ret, cov)
blw, blmu = black_litterman(train_ret, cov, rf)
weights["Black-Litterman"] = pd.Series(blw, index=names)

st.subheader("Data split")
c1,c2,c3,c4 = st.columns(4)
c1.metric("Stocks", len(names))
c2.metric("Common weekly observations", len(prices))
c3.metric("Training observations", len(train_prices))
c4.metric("Testing observations", len(test_prices))
st.write(f"**Training:** {train_prices.index.min().date()} to {train_prices.index.max().date()}  |  **Testing:** {test_prices.index.min().date()} to {test_prices.index.max().date()}")
st.caption("All price series are aligned to common dates. Up to two consecutive missing observations are forward-filled; no leading values are back-filled. NIFTY is excluded from the investable universe and shown only as a benchmark.")

results, port_series, wealth_series = {}, {}, {}
for method, w in weights.items():
    results[method], port_series[method], wealth_series[method] = performance(w, test_ret, rf)

metrics = pd.DataFrame(results).T
metrics_display = metrics.copy()
for c in ["Total Return","Annualised Return (CAGR)","Annualised Risk","Max Drawdown"]:
    metrics_display[c] = metrics_display[c].map(lambda x: f"{x:.2%}" if pd.notna(x) else "—")
metrics_display["Sharpe Ratio"] = metrics_display["Sharpe Ratio"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "—")

st.subheader("Out-of-sample strategy comparison")
st.dataframe(metrics_display, use_container_width=True)
best = metrics["Sharpe Ratio"].idxmax()
st.success(f"Highest test-period Sharpe ratio: {best} ({metrics.loc[best,'Sharpe Ratio']:.2f})")

st.subheader("Growth of ₹1 invested — testing period")
growth = pd.DataFrame({k: v for k,v in wealth_series.items()})
fig = px.line(growth, x=growth.index, y=growth.columns, labels={"value":"Portfolio value (₹1 initial)","index":"Date","variable":"Strategy"})
fig.update_layout(legend_title_text="Strategy", hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Risk versus return")
scatter = metrics.reset_index(names="Strategy")
fig2 = px.scatter(scatter, x="Annualised Risk", y="Annualised Return (CAGR)", text="Strategy",
                  hover_data=["Sharpe Ratio","Total Return"], labels={"Annualised Risk":"Annualised volatility","Annualised Return (CAGR)":"Annualised return (CAGR)"})
fig2.update_traces(textposition="top center", marker_size=13)
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Portfolio allocations")
alloc_tabs = st.tabs(list(weights.keys()))
for tab, (method, w) in zip(alloc_tabs, weights.items()):
    with tab:
        alloc = w.sort_values(ascending=False)
        left,right = st.columns([1,1])
        with left:
            figp = px.pie(names=alloc.index, values=alloc.values, hole=0.25, title=f"{method} allocation")
            st.plotly_chart(figp, use_container_width=True)
        with right:
            table = pd.DataFrame({"Stock":alloc.index, "Weight":alloc.values,
                                  "Allocation (₹)":alloc.values*capital})
            table["Weight"] = table["Weight"].map(lambda x:f"{x:.2%}")
            table["Allocation (₹)"] = table["Allocation (₹)"].map(lambda x:f"₹{x:,.2f}")
            st.dataframe(table, hide_index=True, use_container_width=True)
            st.download_button(f"Download {method} allocation CSV",
                data=pd.DataFrame({"Stock":alloc.index,"Weight":alloc.values,"Allocation_INR":alloc.values*capital}).to_csv(index=False),
                file_name=f"{method.lower().replace(' ','_').replace('-','_')}_allocation.csv", mime="text/csv")

st.subheader("Strategy weights — full comparison")
weight_table = pd.DataFrame(weights).fillna(0)
st.dataframe(weight_table.style.format("{:.2%}"), use_container_width=True)
st.download_button("Download all portfolio weights", weight_table.to_csv().encode("utf-8"),
                   "all_strategy_weights.csv", "text/csv")

if nifty is not None:
    bench = nifty.reindex(test_ret.index).ffill().pct_change().dropna()
    if len(bench):
        bvol = bench.std()*np.sqrt(ANN_FACTOR)
        bret = (1+bench).prod()**(ANN_FACTOR/len(bench))-1
        bsharpe = (bench.mean()*ANN_FACTOR-rf)/bvol if bvol else np.nan
        st.subheader("NIFTY benchmark")
        st.write(f"**NIFTY test CAGR:** {bret:.2%}  |  **Annualised risk:** {bvol:.2%}  |  **Sharpe:** {bsharpe:.2f}")
        # Benchmark rebased to ₹1 and aligned to test observations.
        bw = (1+bench).cumprod()
        compare = growth.copy()
        compare["NIFTY"] = bw.reindex(compare.index).ffill()
        st.plotly_chart(px.line(compare, x=compare.index, y=compare.columns,
                                labels={"value":"Growth of ₹1","index":"Date","variable":"Portfolio"}),
                        use_container_width=True)

with st.expander("Methodology and assumptions"):
    st.markdown("""
    - **MVP:** long-only minimum variance optimisation using SLSQP.
    - **HRP:** hierarchical clustering, quasi-diagonal ordering and recursive bisection.
    - **Eigen Portfolio:** absolute loadings of the largest covariance eigenvector, normalised to long-only weights.
    - **HERC:** hierarchical clusters, inverse-volatility allocation across clusters, and inverse-variance allocation within each cluster. This is a practical HERC-style implementation.
    - **Black–Litterman:** equilibrium prior uses an equal-weight market proxy because market capitalisations are not included in the workbook. No subjective views are assumed by default; the posterior is therefore the equilibrium prior, followed by long-only maximum-Sharpe optimisation.
    - **Annualisation:** 52 periods per year, matching the weekly price observations in this workbook.
    - **Sharpe:** (annualised arithmetic return − annual risk-free rate) / annualised volatility.
    - **Testing:** weights are fixed at the training-period estimates; no rebalancing or transaction costs are modelled.
    - **Important:** the model is an educational research/backtesting tool, not investment advice. Results depend on the selected universe, data frequency and assumptions.
    """)
