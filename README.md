# Portfolio Optimisation Lab

## Run locally
1. Install Python 3.10 or newer.
2. Put `app.py`, `requirements.txt` and `Beta calculation.xlsx` in one folder.
3. Open a terminal in that folder and run:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

4. The included `Beta calculation.xlsx` loads automatically; no upload is needed.

## Updating the price data
- Close the Streamlit app.
- Replace `Beta calculation.xlsx` in the same folder as `app.py` with your updated workbook. Keep the exact filename.
- Start the app again with `streamlit run app.py`.
- Alternatively, use the optional replacement-workbook control in the sidebar for a temporary dataset change.
- The workbook must have one worksheet per stock with `Date` and `Close` columns. An optional worksheet named `NIFTY` is treated as the benchmark.

## Data handling
- Reads only `Date` and `Close` columns from each worksheet.
- `NIFTY` is treated as benchmark, not as a portfolio constituent.
- Uses the last six calendar months as the out-of-sample test period.
- All earlier common observations form the training period.
- The workbook is weekly and annualisation uses 52 observations/year.
- The stock panel is aligned to common dates; up to two consecutive internal missing prices are forward-filled. No leading values are back-filled.

## Models
MVP, HRP, Eigen Portfolio, HERC-style hierarchical allocation and Black–Litterman are implemented in `app.py`. The Black–Litterman prior uses equal weights as a market proxy because market capitalisation data is not provided. The default view set is neutral/no-view.

## Notes
This is an educational backtesting implementation, not investment advice. It does not model taxes, slippage, transaction costs, liquidity or periodic rebalancing.
