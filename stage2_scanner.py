#!/usr/bin/env python3
"""
Stage 2 Scanner - NSE Nifty Midcap 150 + Nifty Smallcap 250
Local:  python stage2_scanner.py   |  Hosted: GitHub Actions + Pages (see .github/workflows/stage2.yml)
Output:     stage2_results.html (opens automatically)

Rule implemented
  "Stage 2 begins when a stock makes a powerful move higher - price surges away
   from the 20 and 30 SMA on expanding volume. The stock then pauses, forming a
   controlled base while remaining above the 30 SMA."

Requires:  pip install yfinance pandas numpy requests
"""
import io
import os
import html
import datetime as dt
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# --------------------------------------------------------------------------- #
# CONFIG - every threshold that defines the rule lives here
# --------------------------------------------------------------------------- #
CFG = dict(
    INDEX_URLS={
        "Midcap": "https://www.niftyindices.com/IndexConstituent/ind_niftymidcap150list.csv",
        "Smallcap": "https://www.niftyindices.com/IndexConstituent/ind_niftysmallcap250list.csv",
    },  # fallback: put midcap.csv / smallcap.csv (with a 'Symbol' column) next to this file
    HISTORY="1y",
    CHUNK=100,
    # 1) Powerful move
    SURGE_WINDOW=15,        # bars leading up to the peak that make up the surge
    MIN_SURGE_PCT=20.0,     # low-to-peak gain inside that window
    MIN_EXTENSION_PCT=10.0, # close must get this far above BOTH 20 & 30 SMA during the surge
    MIN_SURGE_VOL_X=1.5,    # avg surge volume vs 50-day avg volume at surge start
    # 2) Controlled base
    BASE_MIN_DAYS=5,
    BASE_MAX_DAYS=45,
    MAX_BASE_RANGE_PCT=20.0,  # (base high - base low) / base high
    MAX_PULLBACK_PCT=15.0,    # current close vs surge peak
    MAX_DAYS_BELOW_30SMA_PCT=10.0,  # share of base closes allowed under the 30 SMA (0 = strict)
    MAX_DIP_BELOW_30SMA_PCT=3.0,    # deepest allowed close under the 30 SMA
    # 3) Output
    OUT=Path(os.environ.get("OUT", "stage2_results.html")),
)


# --------------------------------------------------------------------------- #
# Universe + prices
# --------------------------------------------------------------------------- #
def get_universe():
    hdr = {"User-Agent": "Mozilla/5.0"}
    rows = []
    for seg, url in CFG["INDEX_URLS"].items():
        local = Path(f"{seg.lower()}.csv")
        try:
            r = requests.get(url, headers=hdr, timeout=30)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
        except Exception as e:
            if local.exists():
                df = pd.read_csv(local)
            else:
                print(f"[warn] could not load {seg} list ({e}); skipping")
                continue
        for _, x in df.iterrows():
            rows.append((str(x["Symbol"]).strip(), str(x.get("Company Name", x["Symbol"])), seg))
    uni = pd.DataFrame(rows, columns=["symbol", "name", "cap"]).drop_duplicates("symbol")
    return uni


def load_prices(symbols):
    out = {}
    tickers = [s + ".NS" for s in symbols]
    for i in range(0, len(tickers), CFG["CHUNK"]):
        chunk = tickers[i:i + CFG["CHUNK"]]
        print(f"  downloading {i + 1}-{i + len(chunk)} of {len(tickers)}")
        data = yf.download(chunk, period=CFG["HISTORY"], interval="1d", group_by="ticker",
                           auto_adjust=True, threads=True, progress=False)
        for t in chunk:
            try:
                df = data[t] if isinstance(data.columns, pd.MultiIndex) else data
                df = df.dropna(how="all")
                if len(df):
                    out[t[:-3]] = df
            except KeyError:
                pass
    return out


# --------------------------------------------------------------------------- #
# The rule
# --------------------------------------------------------------------------- #
def stage2(df):
    """Try every bar that is BASE_MIN_DAYS..BASE_MAX_DAYS old as the surge peak; keep the best match."""
    d = df[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
    if len(d) < 100:
        return None
    c = d["Close"]
    s20, s30 = c.rolling(20).mean(), c.rolling(30).mean()
    v50 = d["Volume"].rolling(50).mean()
    n = len(d)
    best = None
    for p in range(n - 1 - CFG["BASE_MAX_DAYS"], n - CFG["BASE_MIN_DAYS"]):
        if p < 65:
            continue
        peak = d["High"].iloc[p]
        if peak < d["High"].iloc[p - CFG["SURGE_WINDOW"]:p + 1].max():   # peak = top of the surge leg
            continue

        # 1) powerful move, away from 20 & 30 SMA, on expanding volume
        w0 = p - CFG["SURGE_WINDOW"]
        seg = d.iloc[w0:p + 1]
        surge = (peak / seg["Low"].min() - 1) * 100
        if surge < CFG["MIN_SURGE_PCT"]:
            continue
        ext = ((seg["Close"] / np.maximum(s20.iloc[w0:p + 1], s30.iloc[w0:p + 1]) - 1) * 100).max()
        if ext < CFG["MIN_EXTENSION_PCT"]:
            continue
        volx = seg["Volume"].mean() / v50.iloc[w0]
        if not volx >= CFG["MIN_SURGE_VOL_X"]:
            continue

        # 2) controlled base above the 30 SMA
        base = d.iloc[p + 1:]
        top = max(peak, base["High"].max())
        base_rng = (top - base["Low"].min()) / top * 100
        if base_rng > CFG["MAX_BASE_RANGE_PCT"]:
            continue
        rel = (base["Close"].values / s30.iloc[p + 1:].values - 1) * 100
        if (rel < 0).mean() * 100 > CFG["MAX_DAYS_BELOW_30SMA_PCT"] or rel.min() < -CFG["MAX_DIP_BELOW_30SMA_PCT"] or rel[-1] <= 0:
            continue
        last = c.iloc[-1]
        pull = (1 - last / top) * 100
        if pull > CFG["MAX_PULLBACK_PCT"]:
            continue
        if base["Volume"].mean() >= seg["Volume"].mean():   # volume should dry up in the base
            continue

        hit = dict(close=last, surge=surge, ext=ext, volx=volx, days=n - 1 - p,
                   pull=pull, rng=base_rng, above30=(last / s30.iloc[-1] - 1) * 100,
                   asof=d.index[-1].strftime("%d %b %Y"))
        if best is None or surge > best["surge"]:
            best = hit
    return best


# --------------------------------------------------------------------------- #
# Fundamentals: last quarter EPS and sales (+ growth %)
# --------------------------------------------------------------------------- #
def growth(a, b):
    if a is None or b is None or pd.isna(a) or pd.isna(b) or b == 0:
        return None
    return (a - b) / abs(b) * 100


def fundamentals(sym):
    out = dict(qtr="", eps=None, eps_yoy=None, eps_qoq=None,
               sales=None, sales_yoy=None, sales_qoq=None)
    try:
        q = yf.Ticker(sym + ".NS").quarterly_income_stmt
        if q is None or q.empty:
            return out
        q = q.sort_index(axis=1, ascending=False)

        def row(names):
            for nm in names:
                if nm in q.index:
                    return q.loc[nm]
            return None

        def val(s, i):
            if s is None or len(s) <= i or pd.isna(s.iloc[i]):
                return None
            return float(s.iloc[i])

        eps, rev = row(["Diluted EPS", "Basic EPS"]), row(["Total Revenue", "Operating Revenue"])
        out["qtr"] = q.columns[0].strftime("%b %Y")
        out["eps"] = val(eps, 0)
        out["eps_qoq"], out["eps_yoy"] = growth(val(eps, 0), val(eps, 1)), growth(val(eps, 0), val(eps, 4))
        s0 = val(rev, 0)
        out["sales"] = s0 / 1e7 if s0 is not None else None   # INR -> crore
        out["sales_qoq"], out["sales_yoy"] = growth(s0, val(rev, 1)), growth(s0, val(rev, 4))
    except Exception:
        pass
    return out


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage 2 Scanner - __ASOF__</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;600&display=swap" rel="stylesheet">
<style>
:root{--bg:#0b2a4a;--line:#2f5f8f;--ink:#e6f3ff;--mut:#8fb4d6;--acc:#7fd1ff;--up:#5fe3a1;--dn:#ff8a8a}
*{box-sizing:border-box}
body{margin:0;padding:24px;font-family:'IBM Plex Sans',system-ui,sans-serif;color:var(--ink);background-color:var(--bg);
background-image:linear-gradient(#ffffff0d 1px,transparent 1px),linear-gradient(90deg,#ffffff0d 1px,transparent 1px);background-size:24px 24px}
h1{font-family:'IBM Plex Mono',monospace;font-weight:600;margin:0 0 4px;letter-spacing:.5px}
.sub{color:var(--mut);font-size:14px;margin-bottom:16px}
.stats{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px}
.stat{border:1px solid var(--line);padding:8px 14px;background:#0b2a4acc;font-family:'IBM Plex Mono',monospace;font-size:13px}
.stat b{display:block;font-size:20px;color:var(--acc)}
.wrap{overflow-x:auto;border:1px solid var(--line);background:#0b2a4af2}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:8px 10px;border-bottom:1px solid #2f5f8f66;text-align:right;white-space:nowrap}
th{position:sticky;top:0;background:#103a63;color:var(--acc);font-family:'IBM Plex Mono',monospace;font-weight:600;cursor:pointer;user-select:none}
th:nth-child(-n+3),td:nth-child(-n+3){text-align:left}
td{font-family:'IBM Plex Mono',monospace}
td:nth-child(2) a{color:var(--acc);text-decoration:none}
td:nth-child(3){font-family:'IBM Plex Sans',sans-serif}
tr:hover td{background:#7fd1ff14}
.up{color:var(--up)}.dn{color:var(--dn)}.na{color:var(--mut)}
details{margin-top:16px;color:var(--mut);font-size:13px;max-width:900px}
summary{cursor:pointer;color:var(--acc)}
</style></head><body>
<h1>STAGE 2 SCANNER</h1>
<div class="sub">Nifty Midcap 150 + Smallcap 250 &middot; price data as of __ASOF__ &middot; generated __GEN__ __RUNLINK__</div>
<div class="stats"><div class="stat"><b>__COUNT__</b>matches</div><div class="stat"><b>__SCANNED__</b>scanned</div></div>
<div class="wrap"><table id="t"><thead><tr>
<th>#</th><th>Symbol</th><th>Company</th><th>Cap</th><th>Close &#8377;</th><th>Surge %</th><th>Ext vs SMA %</th><th>Surge Vol x</th>
<th>Days in base</th><th>Pullback %</th><th>Base range %</th><th>Above 30SMA %</th>
<th>Qtr</th><th>EPS &#8377;</th><th>EPS YoY %</th><th>EPS QoQ %</th><th>Sales &#8377;Cr</th><th>Sales YoY %</th><th>Sales QoQ %</th>
</tr></thead><tbody>__ROWS__</tbody></table></div>
<details><summary>Rule definition used</summary><p>__CRITERIA__</p>
<p>EPS/sales come from Yahoo Finance quarterly statements and can be missing or lag the latest result - confirm on Screener.in before acting. Research aid only, not investment advice.</p></details>
<script>
const t=document.getElementById('t'),b=t.tBodies[0];let dir=1,last=-1;
t.tHead.addEventListener('click',e=>{const th=e.target.closest('th');if(!th)return;const i=th.cellIndex;dir=(i===last)?-dir:1;last=i;
[...b.rows].sort((x,y)=>{const a=x.cells[i].dataset.v,c=y.cells[i].dataset.v;const na=parseFloat(a),nc=parseFloat(c);
if(isNaN(na)&&isNaN(nc))return a.localeCompare(c)*dir;if(isNaN(na))return 1;if(isNaN(nc))return -1;return (na-nc)*dir}).forEach(r=>b.appendChild(r))});
</script></body></html>"""


def cell(v, fmt="{:,.1f}", color=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '<td class="na" data-v="">-</td>'
    cls = (" up" if v > 0 else " dn" if v < 0 else "") if color else ""
    return f'<td class="{cls.strip()}" data-v="{v:.4f}">{fmt.format(v)}</td>'


def render(results, scanned, asof):
    rows = []
    for i, r in enumerate(results, 1):
        sym = html.escape(r["symbol"])
        rows.append(
            f'<tr><td data-v="{i}">{i}</td>'
            f'<td data-v="{sym}"><a href="https://www.screener.in/company/{sym}/" target="_blank" rel="noopener">{sym}</a></td>'
            f'<td data-v="{html.escape(r["name"])}">{html.escape(r["name"])}</td>'
            f'<td data-v="{r["cap"]}">{r["cap"]}</td>'
            + cell(r["close"], "{:,.2f}") + cell(r["surge"]) + cell(r["ext"]) + cell(r["volx"], "{:.2f}")
            + cell(float(r["days"]), "{:.0f}") + cell(r["pull"]) + cell(r["rng"]) + cell(r["above30"])
            + f'<td data-v="{r["qtr"]}">{r["qtr"] or "-"}</td>'
            + cell(r["eps"], "{:,.2f}", True) + cell(r["eps_yoy"], "{:+,.1f}", True) + cell(r["eps_qoq"], "{:+,.1f}", True)
            + cell(r["sales"], "{:,.0f}") + cell(r["sales_yoy"], "{:+,.1f}", True) + cell(r["sales_qoq"], "{:+,.1f}", True)
            + "</tr>")
    c = CFG
    crit = (f"<b>Powerful move:</b> within the last {c['SURGE_WINDOW']} bars before the peak, price gains at least "
            f"{c['MIN_SURGE_PCT']:.0f}% low-to-peak, closes at least {c['MIN_EXTENSION_PCT']:.0f}% above both the 20 and 30 SMA, "
            f"and average volume is at least {c['MIN_SURGE_VOL_X']}x the 50-day average. "
            f"<b>Controlled base:</b> the peak is {c['BASE_MIN_DAYS']}-{c['BASE_MAX_DAYS']} sessions old, the stock holds above the 30 SMA (at most {c['MAX_DAYS_BELOW_30SMA_PCT']:.0f}% of base closes below it, never more than {c['MAX_DIP_BELOW_30SMA_PCT']:.0f}% under, and today above), "
            f"the base range is at most {c['MAX_BASE_RANGE_PCT']:.0f}%, the pullback from the peak is at most {c['MAX_PULLBACK_PCT']:.0f}%, "
            f"and base volume is lower than surge volume.")
    repo = os.environ.get("GITHUB_REPOSITORY")
    runlink = (f'&middot; <a style="color:var(--acc)" href="https://github.com/{repo}/actions/workflows/stage2.yml">Run scan now</a>'
               if repo else "")
    page = (PAGE.replace("__ROWS__", "".join(rows)).replace("__COUNT__", str(len(results)))
            .replace("__SCANNED__", str(scanned)).replace("__ASOF__", asof)
            .replace("__GEN__", dt.datetime.now(dt.timezone(dt.timedelta(hours=5, minutes=30))).strftime("%d %b %Y %H:%M IST"))
            .replace("__RUNLINK__", runlink).replace("__CRITERIA__", crit))
    CFG["OUT"].parent.mkdir(parents=True, exist_ok=True)
    CFG["OUT"].write_text(page, encoding="utf-8")


# --------------------------------------------------------------------------- #
def main():
    print("Loading universe...")
    uni = get_universe()
    print(f"{len(uni)} stocks. Downloading prices...")
    prices = load_prices(uni["symbol"].tolist())

    print("Scanning...")
    results, asof = [], ""
    for _, u in uni.iterrows():
        df = prices.get(u["symbol"])
        if df is None:
            continue
        hit = stage2(df)
        if hit:
            asof = hit["asof"]
            results.append(dict(symbol=u["symbol"], name=u["name"], cap=u["cap"], **hit))
    results.sort(key=lambda r: r["surge"], reverse=True)

    print(f"{len(results)} matches. Fetching quarterly EPS & sales...")
    for r in results:
        r.update(fundamentals(r["symbol"]))

    if not asof and prices:
        asof = max(df.index[-1] for df in prices.values()).strftime("%d %b %Y")
    render(results, len(prices), asof or "n/a")
    print(f"Done -> {CFG['OUT'].resolve()}")
    if not os.environ.get("CI"):
        webbrowser.open(CFG["OUT"].resolve().as_uri())


if __name__ == "__main__":
    main()
