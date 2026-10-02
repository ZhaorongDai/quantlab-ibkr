# CRSP price factor vs share factor for a held position

Research for issue #16, surfaced by #15 / ADR 0009. Question: is CRSP's price-adjustment factor
the right factor for changing a held position's share count? When do the price and share factors
differ, which one should ADR 0009's split venue fill use for the share count and for cash in lieu,
must quantlab expose a share-factor column, and what does a spin-off's new PERMNO mean for the
backtest?

Date: 2026-10-01.

## Sources and how to read the citations

- **[CRSP-DDG]** CRSP, *Data Descriptions Guide, CRSP US Stock & US Index Databases* (legacy
  CRSPAccess/SIZ definitions), public copy at
  <https://clouddc.chass.utoronto.ca/ds/crsp/en/manuals/data_descriptions_guide.pdf>, pp. 24, 32,
  39 (entries "Cumulative Factor to Adjust Prices/Shares", "Dividend Cash Amount", "Factor to
  Adjust Price in Period", "Factor to Adjust Shares Outstanding"). The same "Factor to Adjust
  Price" text appears in the chapter excerpt at
  <https://leiq.bus.umich.edu/docs/crsp_factor_adjustment.pdf>, p. 61-62.
- **[WRDS-CIZ]** WRDS Research Webinar, *CRSP CIZ Data* (F. Song Drechsler),
  <https://wrds-www.wharton.upenn.edu/documents/2084/Webinar.pdf>, pp. 9, 11: the SIZ `DISTCD` ->
  CIZ `StkDistributions` flag mapping and the `DisType` codes from `crsp.metaflagcoverage`.
- **[quantlab]** `~/projects/quantlab` at `f16dd10`: `quantlab/dataset/crsp/__init__.py`,
  `quantlab/dataset/crsp/reference.py`, `quantlab/acquisition/wrds/crsp.py`,
  `docs/wrds_crsp.md`.
- **[ADR]** this repo's ADR 0002, 0006, 0009, and #15's resolution comment.

The CIZ (v2) *Data Descriptions Guide* and *Metadata Guide* PDFs on crsp.org returned 404 when
fetched, and the WRDS variable pages are login-gated. The factor **definitions** below are
therefore the legacy (SIZ) ones. CIZ keeps the same concepts under new names
(`DisFacPr`/`DisFacShr`, `DlyCumFacPr`/`DlyCumFacShr`) and decomposes the legacy 4-digit
`DISTCD` into flags [WRDS-CIZ]; that the CIZ *values* follow the legacy rules case by case is
**unverified**. No WRDS query was run, and no CRSP rows were inspected (the server's
reference tier was not read for this note), so **frequencies are not measured** here; see
"Open measurement" below.

## Summary

| Question | Answer |
|---|---|
| Do the price and share factors differ? | Yes, by design, for spin-offs, rights, non-final liquidating distributions, issuances and limited tender offers [CRSP-DDG]. They are equal for splits and stock dividends. |
| Is `splitFactor` (price factor) right for the share count? | Only on days whose event is a split or stock dividend. On a spin-off/rights/partial-liquidation day it is `1 + DIVAMT/P(t) > 1` and would add parent shares the holder never receives. |
| Is the share factor right instead? | No, not in general either: it moves on issuances and rights (all holders assumed to exercise) that leave a holder's count unchanged, and it is 0 for spin-offs. |
| What should the venue use? *(delegated)* | Treat a day as a holder split only when the price ratio and the share ratio agree: `k = splitFactor` iff `splitFactor != 1` and `splitFactor ≈ shareFactor`, with `shareFactor = cumfacshr[t-1]/cumfacshr[t]`. Cash in lieu stays `pre-split close / k`, only for such days. |
| Must quantlab expose a share-factor column? *(delegated)* | No. `cumfacshr` (and `cumfacpr`, `facprc`) are already panel variables of `CrspStockDataset`; trader derives the ratio. trader's required-variable list (ADR 0006) gains `cumfacshr`. |
| Spin-off / new PERMNO *(delegated)* | v1: no position in the new PERMNO. The holder keeps its parent shares and the venue credits the distribution's value as cash, `q × (splitFactor - 1) × close[t]`, at the next open, unless `divCash` already carries it (unverified, check first). |

## 1. What quantlab derives today

From [quantlab] `quantlab/dataset/crsp/__init__.py`:

- The raw tier is `dsf_v2` rows, including `dlyfacprc`, `dlycumfacpr`, `dlycumfacshr`,
  `dlyorddivamt`, `dlynonorddivamt` (`quantlab/acquisition/wrds/crsp.py:674-677`).
- `splitFactor = dlycumfacpr[t-1] / dlycumfacpr[t]`, 1.0 on a PERMNO's first row in the window
  (module docstring line 32; `_finalise`, lines 1288-1295).
- `divCash = dlyorddivamt + dlynonorddivamt`, nulls as 0 (lines 1279-1287).
- `adjClose` compounds `dlyret` from an anchor close (lines 25-29), so it does not use the price
  factor at all; `adjVolume` scales by `dlycumfacshr` relative to the anchor (line 30).
- `cumfacpr`, `cumfacshr` and `facprc` (`dlyfacprc`, "the day's own price factor", 4.0 on a 4:1
  split) are exported unchanged as panel variables (`CRSP_EXTRA_VARIABLES`, lines 66-96;
  `docs/wrds_crsp.md:86,97`).
- `stkdistributions` is downloaded into the reference tier with `distype`, `disdetailtype`,
  `disdivamt`, `disfacpr`, `disfacshr`, `dispermno`, `dispermco`
  (`quantlab/dataset/crsp/reference.py:200-223`). Its comment says the panel's `divCash` and
  `splitFactor` come from the daily table and this table "is kept to audit them". Nothing in
  quantlab's backtest reads `splitFactor`, `divCash` or `cumfacshr` (grep of
  `quantlab/backtest`, `quantlab/base/backtest.py`, `quantlab/base/portfolio.py`): the vectorbt
  engine trades adjusted prices (ADR 0002).

The direction is right for a split: a 4:1 split gives `splitFactor = 4` (AAPL 2020-08-31 in
`docs/wrds_crsp.md:101-106`).

## 2. When CRSP's two factors differ

Per-event factors, verbatim in substance from [CRSP-DDG] p. 39 ("Factor to Adjust Price in
Period", "Factor to Adjust Shares Outstanding"):

| Event | Factor to Adjust Price (`facpr`) | Factor to Adjust Shares (`facshr`) | Holder's share count |
|---|---|---|---|
| Ordinary cash dividend, partial liquidating payment | 0 | 0 | unchanged |
| Merger, total liquidation, full exchange, security disappears | -1 by convention | (same) | position ends (delisting path) |
| Split, stock dividend | `s(t)/s(t') - 1` (reverse split in (-1, 0)) | equal to `facpr` | × `1 + facpr` |
| Spin-off | `DIVAMT / P(t)`, P(t) the ex-date price | **0** | unchanged; receives shares of another security |
| Rights | `DIVAMT / P(t)` | computed as if all holders exercise (or 0) | unchanged unless the holder exercises |
| Non-total / non-final liquidating distribution | `DIVAMT / P(t)` | not equal (value not given) | unchanged |
| Issuance | 0 | nonzero possible, computed like a split | unchanged |
| Limited tender offer | `-(fraction of shares accepted)` | not equal | changes only for shares the holder tendered |

"Dividend Cash Amount" for a distribution paid in shares of a trading security is "the exchange
ratio times the price of the security at the close of the Ex-Distribution Date" [CRSP-DDG] p. 32.
The cumulative factors are the running products of these per-event factors from a base date, the
price one "used to adjust prices after distributions", the share one "to adjust shares and
volume ... as a ratio of the additional shares out expected after the distribution to the last
known observation" [CRSP-DDG] p. 24.

So on an ex-date, with `splitFactor = cumfacpr[t-1]/cumfacpr[t] = 1 + facpr` and
`shareFactor = cumfacshr[t-1]/cumfacshr[t] = 1 + facshr` (the per-day identity is the cumulative
definition applied to one event; **unverified** for CIZ on days with several events):

- split / stock dividend: `splitFactor = shareFactor = k`;
- spin-off, rights, partial liquidation: `splitFactor > 1`, `shareFactor` = 1 (spin-off) or
  something else (rights);
- issuance / share-count observation: `splitFactor = 1`, `shareFactor` possibly != 1;
- limited tender: `splitFactor < 1`, `shareFactor` != `splitFactor`.

CIZ `DisType` codes [WRDS-CIZ] p. 11: `CD` cash dividend, `CG` capital gains, `CP` cash payment,
`FRS` forward or reverse split, `IN` issuer notification, `ROC` return of capital, `SD` special
dividends, `SP` security payment, `TSOO` total shares outstanding observation, `N/A`. The example
on p. 9 maps legacy `DISTCD 5523` to `DisType FRS`, `DisDetailType STKSPL`. That spin-offs are
coded `SP` (with `DisPERMNO` the distributed security) and stock dividends `FRS` is the natural
reading of the labels but is **unverified**; the `DisDetailType` code list was not found in a
public source.

## 3. What goes wrong with `splitFactor` alone

ADR 0009's split fill is `floor(q * k) - q` shares at price 0, plus cash in lieu at
`pre-split close / k`, with `k = splitFactor`. On a spin-off day:

- the holder gets `q × DIVAMT/P(t)` extra **parent** shares. Their value at the ex-date close is
  `q × DIVAMT`, the spin-off's value, so equity is right at that close; afterwards they move with
  the parent instead of the spun-off company, the share count differs from what a broker shows,
  and per-share fees and ADR 0002's "the backtest behaves as live trading will" break;
- the fractional remainder is paid as "cash in lieu" of a split that did not happen;
- **ADR 0009's queued-order rescaling multiplies a pre-ex-date exit order by `splitFactor`**. Across
  a spin-off this inflates a full exit by `1 + facpr` and, with the spurious parent shares, still
  closes the position, but a partial order is sized wrong. If the share count is fixed (no
  spurious shares) and rescaling still uses `splitFactor`, a full exit oversells and opens a
  short. The rescale must use the same holder factor as the fill.

On a limited-tender day `splitFactor < 1` would *remove* shares from every holder, although only
tendered shares are bought back. On an issuance day nothing happens under `splitFactor`, which is
correct; the same day under `shareFactor` would wrongly add shares. Neither cumulative factor is a
holder's share factor on its own.

A further caveat: `facpr = -1` on a final event means `cumfacpr` can reach 0, so `splitFactor`
can be non-finite on or after such a row (**unverified** whether CIZ writes it on the delisting
row). ADR 0009's schedule must skip non-finite factors and leave the position to the delisting
path.

## 4. Recommendation *(delegated)*

1. **Holder split rule.** The venue's split schedule uses
   `k = splitFactor` only where `splitFactor != 1` and
   `abs(splitFactor / shareFactor - 1) < 1e-6`, `shareFactor = cumfacshr[t-1]/cumfacshr[t]`.
   Cash in lieu stays `pre-split close / k` for those days only. The queued-order rescale uses the
   same `k`. Any other day where either ratio != 1 changes no share count and is written to
   `events.json` with both ratios, so mismatches are visible.
2. **Value distributions** (`splitFactor > 1`, `shareFactor = 1`; spin-offs, partial liquidations;
   rights treated as not exercised): the holder keeps `q` and is credited
   `q × (splitFactor - 1) × close[t]` through `exchange.adjust_account`, which equals
   `q × DIVAMT` by CRSP's definition. Credit it at the next bar's open (as ADR 0009 does for a
   delisting on b+1), because `close[t]` is not known at the ex-date open. A short pays it, like
   a dividend. **Guard against double counting:** if `divCash[t]` is already nonzero on such a
   day, CIZ may already carry the distribution's value in `dlynonorddivamt` (**unverified**); the
   open measurement below decides which of the two the venue books.
3. **Limited tender** (`splitFactor < 1`, `shareFactor != splitFactor`): no action; the holder did
   not tender. Logged.
4. **No new quantlab column.** `cumfacshr` is already exported; trader derives `shareFactor`.
   ADR 0006's required variables for a CRSP price dataset gain `cumfacshr`. A Tiingo-sourced
   store has no `cumfacshr`; for it trader falls back to `splitFactor` alone and logs that it
   cannot tell a split from a value distribution. Adding a `shareFactor` (or holder-split)
   variable to quantlab is deferred until a second consumer needs it.
5. **Spin-off PERMNO.** v1 does not open a position in `DisPERMNO`. The cash credit above is
   equivalent to selling the distributed shares at their ex-date close with zero fee. Opening
   the new position would need the new PERMNO's prices and a strategy decision about a security
   that is in no prediction panel. This matches quantlab's vectorbt run in value on the ex-date
   (CRSP's `dlyret`, which `adjClose` compounds, includes the distribution; **unverified** for
   the exact CIZ return formula) and differs afterwards only by reinvestment, as ordinary
   dividends already do (ADR 0002). Live, IBKR will deliver real shares of the new security;
   handling them is for the live effort.

## Open measurement

Not run here. A one-off check against the existing reference tier
(`_reference/stkdistributions.parquet`) and a CRSP store would settle the unverified points:

- counts by `distype`/`disdetailtype` of rows with `disfacpr != disfacshr`, restricted to common
  stock (join `stksecurityinfohist`) and 2000-2025, to give the frequency #16 asks for;
- for spin-off rows (`dispermno` not null), whether `dlynonorddivamt` on the ex-date equals
  `disdivamt` (decides recommendation 2's double-counting guard);
- that the panel's `cumfacpr`/`cumfacshr` ratios equal `1 + disfacpr`/`1 + disfacshr` on those
  dates, and what they read on days with several events and on a final (`disfacpr = -1`) row.
  Known spin-offs to spot-check: 3M -> Solventum (2024-04-01), GE -> GE Vernova (2024-04-02).
