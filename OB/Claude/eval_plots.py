"""
eval_plots.py -- evaluation graphics for APS.

Expected input: ONE tidy DataFrame `df`, one row per (trial, method, noise, missing_rate):

    method        str    "baseline" | "logreg" | "random_forest" | "static_extra" | "adaptive"
    trial         int    experiment / seed id (same trial id across methods => paired comparison)
    noise         float  sensor noise level used for that run
    missing_rate  float  fraction of readings dropped
    true_zone     str/int
    pred_zone     str/int
    correct       0/1    pred_zone == true_zone
    graph_dist    int    graph distance between pred_zone and true_zone
    confidence    float  model confidence in pred_zone (NaN for the baseline)

For the adaptive-vs-static transition chart, pass a second DataFrame `adf`:
    trial, correct_before (0/1), correct_after (0/1)

`static_extra` is the fair control for "adaptive": the SAME model given one extra sensor
at a fixed/random location instead of the one the adaptive rule picked.
"""
from __future__ import annotations

import os
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ORDER = ["baseline", "logreg", "random_forest", "static_extra", "adaptive"]
LABEL = {"baseline": "Strongest-sensor baseline", "logreg": "Logistic Regression",
         "random_forest": "Random Forest", "static_extra": "Static + 1 extra sensor",
         "adaptive": "Adaptive (+1 chosen sample)"}
COLOR = {"baseline": "#7f7f7f", "logreg": "#E69F00", "random_forest": "#56B4E9",
         "static_extra": "#CC79A7", "adaptive": "#009E73"}


def _wrap(m, width=14):
    return textwrap.fill(LABEL[m], width)


def _methods(df):
    return [m for m in ORDER if m in set(df["method"])]


def wilson(k, n, z=1.96):
    """95% Wilson interval for a proportion (better than normal approx near 0 or 1)."""
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return p, c - h, c + h


def _acc_table(df, by):
    g = df.groupby(by)["correct"].agg(["sum", "count"]).reset_index()
    g[["p", "lo", "hi"]] = [wilson(k, n) for k, n in zip(g["sum"], g["count"])]
    return g


def plot_accuracy_bars(df, ax=None, title="Exact-source accuracy"):
    ax = ax or plt.subplots(figsize=(8, 4.5))[1]
    t = _acc_table(df, "method").set_index("method").loc[_methods(df)]
    x = np.arange(len(t))
    ax.bar(x, t["p"], color=[COLOR[m] for m in t.index],
           yerr=[t["p"] - t["lo"], t["hi"] - t["p"]], capsize=4)
    for i, (p, hi) in enumerate(zip(t["p"], t["hi"])):
        ax.text(i, hi + 0.02, f"{p:.0%}", ha="center", fontsize=9)
    ax.set_xticks(x, [_wrap(m) for m in t.index], fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("accuracy (95% CI)")
    ax.set_title(title)
    return ax


def plot_distance_distribution(df, ax=None, max_hops=4, title="How far off are the wrong guesses?"):
    """Stacked bars: share of trials at each graph distance (0 = exact hit)."""
    ax = ax or plt.subplots(figsize=(8, 4.5))[1]
    ms = _methods(df)
    d = df["graph_dist"].clip(upper=max_hops)
    share = pd.crosstab(df["method"], d, normalize="index").reindex(ms).fillna(0)
    shades = plt.get_cmap("RdYlGn_r")(np.linspace(0.05, 0.9, max_hops + 1))
    bottom = np.zeros(len(ms))
    for h, col in zip(range(max_hops + 1), shades):
        v = share.get(h, pd.Series(0, index=ms)).values
        ax.bar(range(len(ms)), v, bottom=bottom, color=col,
               label=f"{h} hops" if h < max_hops else f"{h}+ hops")
        bottom += v
    ax.set_xticks(range(len(ms)), [_wrap(m) for m in ms], fontsize=8)
    ax.set_ylabel("share of trials")
    ax.set_title(title)
    ax.legend(title="graph distance", fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    return ax


def plot_sensitivity(df, x="noise", hold=None, ax=None, title=None):
    """Accuracy vs noise (or missing_rate), one line per method, 95% CI band.
    `hold` pins the other variable, e.g. hold={"missing_rate": 0.0}."""
    ax = ax or plt.subplots(figsize=(8, 4.5))[1]
    for k, v in (hold or {}).items():
        df = df[np.isclose(df[k], v)]
    g = _acc_table(df, ["method", x])
    for m in _methods(df):
        s = g[g["method"] == m].sort_values(x)
        ax.plot(s[x], s["p"], marker="o", color=COLOR[m], label=LABEL[m])
        ax.fill_between(s[x], s["lo"], s["hi"], color=COLOR[m], alpha=0.15)
    ax.set_xlabel({"noise": "sensor noise level", "missing_rate": "fraction of readings missing"}.get(x, x))
    ax.set_ylabel("exact-source accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_title(title or f"Robustness to {x.replace('_', ' ')}"
                 + (f" ({', '.join(f'{k}={v}' for k, v in hold.items())})" if hold else ""))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return ax


def plot_noise_missing_heatmap(df, method="random_forest", ax=None):
    """Accuracy of one method over the noise x missing-rate grid."""
    ax = ax or plt.subplots(figsize=(6, 4.5))[1]
    t = _acc_table(df[df["method"] == method], ["noise", "missing_rate"])
    pv = t.pivot(index="noise", columns="missing_rate", values="p").sort_index(ascending=False)
    im = ax.imshow(pv.values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(pv.shape[1]), [f"{c:g}" for c in pv.columns])
    ax.set_yticks(range(pv.shape[0]), [f"{i:g}" for i in pv.index])
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            ax.text(j, i, f"{pv.values[i, j]:.0%}", ha="center", va="center",
                    color="w" if pv.values[i, j] < 0.6 else "k", fontsize=8)
    ax.set_xlabel("missing rate")
    ax.set_ylabel("noise")
    ax.set_title(f"{LABEL[method]}: accuracy by condition")
    plt.colorbar(im, ax=ax, label="accuracy")
    return ax


def plot_adaptive_transitions(adf, ax=None):
    """What did the one extra sample actually do, trial by trial?"""
    ax = ax or plt.subplots(figsize=(7, 4))[1]
    b, a = adf["correct_before"].astype(bool), adf["correct_after"].astype(bool)
    counts = {"Still correct": (b & a).sum(), "Fixed by sample": (~b & a).sum(),
              "Broken by sample": (b & ~a).sum(), "Still wrong": (~b & ~a).sum()}
    cols = ["#9ecae1", "#009E73", "#D55E00", "#bdbdbd"]
    ax.bar(counts.keys(), counts.values(), color=cols)
    n = len(adf)
    for i, v in enumerate(counts.values()):
        ax.text(i, v, f"{v}\n({v / n:.0%})", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(counts.values()) * 1.25)
    ax.set_ylabel("trials")
    ax.set_title("Effect of adaptive sampling on each trial")
    return ax


def plot_calibration(df, bins=5, ax=None):
    """Reliability: when the model says X% confident, is it right X% of the time?
    This is what justifies the 'sample again when confidence is low' threshold."""
    ax = ax or plt.subplots(figsize=(5.5, 5))[1]
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfectly calibrated")
    for m in _methods(df):
        s = df[(df["method"] == m)].dropna(subset=["confidence"])
        if s.empty:
            continue
        s = s.assign(bin=pd.cut(s["confidence"], np.linspace(0, 1, bins + 1), include_lowest=True))
        g = s.groupby("bin", observed=True).agg(conf=("confidence", "mean"), acc=("correct", "mean"))
        ax.plot(g["conf"], g["acc"], marker="o", color=COLOR[m], label=LABEL[m])
    ax.set_xlabel("mean predicted confidence")
    ax.set_ylabel("observed accuracy")
    ax.set_title("Confidence calibration")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return ax


def plot_confusion(df, method, zones=None, ax=None):
    """Row-normalized confusion matrix: which zones get mistaken for which."""
    ax = ax or plt.subplots(figsize=(5.5, 5))[1]
    s = df[df["method"] == method]
    zones = zones if zones is not None else sorted(set(s["true_zone"]) | set(s["pred_zone"]))
    cm = pd.crosstab(s["true_zone"], s["pred_zone"]).reindex(index=zones, columns=zones, fill_value=0)
    cm = cm.div(cm.sum(axis=1).replace(0, 1), axis=0)
    im = ax.imshow(cm.values, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(zones)), zones)
    ax.set_yticks(range(len(zones)), zones)
    ax.set_xlabel("predicted zone")
    ax.set_ylabel("true source zone")
    ax.set_title(f"{LABEL[method]}: confusion")
    plt.colorbar(im, ax=ax, label="share of true-source trials")
    return ax


def save(fig, name, outdir="figures"):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{name}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------
# Smoke test with FAKE numbers so the charts can be built before real results
# exist. These values are placeholders and must never appear in the report.
# ----------------------------------------------------------------------------
def make_fake_results(n_trials=150, zones=tuple(f"Z{i}" for i in range(8)), seed=0):
    rng = np.random.default_rng(seed)
    base = {"baseline": 0.55, "logreg": 0.68, "random_forest": 0.74, "static_extra": 0.76, "adaptive": 0.84}
    rows = []
    for noise in (0.05, 0.1, 0.2, 0.4):
        for miss in (0.0, 0.1, 0.2, 0.4):
            for t in range(n_trials):
                true = rng.choice(zones)
                for m, b in base.items():
                    p = np.clip(b - 0.9 * noise - 0.5 * miss, 0.1, 0.99)
                    ok = rng.random() < p
                    pred = true if ok else rng.choice([z for z in zones if z != true])
                    conf = np.nan if m == "baseline" else float(np.clip(rng.normal(0.8 if ok else 0.5, 0.15), 0.05, 0.99))
                    rows.append((m, t, noise, miss, true, pred, int(ok), 0 if ok else int(rng.integers(1, 4)), conf))
    return pd.DataFrame(rows, columns=["method", "trial", "noise", "missing_rate", "true_zone",
                                       "pred_zone", "correct", "graph_dist", "confidence"])


if __name__ == "__main__":
    df = make_fake_results()
    ref = df[(df.noise == 0.1) & (df.missing_rate == 0.1)]
    print(save(plot_accuracy_bars(ref).figure, "accuracy_bars"))
    print(save(plot_distance_distribution(ref).figure, "distance_distribution"))
    print(save(plot_sensitivity(df, "noise", hold={"missing_rate": 0.0}).figure, "noise_sensitivity"))
    print(save(plot_sensitivity(df, "missing_rate", hold={"noise": 0.1}).figure, "missing_sensitivity"))
    print(save(plot_noise_missing_heatmap(df, "adaptive").figure, "heatmap_adaptive"))
    print(save(plot_calibration(ref).figure, "calibration"))
    print(save(plot_confusion(ref, "random_forest").figure, "confusion_rf"))
    rng = np.random.default_rng(1)
    b = rng.random(300) < 0.7
    a = np.where(b, rng.random(300) < 0.95, rng.random(300) < 0.45)
    adf = pd.DataFrame({"trial": range(300), "correct_before": b.astype(int), "correct_after": a.astype(int)})
    print(save(plot_adaptive_transitions(adf).figure, "adaptive_transitions"))
