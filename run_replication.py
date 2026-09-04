#!/usr/bin/env python
"""
Standalone replication of Araujo et al. (2023), PNAS 120(46) e2312451120.
=========================================================================

    python run_replication.py --demo    # synthetic panel, no data needed, ~2 s
    python run_replication.py --full    # the real 1985-2013 Amazon panel, ~5 min

Both modes run the same estimation code and report the same four estimates, so
the demo is a genuine test of the machinery rather than an illustration.

THREE SPECIFICATIONS x TWO INSTRUMENTS
--------------------------------------
The package reports a 3x2, because in each direction the difference is itself a
result rather than a choice to be buried in a flag.

                      | binary instrument      composition instrument
                      | Delta(W_t^[k] Y_0)     Delta(Wtilde_t^[k] Y_0)
  --------------------+----------------------------------------------
  paper-literal Y_0   | fixed cross-section     attenuation-corrected
  season-matched Y_0  | AUTHOR-CONFIRMED,       both corrections
                      |   overlap-excluded
  authors-346         | AUTHOR-CONFIRMED,       (their sample, no count
                      |   their own sample        factor)

  * The Y_0 axis is how the baseline is defined and, given that, which rows are
    admissible.  The authors have confirmed, in reply to a replication query in
    August 2026, that Y_0 is fixed at the sample's first year but MATCHED BY
    CALENDAR MONTH, which selects the season-matched rows; paper-literal is
    retained as a diagnostic.  The two
    season-matched rows differ only in the baseline year: `season-matched` drops
    it because there the instrument is identically the regressor, `authors-346`
    retains it because Table S1's 3,300,494 = 9,539 x 346 says the paper did.
    See data_loader.Spec.
  * binary vs composition is whether the shared cell-count factor is left in
    the instrument or split out.  The same reply confirmed the BINARY form: a
    binary sum over the upwind pixels, with no row normalisation and no
    additional distance weighting, and with the own pixel excluded.  Composition
    is therefore a stated deviation rather than a candidate reading.  See
    estimation.bennet_split.

`--spec` selects one; the default runs all three.  There are no other estimation
switches: anything that would change a coefficient is a named specification, not
an option.

Outputs, written to outputs/:
    summary.txt                 the comparison table, as printed
    coefficients.csv            beta_k for every spec x instrument, with SEs
    table_s1_comparison.tex     LaTeX table, booktabs
    fig1_decay_profile.png      beta_1..beta_K against Table S1, one panel per spec
    fig2_multiplier.png         implied Omega^(Delta+1) margins
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

if hasattr(sys.stdout, "reconfigure"):          # Greek letters on CP1252 consoles
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src import data_loader as dl                                  # noqa: E402
from src import estimation as est                                  # noqa: E402
from src import matrices as mx                                     # noqa: E402

OUT = HERE / "outputs"

# ── figure styling ───────────────────────────────────────────────────────────
# Three categorical slots from a palette validated for colour-vision deficiency
# (worst all-pairs CVD dE 9.2, normal-vision dE 24.0).  Each series also carries
# a distinct marker and dash pattern, so identity never rests on hue alone and
# the figures survive greyscale printing.  The two specifications are faceted
# rather than overlaid, which keeps three series per panel.
SURFACE = "#fcfcfb"
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8880"
SERIES = {"paper": "#2a78d6", "binary": "#eb6834", "composition": "#1baf7a"}
MARKER = {"paper": "o", "binary": "s", "composition": "^"}
DASH = {"paper": (None, None), "binary": (5, 2), "composition": (1.5, 1.5)}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_3)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_2, labelsize=9, length=3, width=0.8)
    ax.grid(axis="y", color=INK_3, alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


def figure_decay(fits: dict, K: int, path: Path) -> None:
    """One panel per specification; three series each, on a shared y-axis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    specs = [s for s in dl.SPECS if (s, "binary") in fits]
    ks = np.arange(1, K + 1)
    fig, axes = plt.subplots(1, len(specs), figsize=(4.2 * len(specs) + 1.4, 4.2),
                             dpi=200, sharey=True, squeeze=False)
    fig.set_facecolor(SURFACE)

    for ax, spec in zip(axes[0], specs):
        _style(ax)
        ax.axhline(0.0, color=INK_3, linewidth=0.8, zorder=1)
        rows = [("paper", "Table S1", est.PAPER_BETA[:K])]
        for inst in ("binary", "composition"):
            rows.append((inst, inst, fits[(spec, inst)].beta))
        for key, label, vals in rows:
            ax.plot(ks, vals, color=SERIES[key], linewidth=2.0,
                    marker=MARKER[key], markersize=4.0,
                    markeredgecolor=SURFACE, markeredgewidth=0.7,
                    dashes=DASH[key], label=label, zorder=3,
                    solid_capstyle="round")
        ax.set_title(spec, color=INK, fontsize=10, loc="left", pad=8)
        ax.set_xlabel("spatial lag order $k$", color=INK, fontsize=9.5)
        ax.set_xticks(ks if K <= 10 else np.arange(4, K + 1, 4))

    axes[0][0].set_ylabel(r"$\beta_k$", color=INK, fontsize=10)
    leg = axes[0][-1].legend(frameon=False, fontsize=9, loc="upper right")
    for t in leg.get_texts():
        t.set_color(INK_2)
    fig.suptitle("Spatial lag coefficients: published vs. replicated",
                 color=INK, fontsize=11.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def figure_multiplier(rows, path: Path) -> None:
    """
    Implied Omega^(Delta+1) margins: a dot at the mean with a min-max bar.

    Not overlaid histograms, because the paper publishes only a mean and a
    maximum -- a distribution cannot honestly be drawn for it, and this compares
    exactly what is comparable across all of them.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.6, 0.55 * len(rows) + 1.9), dpi=200)
    fig.set_facecolor(SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=INK_3, alpha=0.25, linewidth=0.6)

    ys = np.arange(len(rows))[::-1]
    for y, (key, label, mean, lo, hi) in zip(ys, rows):
        col = SERIES[key]
        if np.isfinite(lo) and np.isfinite(hi):
            ax.plot([lo, hi], [y, y], color=col, linewidth=2.0, zorder=2,
                    solid_capstyle="round")
        elif np.isfinite(hi):
            ax.plot([hi], [y], marker=MARKER[key], markersize=7.5, color=col,
                    markerfacecolor="none", markeredgewidth=1.6, zorder=3)
            ax.annotate(f"{hi:.2f} max", xy=(hi, y), xytext=(0, 8),
                        textcoords="offset points", ha="center",
                        color=INK_2, fontsize=8)
        if np.isfinite(mean):
            ax.plot([mean], [y], marker=MARKER[key], markersize=7.5, color=col,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
            ax.annotate(f"{mean:.3f}", xy=(mean, y), xytext=(0, 8),
                        textcoords="offset points", ha="center",
                        color=INK, fontsize=8.5)

    ax.set_yticks(ys)
    ax.set_yticklabels([r[1] for r in rows], color=INK, fontsize=9)
    ax.set_xlabel(r"$\Omega^{\Delta+1}$ row-sum margin  ($\Delta=1$)",
                  color=INK, fontsize=10)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_title("Implied spatial multiplier", color=INK, fontsize=11.5,
                 loc="left", pad=10)
    fig.text(0.005, 0.015,
             "Dot = mean across pixels, bar = min to max. The paper publishes "
             "summary statistics only, so no range is drawn for it.",
             color=INK_3, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# ── reporting ────────────────────────────────────────────────────────────────

QUANTITIES = [
    # key,          latex,                 fmt
    ("beta_1",      r"$\beta_1$",          "{:.4f}"),
    ("beta_2",      r"$\beta_2$",          "{:.4f}"),
    ("sum_beta",    r"$\sum_k \beta_k$",   "{:.4f}"),
    ("alpha",       r"$\alpha$",           "{:.4f}"),
    ("share_k1",    r"share in $k=1$",     "{:.3f}"),
    ("eff_lags",    "effective lags",      "{:.2f}"),
    ("corr_paper",  "corr.\\ Table S1",    "{:.3f}"),
    ("n_pos",       "positive coefs",      "{:.0f}"),
    ("omega2_mean", r"$\Omega^2$ mean",    "{:.3f}"),
]


def collect(fits: dict, mult: dict, K: int) -> pd.DataFrame:
    """One column per (spec, instrument), plus the published column."""
    paper = est.PAPER_BETA[:K]
    ps = est.shape_metrics(paper)
    cols = {"Table S1": {
        "beta_1": paper[0], "beta_2": paper[1] if K > 1 else np.nan,
        "sum_beta": float(paper.sum()), "alpha": est.PAPER_ALPHA,
        "share_k1": ps["share_k1"], "eff_lags": ps["eff_lags"],
        "corr_paper": 1.0, "n_pos": float(ps["n_pos"]),
        "omega2_mean": est.PAPER_OMEGA2_MEAN}}
    for (spec, inst), f in fits.items():
        sh = f.shape()
        cols[f"{spec} / {inst}"] = {
            "beta_1": f.beta[0], "beta_2": f.beta[1] if K > 1 else np.nan,
            "sum_beta": f.sum_beta, "alpha": f.alpha,
            "share_k1": sh["share_k1"], "eff_lags": sh["eff_lags"],
            "corr_paper": sh["corr_paper"], "n_pos": float(sh["n_pos"]),
            "omega2_mean": mult[(spec, inst)][0]}
    out = pd.DataFrame(cols)
    out.insert(0, "fmt", [f for _, _, f in QUANTITIES])
    out.insert(0, "latex", [l for _, l, _ in QUANTITIES])
    return out.reindex([k for k, _, _ in QUANTITIES])


def format_table(tab: pd.DataFrame) -> str:
    data = [c for c in tab.columns if c not in ("fmt", "latex")]
    w = max(len(q) for q in tab.index)
    width = max(16, max(len(c) for c in data) + 2)
    head = f"{'':<{w}}" + "".join(f"{c:>{width}s}" for c in data)
    lines = [head]
    for q, r in tab.iterrows():
        cells = "".join(
            f"{(r['fmt'].format(r[c]) if np.isfinite(r[c]) else '-'):>{width}s}"
            for c in data)
        lines.append(f"{q:<{w}}" + cells)
    return "\n".join(lines)


def to_latex(tab: pd.DataFrame, fits: dict, mode: str) -> str:
    data = [c for c in tab.columns if c not in ("fmt", "latex")]
    body = []
    for _, r in tab.iterrows():
        cells = " & ".join(
            (r["fmt"].format(r[c]) if np.isfinite(r[c]) else "--")
            for c in data)
        body.append(f"    {r['latex']} & {cells} \\\\")
    n_row = " & ".join(
        [f"{est.PAPER_N_OBS:,}"] +
        [f"{fits[tuple(c.split(' / '))].n_obs:,}" for c in data[1:]])
    return (
        f"% Generated by run_replication.py --{mode}\n"
        "% Requires \\usepackage{booktabs}.\n"
        "\\begin{table}[htbp]\n  \\centering\\small\n"
        "  \\caption{Replication of Araujo et al.\\ (2023), Table S1. "
        "Columns are the three $Y_0$/sample specifications crossed with the "
        "binary and composition instruments. The authors' confirmed "
        "construction is season-matched $Y_0$ with the binary instrument; "
        "\\texttt{authors-346} additionally reconstructs their estimation "
        "sample by retaining the baseline year.}\n"
        "  \\label{tab:replication}\n"
        f"  \\begin{{tabular}}{{l{'r' * len(data)}}}\n    \\toprule\n"
        "     & " + " & ".join(data) + " \\\\\n    \\midrule\n"
        + "\n".join(body) + "\n    \\midrule\n"
        f"    observations & {n_row} \\\\\n"
        "    \\bottomrule\n  \\end{tabular}\n\\end{table}\n")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--demo", action="store_true",
                   help="synthetic panel, no data required (default)")
    g.add_argument("--full", action="store_true",
                   help="the real panel, from a Step-4/6 checkpoint")
    # "both" is kept as an accepted alias for "all" so that the documented
    # invocations from before the third specification existed still run.
    ap.add_argument("--spec", choices=list(dl.SPECS) + ["all", "both"],
                    default="all",
                    help="which specification to estimate (default: all three)")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="path to the built input (default: "
                         f"data/{dl.DEFAULT_CHECKPOINT.name}, written by "
                         "--build-inputs)")
    ap.add_argument("--K", type=int, default=20, help="spatial lag orders")
    ap.add_argument("--seed", type=int, default=20260813)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--outputs-dir", type=Path, default=None,
                    help="where tables and figures go. Default: outputs/ for "
                         "--full, outputs_demo/ for --demo, so that a demo run "
                         "can never overwrite the committed full-mode results")
    g.add_argument("--build-inputs", action="store_true",
                   help="rebuild the full-mode input from monthly LAI and wind "
                        "NetCDFs; remaining arguments go to src.build_inputs")
    g.add_argument("--fetch-raw", action="store_true",
                   help="download those two NetCDFs from CDS and Earth Engine; "
                        "remaining arguments go to src.fetch_raw")
    args, extra = ap.parse_known_args()

    if args.build_inputs:
        from src import build_inputs
        return build_inputs.main(extra)
    if args.fetch_raw:
        from src import fetch_raw
        return fetch_raw.main(extra)
    if extra:
        ap.error(f"unrecognised arguments: {' '.join(extra)}")

    mode = "full" if args.full else "demo"
    specs = (list(dl.SPECS.values()) if args.spec in ("all", "both")
             else [dl.SPECS[args.spec]])

    # Demo and full write to different directories on purpose.  They used to
    # share one, so `--demo` silently overwrote the committed full-mode summary
    # and figures with synthetic numbers that look superficially similar.
    global OUT
    OUT = (args.outputs_dir if args.outputs_dir is not None
           else HERE / ("outputs" if mode == "full" else "outputs_demo"))
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    print("=" * 86)
    print("  Araujo et al. (2023) — standalone replication")
    print(f"  mode: {mode} | specifications: {', '.join(s.name for s in specs)}")
    print("=" * 86)

    panel = dl.resolve_panel(mode, args.checkpoint, args.K, args.seed)
    K = panel.K
    print(f"\nPanel: {panel.label}")
    print(f"  {panel.N:,} pixels x {panel.T} months, K = {K}")
    if panel.truth:
        print(f"  planted alpha = {panel.truth['alpha']:+.4f} | "
              f"planted Sum beta_k = {float(np.sum(panel.truth['beta'])):+.4f}")

    print("\nTime-averaging W for the multiplier …", flush=True)
    shells = mx.geodesic_shells(panel.G, K)
    gen = (mx.upwind_matrices(panel.C_t[ts], shells, K) for ts in panel.times)
    Wbar = {"binary": mx.time_average(gen, K, panel.N, standardise=False)}
    gen = (mx.upwind_matrices(panel.C_t[ts], shells, K) for ts in panel.times)
    Wbar["composition"] = mx.time_average(gen, K, panel.N, standardise=True)
    rs = mx.mean_row_sums(Wbar["binary"], K)
    print(f"  mean row sums of Wbar^[k]: k=1 {rs[0]:.3f} | k={K} {rs[-1]:.3f} "
          "(about one cell per shell)")

    fits, mult, overid = {}, {}, {}
    for spec in specs:
        print(f"\n--- {spec.name} ---", flush=True)
        Y0 = panel.Y0_table if panel.Y0_table is not None else spec.baseline(
            panel.Y, panel.times)
        if panel.Y0_table is not None and spec.y0_kind == "first-obs":
            Y0 = np.repeat(panel.Y0_table[:, [0]], 12, axis=1)
        Nc, M, M0 = est.build_channels(panel, Y0, progress=(mode == "full"))
        sample = est.assemble(panel, Nc, M, M0, spec)
        del Nc, M, M0
        err_x, err_z = sample["bennet_err"]
        print(f"    Bennet identity holds to {max(err_x, err_z):.1e}; "
              f"{len(sample['dy']):,} observations")
        m_bin, m_cmp = est.fit_models(sample)
        overid[spec.name] = est.fit_overidentified(sample)
        del sample
        for inst, f in (("binary", m_bin), ("composition", m_cmp)):
            fits[(spec.name, inst)] = f
            r, _ = mx.omega_margins(f.beta, Wbar[inst], K, delta=1)
            mult[(spec.name, inst)] = ((float(r.mean()), float(r.min()),
                                        float(r.max()))
                                       if r is not None else (np.nan,) * 3)
            print(f"    {inst:<12s} alpha {f.alpha:+.4f} | beta_1 "
                  f"{f.beta[0]:+.5f} (t {f.beta[0] / max(f.beta_se[0], 1e-300):6.1f})"
                  f" | Sum beta {f.sum_beta:+.5f} | Omega^2 "
                  f"{mult[(spec.name, inst)][0]:.3f}", flush=True)

    tab = collect(fits, mult, K)
    # The rules follow the table's own width: with three specifications it is
    # ~220 characters, and a fixed 86-wide rule would sit under a third of it.
    body = format_table(tab)
    rule = max(86, max(len(ln) for ln in body.splitlines()))
    lines = ["", "=" * rule, "  REPLICATION SUMMARY", "=" * rule,
             body, "-" * rule]
    for (spec, inst), f in fits.items():
        ar = (f"AR(2) m={f.ar2_m:+6.2f} p={f.ar2_p:.3f}"
              if np.isfinite(f.ar2_m) else "AR(2) n/a")
        lines.append(f"  {spec:<15s} {inst:<12s} n={f.n_obs:>10,} "
                     f"clusters={f.n_clust:>7,} min SW F={f.sw_f_min:>10.4g} "
                     f"cond(Z)={f.cond_Z:.3g}  {ar}")
    lines.append(f"  {'Table S1':<15s} {'(published)':<12s} "
                 f"n={est.PAPER_N_OBS:>10,} clusters={est.PAPER_N_CLUSTERS:>7,}")
    lines += ["",
              "  AR(2) is the test that licenses Y_it-2 as an instrument for "
              "dY_it-1: it must NOT reject.",
              "  AR(1) is expected to reject by construction and is reported "
              "in the run log, not here.",
              "  Hansen J does not exist for the specifications above -- K+1 "
              "endogenous against K+1 excluded",
              "  instruments is EXACT identification, q - p = 0.  The rows "
              "below add Y_it-3 to buy one degree of",
              "  freedom; they are a diagnostic on a SHORTER sample and their "
              "coefficients are not the headline ones."]
    for spec_name, od in overid.items():
        if od is None:
            lines.append(f"  {spec_name:<15s} {'overid GMM':<12s} "
                         "not computable (panel too short for Y_it-3)")
            continue
        lines.append(
            f"  {spec_name:<15s} {'overid GMM':<12s} n={int(od['n']):>10,} "
            f"(-{int(od['n_dropped']):,}) Hansen J={od['J']:.2f} on "
            f"{int(od['df'])} df, p={od['p']:.4f} | alpha {od['alpha']:+.4f} "
            f"| Sum beta {od['sum_beta']:+.4f}")
    lines += ["", "  Author-confirmed construction (authors' reply, August 2026):"
                  " season-matched Y_0, binary W.",
              "  'authors-346' is the same construction on their own sample "
              "(baseline year retained; see data_loader.Spec)."]
    if panel.truth:
        tb = float(np.sum(panel.truth["beta"]))
        f = fits[(specs[0].name, "binary")]
        lines += ["", "  DEMO RECOVERY CHECK (what --demo exists to show)",
                  f"    alpha      planted {panel.truth['alpha']:+.4f} -> "
                  f"recovered {f.alpha:+.4f} (SE {f.alpha_se:.4f})",
                  f"    Sum beta_k planted {tb:+.4f} -> recovered {f.sum_beta:+.4f}"]
    lines.append("=" * rule)
    summary = "\n".join(lines)
    print(summary)

    coefs = pd.DataFrame({"k": np.arange(1, K + 1),
                          "paper_beta": est.PAPER_BETA[:K]})
    for (spec, inst), f in fits.items():
        coefs[f"{spec}__{inst}__beta"] = f.beta
        coefs[f"{spec}__{inst}__se"] = f.beta_se
    (OUT / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    coefs.to_csv(OUT / "coefficients.csv", index=False)
    (OUT / "table_s1_comparison.tex").write_text(
        to_latex(tab, fits, mode), encoding="utf-8")

    if not args.no_figures:
        try:
            figure_decay(fits, K, OUT / "fig1_decay_profile.png")
            rows = [("paper", "Table S1 (published)", est.PAPER_OMEGA2_MEAN,
                     np.nan, est.PAPER_OMEGA2_MAX)]
            for (spec, inst), m in mult.items():
                rows.append((inst, f"{spec} / {inst}", m[0], m[1], m[2]))
            figure_multiplier(rows, OUT / "fig2_multiplier.png")
            print(f"\nFigures  -> {OUT / 'fig1_decay_profile.png'}")
            print(f"         -> {OUT / 'fig2_multiplier.png'}")
        except ImportError:
            print("\n[matplotlib not installed — figures skipped]")

    print(f"Tables   -> {OUT / 'table_s1_comparison.tex'}")
    print(f"         -> {OUT / 'coefficients.csv'}")
    print(f"Summary  -> {OUT / 'summary.txt'}")
    print(f"\nCompleted in {time.perf_counter() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
