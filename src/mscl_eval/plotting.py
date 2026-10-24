"""Vector/raster plots from supplied predictions, with computed annotations."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from lifelines import KaplanMeierFitter
from sklearn.metrics import roc_curve, roc_auc_score

from .io import aligned
from .metrics import available_case_roc, km_summary

COLOURS = {"MSCL_HE": "#1F77B4", "MSCL_HE_IHC": "#2CA02C",
           "MSCL_HE_predIHC": "#FF7F0E", "GigaPath": "#9E9E9E",
           "PRISM": "#6F6F6F", "TITAN": "#BDBDBD"}


def setup():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})


def colour(model, index):
    # Reserve the established blue/green/orange/gray colours for named models.
    other = ["#9467BD", "#8C564B", "#E377C2", "#17BECF", "#BCBD22", "#D62728"]
    return COLOURS.get(model, other[index % len(other)])


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)


def save(fig, path):
    fig.tight_layout()
    for extension in ("pdf", "png", "svg"):
        fig.savefig(str(path) + "." + extension, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_comparisons(comparisons, output):
    if comparisons.empty:
        return
    fig, ax = plt.subplots(figsize=(9, max(3, len(comparisons) * 0.5)))
    labels = []
    for i, row in enumerate(comparisons.itertuples()):
        ax.plot([row.ci_low, row.ci_high], [i, i], color="#444444")
        ax.scatter(row.delta, i, color="#D55E00" if row.q_value < 0.05 else "#4477AA")
        context = row.cohort + (": " + row.marker if hasattr(row, "marker") else "")
        labels.append(f"{context}: {row.candidate} − {row.baseline} (q={row.q_value:.3g})")
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("Metric difference (candidate − baseline), 95% bootstrap CI")
    ax.set_title("Paired comparisons; BH correction across the supplied comparison family")
    style(ax)
    save(fig, output / "paired_comparisons")


def plot_survival(frame, summary, comparisons, horizons, output):
    setup()
    cohorts = sorted(frame.cohort.unique())
    models = sorted(frame.model.unique())
    fig, axes = plt.subplots(1, len(cohorts), figsize=(5 * len(cohorts), 4), squeeze=False)
    for ax, cohort in zip(axes.flat, cohorts):
        rows = summary[summary.cohort.eq(cohort)].sort_values("model")
        for i, row in enumerate(rows.itertuples()):
            ax.plot([i, i], [row.ci_low, row.ci_high], color=colour(row.model, i))
            ax.scatter(i, row.c_index, color=colour(row.model, i))
            ax.annotate(f"{row.c_index:.3f}", (i, row.ci_high), xytext=(0, 5),
                        textcoords="offset points", ha="center", fontsize=8)
        ax.set_xticks(range(len(rows)), rows.model, rotation=35, ha="right")
        ax.set_ylim(0, 1.07)
        ax.set_ylabel("Harrell's C-index (95% bootstrap CI)")
        ax.set_title(f"{cohort} (n={rows.n.iloc[0]}, events={rows.events.iloc[0]})")
        style(ax)
    save(fig, output / "cindex")

    fig, axes = plt.subplots(len(models), len(cohorts),
                             figsize=(4.4 * len(cohorts), 3.5 * len(models)), squeeze=False)
    for col, cohort in enumerate(cohorts):
        rows, present, risks = aligned(frame[frame.cohort.eq(cohort)], "risk")
        time, event = rows.time.to_numpy(), rows.event.to_numpy()
        for row, model in enumerate(models):
            ax = axes[row, col]
            if model not in present:
                ax.set_axis_off()
                continue
            risk = risks[:, present.index(model)]
            info = km_summary(time, event, risk)
            high = risk >= info["threshold"]
            for mask, label, c in [(~high, "Low risk", "#2C7BB6"), (high, "High risk", "#B64342")]:
                if mask.any():
                    km = KaplanMeierFitter().fit(time[mask], event[mask], label=f"{label} (n={mask.sum()})")
                    km.plot_survival_function(ax=ax, ci_show=True, show_censors=True, color=c)
            p = "not estimable" if info["logrank_p"] is None else f"{info['logrank_p']:.3g}"
            ax.text(0.04, 0.08, f"Log-rank P = {p}", transform=ax.transAxes)
            ax.set(title=f"{cohort}: {model}", xlabel="Time (months)",
                   ylabel="Overall survival probability", ylim=(0, 1.05))
            ax.legend(loc="upper right", frameon=False, fontsize=8)
            style(ax)
    save(fig, output / "kaplan_meier")

    fig, axes = plt.subplots(len(cohorts), len(horizons),
                             figsize=(4.5 * len(horizons), 4 * len(cohorts)), squeeze=False)
    for row, cohort in enumerate(cohorts):
        rows, present, risks = aligned(frame[frame.cohort.eq(cohort)], "risk")
        for col, horizon in enumerate(horizons):
            ax = axes[row, col]
            for i, model in enumerate(present):
                roc = available_case_roc(rows.time, rows.event, risks[:, i], horizon)
                if roc["auc"] is not None:
                    ax.plot(roc["fpr"], roc["tpr"], color=colour(model, i),
                            label=f"{model} ({roc['auc']:.3f})")
            ax.plot([0, 1], [0, 1], "--", color="silver", linewidth=0.8)
            ax.set(title=f"{cohort}: {horizon:g} months", xlabel="1 − Specificity",
                   ylabel="Sensitivity", xlim=(0, 1), ylim=(0, 1.02))
            if ax.get_legend_handles_labels()[0]:
                ax.legend(loc="lower right", frameon=False, fontsize=7)
            style(ax)
    fig.suptitle("Available-case ROC (early-censored observations excluded; not IPCW)", y=1.02)
    save(fig, output / "available_case_roc")
    plot_comparisons(comparisons, output)


def plot_markers(frame, comparisons, output):
    setup()
    pairs = list(frame.groupby(["cohort", "marker"], sort=True))
    cols = min(2, len(pairs))
    fig, axes = plt.subplots(int(np.ceil(len(pairs) / cols)), cols,
                             figsize=(5 * cols, 4.5 * np.ceil(len(pairs) / cols)), squeeze=False)
    for ax, ((cohort, marker), part) in zip(axes.flat, pairs):
        rows, models, scores = aligned(part, "score")
        for i, model in enumerate(models):
            fpr, tpr, _ = roc_curve(rows.label, scores[:, i])
            auc = roc_auc_score(rows.label, scores[:, i])
            ax.plot(fpr, tpr, color=colour(model, i), label=f"{model} ({auc:.3f})")
        ax.plot([0, 1], [0, 1], "--", color="silver")
        ax.set(title=f"{cohort}: {marker} (n={len(rows)}, positive={int(rows.label.sum())})",
               xlabel="1 − Specificity", ylabel="Sensitivity", xlim=(0, 1), ylim=(0, 1.02))
        ax.legend(loc="lower right", fontsize=8, frameon=False)
        style(ax)
    for ax in list(axes.flat)[len(pairs):]:
        ax.set_axis_off()
    save(fig, output / "marker_roc")
    plot_comparisons(comparisons, output)
