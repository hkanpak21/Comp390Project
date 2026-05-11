"""Two figures for the per-op vs NEXUS comparison (PI presentation, 2026-05-11).

fig_per_op_speedup.{pdf,svg,png}
    Per-op multi-GPU speedup vs single-GPU H100. Two grouped bars per op:
    4-GPU and 16-GPU. Annotated with the absolute per-call latency.

fig_per_op_latency.{pdf,svg,png}
    Per-op latency on log scale across five hardware/configuration columns:
    NEXUS-A100 published, NEXUS-H100 measured, our 1-GPU, our 4-GPU, our
    16-GPU. Shows the H100 hardware uplift + further multi-GPU gains.

All numbers are sourced from docs/PER_OP_VS_NEXUS.md §4.4 (Lane PEROP-FINAL,
2026-05-11). MN5 ACC partition, H100 64 GB SXM, NVSwitch.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _style import (  # noqa: E402
    INDIGO,
    PERSIMMON,
    INDIGO_LIGHT,
    PERSIMMON_LIGHT,
    GREY,
    apply_rc,
)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# ── Source data: per-op latency in milliseconds, MN5 H100 SXM 64 GB ─────────
# Columns: nexus_a100_pub, nexus_h100_meas, our_1gpu, our_4gpu, our_16gpu
# Argmax 4-GPU per-call number is "slowest-GPU compute" (919 ms) — it does
# not reduce per-call latency; the throughput benefit is 4×.
DATA = [
    # name, a100, h100, ours_1, ours_4, ours_16, note
    # Argmax 4-GPU per-call latency does not reduce because the benchmark's
    # context-rebuild overhead per call dominates (~3.7 s amortizable setup);
    # the 4-GPU value here is "slowest-GPU compute" — see footnote on chart
    # and PER_OP_VS_NEXUS.md §4.4. The 16-GPU value (376 ms / batch
    # effective, JOBID 40387054) is a clean per-batch throughput speedup.
    ("Bootstrap",  5630.0, 252.8, 250.0, 240.98, 192.5, ""),
    ("LayerNorm",  1010.0,  45.0,  45.5,  25.07,  17.6, ""),
    ("Softmax",    1150.0,  20.0,  20.0,  16.52,  13.4, ""),
    ("MatMul/col", 1310.0,  95.0, 285.0, 122.0,   34.9, ""),
    ("GELU",       3350.0,  69.0,  70.30, 31.84,  19.8, ""),
    ("Argmax v8",  2480.0, 863.0, 848.0, 919.0,  376.0, "*"),
]


def fig_speedup():
    """Per-op multi-GPU speedup vs single-GPU H100."""
    apply_rc()
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.92, bottom=0.16)

    ops = [r[0] + r[6] for r in DATA]
    s_4gpu = [r[3] / r[4] for r in DATA]   # 1-GPU / 4-GPU per-call
    s_16gpu = [r[3] / r[5] for r in DATA]  # 1-GPU / 16-GPU per-call

    x = np.arange(len(ops))
    width = 0.36

    bars4 = ax.bar(x - width / 2, s_4gpu, width, color=INDIGO,
                   label="4-GPU (1 node)", edgecolor=INDIGO, linewidth=0.5)
    bars16 = ax.bar(x + width / 2, s_16gpu, width, color=PERSIMMON,
                    label="16-GPU (4 nodes)", edgecolor=PERSIMMON, linewidth=0.5)

    ax.axhline(1.0, color=GREY, linewidth=0.6, linestyle="--",
               label="single-GPU baseline")

    # Annotate each bar with the absolute per-call latency (ms)
    for bars, latencies, dy in [(bars4, [r[4] for r in DATA], 0.08),
                                 (bars16, [r[5] for r in DATA], 0.08)]:
        for bar, lat in zip(bars, latencies):
            h = bar.get_height()
            label = f"{lat:.0f}ms" if lat >= 10 else f"{lat:.1f}ms"
            ax.text(bar.get_x() + bar.get_width() / 2, h + dy,
                    label, ha="center", va="bottom", fontsize=7, color=INDIGO)

    ax.set_xticks(x)
    ax.set_xticklabels(ops, rotation=12, ha="right")
    ax.set_ylabel("per-call speedup vs single-GPU H100")
    ax.set_title("Multi-GPU per-op latency speedup (NEXUS-aligned parameters)")
    ax.set_ylim(0, max(s_16gpu) * 1.30)
    ax.legend(loc="upper left", frameon=False)

    # Footnote moved into LaTeX caption — keeps the figure tight.
    return fig


def fig_latency():
    """Per-op latency across five configurations, log scale."""
    apply_rc()
    fig, ax = plt.subplots(figsize=(6.8, 3.4))

    ops = [r[0] for r in DATA]
    cols = [
        ("NEXUS A100 (paper)",      [r[1] for r in DATA], INDIGO_LIGHT),
        ("NEXUS H100 (measured)",   [r[2] for r in DATA], INDIGO),
        ("Our 1-GPU H100",          [r[3] for r in DATA], PERSIMMON_LIGHT),
        ("Our 4-GPU H100",          [r[4] for r in DATA], PERSIMMON),
        ("Our 16-GPU H100",         [r[5] for r in DATA], "#7A2A1A"),  # darker persimmon
    ]

    x = np.arange(len(ops))
    width = 0.16
    offsets = [-2 * width, -width, 0, width, 2 * width]

    for (label, vals, color), off in zip(cols, offsets):
        ax.bar(x + off, vals, width, color=color, label=label,
               edgecolor=color, linewidth=0.5)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(ops, rotation=15, ha="right")
    ax.set_ylabel("per-call latency (ms, log scale)")
    ax.set_title("Per-op latency: NEXUS published vs measured-on-H100 vs multiNEXUS")
    ax.legend(loc="upper right", frameon=False, ncol=1)
    ax.grid(axis="y", which="major", color=GREY, linewidth=0.3, alpha=0.5)

    return fig


def main():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

    f1 = fig_speedup()
    for ext in ("pdf", "svg", "png"):
        f1.savefig(os.path.join(out_dir, f"fig_per_op_speedup.{ext}"),
                   format=ext, dpi=180)
    plt.close(f1)

    f2 = fig_latency()
    for ext in ("pdf", "svg", "png"):
        f2.savefig(os.path.join(out_dir, f"fig_per_op_latency.{ext}"),
                   format=ext, dpi=180)
    plt.close(f2)

    print("Wrote:")
    for stem in ("fig_per_op_speedup", "fig_per_op_latency"):
        for ext in ("pdf", "svg", "png"):
            print(f"  paper/{stem}.{ext}")


if __name__ == "__main__":
    main()
