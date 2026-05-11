"""Shared matplotlib style for multiNEXUS paper figures.

Palette: indigo + persimmon, cream background.  At most three colours per
figure; no rainbows.  Column-width sized for IEEEtran two-column.
"""

import matplotlib

# Force a non-interactive backend so headless invocations behave.
matplotlib.use("Agg")  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

INDIGO = "#1B3A55"
PERSIMMON = "#C8723A"
CREAM = "#F0E8D0"
INDIGO_LIGHT = "#5D7F9A"  # 55%-blend with cream, used sparingly for stacks
PERSIMMON_LIGHT = "#E0AD8A"
GREY = "#8A8A8A"


def apply_rc():
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.size"] = 9
    matplotlib.rcParams["axes.labelsize"] = 9
    matplotlib.rcParams["axes.titlesize"] = 10
    matplotlib.rcParams["legend.fontsize"] = 8
    matplotlib.rcParams["xtick.labelsize"] = 8
    matplotlib.rcParams["ytick.labelsize"] = 8
    matplotlib.rcParams["axes.spines.top"] = False
    matplotlib.rcParams["axes.spines.right"] = False
    matplotlib.rcParams["axes.linewidth"] = 0.8
    matplotlib.rcParams["xtick.major.width"] = 0.8
    matplotlib.rcParams["ytick.major.width"] = 0.8
    matplotlib.rcParams["axes.edgecolor"] = INDIGO
    matplotlib.rcParams["xtick.color"] = INDIGO
    matplotlib.rcParams["ytick.color"] = INDIGO
    matplotlib.rcParams["axes.labelcolor"] = INDIGO
    matplotlib.rcParams["text.color"] = INDIGO
    matplotlib.rcParams["figure.figsize"] = (3.4, 2.4)
    matplotlib.rcParams["savefig.bbox"] = "tight"
    matplotlib.rcParams["savefig.pad_inches"] = 0.02


def save(fig, out_pdf):
    """Save the figure to a PDF using consistent settings."""
    fig.savefig(out_pdf, format="pdf")
    plt.close(fig)
