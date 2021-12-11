import imageio
import matplotlib.gridspec as grid_spec
import matplotlib.pyplot as plt
import numpy as np


def plot(
    intentions,
    colors,
    dates="",
    pollster="",
    title="Intentions de vote au premier tour",
    sample_size=None,
    base="Inscrits",
    logo_path="./images/logo.png",
    max_intentions=30,
    ranks=[5, 95],
):
    """Use a forest plot to represent the voting intentions.

    In ASCII format:

    [LOGO] Intentions de vote au premier tour
    [    ] {date} | {pollster}

    | 10%                    | 15%               | 20%
    |         Candidat       |                   |
    |   13.5 ----o---- 14.2  |                   |
    |                        |                   |
    |                        |                   |
    """
    num_candidates = len(intentions)

    for candidate in intentions:
        try:
            colors[candidate]
        except KeyError:
            raise KeyError(f"You need to provide a color for candidate {candidate}")

    gs = grid_spec.GridSpec(num_candidates, 1)
    fig = plt.figure(figsize=(8, 10))
    axes = []

    for i, (c, samples) in enumerate(intentions.items()):
        axes.append(fig.add_subplot(gs[i : i + 1, 0:]))

        samples_r = 100 * samples
        percentiles = np.percentile(samples_r, ranks)
        axes[-1].plot(percentiles, [0.15, 0.15], lw=1, color=colors[c])
        axes[-1].scatter([np.mean(samples_r)], [0.15], color=colors[c])

        # setting uniform x and y lims
        axes[-1].set_xlim(0, max_intentions)
        axes[-1].set_ylim(0, 0.5)

        # transparent background
        rect = axes[-1].patch
        rect.set_alpha(0)

        # remove borders, ticks and labels
        axes[-1].set_yticklabels([])
        axes[-1].set_ylabel("")
        axes[-1].yaxis.set_ticks_position("none")

        axes[-1].set_xticklabels([])
        axes[-1].xaxis.set_ticks_position("none")

        axes[-1].axvline(5, lw=0.3, color="lightgray", ls="--")
        axes[-1].axvline(10, lw=0.3, color="lightgray", ls="--")
        axes[-1].axvline(15, lw=0.3, color="lightgray", ls="--")
        axes[-1].axvline(20, lw=0.3, color="lightgray", ls="--")
        axes[-1].axvline(25, lw=0.3, color="lightgray", ls="--")
        if i == 0:
            axes[-1].text(
                5.2,
                0.45,
                "5%",
                fontweight="bold",
                fontname="Futura PT",
                color="lightgray",
            )
            axes[-1].text(
                10.2,
                0.45,
                "10%",
                fontweight="bold",
                fontname="Futura PT",
                color="lightgray",
            )
            axes[-1].text(
                15.2,
                0.45,
                "15%",
                fontweight="bold",
                fontname="Futura PT",
                color="lightgray",
            )
            axes[-1].text(
                20.2,
                0.45,
                "20%",
                fontweight="bold",
                fontname="Futura PT",
                color="lightgray",
            )
            axes[-1].text(
                25.2,
                0.45,
                "25%",
                fontweight="bold",
                fontname="Futura PT",
                color="lightgray",
            )

        spines = ["top", "right", "left", "bottom"]
        for s in spines:
            axes[-1].spines[s].set_visible(False)

        axes[-1].text(
            np.mean(samples_r),
            0.3,
            f"{c}",
            fontweight="bold",
            fontname="Futura PT",
            va="center",
            ha="center",
            fontsize=12,
            color=colors[c],
        )
        axes[-1].text(
            percentiles[0] - 1,
            0.15,
            f"{percentiles[0]:.1f}",
            fontweight="normal",
            fontname="Futura PT",
            va="center",
            ha="center",
            fontsize=10,
            color=colors[c],
        )
        axes[-1].text(
            percentiles[1] + 1,
            0.15,
            f"{percentiles[1]:.1f}",
            fontweight="normal",
            fontname="Futura PT",
            va="center",
            ha="center",
            fontsize=10,
            color=colors[c],
        )

    axes.append(fig.add_axes([0.07, 0.9, 0.1, 0.1]))
    im = imageio.imread(logo_path)
    axes[-1].imshow(im)
    axes[-1].axis("off")

    fig.text(
        0.18, 0.942, f"{title}", fontsize=25, fontweight="bold", fontname="Futura PT"
    )
    fig.text(
        0.18,
        0.92,
        f"{dates} | {pollster} | N={sample_size} | {base}",
        fontsize=10,
        fontweight="normal",
        fontname="Futura PT",
        color="darkgray",
    )
    fig.text(
        0.93,
        0.08,
        "Tracé avec soin par @pollsposition",
        ha="right",
        va="bottom",
        fontsize=10,
        fontweight="normal",
        fontname="Futura PT",
        color="darkgray",
    )
    fig.text(
        0.93,
        0.067,
        "Barres et chiffres représentent les intervalles de crédibilité à 95%",
        ha="right",
        va="bottom",
        fontsize=8,
        fontweight="normal",
        fontname="Futura PT",
        color="darkgray",
    )

    gs.update(hspace=-0.1)
    return fig
