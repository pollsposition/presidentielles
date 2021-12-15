from math import cos, sin, floor, sqrt, pi, ceil

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


def plot_pair(intentions, colors, c1, c2, scores=None, title="", num_points=100):
    """Plot pairwise difference in voting intentions."""

    srng = np.random.default_rng(0)
    diff = intentions[c1] - intentions[c2]
    x = 100 * srng.choice(diff, num_points)

    # Attribute candidate color to each posterior sample
    condlist = [x > 0, x < 0]
    choicelist = [colors[c1], colors[c2]]
    colors_points = np.select(condlist, choicelist)

    # Attribute position to each posterior sample
    limit = np.ceil(100 * np.max(abs(diff)))

    a = generate_blue_noise((2 * limit, 3), 0.2, seed=0)
    a[:, 0] = a[:, 0] - limit
    a[:, 1] = a[:, 1] + 0.1

    if scores is None:
        num_wins = int(100 * np.sum(diff > 0) / len(diff))
        scores = {c1: f"{num_wins:.0f} sur 100", c2: f"{100-num_wins:.0f} sur 100"}

    positions = []
    for x_i in x:
        idx = np.abs(a[:, 0] - x_i).argmin()  # it needs to stay on the same side
        positions.append(a[idx])
        a = np.delete(a, idx, 0)

    x_i = np.array([p[0] for p in positions])
    y_i = np.array([p[1] for p in positions])

    fig, ax = plt.subplots()

    # Plot the samples
    s = (ax.get_window_extent().width / 28 * 72.0 / fig.dpi) ** 2
    ax.scatter(x_i, y_i, s=s, edgecolor="white", c=colors_points)

    # Plot the vertical line that marks equality
    ax.axvline(0, color="black", lw=2)

    # And now plot the % ticks
    ax.axvline(5, ymin=0.1, color=colors[c1], lw=0.5)
    ax.axvline(-5, ymin=0.1, color=colors[c2], lw=0.5)

    m = limit // 5
    if m >= 2:
        ax.axvline(10, ymin=0.1, color=colors[c1], lw=0.5)
        ax.axvline(-10, ymin=0.1, color=colors[c2], lw=0.5)
        if m >= 3:
            ax.axvline(15, ymin=0.1, color=colors[c1], lw=0.5)
            ax.axvline(-15, ymin=0.1, color=colors[c2], lw=0.5)
            if m >= 4:
                ax.axvline(20, ymin=0.1, color=colors[c1], lw=0.5)
                ax.axvline(-20, ymin=0.1, color=colors[c2], lw=0.5)

    ax.set_xlim(-limit - 3, limit + 3)
    ax.set_ylim(-0.3, 3.2)

    # transparent background
    rect = ax.patch
    rect.set_alpha(0)

    # remove borders, ticks and labels
    ax.set_yticklabels([])
    ax.set_ylabel("")
    ax.yaxis.set_ticks_position("none")

    ax.set_xticklabels([])
    ax.xaxis.set_ticks_position("none")

    spines = ["top", "right", "left", "bottom"]
    for s in spines:
        ax.spines[s].set_visible(False)

    num_wins = np.ceil(100 * np.sum(x_i > 0) / len(x_i))

    ax.text(
        limit + 3,
        1.7,
        f"{c1} en tête",
        fontweight="normal",
        fontname="Futura PT",
        va="center",
        ha="left",
        fontsize=15,
        color=colors[c1],
    )
    ax.text(
        limit + 3,
        1.5,
        scores[c1],
        fontweight="bold",
        fontname="Futura PT",
        va="center",
        ha="left",
        fontsize=20,
        color=colors[c1],
    )

    ax.text(
        -limit - 3,
        1.7,
        f"{c2} en tête",
        fontweight="normal",
        fontname="Futura PT",
        va="center",
        ha="right",
        fontsize=15,
        color=colors[c2],
    )
    ax.text(
        -limit - 3,
        1.5,
        scores[c2],
        fontweight="bold",
        fontname="Futura PT",
        va="center",
        ha="right",
        fontsize=20,
        color=colors[c2],
    )

    ax.text(
        0,
        -0.31,
        f"Égalité",
        fontweight="light",
        fontname="Futura PT",
        va="top",
        ha="center",
        fontsize=16,
        color="black",
    )
    ax.text(
        5,
        -0.05,
        f"+5%",
        fontweight="light",
        fontname="Futura PT",
        va="top",
        ha="center",
        fontsize=10,
        color=colors[c1],
    )
    ax.text(
        -5,
        -0.05,
        f"+5%\nd'avance",
        fontweight="light",
        fontname="Futura PT",
        va="top",
        ha="center",
        fontsize=10,
        color=colors[c2],
    )
    m = limit // 5
    if m >= 2:
        ax.text(
            10,
            -0.05,
            f"+10%",
            fontweight="light",
            fontname="Futura PT",
            va="top",
            ha="center",
            fontsize=10,
            color=colors[c1],
        )
        ax.text(
            -10,
            -0.05,
            f"+10%",
            fontweight="light",
            fontname="Futura PT",
            va="top",
            ha="center",
            fontsize=10,
            color=colors[c2],
        )
        if m >= 3:
            ax.text(
                15,
                -0.05,
                f"+15%",
                fontweight="light",
                fontname="Futura PT",
                va="top",
                ha="center",
                fontsize=10,
                color=colors[c1],
            )
            ax.text(
                -15,
                -0.05,
                f"+15%",
                fontweight="light",
                fontname="Futura PT",
                va="top",
                ha="center",
                fontsize=10,
                color=colors[c2],
            )
            if m >= 4:
                ax.text(
                    20,
                    -0.05,
                    f"+20%",
                    fontweight="light",
                    fontname="Futura PT",
                    va="top",
                    ha="center",
                    fontsize=10,
                    color=colors[c1],
                )
                ax.text(
                    -20,
                    -0.05,
                    f"+20%",
                    fontweight="light",
                    fontname="Futura PT",
                    va="top",
                    ha="center",
                    fontsize=10,
                    color=colors[c2],
                )

    fig.text(
        0.5,
        1.06,
        "Différence d'intentions de vote au 1er tour",
        fontname="Futura PT",
        fontweight="bold",
        fontsize=20,
        ha="center",
    )
    fig.text(
        0.5,
        0.95,
        "Nous simulons le premier tour de l'élection 10000 fois et calculons\nla différence entre le nombre d'intention de votes obtenues pour les deux candidats.\nChaque point représente le résultat d'une simulation.",
        fontname="Futura PT",
        fontweight="normal",
        fontsize=12,
        ha="center",
        linespacing=1,
    )

    fig.text(
        1,
        0,
        "Tracé avec soin par @pollsposition",
        ha="right",
        va="bottom",
        fontsize=10,
        fontweight="normal",
        fontname="Futura PT",
        color="darkgray",
    )
    return fig


def generate_blue_noise(shape, radius, k=32, seed=None):
    """Generate blue noise over a two-dimensional rectangle of size (width,height)

    This code was copied from Nicolas Rougier's repo at https://github.com/rougier/scientific-visualization-book.

    Parameters
    ----------
    shape : tuple
        Two-dimensional domain (width x height)
    radius : float
        Minimum distance between samples
    k : int, optional
        Limit of samples to choose before rejection (typically k = 30)
    seed : int, optional
        If provided, this will set the random seed before generating noise,
        for valid pseudo-random comparisons.

    References
    ----------
    .. [1] Fast Poisson Disk Sampling in Arbitrary Dimensions, Robert Bridson,
           Siggraph, 2007. :DOI:`10.1145/1278780.1278807`

    """

    def sqdist(a, b):
        """Squared Euclidean distance"""
        dx, dy = a[0] - b[0], a[1] - b[1]
        return dx * dx + dy * dy

    def grid_coords(p):
        """Return index of cell grid corresponding to p"""
        return int(floor(p[0] / cellsize)), int(floor(p[1] / cellsize))

    def fits(p, radius):
        """Check whether p can be added to the queue"""

        radius2 = radius * radius
        gx, gy = grid_coords(p)
        for x in range(max(gx - 2, 0), min(gx + 3, grid_width)):
            for y in range(max(gy - 2, 0), min(gy + 3, grid_height)):
                g = grid[x + y * grid_width]
                if g is None:
                    continue
                if sqdist(p, g) <= radius2:
                    return False
        return True

    # When given a seed, we use a private random generator in order to not
    # disturb the default global random generator
    if seed is not None:
        from numpy.random.mtrand import RandomState

        rng = RandomState(seed=seed)
    else:
        rng = np.random

    width, height = shape
    cellsize = radius / sqrt(2)
    grid_width = int(ceil(width / cellsize))
    grid_height = int(ceil(height / cellsize))
    grid = [None] * (grid_width * grid_height)

    p = rng.uniform(0, shape, 2)
    queue = [p]
    grid_x, grid_y = grid_coords(p)
    grid[grid_x + grid_y * grid_width] = p

    while queue:
        qi = rng.randint(len(queue))
        qx, qy = queue[qi]
        queue[qi] = queue[-1]
        queue.pop()
        for _ in range(k):
            theta = rng.uniform(0, 2 * pi)
            r = radius * np.sqrt(rng.uniform(1, 4))
            p = qx + r * cos(theta), qy + r * sin(theta)
            if not (0 <= p[0] < width and 0 <= p[1] < height) or not fits(p, radius):
                continue
            queue.append(p)
            gx, gy = grid_coords(p)
            grid[gx + gy * grid_width] = p

    return np.array([p for p in grid if p is not None])
