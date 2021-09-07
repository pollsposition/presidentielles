import math
from typing import Dict

import altair as alt
import numpy as np
import pandas as pd


def sample_uncertainty(
    sample_size: int,
    probabilities: Dict[str, float],
    num_candidates=None,
    samples=1000,
    prior=0.5,
):
    """Return sample from the voting shares' posterior distributions."""

    if not num_candidates:
        num_candidates = len(probabilities)

    votes = {
        candidate: int(share * sample_size / 100)
        for candidate, share in probabilities.items()
    }
    alphas = {candidate: prior + v for candidate, v in votes.items()}
    betas = {
        candidate: sample_size - v + (num_candidates - 1) * prior
        for candidate, v in votes.items()
    }
    posterior_samples = {
        c: 100 * np.random.beta(alphas[c], betas[c], samples) for c in votes
    }
    return posterior_samples


def stripplot(
    results: Dict[str, np.ndarray],
    *,
    title: str,
    subtitle: str,
    source: str,
    sort=None,
    domain=[0, 34],
    xaxis="Intentions de votes (en %)",
):

    # If we don't specify any particular order we sort
    # them in the order in which they appear in the dataset.
    if not sort:
        sort = list(results.keys())

    data = pd.DataFrame(results).melt()

    stripplot = (
        alt.Chart(
            data,
            height=15,
        )
        .transform_calculate(jitter="sqrt(-2*log(random()))*cos(2*PI*random())")
        .mark_circle(size=1)
        .encode(
            x=alt.X(
                "value:Q",
                scale=alt.Scale(domain=domain),
                title=f"{xaxis}",
                axis=alt.Axis(
                    ticks=False,
                    grid=True,
                    labels=True,
                    domain=False,
                    labelFont="Lato",
                    labelFontSize=12,
                ),
            ),
            y=alt.Y(
                "jitter:Q",
                title=None,
                axis=None,
                scale=alt.Scale(range=[25, -1.0]),
            ),
            color=alt.Color("variable:N", legend=None),
            opacity=alt.value(0.5),
        )
        .facet(
            row=alt.Row(
                "variable",
                title=None,
                header=alt.Header(
                    labelAngle=0,
                    labelAlign="left",
                    labelFontWeight="normal",
                    labelFontSize=12,
                    labelFont="Lato",
                ),
                sort=sort,
            ),
        )
        .properties(
            title=alt.TitleParams(
                text="@pollsposition",
                font="Lato",
                color="darkgray",
                subtitle=f"Source: {source}",
                subtitleColor="lightgray",
                baseline="bottom",
                orient="bottom",
                anchor="end",
                dy=20,
            ),
        )
    )

    chart_with_title = (
        alt.concat(
            stripplot,
            title=alt.TitleParams(
                dy=-30,
                subtitle=f"{subtitle}",
                fontSize=19,
                text=f"{title}",
                subtitleFontSize=12,
                font="Lato",
            ),
            background="#fAfAfA",
        )
        .configure_facet(spacing=0)
        .configure_view(
            stroke=None,
            width=500,
            height=800,
        )
    )

    return chart_with_title


def comparisonplot(
    reference: Dict[str, np.ndarray],
    results: Dict[str, np.ndarray],
    *,
    title: str,
    subtitle: str,
    source: str,
):

    # First compute the difference in percentage points and save
    # a DataFrame
    data = {}
    for key, value in reference.items():
        if key in results:
            data[key] = results[key] - value

    df_data = (
        pd.DataFrame.from_dict(data, orient="index", columns=["intentions"])
        .reset_index()
        .reset_index()
    )

    # Then compute the x-domain and the x-ticks based on
    # the difference's range
    min_value = math.floor(min(data.values()))
    max_value = math.ceil(max(data.values()))
    domain = [min_value - 1, max_value + 1]
    xticks = [min_value + i for i in range(1 + max_value - min_value)]

    # Build the chart
    chart = (
        alt.Chart(df_data)
        .mark_circle(size=130)
        .encode(
            x=alt.X(
                "intentions:Q",
                scale=alt.Scale(domain=domain),
                axis=alt.Axis(
                    labelFont="Lato",
                    labelFontSize=13,
                    labelFontWeight="bold",
                    grid=True,
                    values=xticks,
                    domain=False,
                    format="+20",
                    ticks=False,
                    orient="top",
                    offset=10,
                ),
                title=None,
            ),
            y=alt.Y(
                "index",
                axis=alt.Axis(
                    labelFont="Lato",
                    labelFontSize=12,
                    grid=False,
                    ticks=False,
                    domain=False,
                ),
                title=None,
                sort=None,
            ),
            color=alt.Color("index", legend=None),
        )
        .properties(
            title=alt.TitleParams(
                text="@pollsposition",
                fontSize=15,
                color="darkgray",
                fontWeight="bold",
                subtitle=source,
                subtitleColor="lightgray",
                baseline="bottom",
                orient="bottom",
                anchor="end",
                dy=30,
                font="Lato",
            ),
        )
    )

    chart_with_title = alt.concat(
        chart,
        title=alt.TitleParams(
            dy=-30,
            text=title,
            fontSize=17,
            subtitle=subtitle,
            subtitleFontStyle="italic",
            font="Lato",
        ),
        background="#ffffff",
    ).configure_view(
        strokeWidth=0,
    )

    return chart_with_title
