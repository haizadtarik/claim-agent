from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from fraud_detection.data.load import load_data

REPORT_PATH = Path(__file__).resolve().parents[3] / "reports" / "eda_report.html"

TARGET_COLUMN = "fraud_reported"

NUMERIC_DISTRIBUTION_COLUMNS = [
    "age",
    "months_as_customer",
    "total_claim_amount",
    "policy_annual_premium",
    "incident_hour_of_the_day",
    "vehicle_claim",
]

CORRELATION_COLUMNS = [
    "months_as_customer",
    "age",
    "policy_annual_premium",
    "number_of_vehicles_involved",
    "witnesses",
    "bodily_injuries",
    "incident_hour_of_the_day",
    "total_claim_amount",
    "injury_claim",
    "property_claim",
    "vehicle_claim",
]

FRAUD_RATE_COLUMNS = [
    "incident_type",
    "incident_severity",
    "authorities_contacted",
    "insured_hobbies",
]

CLAIM_COMPONENT_COLUMNS = ["injury_claim", "property_claim", "vehicle_claim"]

# Reference palette (see dataviz skill references/palette.md) — light mode only.
SURFACE = "#fcfcfb"
PRIMARY_INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

CATEGORICAL = [
    "#2a78d6",  # blue
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
    "#e87ba4",  # magenta
    "#eb6834",  # orange
]
SEQUENTIAL_BLUE = "#2a78d6"
DIVERGING_SCALE = [[0, "#e34948"], [0.5, "#f0efec"], [1, "#2a78d6"]]
STATUS_GOOD = "#0ca30c"
STATUS_CRITICAL = "#d03b3b"

FONT_FAMILY = 'system-ui, -apple-system, "Segoe UI", sans-serif'

_template = go.layout.Template()
_template.layout = go.Layout(
    paper_bgcolor=SURFACE,
    plot_bgcolor=SURFACE,
    font=dict(family=FONT_FAMILY, color=PRIMARY_INK, size=13),
    colorway=CATEGORICAL,
    title=dict(font=dict(size=16, color=PRIMARY_INK)),
    xaxis=dict(
        gridcolor=GRIDLINE,
        linecolor=BASELINE,
        zerolinecolor=BASELINE,
        showgrid=True,
        gridwidth=1,
        tickfont=dict(color=MUTED_INK),
        automargin=True,
    ),
    yaxis=dict(
        gridcolor=GRIDLINE,
        linecolor=BASELINE,
        zerolinecolor=BASELINE,
        showgrid=True,
        gridwidth=1,
        tickfont=dict(color=MUTED_INK),
        automargin=True,
    ),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=SECONDARY_INK)),
    hoverlabel=dict(
        bgcolor=SURFACE,
        bordercolor=BASELINE,
        font=dict(color=PRIMARY_INK, family=FONT_FAMILY),
    ),
    margin=dict(t=60, r=30, b=50, l=60),
    bargap=0.3,
)
pio.templates["claim_eda"] = _template
pio.templates.default = "claim_eda"


def clean_for_analysis(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace("?", np.nan)
    df = df.drop(columns=[c for c in df.columns if c.startswith("_c")], errors="ignore")
    return df


def build_kpi_row(df: pd.DataFrame) -> str:
    total = len(df)
    fraud_count = int((df[TARGET_COLUMN] == "Y").sum())
    fraud_rate = fraud_count / total * 100 if total else 0.0
    missing_cols = int(df.isna().any().sum())

    tiles = [
        ("Total claims", f"{total:,}"),
        ("Fraud reported", f"{fraud_count:,}"),
        ("Fraud rate", f"{fraud_rate:.1f}%"),
        ("Columns with missing values", f"{missing_cols}"),
    ]
    cells = "".join(
        f'<div class="kpi-tile"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div></div>'
        for label, value in tiles
    )
    return f'<div class="kpi-row">{cells}</div>'


def plot_target_distribution(df: pd.DataFrame) -> go.Figure:
    counts = df[TARGET_COLUMN].value_counts().reindex(["N", "Y"]).fillna(0)
    labels = ["Not fraud (N)", "Fraud (Y)"]
    colors = [STATUS_GOOD, STATUS_CRITICAL]
    total = counts.sum()
    percentages = counts / total * 100

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=counts.values,
            marker_color=colors,
            text=[
                f"{c:,.0f} ({p:.1f}%)"
                for c, p in zip(counts.values, percentages.values)
            ],
            textposition="outside",
            hovertemplate="%{x}: %{y:,.0f} claims<extra></extra>",
        )
    )
    fig.update_layout(
        title="Fraud reported — class balance",
        showlegend=False,
        yaxis_title="Claims",
        xaxis=dict(showgrid=False),
    )
    return fig


def plot_missing_values(df: pd.DataFrame) -> go.Figure | None:
    missing_pct = (df.isna().mean() * 100).sort_values(ascending=True)
    missing_pct = missing_pct[missing_pct > 0]
    if missing_pct.empty:
        return None

    fig = go.Figure(
        go.Bar(
            x=missing_pct.values,
            y=missing_pct.index,
            orientation="h",
            marker_color=SEQUENTIAL_BLUE,
            text=[f"{v:.1f}%" for v in missing_pct.values],
            textposition="outside",
            hovertemplate="%{y}: %{x:.1f}%% missing<extra></extra>",
        )
    )
    fig.update_layout(
        title="Missing values by column",
        xaxis_title="Missing (%)",
        yaxis=dict(showgrid=False),
        height=max(300, 40 * len(missing_pct)),
    )
    return fig


def plot_numeric_distributions(df: pd.DataFrame, columns: list[str]) -> go.Figure:
    columns = [c for c in columns if c in df.columns]
    n_cols = 3
    n_rows = -(-len(columns) // n_cols)
    fig = make_subplots(
        rows=n_rows, cols=n_cols, subplot_titles=columns, vertical_spacing=0.5 / n_rows
    )

    groups = {"N": STATUS_GOOD, "Y": STATUS_CRITICAL}
    for i, column in enumerate(columns):
        row, col = divmod(i, n_cols)
        row, col = row + 1, col + 1
        for label, color in groups.items():
            values = pd.to_numeric(
                df.loc[df[TARGET_COLUMN] == label, column], errors="coerce"
            ).dropna()
            fig.add_trace(
                go.Histogram(
                    x=values,
                    name="Fraud (Y)" if label == "Y" else "Not fraud (N)",
                    marker_color=color,
                    opacity=0.65,
                    legendgroup=label,
                    showlegend=(i == 0),
                ),
                row=row,
                col=col,
            )

    fig.update_layout(
        barmode="overlay",
        height=320 * n_rows,
        margin=dict(t=40),
    )
    return fig


def plot_fraud_rate_by_category(
    df: pd.DataFrame, column: str, top_n: int = 10
) -> go.Figure | None:
    if column not in df.columns:
        return None
    working = df[[column, TARGET_COLUMN]].dropna(subset=[column])
    counts = working[column].value_counts().head(top_n)
    rates = (
        working[working[column].isin(counts.index)]
        .assign(is_fraud=lambda d: d[TARGET_COLUMN] == "Y")
        .groupby(column)["is_fraud"]
        .mean()
        .mul(100)
        .sort_values()
    )

    fig = go.Figure(
        go.Bar(
            x=rates.values,
            y=rates.index,
            orientation="h",
            marker_color=SEQUENTIAL_BLUE,
            text=[f"{v:.1f}%" for v in rates.values],
            textposition="outside",
            hovertemplate="%{y}: %{x:.1f}%% fraud rate<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Fraud rate by {column.replace('_', ' ')}",
        xaxis_title="Fraud rate (%)",
        yaxis=dict(showgrid=False),
        height=max(300, 40 * len(rates)),
    )
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, columns: list[str]) -> go.Figure:
    columns = [c for c in columns if c in df.columns]
    numeric_df = df[columns].apply(pd.to_numeric, errors="coerce")
    corr = numeric_df.corr()

    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale=DIVERGING_SCALE,
            zmid=0,
            zmin=-1,
            zmax=1,
            text=corr.values,
            texttemplate="%{text:.2f}",
            textfont=dict(color=PRIMARY_INK, size=10),
            colorbar=dict(title="r", outlinewidth=0),
            hovertemplate="%{x} vs %{y}: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Correlation between numeric features",
        xaxis=dict(showgrid=False, tickangle=45),
        yaxis=dict(showgrid=False, autorange="reversed"),
        height=600,
    )
    return fig


def plot_claim_composition(df: pd.DataFrame) -> go.Figure:
    means = (
        df.groupby(TARGET_COLUMN)[CLAIM_COMPONENT_COLUMNS].mean().reindex(["N", "Y"])
    )
    labels = ["Not fraud (N)", "Fraud (Y)"]

    fig = go.Figure()
    for i, component in enumerate(CLAIM_COMPONENT_COLUMNS):
        fig.add_trace(
            go.Bar(
                name=component.replace("_", " "),
                x=labels,
                y=means[component].values,
                marker_color=CATEGORICAL[i],
                hovertemplate="%{x} — "
                + component.replace("_", " ")
                + ": $%{y:,.0f}<extra></extra>",
            )
        )
    fig.update_layout(
        title="Average claim composition by fraud status",
        barmode="stack",
        yaxis_title="Average claim amount ($)",
        xaxis=dict(showgrid=False),
    )
    return fig


def plot_claim_scatter(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    colors = {"N": STATUS_GOOD, "Y": STATUS_CRITICAL}
    labels = {"N": "Not fraud (N)", "Y": "Fraud (Y)"}
    for status, color in colors.items():
        subset = df[df[TARGET_COLUMN] == status]
        fig.add_trace(
            go.Scatter(
                x=subset["policy_annual_premium"],
                y=subset["total_claim_amount"],
                mode="markers",
                name=labels[status],
                marker=dict(color=color, size=8, opacity=0.7),
                hovertemplate="Premium: $%{x:,.0f}<br>Total claim: $%{y:,.0f}<extra></extra>",
            )
        )
    fig.update_layout(
        title="Total claim amount vs. policy annual premium",
        xaxis_title="Policy annual premium ($)",
        yaxis_title="Total claim amount ($)",
        hovermode="closest",
    )
    return fig


def _section(title: str, description: str, fig: go.Figure, include_js: bool) -> str:
    fig_html = fig.to_html(
        full_html=False,
        include_plotlyjs="cdn" if include_js else False,
        config={"displaylogo": False},
    )
    return (
        f'<section class="report-section">'
        f"<h2>{title}</h2>"
        f'<p class="section-description">{description}</p>'
        f"{fig_html}"
        f"</section>"
    )


def generate_report(df: pd.DataFrame, output_path: Path = REPORT_PATH) -> Path:
    clean_df = clean_for_analysis(df)

    sections: list[str] = []
    sections.append(
        _section(
            "Class balance",
            "How many claims in the dataset are labeled fraudulent.",
            plot_target_distribution(clean_df),
            include_js=True,
        )
    )

    missing_fig = plot_missing_values(clean_df)
    if missing_fig is not None:
        sections.append(
            _section(
                "Missing values",
                "Columns with the '?' placeholder converted to missing, by share of rows affected.",
                missing_fig,
                include_js=False,
            )
        )

    sections.append(
        _section(
            "Numeric distributions",
            "Distribution of key numeric features, split by fraud status.",
            plot_numeric_distributions(clean_df, NUMERIC_DISTRIBUTION_COLUMNS),
            include_js=False,
        )
    )

    for column in FRAUD_RATE_COLUMNS:
        fig = plot_fraud_rate_by_category(clean_df, column)
        if fig is not None:
            sections.append(
                _section(
                    f"Fraud rate by {column.replace('_', ' ')}",
                    "Share of claims labeled fraudulent within each category (top 10 by frequency).",
                    fig,
                    include_js=False,
                )
            )

    sections.append(
        _section(
            "Feature correlation",
            "Pearson correlation between numeric features, including claim amount components.",
            plot_correlation_heatmap(clean_df, CORRELATION_COLUMNS),
            include_js=False,
        )
    )

    sections.append(
        _section(
            "Claim composition",
            "Average injury, property, and vehicle claim amounts by fraud status.",
            plot_claim_composition(clean_df),
            include_js=False,
        )
    )

    sections.append(
        _section(
            "Claim amount vs. premium",
            "Relationship between policy annual premium and total claim amount, colored by fraud status.",
            plot_claim_scatter(clean_df),
            include_js=False,
        )
    )

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Insurance Claim Fraud — Exploratory Data Analysis</title>
<style>
  body {{
    margin: 0;
    padding: 0 24px 48px;
    background: #f9f9f7;
    color: {PRIMARY_INK};
    font-family: {FONT_FAMILY};
  }}
  header {{
    padding: 32px 0 16px;
  }}
  h1 {{
    font-size: 24px;
    margin: 0 0 4px;
  }}
  .subtitle {{
    color: {SECONDARY_INK};
    font-size: 14px;
    margin: 0;
  }}
  .kpi-row {{
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    margin: 24px 0;
  }}
  .kpi-tile {{
    background: {SURFACE};
    border: 1px solid {GRIDLINE};
    border-radius: 8px;
    padding: 16px 20px;
    min-width: 160px;
    flex: 1;
  }}
  .kpi-label {{
    color: {MUTED_INK};
    font-size: 12px;
    margin-bottom: 6px;
  }}
  .kpi-value {{
    font-size: 28px;
    font-weight: 600;
  }}
  .report-section {{
    background: {SURFACE};
    border: 1px solid {GRIDLINE};
    border-radius: 8px;
    padding: 16px 20px 8px;
    margin-bottom: 24px;
  }}
  .report-section h2 {{
    font-size: 16px;
    margin: 0 0 4px;
  }}
  .section-description {{
    color: {SECONDARY_INK};
    font-size: 13px;
    margin: 0 0 8px;
  }}
</style>
</head>
<body>
<header>
  <h1>Insurance Claim Fraud — Exploratory Data Analysis</h1>
  <p class="subtitle">Generated {generated_at} · {len(clean_df):,} claims</p>
</header>
{build_kpi_row(clean_df)}
{"".join(sections)}
</body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def main() -> None:
    df = load_data()
    output_path = generate_report(df)
    print(f"EDA report written to {output_path}")


if __name__ == "__main__":
    main()
