import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from IPython.display import display
import plotly.express as px

DAYS_IN_MONTH = 30.4


def correlation_pearson(df: pd.DataFrame, target: str) -> None:
    """Calculate and plot Pearson correlations of all features with the target variable."""

    # Verify that the target column exists
    if target not in df.columns:
        raise ValueError(f"La target column {target} is not in the DataFrame.")

    # Separate numeric and categorical columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    categorical_columns = df.select_dtypes(include=["object", "category"]).columns

    # Initialize a dictionary to store correlations
    correlations = {}

    # Calculate the correlation for each numeric column with the target 'RAN'
    for col in numeric_columns:
        if col != target:
            correlations[col] = df[target].corr(df[col])

    # Calculate the correlation for categorical columns
    for col in categorical_columns:
        # Encoder les catégories comme des entiers pour la corrélation de Spearman
        encoded_col = df[col].astype("category").cat.codes
        corr, _ = spearmanr(df[target], encoded_col)  # Corrélation de Spearman
        correlations[col] = corr  # Pas d'inversion ni d'amplification

    # Convert results in a DataFrame
    corr_df = pd.DataFrame(
        list(correlations.items()), columns=["Variable", "Correlation"]
    )
    corr_df = corr_df.sort_values(by="Correlation", ascending=True)
    display(corr_df)

    # Display results
    print(corr_df.head())

    # Create the plot
    plt.figure(figsize=(10, 15))
    plt.barh(corr_df["Variable"], corr_df["Correlation"], color="skyblue")
    plt.title(f"Correlation of the variables with the target {target}", fontsize=16)
    plt.xlabel("Correlation Coefficient", fontsize=12)
    plt.ylabel("Variables", fontsize=12)
    plt.gca().invert_yaxis()  # Reverse to have the highest correlations on top
    plt.tight_layout()
    plt.show()


def correlation_spearman(df: pd.DataFrame, target: str) -> None:
    """Calculate and plot Spearman correlations of all features with the target variable."""

    # Verify the presence of the target
    if target not in df.columns:
        raise ValueError(f"The target column {target} does not exist in the DataFrame.")

    # Separate numeric and categorical columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    categorical_columns = df.select_dtypes(include=["object", "category"]).columns

    # Calculate Spearman correlations
    correlations = []

    # For numeric variables
    for col in numeric_columns:
        if col != target:
            spearman_corr, _ = spearmanr(df[target], df[col], nan_policy="omit")
            correlations.append(
                {"Variable": col, "Type": "Numeric", "Spearman": spearman_corr}
            )

    # For categorical variables
    for col in categorical_columns:
        encoded_col = df[col].astype("category").cat.codes
        spearman_corr, _ = spearmanr(df[target], encoded_col, nan_policy="omit")
        correlations.append(
            {"Variable": col, "Type": "Categorical", "Spearman": spearman_corr}
        )

    # Convert to DataFrame
    correlation_df = pd.DataFrame(correlations).sort_values(
        by="Spearman", ascending=True
    )

    # Display results
    print("\nCombined Spearman Correlations:\n", correlation_df)

    # Visualization of correlations (numeric and categorical variables)
    plt.figure(figsize=(12, 12))
    plt.barh(correlation_df["Variable"], correlation_df["Spearman"], color="skyblue")
    plt.title("Spearman Correlations of Variables with Rank", fontsize=16)
    plt.xlabel("Spearman Correlation", fontsize=12)
    plt.ylabel("Variables", fontsize=12)
    plt.gca().invert_yaxis()  # Reverse to have the highest correlations on top
    plt.tight_layout()
    plt.savefig("../results/eda_correlations_spearman.png")
    plt.show()


def plot_delai(delai) -> None:
    """Plot the distribution of the time elapsed between the first and last proposal for a patient."""
    fig = px.histogram(
        delai["delai"] / DAYS_IN_MONTH + 1,
        nbins=80,
        title="Distribution: time elapsed between the first and last proposal for a patient"
        "<br>= time spent in the classification system",
        labels={"delai": "Time (in months)"},
        range_x=[0.5, 18],
        opacity=0.7,
        color_discrete_sequence=["indianred"],
    )

    fig.update_layout(
        xaxis_title_text='Time (in months)<br>Example : "2" months means "delay of 1 to 2 months"',
        yaxis_title_text="Frequency",
        bargap=0.2,
        xaxis=dict(tickmode="linear", tick0=0, dtick=1),
        height=500,
        width=1000,
    )

    fig.update_traces(
        marker=dict(line=dict(width=1, color="DarkSlateGrey")),
        selector=dict(type="histogram"),
    )

    fig.show()


def plot_rank(
    df: pd.DataFrame, target: str, x_range, xdtick, color: str, nbins: int
) -> None:
    """Plot the distribution of the rank assigned to a candidate."""
    fig = px.histogram(
        df[target],
        nbins=nbins,
        title="Distribution: the rank assigned to a candidate",
        labels={"rang"},
        range_x=x_range,
        opacity=0.7,
        color_discrete_sequence=[color],
    )

    fig.update_layout(
        xaxis_title_text="Rang",
        yaxis_title_text="Frequency",
        bargap=0.2,
        xaxis=dict(tickmode="linear", tick0=0, dtick=xdtick),
        width=1200,
        height=600,
    )

    fig.update_traces(
        marker=dict(line=dict(width=1, color="DarkSlateGrey")),
        selector=dict(type="histogram"),
    )

    fig.show()
