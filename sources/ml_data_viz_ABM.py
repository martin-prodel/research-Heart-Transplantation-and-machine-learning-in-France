import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import learning_curve
from sklearn.inspection import PartialDependenceDisplay
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from scipy.stats import spearmanr
import shap
from lime.lime_tabular import LimeTabularExplainer
from alibi.explainers import ALE
from tqdm import tqdm
import math
from matplotlib.colors import LinearSegmentedColormap

FIG_SIZE = (8, 4)


def train_vs_test_error_curve(
    X_train, y_train, X_test, y_test, weights_train, model_type: str
):
    train_errors, test_errors = [], []

    # Test model complexity with increasing max_depth
    range_max_depth = [2, 3, 5, 8, 10, 15, 20, 25]
    for max_depth in range_max_depth:
        if model_type == "dt":
            model = DecisionTreeRegressor(
                max_depth=max_depth, random_state=42, min_samples_leaf=50
            )
        elif model_type == "rf":
            model = RandomForestRegressor(
                max_depth=max_depth, random_state=42, n_estimators=30
            )
        elif model_type == "xgb":
            model = XGBRegressor(
                n_estimators=1000,
                max_depth=max_depth,  # for each tree
                eta=0.1,  # learning rate
                subsample=0.7,
                colsample_bytree=0.8,  # subsampling of columns
            )
        else:
            raise NotImplementedError(f"Model type {model_type} not implemented.")
        model.fit(X_train, y_train, sample_weight=weights_train)
        train_errors.append(mean_squared_error(y_train, model.predict(X_train)))
        test_errors.append(mean_squared_error(y_test, model.predict(X_test)))

    # Plot the error curves
    plt.figure(figsize=FIG_SIZE)
    plt.plot(range_max_depth, train_errors, label="Training Error", marker="o")
    plt.plot(range_max_depth, test_errors, label="Testing Error", marker="o")
    plt.title("Training vs Testing Error")
    plt.xlabel("Max Depth")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.grid()
    maxy_y_val = max(train_errors + test_errors) + 0.01
    plt.ylim(0, maxy_y_val)  # Ensure y-axis starts at 0
    plt.show()


def residual_plot(y_test, y_pred):
    residuals = y_test - y_pred

    plt.figure(figsize=FIG_SIZE)
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(y=0, color="r", linestyle="--")
    plt.title("Residual Plot")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.grid()
    plt.show()

    return residuals


def histo_residuals(residuals):
    plt.figure(figsize=FIG_SIZE)
    sns.histplot(residuals, kde=True, color="purple", bins=30)
    plt.title("Residuals Distribution")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.axvline(x=0, color="red", linestyle="--")
    plt.grid()
    plt.show()


def feature_importance(best_model, X):
    importances = best_model.feature_importances_
    features = X.columns

    # Create a DataFrame to sort features by importance
    feature_importance_df = pd.DataFrame(
        {"Feature": features, "Importance": importances}
    )
    feature_importance_df = feature_importance_df.sort_values(
        by="Importance", ascending=False
    )

    # Extract sorted features and importances
    sorted_features = feature_importance_df["Feature"]
    sorted_importances = feature_importance_df["Importance"]

    plt.figure(figsize=(10, 16))  # width, height
    plt.barh(sorted_features, sorted_importances, color="skyblue")
    plt.title("Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.gca().invert_yaxis()
    plt.grid()
    plt.savefig("")
    plt.show()


def predicted_vs_actual(y_test, y_pred, activate_zoom: bool = False):
    plt.figure(figsize=FIG_SIZE)
    sns.scatterplot(x=y_test, y=y_pred, color="blue", alpha=0.6)
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        color="red",
        lw=2,
        linestyle="--",
    )
    plt.title("Predicted vs Actual Values")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.grid()
    if activate_zoom:
        plt.xlim(0.8, 1)
        plt.ylim(0.8, 1)

    plt.show()


def learning_cure(best_model, X, y, groups=None):
    train_sizes, train_scores, test_scores = learning_curve(
        best_model,
        X,
        y,
        groups=groups,
        cv=5,
        scoring="neg_mean_squared_error",
        train_sizes=np.linspace(0.1, 1.0, 15),
    )

    train_scores_mean = -np.mean(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)

    plt.figure(figsize=FIG_SIZE)
    plt.plot(train_sizes, train_scores_mean, label="Training Error", marker="o")
    plt.plot(train_sizes, test_scores_mean, label="Testing Error (CV)", marker="o")
    plt.title("Learning Curve")
    plt.xlabel("Training Set Size")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.grid()
    plt.ylim(0, None)  # Ensure y-axis starts at 0
    plt.show()


def partial_dependence_plot(best_model, X):
    # Since we have 57 features, we do 2 plots (each of 30 features)
    for my_range in tqdm(
        [
            range(15),
            range(15, 30),
            range(30, 45),
            range(45, len(X.columns)),
        ]
    ):
        fig, ax = plt.subplots(figsize=(14, 18))  # width, height
        PartialDependenceDisplay.from_estimator(
            best_model,
            X,
            features=my_range,
            grid_resolution=100,
            n_cols=3,
            ax=ax,
        )


################################################################
# ADDITIONAL DATA VIZ AND EXPLAINABILITY OF ML MODELS
################################################################


# SHAP Interaction Values
def compute_shap_values(model, X, max_nb_feature_show: int = 20):
    """
    Computes SHAP values to explain the contributions of each feature to the predictions.

    Advantages:
    - Provides global and local interpretability.
    - Works well with tree-based models.

    Limitations:
    - Computationally expensive for large datasets.

    Args:
        model: Trained tree-based model (e.g., Decision Tree, Random Forest).
        X: DataFrame of features.
        max_nb_feature_show: nb of features to display on the SHAP graph

    Returns:
        SHAP values for the dataset.
    """
    # Create a TreeExplainer
    explainer = shap.TreeExplainer(model)
    # Compute SHAP values
    shap_values = explainer.shap_values(X)
    # Compute explaination
    explaination = explainer(X)

    # Visualize SHAP values - Summary plot for feature importance
    shap.summary_plot(shap_values, X, max_display=max_nb_feature_show)

    return explainer, shap_values, explaination


def compute_shap_interaction_values(X_test, explainer=None, model=None):
    """Computes SHAP interaction values to analyze feature interactions.

    Advantages:
    - Explains how pairs of features interact to influence predictions.
    - Highlights second-order feature effects.

    Limitations:
    - Computational cost increases with feature size.

    Args:
        model: Trained tree-based model.
        X_test: DataFrame of features.

    Returns: SHAP interaction values for the dataset.
    """
    # Create a TreeExplainer
    if explainer is None:
        if model is None:
            raise ValueError(
                "You must provide a model if you do not provide an explainer!"
            )
        explainer = shap.TreeExplainer(model)

    interaction_values = explainer.shap_interaction_values(X_test)
    shap.summary_plot(interaction_values, X_test)
    return interaction_values


# Partial Dependence Plots
def partial_dependence(model, X_test, two_features):
    """Plots partial dependence to show the average relationship between a feature and the target.

    Advantages:
    - Visualizes trends for single or paired features.
    - Provides actionable insights for feature impact.

    Limitations:
    - Assumes features are independent, which may not hold for correlated features.

    Args:
        model: Trained model.
        X_test: DataFrame of features.
        two_features: List of 2 features to analyze.
    """
    raise NotImplementedError("Not implemented ...")
    # plot_partial_dependence(model, X_test, features=two_features, grid_resolution=50)
    # plt.show()


# Accumulated Local Effects (ALE)
def accumulated_local_effects(model, X_test):
    """Generates ALE plots to analyze feature impacts, robust to correlated features.

    Advantages:
    - Handles correlated features better than partial dependence plots.
    - Focuses on local rather than global effects.

    Limitations:
    - Less intuitive than SHAP or PDPs.

    Args:
        model: Trained model with a predict method.
        X_test: DataFrame of features.
    """

    ale_explainer = ALE(model.predict, feature_names=X_test.columns)
    ale_exp = ale_explainer.explain(X_test.to_numpy())
    ale_exp.plot()


# Decision Path Analysis
def decision_path_analysis(model, feature_names):
    """Extracts decision paths for individual predictions in tree-based models.

    Advantages:
    - Provides rule-based explanations for individual predictions.
    - Easy to interpret for small trees.

    Limitations:
    - Complex for large trees or random forests.

    Args:
        model: Trained Decision Tree model.
        feature_names: List of feature names.
    """
    tree_rules = export_text(model, feature_names=feature_names)
    print(tree_rules)


# 5 Trend and Threshold Detection
def trend_threshold_detection(data, feature, target, bins=5):
    """Detects trends and thresholds by binning continuous features.

    Advantages:
    - Identifies meaningful thresholds in continuous features.
    - Highlights non-linear relationships.
    - Decision Trees: Tree-based models naturally split features into thresholds.
    Analyze the splits to detect thresholds influencing the target.

    Limitations:
    - Sensitive to the choice of bin size.

    Args:
        data: DataFrame containing the feature and target.
        feature: Feature to analyze.
        target: Target variable.
        bins: Number of bins for grouping.
    """

    data = data.copy()

    data["Binned_Feature"] = pd.cut(data[feature], bins=bins)
    print("Q3 value is dispay for each x label:")
    grouped = data.groupby("Binned_Feature")[target].quantile(0.75)
    grouped.plot(kind="bar")
    plt.title(f"Trend and Threshold Detection for {feature}")
    plt.xlabel("Binned Feature")
    plt.ylabel(target)
    plt.show()


# 7. Feature Combination Analysis
def feature_combinations(X, feature_1: str, feature_2: str):
    """Creates interaction terms and ratios between two features.

    Advantages:
    - Explores combined effects of feature pairs.
    - Useful for identifying non-linear relationships.

    Limitations:
    - May introduce irrelevant features.

    Args:
        X: DataFrame of features.
        feature_1: First feature.
        feature_2: Second feature.

    Returns: DataFrame with interaction and ratio columns.
    """
    X["Interaction"] = X[feature_1] * X[feature_2]
    X["Ratio"] = X[feature_1] / (X[feature_2] + 1e-9)
    return X[["Interaction", "Ratio"]]


# 8. LIME (Local Interpretable Model-Agnostic Explanations) Explanations
# LIME is an alternative to SHAP for explaining individual predictions by approximating
# the model locally with a simpler interpretable model.
def lime_explanation(model, X_train, X_test, feature_names: list):
    """Generates local explanations for individual predictions using LIME.

    Advantages:
    - Model-agnostic and locally interpretable.
    - Explains individual predictions.

    Limitations:
    - Requires retraining surrogate models for each instance.

    Args:
        model: Trained model.
        X_train: Training data.
        X_test: Test data.
        feature_names: List of feature names.
    """
    explainer = LimeTabularExplainer(
        X_train.values, feature_names=feature_names, mode="regression"
    )
    lime_exp = explainer.explain_instance(X_test.iloc[0].values, model.predict)
    lime_exp.show_in_notebook()


# 9. Target Encoding and Conditional Feature Analysis
# Analyze how target values vary across feature categories or bins:
def target_encoding(data, one_feature: str, target: str):
    """Performs target encoding by averaging the target per feature category.

    Advantages:
    - Highlights feature-target relationships.

    Limitations:
    - Only applicable for categorical features.

    Args:
        data: DataFrame containing the feature and target.
        one_feature: Categorical feature to analyze.
        target: Target variable.
    """
    grouped = data.groupby(one_feature)[target].mean().sort_values()
    grouped.plot(kind="bar")
    plt.title(f"Target Encoding for {one_feature}")
    plt.ylabel(target)
    plt.show()


# Conditional Analysis: Evaluate how the target changes under specific feature conditions
# (e.g., thresholds in Feature_A while Feature_B is in a certain range).
def conditional_analysis(data, one_feature: str, target: str, threshold):
    """Analyzes the target variable conditionally on a feature threshold.

    Advantages:
    - Highlights target behavior under specific feature conditions.

    Limitations:
    - Relies on manually chosen thresholds.

    Args:
        data: DataFrame containing the feature and target.
        one_feature: Feature to analyze.
        target: Target variable.
        threshold: Threshold value for the feature.
    """
    filtered_data = data[data[one_feature] > threshold]
    filtered_mean = filtered_data[target].mean()
    print(f"Mean of {target} when {one_feature} > {threshold}: {filtered_mean}")


# 10. Surrogate Models
# Train an interpretable surrogate model (e.g., Decision Tree or Linear Regression) on the
# predictions of the original model to understand feature contributions.
def surrogate_model_analysis(model, X_test):
    """Trains a surrogate model to explain predictions from a complex model (on the original test data).

    Advantages:
    - Provides interpretable rule-based explanations.

    Limitations:
    - Explanations may not fully represent the original model.

    Args:
        model: Trained complex model.
        X: Feature data.
        y: Target variable.
    """
    # Here, you can choose another surrogate
    surrogate_model = DecisionTreeRegressor(max_depth=3, random_state=42)
    surrogate_model.fit(X_test, model.predict(X_test))
    tree_rules = export_text(surrogate_model, feature_names=list(X_test.columns))
    print(tree_rules)


# 11. Correlation and Statistical Analysis
def correlation_analysis(X, y_vector):
    """Computes feature-target correlations using Spearman's rank correlation.

    Advantages:
    - Quantifies monotonic relationships.
    - Handles non-linear associations.

    Limitations:
    - Does not capture interactions or causality.

    Args:
        X: DataFrame of features.
        y_vector: Target column.

    Returns: DataFrame of feature correlations.
    """
    correlations = {col: spearmanr(X[col], y_vector)[0] for col in X.columns}
    correlation_df = pd.DataFrame(
        correlations.items(), columns=["Feature", "Correlation"]
    )
    correlation_df = correlation_df.sort_values(by="Correlation", ascending=False)
    print(correlation_df)
    return correlation_df


# 12. Recursive Feature Elimination (RFE)
# Use RFE to iteratively rank features by their importance for the model.
def recursive_feature_elimination(model, X, y, n_features_to_select):
    """Performs recursive feature elimination to rank features by importance.

    Advantages:
    - Identifies a minimal set of important features.

    Limitations:
    - Computationally expensive for large datasets.

    Args:
        model: Base model for feature selection.
        X: Feature data.
        y: Target variable.
        n_features_to_select: Number of features to select.

    Returns: List of selected features.
    """
    rfe = RFE(estimator=model, n_features_to_select=n_features_to_select)
    rfe.fit(X, y)
    selected_features = X.columns[rfe.support_]
    print(f"Selected Features: {list(selected_features)}")
    return selected_features


# More graphs
def draw_single_shap_plot(
    one_feature: str,
    interaction_index: str,
    shap_values,
    X_test: pd.DataFrame,
    xlim: tuple,
):
    figsize = (10, 5)  # Figure size (width, height)
    fig = plt.figure(figsize=figsize)
    ax = fig.gca()

    # For categorical variables, we can add a bit of jitter to avoid overlapping points, making it more readable
    # 2nd adjustment: the ticks on the x-axis
    if X_test[one_feature].nunique() < 15:
        x_jitter = 0.3
        # Display all unique values as xticks (after dropping NaN)
        unique_vals = sorted(X_test[one_feature].dropna().unique())
        if unique_vals == [0, 1]:
            ax.set_xticks(unique_vals, labels=["No", "Yes"])
        else:
            ax.set_xticks(unique_vals)
    else:
        x_jitter = 0.1
        # Show 10 evenly spaced xticks between min and max (excluding NaN)
        # feature_no_nan = X_test[one_feature].dropna()
        # xticks = np.linspace(feature_no_nan.min(), feature_no_nan.max(), 10)
        # ax.set_xticks(xticks)

    # If coloring the plot with a continuous variable, it's better to use an appropriate color palette
    if interaction_index not in X_test:
        my_cmap = None

    elif X_test[interaction_index].nunique() > 15:
        # colors = ["#B24745FF", "#00A1D5FF"]  # first color is black, last is red
        colors = ["#00A1D5FF", "#B24745FF"]  # first color is black, last is red
        my_cmap = LinearSegmentedColormap.from_list("Custom", colors, N=100)
        # my_cmap = plt.get_cmap("bwr")
    elif X_test[interaction_index].nunique() == 2:  # binary
        my_cmap = plt.get_cmap("viridis")
    else:
        my_cmap = plt.get_cmap("viridis").reversed()

    # After setting up ax and before shap.dependence_plot
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)

    # my_cmap = None
    shap.dependence_plot(
        one_feature,
        shap_values,
        X_test,
        # plot_size=(18, 9),
        cmap=my_cmap,  # Spectral # PiYG
        interaction_index=interaction_index,
        dot_size=16,
        alpha=0.9,
        x_jitter=x_jitter,
        title="SHAP Dependence Plot",
        ax=ax,
        show=False,  # Do not show the plot yet, we will adjust it
    )
    # Set y-axis limit value (to have the same accross features)
    # Get y-data from scatter plot (PathCollection)
    y_data = []
    for col in ax.collections:
        offsets = col.get_offsets()
        if offsets.size > 0:
            y_data.extend(offsets[:, 1])
    y_data = np.array(y_data)
    ymin = y_data.min()
    ymax = y_data.max()
    # Set y-limits: ymin -0.4 only if all points are above -0.4, ymax 0.4 only if all points are below 0.4
    y_upper_bound = 0.4
    y_lower_bound = -y_upper_bound
    y_axis_min = (
        y_lower_bound if math.isnan(ymin) or ymin >= y_lower_bound else ymin * 1.05
    )
    y_axis_max = (
        y_upper_bound if math.isnan(ymax) or ymax <= y_upper_bound else ymax * 1.05
    )
    ax.set_ylim(y_axis_min, y_axis_max)
    if len(xlim) == 2:
        ax.set_xlim(xlim[0], xlim[1])

    plt.show()


# New graph
def draw_single_shap_plot_uni(
    one_feature: str, shap_values, X_test: pd.DataFrame, xlim: tuple, ylim=(-0.4, 0.4)
):
    figsize = (10, 5)  # Figure size (width, height)
    fig = plt.figure(figsize=figsize)
    ax = fig.gca()

    # For the categorical variables, we can add a bit of jitter to avoid overlapping points, making it more readable
    # 2nd adjustment: the ticks on the x-axis
    if X_test[one_feature].nunique() < 15:
        x_jitter = 0.3
        # Display all unique values as xticks (after dropping NaN)
        unique_vals = sorted(X_test[one_feature].dropna().unique())
        if unique_vals == [0, 1]:
            ax.set_xticks(unique_vals, labels=["No", "Yes"])
        else:
            ax.set_xticks(unique_vals)
    else:
        x_jitter = 0.1
        # Show 10 evenly spaced xticks between min and max (excluding NaN)
        # feature_no_nan = X_test[one_feature].dropna()
        # xticks = np.linspace(feature_no_nan.min(), feature_no_nan.max(), 10)
        # ax.set_xticks(xticks)

    # After setting up ax and before shap.dependence_plot
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)

    # my_cmap = None
    shap.dependence_plot(
        one_feature,
        shap_values,
        X_test,
        # plot_size=(18, 9),
        interaction_index=None,
        dot_size=16,
        alpha=0.9,
        x_jitter=x_jitter,
        title="SHAP Dependence Plot",
        ax=ax,
        show=False,  # Do not show the plot yet, we will adjust it
    )
    ax.set_ylim(ylim[0], ylim[1])
    if len(xlim) == 2:
        ax.set_xlim(xlim[0], xlim[1])

    plt.show()
