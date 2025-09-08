# BENCHMARK of ML models for regression tasks
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer

from ml_training1 import train_cv_regressor_group_effect
from params import decision_tree_grid, random_forest_grid, xgboosst_grid

params_regressors = {
    "LinearRegression": {},
    "Ridge": {"alpha": [0.05, 0.1, 0.3]},
    "Lasso": {"alpha": [0.01, 0.05]},
    "DecisionTree": decision_tree_grid,
    "RandomForest": random_forest_grid,
    "MLPRegressor": {
        "hidden_layer_sizes": [(64, 32)],
        "max_iter": [200],
        "random_state": [42],
    },
    "XGboost": xgboosst_grid,
}

models_bench = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.01),
    "DecisionTree": DecisionTreeRegressor(random_state=42),
    "RandomForest": RandomForestRegressor(random_state=42),
    "MLPRegressor": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42),
    "XGboost": XGBRegressor(n_estimators=100, max_depth=10, eta=0.1, subsample=0.7, colsample_bytree=0.8),
}


def imputation(df_to_impute: pd.DataFrame) -> pd.DataFrame:
    # impute missing values in the DataFrame for the benchmark so that all algorithms
    # are trained on exactly the same data (although the final version of XGboost will
    # not need it for the SHAP and explanatory part)
    df = df_to_impute.copy()

    for col in [
        "Délai CEC2 receveur (en j) (DelaiCEC2)",
        "Délai ass. ventriculaire (DelaiAV2)",
    ]:
        df[col] = df[col].fillna(-1)


    col = "Décile Peptides (DecilePEPT_IMP)"
    df[col] = df[col].fillna(11)
    
    col = "Décile Peptides avant CEC2 ou DRG2 (1 à 10) (DecilePEPT_AVI)"
    df[col] = df[col].fillna(11)


    for col in [
        "Bilirubine receveur (µmol/L) (BILI2)",
        "Bilirubine avant CEC2 ou DRG2 (BILI_AVI)",
        "Débit de filtration glomérulaire (DFG)",
        "Débit de filtration glomérulaire avant CEC2 ou DRG2 (DFG_AVI)",
    ]:
        df[col] = df[col].fillna(df[col].median())

    return df


def run_benchmark(
    models,
    params_models,
    X_train,
    X_test,
    y_train,
    y_test,
    weights_train,
    weights_test,
    groups_train,
    activate_group_mode,
    gkf,
):
    results = {}

    # # Impute missing values
    X_train_imputed = imputation(X_train.copy())
    X_test_imputed = imputation(X_test.copy())

    for name, model in models.items():
        if activate_group_mode:
            # Training CV on the grid
            results[name] = train_cv_regressor_group_effect(
                X_train_imputed,
                y_train,
                X_test_imputed,
                y_test,
                weights_train,
                weights_test,
                groups_train,
                model,
                params_models[name],
                gkf,
            )
            print(f"training complete (with group effect) for {model}.\n")
        else:
            raise ValueError('"activate_group_mode" must be True to use this function.')

    return results


def show_results_bench(bench_results, y_test):
    # Build a DataFrame with each algorithm as a column
    df_results = pd.DataFrame({k: v[0] for k, v in bench_results.items()})
    df_results["y_test"] = list(y_test)
    print(df_results.head())

    from sklearn.metrics import mean_squared_error, r2_score
    import matplotlib.pyplot as plt

    # Compute metrics for each algorithm
    metrics = {}
    for col in df_results.columns:
        if col == "y_test":
            continue
        mse = mean_squared_error(df_results["y_test"], df_results[col])
        r2 = r2_score(df_results["y_test"], df_results[col])
        metrics[col] = {"MSE": mse, "R2": r2}

    metrics_df = pd.DataFrame(metrics).T

    # Plot histograms for both metrics
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    metrics_df["MSE"].plot(kind="bar", ax=axes[0], title="MSE by Algorithm (lower is better)", color="skyblue")
    axes[0].set_ylabel("MSE")
    metrics_df["R2"].plot(kind="bar", ax=axes[1], title="R² by Algorithm (higher is better)", color="orange")
    axes[1].set_ylabel("R²")
    axes[1].set_ylim(0, 1)  # Set R² axis from 0 to 1
    fig.savefig("../results/metrics_comparison.png", dpi=300, bbox_inches="tight")
