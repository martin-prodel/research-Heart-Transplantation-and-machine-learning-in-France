from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import numpy as np


def train_cv_regressor(
    X_train,
    y_train,
    X_test,
    y_test,
    w_train,
    w_test,
    regressor,
    param_grid,
    group_k_fold=None,
    groups=None,
):
    if group_k_fold is None:
        print(
            f"⚠️ Warning, no group effect is used in the cross-validation. You are exposed to data leakage.\n{'-' * 90}"
        )
        cv = 5  # cross validation with 5 k-folds
    else:
        cv = group_k_fold  # special CV with groups effect (repeated measures)

    # Validation croisée et recherche des hyperparamètres
    grid_search = GridSearchCV(
        estimator=regressor,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=cv,  # validation
        verbose=1,
        n_jobs=-1,  # Utiliser tous les cœurs disponibles
    )
    if group_k_fold is None:
        grid_search.fit(X_train, y_train, sample_weight=w_train)
    else:
        if groups is None:
            raise ValueError(
                "'groups' must be defined to use your 'group_k_fold'. Please specify it."
            )
        # Ajuster GridSearchCV
        grid_search.fit(X_train, y_train, sample_weight=w_train, groups=groups)

    # Meilleur modèle
    best_model = grid_search.best_estimator_

    # Évaluation sur l'ensemble de test
    y_pred = best_model.predict(X_test)

    # clip the y value (between 0 & 1)
    y_pred = y_pred.clip(min=0, max=1)

    mse = mean_squared_error(y_test, y_pred, sample_weight=w_test)
    r2 = r2_score(y_test, y_pred)

    # Affichage des résultats
    print(f"Best hyperparameters :\t\t{grid_search.best_params_}")
    print(f"Mean Squared Error (MSE) :\t{mse:.3f}")
    print(f"Coefficient of Determination (R²) :\t{r2:.2f}")

    # Vérification des performances avec validation croisée
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring="r2")
    print(
        f"Scores R² in crossed validation :\t{[round(score, 2) for score in cv_scores]}"
    )
    print(f"Mean R² score in crossed validation :\t{np.mean(cv_scores):.2f}")
    print("---")

    return y_pred, best_model


def train_cv_regressor_group_effect(
    X_train,
    y_train,
    X_test,
    y_test,
    weights_train,
    weights_test,
    groups_train,
    regressor,
    param_grid,
    group_k_fold,
):
    # Crossed validation and hyperparameter search
    grid_search = GridSearchCV(
        estimator=regressor,
        param_grid=param_grid,
        # choose error : https://scikit-learn.org/1.6/modules/model_evaluation.html#string-name-scorers
        scoring="neg_mean_squared_error",
        cv=group_k_fold.split(X_train, y_train, groups=groups_train),
        verbose=1,
        n_jobs=-1,  # Use all available cores
    )

    # case of MLPRegressor : no weight can be used
    if str(regressor).split("(")[0] == "MLPRegressor":
        grid_search.fit(X_train, y_train)

    # Default situation for all other algorithms
    else:
        grid_search.fit(X_train, y_train, sample_weight=weights_train)

    # Best model
    best_model = grid_search.best_estimator_

    # # Evaluation on the test set
    y_pred = best_model.predict(X_test)
    # clip the y value (between 0 & 1)
    y_pred = y_pred.clip(min=0, max=1)

    mse = mean_squared_error(y_test, y_pred, sample_weight=weights_test)
    r2 = r2_score(y_test, y_pred)

    param_converted_str = (
        str(grid_search.best_params_).replace("{", "").replace("}", "")
    )

    # Display results
    print(f"\tBest hyperparameters (train set) :\t{param_converted_str}")
    print(
        f"\tSCORE 'neg_mean_squared_error' (train set) :\t{grid_search.best_score_:.2f}"
    )
    print(f"\tMean Squared Error (MSE) (test set) :\t{mse:.3f}")
    print(f"\tDetermination Coefficient (R²) (test set) :\t{r2:.2f}")

    return y_pred, best_model
