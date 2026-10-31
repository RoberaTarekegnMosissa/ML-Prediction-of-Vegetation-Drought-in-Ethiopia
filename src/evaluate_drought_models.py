from pathlib import Path

import json

import os



import joblib

import matplotlib

matplotlib.use("Agg")



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import xarray as xr



from sklearn.inspection import permutation_importance

from sklearn.metrics import (

    accuracy_score,

    average_precision_score,

    balanced_accuracy_score,

    confusion_matrix,

    f1_score,

    precision_recall_curve,

    precision_score,

    recall_score,

    roc_auc_score,

    roc_curve,

)

from sklearn.model_selection import train_test_split





PROJECT_DIR = Path(__file__).resolve().parents[1]



DATA_FILE = (

    PROJECT_DIR

    / "data"

    / "processed"

    / "ethiopia_drought_ml_feature_cube_2001_2024.nc"

)



METADATA_FILE = PROJECT_DIR / "results" / "model_metadata.json"

MODEL_DIR = PROJECT_DIR / "results" / "models"

RESULT_DIR = PROJECT_DIR / "results"

FIGURE_DIR = PROJECT_DIR / "figures"



FIGURE_DIR.mkdir(exist_ok=True)



RANDOM_SEED = 42

IMPORTANCE_SAMPLES = 30_000

N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))



MODEL_NAMES = {

    "logistic_regression": "Logistic Regression",

    "random_forest": "Random Forest",

    "xgboost": "XGBoost",

}



COLORS = {

    "logistic_regression": "#377eb8",

    "random_forest": "#4daf4a",

    "xgboost": "#984ea3",

}





with open(METADATA_FILE, encoding="utf-8") as metadata_file:

    metadata = json.load(metadata_file)



features = metadata["features"]

thresholds = metadata["selected_thresholds"]



print(f"Loading {len(features)} predictors...")



dataset = xr.open_dataset(

    DATA_FILE

)[features + ["drought", "valid_sample", "data_split"]].load()





test_mask = (

    (dataset["valid_sample"] == 1)

    & (dataset["data_split"] == 2)

)



flat_indices = np.flatnonzero(test_mask.values.ravel())



y_test = (

    dataset["drought"]

    .values

    .ravel()[flat_indices]

    .astype(np.int8)

)



X_test = np.column_stack([

    dataset[feature].values.ravel()[flat_indices]

    for feature in features

]).astype(np.float32)



print(f"Test samples: {len(y_test):,}")

print(f"Drought prevalence: {100 * y_test.mean():.2f}%")





models = {}

probabilities = {}

predictions = {}



for model_key, display_name in MODEL_NAMES.items():

    model_path = MODEL_DIR / f"{model_key}.joblib"



    if not model_path.exists():

        raise FileNotFoundError(f"Missing model: {model_path}")



    models[model_key] = joblib.load(model_path)



    probabilities[model_key] = models[model_key].predict_proba(

        X_test

    )[:, 1]



    predictions[model_key] = (

        probabilities[model_key] >= thresholds[model_key]

    ).astype(np.int8)



    print(f"Loaded: {display_name}")





# Persistence: drought status during the previous month.

ndvi_lag1_index = features.index("ndvi_z_lag1")



persistence_prediction = (

    X_test[:, ndvi_lag1_index] < -1.0

).astype(np.int8)



predictions["persistence"] = persistence_prediction





# -------------------------------------------------------------------

# Final comparison table

# -------------------------------------------------------------------



def calculate_metrics(name, scores, predicted):

    return {

        "method": name,

        "samples": len(y_test),

        "accuracy": accuracy_score(y_test, predicted),

        "balanced_accuracy": balanced_accuracy_score(

            y_test,

            predicted,

        ),

        "precision": precision_score(

            y_test,

            predicted,

            zero_division=0,

        ),

        "recall": recall_score(

            y_test,

            predicted,

            zero_division=0,

        ),

        "f1_score": f1_score(

            y_test,

            predicted,

            zero_division=0,

        ),

        "roc_auc": roc_auc_score(y_test, scores),

        "average_precision": average_precision_score(

            y_test,

            scores,

        ),

    }





metric_rows = []



always_no_drought = np.zeros_like(y_test)



metric_rows.append(

    calculate_metrics(

        "Always no drought",

        always_no_drought,

        always_no_drought,

    )

)



metric_rows.append(

    calculate_metrics(

        "Previous-month persistence",

        persistence_prediction,

        persistence_prediction,

    )

)



for model_key, display_name in MODEL_NAMES.items():

    metric_rows.append(

        calculate_metrics(

            display_name,

            probabilities[model_key],

            predictions[model_key],

        )

    )



final_metrics = pd.DataFrame(metric_rows)



final_metrics.to_csv(

    RESULT_DIR / "final_test_metrics.csv",

    index=False,

)





# -------------------------------------------------------------------

# Precision-recall curves

# -------------------------------------------------------------------



fig, axis = plt.subplots(figsize=(8, 6))



prevalence = float(y_test.mean())



axis.axhline(

    prevalence,

    color="black",

    linestyle="--",

    linewidth=1,

    label=f"No-skill prevalence ({prevalence:.3f})",

)



for model_key, display_name in MODEL_NAMES.items():

    precision, recall, _ = precision_recall_curve(

        y_test,

        probabilities[model_key],

    )



    average_precision = average_precision_score(

        y_test,

        probabilities[model_key],

    )



    axis.plot(

        recall,

        precision,

        color=COLORS[model_key],

        linewidth=2,

        label=f"{display_name} (AP={average_precision:.3f})",

    )



persistence_precision = precision_score(

    y_test,

    persistence_prediction,

)



persistence_recall = recall_score(

    y_test,

    persistence_prediction,

)



axis.scatter(

    persistence_recall,

    persistence_precision,

    color="darkorange",

    marker="X",

    s=110,

    label="Persistence",

    zorder=5,

)



axis.set_xlabel("Recall")

axis.set_ylabel("Precision")

axis.set_title("Test precision–recall performance, 2021–2024")

axis.set_xlim(0, 1)

axis.set_ylim(0, 1)

axis.grid(alpha=0.25)

axis.legend(loc="upper right")



fig.tight_layout()

fig.savefig(

    FIGURE_DIR / "model_precision_recall_curves.png",

    dpi=300,

    bbox_inches="tight",

)

plt.close(fig)





# -------------------------------------------------------------------

# ROC curves

# -------------------------------------------------------------------



fig, axis = plt.subplots(figsize=(8, 6))



axis.plot(

    [0, 1],

    [0, 1],

    color="black",

    linestyle="--",

    linewidth=1,

    label="No skill",

)



for model_key, display_name in MODEL_NAMES.items():

    false_positive_rate, true_positive_rate, _ = roc_curve(

        y_test,

        probabilities[model_key],

    )



    auc_value = roc_auc_score(

        y_test,

        probabilities[model_key],

    )



    axis.plot(

        false_positive_rate,

        true_positive_rate,

        color=COLORS[model_key],

        linewidth=2,

        label=f"{display_name} (AUC={auc_value:.3f})",

    )



persistence_cm = confusion_matrix(

    y_test,

    persistence_prediction,

    labels=[0, 1],

)



persistence_fpr = (

    persistence_cm[0, 1]

    / persistence_cm[0].sum()

)



persistence_tpr = (

    persistence_cm[1, 1]

    / persistence_cm[1].sum()

)



axis.scatter(

    persistence_fpr,

    persistence_tpr,

    color="darkorange",

    marker="X",

    s=110,

    label="Persistence",

    zorder=5,

)



axis.set_xlabel("False-positive rate")

axis.set_ylabel("True-positive rate")

axis.set_title("Test ROC performance, 2021–2024")

axis.set_xlim(0, 1)

axis.set_ylim(0, 1)

axis.grid(alpha=0.25)

axis.legend(loc="lower right")



fig.tight_layout()

fig.savefig(

    FIGURE_DIR / "model_roc_curves.png",

    dpi=300,

    bbox_inches="tight",

)

plt.close(fig)





# -------------------------------------------------------------------

# Confusion matrices

# -------------------------------------------------------------------



confusion_methods = {

    "Persistence": persistence_prediction,

    "Logistic Regression": predictions["logistic_regression"],

    "Random Forest": predictions["random_forest"],

    "XGBoost": predictions["xgboost"],

}



fig, axes = plt.subplots(

    nrows=2,

    ncols=2,

    figsize=(11, 9),

)



for axis, (method_name, predicted) in zip(

    axes.ravel(),

    confusion_methods.items(),

):

    matrix = confusion_matrix(

        y_test,

        predicted,

        labels=[0, 1],

    )



    normalized = matrix / matrix.sum(

        axis=1,

        keepdims=True,

    )



    image = axis.imshow(

        normalized,

        cmap="Blues",

        vmin=0,

        vmax=1,

    )



    for row in range(2):

        for column in range(2):

            text_color = (

                "white"

                if normalized[row, column] > 0.5

                else "black"

            )



            axis.text(

                column,

                row,

                f"{matrix[row, column]:,}\n"

                f"{normalized[row, column]:.1%}",

                ha="center",

                va="center",

                color=text_color,

                fontsize=11,

            )



    axis.set_title(method_name)

    axis.set_xlabel("Predicted class")

    axis.set_ylabel("Observed class")

    axis.set_xticks([0, 1], ["No drought", "Drought"])

    axis.set_yticks([0, 1], ["No drought", "Drought"])



fig.suptitle(

    "Test confusion matrices, 2021–2024",

    fontsize=15,

)



fig.colorbar(

    image,

    ax=axes.ravel().tolist(),

    label="Fraction of observed class",

    shrink=0.8,

)



fig.subplots_adjust(

    top=0.90,

    right=0.88,

    wspace=0.28,

    hspace=0.30,

)



fig.savefig(

    FIGURE_DIR / "test_confusion_matrices.png",

    dpi=300,

    bbox_inches="tight",

)

plt.close(fig)





# -------------------------------------------------------------------

# Random Forest permutation importance

# -------------------------------------------------------------------



positions = np.arange(len(y_test))



if len(positions) > IMPORTANCE_SAMPLES:

    selected_positions, _ = train_test_split(

        positions,

        train_size=IMPORTANCE_SAMPLES,

        stratify=y_test,

        random_state=RANDOM_SEED,

    )

else:

    selected_positions = positions



X_importance = X_test[selected_positions]

y_importance = y_test[selected_positions]



random_forest = models["random_forest"]



# Avoid nested parallel processing.

random_forest.n_jobs = 1



print(

    f"Calculating permutation importance with "

    f"{len(y_importance):,} test samples..."

)



importance_result = permutation_importance(

    random_forest,

    X_importance,

    y_importance,

    scoring="average_precision",

    n_repeats=3,

    random_state=RANDOM_SEED,

    n_jobs=N_JOBS,

)



importance_table = pd.DataFrame({

    "feature": features,

    "permutation_importance_mean":

        importance_result.importances_mean,

    "permutation_importance_std":

        importance_result.importances_std,

    "impurity_importance":

        random_forest.feature_importances_,

})



importance_table = importance_table.sort_values(

    "permutation_importance_mean",

    ascending=False,

)



importance_table.to_csv(

    RESULT_DIR / "random_forest_feature_importance.csv",

    index=False,

)





def readable_feature_name(feature):

    replacements = {

        "precipitation_z": "Precipitation",

        "temperature_z": "Temperature",

        "rootzone_soil_moisture_z": "Root-zone soil moisture",

        "evaporation_z": "Evaporation",

        "ndvi_z": "NDVI",

        "_lag1": " (t−1)",

        "_lag2": " (t−2)",

        "_lag3": " (t−3)",

    }



    label = feature



    for old, new in replacements.items():

        label = label.replace(old, new)



    return label





plot_table = importance_table.sort_values(

    "permutation_importance_mean",

    ascending=True,

).copy()



plot_table["label"] = plot_table["feature"].apply(

    readable_feature_name

)



fig, axis = plt.subplots(figsize=(9, 7))



axis.barh(

    plot_table["label"],

    plot_table["permutation_importance_mean"],

    xerr=plot_table["permutation_importance_std"],

    color="forestgreen",

    alpha=0.85,

    capsize=3,

)



axis.axvline(

    0,

    color="black",

    linewidth=0.8,

)



axis.set_xlabel(

    "Decrease in average precision after permutation"

)

axis.set_title(

    "Random Forest test-set permutation importance"

)

axis.grid(axis="x", alpha=0.25)



fig.tight_layout()

fig.savefig(

    FIGURE_DIR / "random_forest_feature_importance.png",

    dpi=300,

    bbox_inches="tight",

)

plt.close(fig)





print("\nFinal test performance")

print(

    final_metrics[

        [

            "method",

            "balanced_accuracy",

            "precision",

            "recall",

            "f1_score",

            "roc_auc",

            "average_precision",

        ]

    ].round(4).to_string(index=False)

)



print("\nTop five Random Forest predictors")

print(

    importance_table[

        [

            "feature",

            "permutation_importance_mean",

            "permutation_importance_std",

        ]

    ].head(5).round(5).to_string(index=False)

)



print("\nCreated:")

print("  results/final_test_metrics.csv")

print("  results/random_forest_feature_importance.csv")

print("  figures/model_precision_recall_curves.png")

print("  figures/model_roc_curves.png")

print("  figures/test_confusion_matrices.png")

print("  figures/random_forest_feature_importance.png")



dataset.close()
