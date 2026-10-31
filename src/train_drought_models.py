from pathlib import Path

import json

import os



import joblib

import numpy as np

import pandas as pd

import xarray as xr



from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (

    accuracy_score,

    average_precision_score,

    balanced_accuracy_score,

    f1_score,

    precision_recall_curve,

    precision_score,

    recall_score,

    roc_auc_score,

)

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler



from xgboost import XGBClassifier





PROJECT_DIR = Path(__file__).resolve().parents[1]



DATA_FILE = (

    PROJECT_DIR

    / "data"

    / "processed"

    / "ethiopia_drought_ml_feature_cube_2001_2024.nc"

)



RESULT_DIR = PROJECT_DIR / "results"

MODEL_DIR = RESULT_DIR / "models"



RESULT_DIR.mkdir(exist_ok=True)

MODEL_DIR.mkdir(exist_ok=True)



RANDOM_SEED = 42

MAX_TRAINING_SAMPLES = 400_000



N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))



FEATURES = [

    "precipitation_z_lag1",

    "precipitation_z_lag2",

    "precipitation_z_lag3",

    "temperature_z_lag1",

    "temperature_z_lag2",

    "temperature_z_lag3",

    "rootzone_soil_moisture_z_lag1",

    "rootzone_soil_moisture_z_lag2",

    "rootzone_soil_moisture_z_lag3",

    "evaporation_z_lag1",

    "evaporation_z_lag2",

    "evaporation_z_lag3",
    "ndvi_z_lag1",

    "ndvi_z_lag2",

    "ndvi_z_lag3",

]





if not DATA_FILE.exists():

    raise FileNotFoundError(f"Missing feature cube: {DATA_FILE}")





print("Loading feature cube...")



variables_to_load = FEATURES + [

    "drought",

    "valid_sample",

    "data_split",

]



dataset = xr.open_dataset(DATA_FILE)[variables_to_load].load()





def extract_split(split_code, maximum_samples=None):

    """Extract one chronological split as NumPy arrays."""



    split_mask = (

        (dataset["valid_sample"] == 1)

        & (dataset["data_split"] == split_code)

    )



    flat_indices = np.flatnonzero(split_mask.values.ravel())



    target_flat = dataset["drought"].values.ravel()

    targets = target_flat[flat_indices].astype(np.int8)



    original_count = len(flat_indices)



    if maximum_samples is not None and original_count > maximum_samples:

        positions = np.arange(original_count)



        selected_positions, _ = train_test_split(

            positions,

            train_size=maximum_samples,

            stratify=targets,

            random_state=RANDOM_SEED,

        )



        flat_indices = flat_indices[selected_positions]

        targets = targets[selected_positions]



    feature_columns = []



    for feature in FEATURES:

        feature_flat = dataset[feature].values.ravel()

        feature_columns.append(feature_flat[flat_indices])



    predictors = np.column_stack(feature_columns).astype(np.float32)



    if not np.isfinite(predictors).all():

        raise ValueError(

            f"Non-finite predictor values found in split {split_code}."

        )



    return predictors, targets, original_count





print("Extracting chronological splits...")



X_train, y_train, complete_train_count = extract_split(

    split_code=0,

    maximum_samples=MAX_TRAINING_SAMPLES,

)



X_validation, y_validation, complete_validation_count = extract_split(

    split_code=1,

)



X_test, y_test, complete_test_count = extract_split(

    split_code=2,

)





print(f"Training samples used: {len(y_train):,}")

print(f"Complete training samples: {complete_train_count:,}")

print(f"Validation samples: {len(y_validation):,}")

print(f"Testing samples: {len(y_test):,}")

print(f"Number of predictors: {X_train.shape[1]}")



positive_count = int(np.sum(y_train == 1))

negative_count = int(np.sum(y_train == 0))



scale_positive_weight = negative_count / positive_count



print(f"Training drought samples: {positive_count:,}")

print(f"Training non-drought samples: {negative_count:,}")

print(f"XGBoost scale_pos_weight: {scale_positive_weight:.3f}")





models = {

    "logistic_regression": make_pipeline(

        StandardScaler(),

        LogisticRegression(

            class_weight="balanced",

            max_iter=500,

            solver="lbfgs",

            random_state=RANDOM_SEED,

        ),

    ),



    "random_forest": RandomForestClassifier(

        n_estimators=250,

        max_depth=16,

        min_samples_leaf=20,

        max_features="sqrt",

        class_weight="balanced_subsample",

        max_samples=0.8,

        n_jobs=N_JOBS,

        random_state=RANDOM_SEED,

    ),



    "xgboost": XGBClassifier(

        n_estimators=350,

        learning_rate=0.05,

        max_depth=6,

        min_child_weight=5,

        subsample=0.8,

        colsample_bytree=0.8,

        scale_pos_weight=scale_positive_weight,

        objective="binary:logistic",

        eval_metric="aucpr",

        tree_method="hist",

        n_jobs=N_JOBS,

        random_state=RANDOM_SEED,

    ),

}





def select_f1_threshold(y_true, probabilities):

    """Select the probability threshold maximizing validation F1."""



    precision, recall, thresholds = precision_recall_curve(

        y_true,

        probabilities,

    )



    f1_values = (

        2 * precision[:-1] * recall[:-1]

        / (precision[:-1] + recall[:-1] + 1e-12)

    )



    best_index = int(np.nanargmax(f1_values))



    return (

        float(thresholds[best_index]),

        float(f1_values[best_index]),

    )





def calculate_metrics(

    y_true,

    probabilities,

    threshold,

    model_name,

    split_name,

):

    predictions = (probabilities >= threshold).astype(np.int8)



    return {

        "model": model_name,

        "split": split_name,

        "threshold": threshold,

        "samples": len(y_true),

        "accuracy": accuracy_score(y_true, predictions),

        "balanced_accuracy": balanced_accuracy_score(

            y_true,

            predictions,

        ),

        "precision": precision_score(

            y_true,

            predictions,

            zero_division=0,

        ),

        "recall": recall_score(

            y_true,

            predictions,

            zero_division=0,

        ),

        "f1_score": f1_score(

            y_true,

            predictions,

            zero_division=0,

        ),

        "roc_auc": roc_auc_score(y_true, probabilities),

        "average_precision": average_precision_score(

            y_true,

            probabilities,

        ),

    }





metric_rows = []

selected_thresholds = {}





for model_name, model in models.items():

    print(f"\nTraining {model_name}...")



    model.fit(X_train, y_train)



    validation_probabilities = model.predict_proba(

        X_validation

    )[:, 1]



    selected_threshold, validation_best_f1 = select_f1_threshold(

        y_validation,

        validation_probabilities,

    )



    selected_thresholds[model_name] = selected_threshold



    print(f"Selected threshold: {selected_threshold:.4f}")

    print(f"Validation F1 at threshold: {validation_best_f1:.4f}")



    metric_rows.append(

        calculate_metrics(

            y_validation,

            validation_probabilities,

            selected_threshold,

            model_name,

            "validation",

        )

    )



    test_probabilities = model.predict_proba(X_test)[:, 1]



    metric_rows.append(

        calculate_metrics(

            y_test,

            test_probabilities,

            selected_threshold,

            model_name,

            "testing",

        )

    )



    model_path = MODEL_DIR / f"{model_name}.joblib"



    joblib.dump(

        model,

        model_path,

        compress=3,

    )



    print(f"Saved model: {model_path.name}")





metrics_table = pd.DataFrame(metric_rows)



metrics_table.to_csv(

    RESULT_DIR / "model_metrics.csv",

    index=False,

)





metadata = {

    "features": FEATURES,

    "random_seed": RANDOM_SEED,

    "training_period": "2001-2017",

    "validation_period": "2018-2020",

    "testing_period": "2021-2024",

    "maximum_training_samples": MAX_TRAINING_SAMPLES,

    "training_samples_used": int(len(y_train)),

    "complete_training_samples": int(complete_train_count),

    "validation_samples": int(len(y_validation)),

    "testing_samples": int(len(y_test)),

    "threshold_selection": "Maximum F1 score on validation data",

    "selected_thresholds": selected_thresholds,

}



with open(

    RESULT_DIR / "model_metadata.json",

    "w",

    encoding="utf-8",

) as metadata_file:

    json.dump(

        metadata,

        metadata_file,

        indent=2,

    )





print("\nModel performance")

print(

    metrics_table[

        [

            "model",

            "split",

            "balanced_accuracy",

            "precision",

            "recall",

            "f1_score",

            "roc_auc",

            "average_precision",

        ]

    ].round(4).to_string(index=False)

)



print("\nCreated:")

print("  results/model_metrics.csv")

print("  results/model_metadata.json")

print("  results/models/*.joblib")



dataset.close()
