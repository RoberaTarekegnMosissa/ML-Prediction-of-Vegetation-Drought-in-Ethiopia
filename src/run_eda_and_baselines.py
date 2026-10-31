from pathlib import Path



import matplotlib

matplotlib.use("Agg")



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import xarray as xr



from sklearn.metrics import (

    accuracy_score,

    average_precision_score,

    balanced_accuracy_score,

    f1_score,

    precision_score,

    recall_score,

)





PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = (

    PROJECT_DIR

    / "data"

    / "processed"

    / "ethiopia_drought_ml_feature_cube_2001_2024.nc"

)



FIGURE_DIR = PROJECT_DIR / "figures"

RESULT_DIR = PROJECT_DIR / "results"



FIGURE_DIR.mkdir(exist_ok=True)

RESULT_DIR.mkdir(exist_ok=True)



if not DATA_FILE.exists():

    raise FileNotFoundError(f"Missing feature cube: {DATA_FILE}")





dataset = xr.open_dataset(DATA_FILE)



required_variables = [

    "ndvi_z",

    "drought",

    "valid_sample",

    "data_split",

]



for variable in required_variables:

    if variable not in dataset:

        raise ValueError(f"Missing variable: {variable}")





# -------------------------------------------------------------------

# 1. Area-weighted Ethiopia-wide time series

# -------------------------------------------------------------------



latitude_weights = xr.DataArray(

    np.cos(np.deg2rad(dataset["latitude"])),

    coords={"latitude": dataset["latitude"]},

    dims=["latitude"],

)



monthly_ndvi = dataset["ndvi_z"].weighted(latitude_weights).mean(

    dim=["latitude", "longitude"],

    skipna=True,

)



monthly_drought_fraction = dataset["drought"].weighted(

    latitude_weights

).mean(

    dim=["latitude", "longitude"],

    skipna=True,

)





# -------------------------------------------------------------------

# 2. Annual and calendar-month summaries

# -------------------------------------------------------------------



annual_drought = monthly_drought_fraction.groupby("time.year").mean(

    dim="time",

    skipna=True,

)



annual_table = pd.DataFrame({

    "year": annual_drought["year"].values,

    "drought_fraction": annual_drought.values,

})



annual_table["drought_percentage"] = (

    100 * annual_table["drought_fraction"]

)



annual_table.to_csv(

    RESULT_DIR / "annual_drought_frequency.csv",

    index=False,

)





monthly_climatology = monthly_drought_fraction.groupby(

    "time.month"

).mean(

    dim="time",

    skipna=True,

)



monthly_table = pd.DataFrame({

    "month": monthly_climatology["month"].values,

    "mean_drought_fraction": monthly_climatology.values,

})



monthly_table["mean_drought_percentage"] = (

    100 * monthly_table["mean_drought_fraction"]

)



monthly_table.to_csv(

    RESULT_DIR / "monthly_drought_climatology.csv",

    index=False,

)





# -------------------------------------------------------------------

# 3. Class distribution by chronological split

# -------------------------------------------------------------------



split_names = {

    0: "training",

    1: "validation",

    2: "testing",

}



class_rows = []



for split_code, split_name in split_names.items():

    mask = (

        (dataset["valid_sample"] == 1)

        & (dataset["data_split"] == split_code)

    )



    sample_count = int(mask.sum().values)



    drought_count = int(

        (

            mask

            & (dataset["drought"] == 1)

        ).sum().values

    )



    non_drought_count = sample_count - drought_count



    class_rows.append({

        "split": split_name,

        "split_code": split_code,

        "valid_samples": sample_count,

        "drought_samples": drought_count,

        "non_drought_samples": non_drought_count,

        "drought_percentage": (

            100 * drought_count / sample_count

            if sample_count > 0

            else np.nan

        ),

    })



class_table = pd.DataFrame(class_rows)



class_table.to_csv(

    RESULT_DIR / "split_class_distribution.csv",

    index=False,

)





# -------------------------------------------------------------------

# 4. Baseline predictions

# -------------------------------------------------------------------



# Baseline 1: always predict no drought.

always_no_drought = xr.zeros_like(dataset["drought"])



# Baseline 2: predict that drought persists from the previous month.

previous_ndvi = dataset["ndvi_z"].shift(time=1)



persistence_prediction = xr.where(

    previous_ndvi.notnull(),

    xr.where(previous_ndvi < -1.0, 1.0, 0.0),

    np.nan,

)



baseline_predictions = {

    "always_no_drought": always_no_drought,

    "previous_month_persistence": persistence_prediction,

}



metric_rows = []



# Baselines are evaluated on validation and testing periods.

for split_code in [1, 2]:

    split_name = split_names[split_code]



    for baseline_name, prediction in baseline_predictions.items():

        evaluation_mask = (

            (dataset["valid_sample"] == 1)

            & (dataset["data_split"] == split_code)

            & prediction.notnull()

        )



        y_true = dataset["drought"].where(

            evaluation_mask

        ).values.ravel()



        y_pred = prediction.where(

            evaluation_mask

        ).values.ravel()



        finite = np.isfinite(y_true) & np.isfinite(y_pred)



        y_true = y_true[finite].astype(int)

        y_pred = y_pred[finite].astype(int)



        metric_rows.append({

            "split": split_name,

            "baseline": baseline_name,

            "samples": len(y_true),

            "accuracy": accuracy_score(y_true, y_pred),

            "balanced_accuracy": balanced_accuracy_score(

                y_true,

                y_pred,

            ),

            "precision": precision_score(

                y_true,

                y_pred,

                zero_division=0,

            ),

            "recall": recall_score(

                y_true,

                y_pred,

                zero_division=0,

            ),

            "f1_score": f1_score(

                y_true,

                y_pred,

                zero_division=0,

            ),

            "average_precision": average_precision_score(

                y_true,

                y_pred,

            ),

        })



metrics_table = pd.DataFrame(metric_rows)



metrics_table.to_csv(

    RESULT_DIR / "baseline_metrics.csv",

    index=False,

)





# -------------------------------------------------------------------

# 5. Create exploratory figure

# -------------------------------------------------------------------



fig, axes = plt.subplots(

    nrows=2,

    ncols=1,

    figsize=(13, 8),

    sharex=True,

)



axes[0].plot(

    monthly_ndvi["time"],

    monthly_ndvi,

    color="forestgreen",

    linewidth=1.2,

)



axes[0].axhline(

    0,

    color="black",

    linewidth=0.8,

    linestyle="--",

)



axes[0].set_ylabel("Mean NDVI anomaly")

axes[0].set_title(

    "Ethiopia vegetation condition and vegetation-drought extent"

)

axes[0].grid(alpha=0.25)





axes[1].plot(

    monthly_drought_fraction["time"],

    100 * monthly_drought_fraction,

    color="firebrick",

    linewidth=1.2,

)



axes[1].set_ylabel("Drought-affected area (%)")

axes[1].set_xlabel("Year")

axes[1].grid(alpha=0.25)





for axis in axes:

    axis.axvline(

        pd.Timestamp("2018-01-01"),

        color="royalblue",

        linestyle="--",

        linewidth=1,

        label="Validation begins",

    )



    axis.axvline(

        pd.Timestamp("2021-01-01"),

        color="darkorange",

        linestyle="--",

        linewidth=1,

        label="Testing begins",

    )



axes[0].legend(loc="upper right")



fig.tight_layout()



figure_path = FIGURE_DIR / "eda_monthly_ndvi_drought.png"



fig.savefig(

    figure_path,

    dpi=300,

    bbox_inches="tight",

)



plt.close(fig)





# -------------------------------------------------------------------

# 6. Print results

# -------------------------------------------------------------------



print("\nClass distribution")

print(

    class_table[

        [

            "split",

            "valid_samples",

            "drought_samples",

            "drought_percentage",

        ]

    ].to_string(index=False)

)



print("\nBaseline performance")

print(

    metrics_table[

        [

            "split",

            "baseline",

            "balanced_accuracy",

            "precision",

            "recall",

            "f1_score",

            "average_precision",

        ]

    ].round(4).to_string(index=False)

)



print("\nCreated:")

print("  results/annual_drought_frequency.csv")

print("  results/monthly_drought_climatology.csv")

print("  results/split_class_distribution.csv")

print("  results/baseline_metrics.csv")

print("  figures/eda_monthly_ndvi_drought.png")



dataset.close()
