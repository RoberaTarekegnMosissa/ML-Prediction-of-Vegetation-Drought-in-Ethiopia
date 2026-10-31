from pathlib import Path

import json



import joblib

import matplotlib

matplotlib.use("Agg")



import matplotlib.pyplot as plt

from matplotlib.colors import BoundaryNorm, ListedColormap

import numpy as np

import pandas as pd

import xarray as xr





PROJECT_DIR = Path(__file__).resolve().parents[1]



DATA_FILE = (

    PROJECT_DIR

    / "data"

    / "processed"

    / "ethiopia_drought_ml_feature_cube_2001_2024.nc"

)



MODEL_FILE = (

    PROJECT_DIR

    / "results"

    / "models"

    / "random_forest.joblib"

)



METADATA_FILE = (

    PROJECT_DIR

    / "results"

    / "model_metadata.json"

)



OUTPUT_NETCDF = (

    PROJECT_DIR

    / "data"

    / "processed"

    / "random_forest_test_predictions_2021_2024.nc"

)



FIGURE_DIR = PROJECT_DIR / "figures"

RESULT_DIR = PROJECT_DIR / "results"



FIGURE_DIR.mkdir(exist_ok=True)

RESULT_DIR.mkdir(exist_ok=True)





with open(METADATA_FILE, encoding="utf-8") as file:

    metadata = json.load(file)



features = metadata["features"]

threshold = metadata["selected_thresholds"]["random_forest"]



print(f"Random Forest threshold: {threshold:.4f}")

print(f"Predictors: {len(features)}")





dataset = xr.open_dataset(

    DATA_FILE

)[features + ["drought", "valid_sample", "data_split"]].load()



test_mask = (

    (dataset["valid_sample"] == 1)

    & (dataset["data_split"] == 2)

)



flat_indices = np.flatnonzero(test_mask.values.ravel())



X_test = np.column_stack([

    dataset[feature].values.ravel()[flat_indices]

    for feature in features

]).astype(np.float32)



random_forest = joblib.load(MODEL_FILE)



print(f"Predicting {len(flat_indices):,} test samples...")



probabilities = random_forest.predict_proba(X_test)[:, 1]





# Reconstruct predictions on the original time–latitude–longitude grid.

probability_values = np.full(

    dataset["drought"].shape,

    np.nan,

    dtype=np.float32,

)



probability_values.ravel()[flat_indices] = probabilities.astype(

    np.float32

)



probability_cube = xr.DataArray(

    probability_values,

    coords=dataset["drought"].coords,

    dims=dataset["drought"].dims,

    name="drought_probability_score",

)



predicted_drought = xr.where(

    probability_cube.notnull(),

    (probability_cube >= threshold).astype(np.float32),

    np.nan,

).rename("predicted_drought")



observed_drought = dataset["drought"].where(

    test_mask

).astype("float32").rename("observed_drought")





prediction_dataset = xr.Dataset({

    "observed_drought": observed_drought,

    "predicted_drought": predicted_drought,

    "drought_probability_score": probability_cube,

}).sel(

    time=slice("2021-01-01", "2024-12-31")

)



prediction_dataset["observed_drought"].attrs = {

    "long_name": "Observed vegetation-drought class",

    "definition": "1 when NDVI standardized anomaly is below -1",

}



prediction_dataset["predicted_drought"].attrs = {

    "long_name": "Random Forest predicted drought class",

    "threshold": threshold,

}



prediction_dataset["drought_probability_score"].attrs = {

    "long_name": "Random Forest drought probability score",

    "note": "Model score is not explicitly probability-calibrated",

}



prediction_dataset.attrs = {

    "title": "Random Forest vegetation-drought predictions for Ethiopia",

    "test_period": "2021-01 to 2024-12",

    "model": "Random Forest",

    "decision_threshold": str(threshold),

}



encoding = {

    variable: {

        "dtype": "float32",

        "zlib": True,

        "complevel": 4,

        "_FillValue": -9999.0,

    }

    for variable in prediction_dataset.data_vars

}



prediction_dataset.to_netcdf(

    OUTPUT_NETCDF,

    format="NETCDF4",

    encoding=encoding,

)



print(f"Created: {OUTPUT_NETCDF.name}")





# -------------------------------------------------------------------

# Spatial drought-frequency maps

# -------------------------------------------------------------------



observed_frequency = (

    prediction_dataset["observed_drought"].mean(

        dim="time",

        skipna=True,

    )

    * 100

)



predicted_frequency = (

    prediction_dataset["predicted_drought"].mean(

        dim="time",

        skipna=True,

    )

    * 100

)



frequency_bias = predicted_frequency - observed_frequency



valid_spatial_mask = (

    prediction_dataset["observed_drought"]

    .notnull()

    .any(dim="time")

)



longitude = prediction_dataset["longitude"]

latitude = prediction_dataset["latitude"]



mean_latitude = float(latitude.mean())

map_aspect = 1 / np.cos(np.deg2rad(mean_latitude))





def decorate_map(axis, title):

    axis.contour(

        longitude,

        latitude,

        valid_spatial_mask.astype(float),

        levels=[0.5],

        colors="black",

        linewidths=0.7,

    )



    axis.set_title(title)

    axis.set_xlabel("Longitude")

    axis.set_ylabel("Latitude")

    axis.set_aspect(map_aspect)

    axis.grid(alpha=0.15)





frequency_max = max(

    10,

    float(

        np.nanpercentile(

            np.stack([

                observed_frequency.values,

                predicted_frequency.values,

            ]),

            99,

        )

    ),

)



frequency_max = np.ceil(frequency_max / 5) * 5



bias_max = max(

    5,

    float(

        np.nanpercentile(

            np.abs(frequency_bias.values),

            99,

        )

    ),

)



bias_max = np.ceil(bias_max / 5) * 5





fig, axes = plt.subplots(

    nrows=1,

    ncols=3,

    figsize=(15, 6),

)



observed_image = axes[0].pcolormesh(

    longitude,

    latitude,

    observed_frequency,

    cmap="YlOrRd",

    vmin=0,

    vmax=frequency_max,

    shading="auto",

)



decorate_map(

    axes[0],

    "Observed drought frequency",

)



fig.colorbar(

    observed_image,

    ax=axes[0],

    label="Drought months (%)",

    shrink=0.75,

)





predicted_image = axes[1].pcolormesh(

    longitude,

    latitude,

    predicted_frequency,

    cmap="YlOrRd",

    vmin=0,

    vmax=frequency_max,

    shading="auto",

)



decorate_map(

    axes[1],

    "Random Forest predicted frequency",

)



fig.colorbar(

    predicted_image,

    ax=axes[1],

    label="Predicted drought months (%)",

    shrink=0.75,

)





bias_image = axes[2].pcolormesh(

    longitude,

    latitude,

    frequency_bias,

    cmap="RdBu_r",

    vmin=-bias_max,

    vmax=bias_max,

    shading="auto",

)



decorate_map(

    axes[2],

    "Prediction bias",

)



fig.colorbar(

    bias_image,

    ax=axes[2],

    label="Predicted − observed (percentage points)",

    shrink=0.75,

)



fig.suptitle(

    "Vegetation-drought frequency over Ethiopia, 2021–2024",

    fontsize=15,

)



fig.tight_layout()



fig.savefig(

    FIGURE_DIR / "spatial_drought_frequency_2021_2024.png",

    dpi=300,

    bbox_inches="tight",

)



plt.close(fig)





# -------------------------------------------------------------------

# Select three major spatial drought events

# -------------------------------------------------------------------



latitude_weights = xr.DataArray(

    np.cos(np.deg2rad(latitude)),

    coords={"latitude": latitude},

    dims=["latitude"],

)



observed_area = (

    prediction_dataset["observed_drought"]

    .weighted(latitude_weights)

    .mean(

        dim=["latitude", "longitude"],

        skipna=True,

    )

    * 100

)



predicted_area = (

    prediction_dataset["predicted_drought"]

    .weighted(latitude_weights)

    .mean(

        dim=["latitude", "longitude"],

        skipna=True,

    )

    * 100

)



mean_probability = (

    prediction_dataset["drought_probability_score"]

    .weighted(latitude_weights)

    .mean(

        dim=["latitude", "longitude"],

        skipna=True,

    )

)



sorted_events = (

    observed_area

    .to_series()

    .dropna()

    .sort_values(ascending=False)

)



selected_dates = []



# Keep selected months at least approximately two months apart.

for timestamp in sorted_events.index:

    timestamp = pd.Timestamp(timestamp)



    sufficiently_separated = all(

        abs((timestamp - previous).days) >= 60

        for previous in selected_dates

    )



    if sufficiently_separated:

        selected_dates.append(timestamp)



    if len(selected_dates) == 3:

        break





event_rows = []



for timestamp in selected_dates:

    event_rows.append({

        "month": timestamp.strftime("%Y-%m"),

        "observed_drought_area_percent": float(

            observed_area.sel(time=timestamp).values

        ),

        "predicted_drought_area_percent": float(

            predicted_area.sel(time=timestamp).values

        ),

        "mean_drought_probability_score": float(

            mean_probability.sel(time=timestamp).values

        ),

    })



event_table = pd.DataFrame(event_rows)



event_table.to_csv(

    RESULT_DIR / "selected_drought_events.csv",

    index=False,

)





# -------------------------------------------------------------------

# Event maps

# -------------------------------------------------------------------



binary_cmap = ListedColormap([

    "#eeeeee",

    "#b2182b",

])



binary_norm = BoundaryNorm(

    [-0.5, 0.5, 1.5],

    binary_cmap.N,

)



fig, axes = plt.subplots(

    nrows=2,

    ncols=3,

    figsize=(14, 9),

)



for column, timestamp in enumerate(selected_dates):

    observed_month = prediction_dataset[

        "observed_drought"

    ].sel(time=timestamp)



    probability_month = prediction_dataset[

        "drought_probability_score"

    ].sel(time=timestamp)



    observed_image = axes[0, column].pcolormesh(

        longitude,

        latitude,

        observed_month,

        cmap=binary_cmap,

        norm=binary_norm,

        shading="auto",

    )



    decorate_map(

        axes[0, column],

        f"{timestamp:%Y-%m}\nObserved drought",

    )



    probability_image = axes[1, column].pcolormesh(

        longitude,

        latitude,

        probability_month,

        cmap="YlOrRd",

        vmin=0,

        vmax=1,

        shading="auto",

    )



    decorate_map(

        axes[1, column],

        "Random Forest drought score",

    )





observed_colorbar = fig.colorbar(

    observed_image,

    ax=axes[0, :].tolist(),

    ticks=[0, 1],

    shrink=0.75,

    label="Observed class",

)



observed_colorbar.ax.set_yticklabels([

    "No drought",

    "Drought",

])



fig.colorbar(

    probability_image,

    ax=axes[1, :].tolist(),

    shrink=0.75,

    label="Drought probability score",

)



fig.suptitle(

    "Major vegetation-drought events during the test period",

    fontsize=15,

)


# Reserve space on the right for two dedicated colorbar axes.

fig.subplots_adjust(

    top=0.88,

    bottom=0.07,

    left=0.06,

    right=0.84,

    wspace=0.25,

    hspace=0.32,

)



observed_colorbar_axis = fig.add_axes([

    0.87,

    0.56,

    0.018,

    0.25,

])



observed_colorbar = fig.colorbar(

    observed_image,

    cax=observed_colorbar_axis,

    ticks=[0, 1],

)



observed_colorbar.set_label("Observed class")



observed_colorbar.ax.set_yticklabels([

    "No drought",

    "Drought",

])





probability_colorbar_axis = fig.add_axes([

    0.87,

    0.14,

    0.018,

    0.25,

])



probability_colorbar = fig.colorbar(

    probability_image,

    cax=probability_colorbar_axis,

)



probability_colorbar.set_label(

    "Drought probability score"

)



fig.savefig(

    FIGURE_DIR / "major_drought_event_maps.png",

    dpi=300,

    bbox_inches="tight",

)



plt.close(fig)





print("\nSelected major drought events")

print(

    event_table.round(3).to_string(index=False)

)



print("\nCreated:")

print("  data/processed/random_forest_test_predictions_2021_2024.nc")

print("  results/selected_drought_events.csv")

print("  figures/spatial_drought_frequency_2021_2024.png")

print("  figures/major_drought_event_maps.png")



dataset.close()
