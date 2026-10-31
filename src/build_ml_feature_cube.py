from pathlib import Path



import numpy as np

import xarray as xr





PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_DIR / "data" / "processed"



INPUT_FILES = {

    "precipitation_z":

        "chirps_precipitation_standardized_anomaly_ethiopia_2001_2024.nc",



    "temperature_z":

        "era5land_temperature_standardized_anomaly_ethiopia_2001_2024.nc",



    "rootzone_soil_moisture_z":

        "era5land_rootzone_soil_moisture_standardized_anomaly_ethiopia_2001_2024.nc",



    "evaporation_z":

        "era5land_evaporation_standardized_anomaly_ethiopia_2001_2024.nc",



    "ndvi_z":

        "modis_ndvi_standardized_anomaly_ethiopia_2001_2024.nc",

}



CLIMATE_VARIABLES = [

    "precipitation_z",

    "temperature_z",

    "rootzone_soil_moisture_z",

    "evaporation_z",

]



LAGS = [1, 2, 3]



OUTPUT_FILE = DATA_DIR / "ethiopia_drought_ml_feature_cube_2001_2024.nc"





opened_datasets = []



for variable_name, filename in INPUT_FILES.items():

    path = DATA_DIR / filename



    if not path.exists():

        raise FileNotFoundError(f"Missing file: {path}")



    dataset = xr.open_dataset(path)



    if variable_name not in dataset.data_vars:

        raise ValueError(

            f"{variable_name} was not found in {filename}. "

            f"Available variables: {list(dataset.data_vars)}"

        )



    opened_datasets.append(dataset[[variable_name]])





# Require identical time and spatial coordinates.

anomalies = xr.merge(

    opened_datasets,

    join="exact",

    compat="equals",

)



if anomalies.sizes["time"] != 288:

    raise ValueError(

        f"Expected 288 months, found {anomalies.sizes['time']}."

    )





feature_cube = xr.Dataset(coords=anomalies.coords)



# Keep NDVI anomaly for target construction and later diagnostics.

feature_cube["ndvi_z"] = anomalies["ndvi_z"].astype("float32")





# Create predictors for months t-1, t-2 and t-3.

feature_names = []



for variable in CLIMATE_VARIABLES:

    for lag in LAGS:

        feature_name = f"{variable}_lag{lag}"



        feature_cube[feature_name] = (

            anomalies[variable]

            .shift(time=lag)

            .astype("float32")

        )



        feature_cube[feature_name].attrs = {

            "long_name": f"{variable} lagged by {lag} month(s)",

            "prediction_time": f"Predictor from t-{lag} for drought at time t",

            "units": "1",

        }



        feature_names.append(feature_name)





# Target: vegetation drought when standardized NDVI anomaly is below -1.

feature_cube["drought"] = xr.where(

    anomalies["ndvi_z"].notnull(),

    xr.where(anomalies["ndvi_z"] < -1.0, 1.0, 0.0),

    np.nan,

).astype("float32")



feature_cube["drought"].attrs = {

    "long_name": "Binary vegetation drought indicator",

    "definition": "1 when NDVI standardized anomaly < -1; 0 otherwise",

    "units": "1",

}





# Identify samples having a valid target and all 12 predictors.

valid_sample = feature_cube["drought"].notnull()



for feature_name in feature_names:

    valid_sample = valid_sample & feature_cube[feature_name].notnull()



feature_cube["valid_sample"] = valid_sample.astype("int8")



feature_cube["valid_sample"].attrs = {

    "long_name": "Valid machine-learning sample",

    "definition": "1 when drought target and all predictors are available",

}





# Chronological data split: no random temporal leakage.

year = feature_cube["time"].dt.year



feature_cube["data_split"] = xr.where(

    year <= 2017,

    0,

    xr.where(year <= 2020, 1, 2),

).astype("int8")



feature_cube["data_split"].attrs = {

    "long_name": "Chronological machine-learning split",

    "codes": "0=training, 1=validation, 2=testing",

    "training_period": "2001-2017",

    "validation_period": "2018-2020",

    "testing_period": "2021-2024",

}





feature_cube.attrs = {

    "title": "Ethiopia vegetation-drought machine-learning feature cube",

    "temporal_extent": "2001-01 to 2024-12",

    "target_definition": "NDVI standardized anomaly < -1",

    "prediction_horizon": "Climate predictors from the previous 1-3 months",

    "number_of_predictors": str(len(feature_names)),

    "history": "Created by src/build_ml_feature_cube.py",

}





float_variables = ["ndvi_z", "drought"] + feature_names



encoding = {

    variable: {

        "dtype": "float32",

        "zlib": True,

        "complevel": 4,

        "_FillValue": -9999.0,

    }

    for variable in float_variables

}



encoding["valid_sample"] = {

    "dtype": "int8",

    "zlib": True,

    "complevel": 4,

}



encoding["data_split"] = {

    "dtype": "int8",

    "zlib": True,

    "complevel": 4,

}





feature_cube.to_netcdf(

    OUTPUT_FILE,

    format="NETCDF4",

    encoding=encoding,

)





print(f"\nCreated: {OUTPUT_FILE.name}")

print(f"Dimensions: {dict(feature_cube.sizes)}")

print(f"Predictors: {len(feature_names)}")

print("Target: NDVI standardized anomaly < -1\n")



split_names = {

    0: "Training 2001-2017",

    1: "Validation 2018-2020",

    2: "Testing 2021-2024",

}



for split_code, split_name in split_names.items():

    split_mask = (

        valid_sample

        & (feature_cube["data_split"] == split_code)

    )



    sample_count = int(split_mask.sum().values)



    drought_count = int(

        (

            split_mask

            & (feature_cube["drought"] == 1)

        ).sum().values

    )



    drought_percentage = (

        100 * drought_count / sample_count

        if sample_count > 0

        else np.nan

    )



    print(split_name)

    print(f"  Valid samples: {sample_count:,}")

    print(f"  Drought samples: {drought_count:,}")

    print(f"  Drought frequency: {drought_percentage:.2f}%")



for dataset in opened_datasets:

    dataset.close()



print("\nFeature cube created successfully.")
