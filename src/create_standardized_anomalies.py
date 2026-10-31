from pathlib import Path



import xarray as xr





PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_DIR = PROJECT_DIR / "data" / "processed"



TRAIN_START = "2001-01-01"

TRAIN_END = "2017-12-31"



DATASETS = {

    "chirps_precipitation_monthly_ethiopia_2001_2024.nc": (

        "precipitation_z",

        "chirps_precipitation_standardized_anomaly_ethiopia_2001_2024.nc",

    ),

    "era5land_temperature_monthly_ethiopia_2001_2024.nc": (

        "temperature_z",

        "era5land_temperature_standardized_anomaly_ethiopia_2001_2024.nc",

    ),

    "era5land_rootzone_soil_moisture_monthly_ethiopia_2001_2024.nc": (

        "rootzone_soil_moisture_z",

        "era5land_rootzone_soil_moisture_standardized_anomaly_ethiopia_2001_2024.nc",

    ),

    "era5land_evaporation_monthly_ethiopia_2001_2024.nc": (

        "evaporation_z",

        "era5land_evaporation_standardized_anomaly_ethiopia_2001_2024.nc",

    ),

    "modis_ndvi_monthly_ethiopia_2001_2024.nc": (

        "ndvi_z",

        "modis_ndvi_standardized_anomaly_ethiopia_2001_2024.nc",

    ),

}





def find_main_variable(dataset):

    """Find the variable containing time, latitude and longitude."""

    candidates = [

        name

        for name, variable in dataset.data_vars.items()

        if {"time", "latitude", "longitude"}.issubset(variable.dims)

    ]



    if len(candidates) != 1:

        raise ValueError(

            f"Expected one spatiotemporal variable, found: {candidates}"

        )



    return candidates[0]





for input_name, (output_variable, output_name) in DATASETS.items():

    input_path = INPUT_DIR / input_name

    output_path = INPUT_DIR / output_name



    if not input_path.exists():

        raise FileNotFoundError(f"Missing input file: {input_path}")



    with xr.open_dataset(input_path) as dataset:

        source_variable = find_main_variable(dataset)



        data = (

            dataset[source_variable]

            .sortby("time")

            .sortby("latitude")

            .sortby("longitude")

        )



        if data.sizes["time"] != 288:

            raise ValueError(

                f"{input_name} contains {data.sizes['time']} months; expected 288."

            )



        training_data = data.sel(time=slice(TRAIN_START, TRAIN_END))



        monthly_mean = training_data.groupby("time.month").mean(

            dim="time",

            skipna=True,

        )



        monthly_std = training_data.groupby("time.month").std(

            dim="time",

            skipna=True,

            ddof=1,

        )



        # Prevent division by zero in pixels with no temporal variability.

        monthly_std = monthly_std.where(monthly_std > 1e-6)



        standardized = (

            (data.groupby("time.month") - monthly_mean)

            .groupby("time.month")

            / monthly_std

        )



        standardized = standardized.astype("float32").rename(output_variable)



        standardized.attrs = {

            "long_name": (

                f"Calendar-month standardized anomaly of {source_variable}"

            ),

            "units": "1",

            "standardization": "(value - monthly mean) / monthly standard deviation",

            "reference_period": "2001-01 to 2017-12",

            "source_variable": source_variable,

        }



        output_dataset = standardized.to_dataset()



        output_dataset.attrs = {

            "title": f"Monthly standardized anomalies for {source_variable}",

            "spatial_extent": "Ethiopia",

            "temporal_extent": "2001-01 to 2024-12",

            "climatology_reference_period": "2001-01 to 2017-12",

            "history": "Created by src/create_standardized_anomalies.py",

        }



        encoding = {

            output_variable: {

                "dtype": "float32",

                "zlib": True,

                "complevel": 4,

                "_FillValue": -9999.0,

            }

        }



        output_dataset.to_netcdf(

            output_path,

            format="NETCDF4",

            encoding=encoding,

        )



        training_standardized = standardized.sel(

            time=slice(TRAIN_START, TRAIN_END)

        )



        mean_check = float(training_standardized.mean(skipna=True).values)

        std_check = float(training_standardized.std(skipna=True).values)

        missing = float(standardized.isnull().mean().values) * 100



        print(f"\nCreated: {output_name}")

        print(f"  Variable: {output_variable}")

        print(f"  Shape: {standardized.shape}")

        print(f"  Training anomaly mean: {mean_check:.4f}")

        print(f"  Training anomaly std: {std_check:.4f}")

        print(f"  Missing fraction: {missing:.4f}%")



print("\nAll five standardized-anomaly datasets were created successfully.")
