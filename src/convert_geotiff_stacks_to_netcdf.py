from pathlib import Path



import numpy as np

import pandas as pd

import rasterio

import xarray as xr





# ------------------------------------------------------------

# Project directories

# ------------------------------------------------------------



PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"



PROCESSED_DIR.mkdir(parents=True, exist_ok=True)





# January 2001 to December 2024

EXPECTED_TIME = pd.date_range(

    start="2001-01-01",

    end="2024-12-01",

    freq="MS",

)



EXPECTED_MONTHS = len(EXPECTED_TIME)





# ------------------------------------------------------------

# Dataset configuration

# ------------------------------------------------------------



DATASETS = [

    {

        "input": "chirps_precipitation_monthly_ethiopia_2001_2024.tif",

        "output": "chirps_precipitation_monthly_ethiopia_2001_2024.nc",

        "variable": "precipitation",

        "long_name": "Monthly total precipitation",

        "units": "mm month-1",

        "source": "CHIRPS v2 pentad precipitation",

    },

    {

        "input": "era5land_temperature_monthly_ethiopia_2001_2024.tif",

        "output": "era5land_temperature_monthly_ethiopia_2001_2024.nc",

        "variable": "temperature",

        "long_name": "Monthly mean 2-metre air temperature",

        "units": "degree_Celsius",

        "source": "ERA5-Land monthly aggregated data",

    },

    {

        "input": (

            "era5land_rootzone_soil_moisture_monthly_"

            "ethiopia_2001_2024.tif"

        ),

        "output": (

            "era5land_rootzone_soil_moisture_monthly_"

            "ethiopia_2001_2024.nc"

        ),

        "variable": "rootzone_soil_moisture",

        "long_name": "Monthly mean root-zone soil moisture for 0-100 cm",

        "units": "m3 m-3",

        "source": (

            "ERA5-Land soil-water layers 1-3; "

            "depth-weighted over 0-100 cm"

        ),

    },

    {

        "input": "era5land_evaporation_monthly_ethiopia_2001_2024.tif",

        "output": "era5land_evaporation_monthly_ethiopia_2001_2024.nc",

        "variable": "evaporation",

        "long_name": "Monthly total evaporation",

        "units": "mm month-1",

        "source": "ERA5-Land monthly total evaporation",

    },

    {

        "input": "modis_ndvi_monthly_ethiopia_2001_2024.tif",

        "output": "modis_ndvi_monthly_ethiopia_2001_2024.nc",

        "variable": "ndvi",

        "long_name": "Monthly Normalized Difference Vegetation Index",

        "units": "1",

        "source": "MODIS MOD13A3 Version 6.1",

    },

]





def coordinate_centres(transform, width, height):

    """Calculate longitude and latitude pixel-centre coordinates."""



    if transform.b != 0 or transform.d != 0:

        raise ValueError("Rotated raster grids are not supported.")



    longitude = (

        transform.c

        + transform.a * (np.arange(width) + 0.5)

    )



    latitude = (

        transform.f

        + transform.e * (np.arange(height) + 0.5)

    )



    return latitude, longitude





def convert_dataset(config):

    """Convert one monthly multiband GeoTIFF into NetCDF."""



    input_path = RAW_DIR / config["input"]

    output_path = PROCESSED_DIR / config["output"]



    if not input_path.exists():

        raise FileNotFoundError(f"Missing input file: {input_path}")



    with rasterio.open(input_path) as source:

        if source.count != EXPECTED_MONTHS:

            raise ValueError(

                f"{input_path.name} contains {source.count} bands, "

                f"but {EXPECTED_MONTHS} were expected."

            )



        if source.crs is None:

            raise ValueError(f"{input_path.name} has no CRS information.")



        raster = source.read(masked=True).astype("float32")

        data = raster.filled(np.nan)



        latitude, longitude = coordinate_centres(

            source.transform,

            source.width,

            source.height,

        )



        crs_text = source.crs.to_string()



    data_array = xr.DataArray(

        data=data,

        dims=("time", "latitude", "longitude"),

        coords={

            "time": EXPECTED_TIME,

            "latitude": latitude,

            "longitude": longitude,

        },

        name=config["variable"],

    )



    # Use ascending coordinates

    data_array = data_array.sortby("latitude")

    data_array = data_array.sortby("longitude")



    data_array.attrs = {

        "long_name": config["long_name"],

        "units": config["units"],

        "source": config["source"],

        "grid_mapping": "crs",

    }



    dataset = data_array.to_dataset()



    dataset["latitude"].attrs = {

        "long_name": "latitude",

        "standard_name": "latitude",

        "units": "degrees_north",

        "axis": "Y",

    }



    dataset["longitude"].attrs = {

        "long_name": "longitude",

        "standard_name": "longitude",

        "units": "degrees_east",

        "axis": "X",

    }



    dataset["time"].attrs = {

        "long_name": "time",

        "standard_name": "time",

        "axis": "T",

    }



    dataset["crs"] = xr.DataArray(0)

    dataset["crs"].attrs = {

        "grid_mapping_name": "latitude_longitude",

        "spatial_ref": crs_text,

        "epsg_code": "EPSG:4326",

    }



    dataset.attrs = {

        "title": config["long_name"] + " over Ethiopia",

        "temporal_coverage": "2001-01 to 2024-12",

        "spatial_resolution": "0.1 degree",

        "Conventions": "CF-1.8",

        "processing": (

            "Clipped to Ethiopia and aggregated to a common "

            "0.1-degree grid in Google Earth Engine."

        ),

    }



    valid_months = dataset[config["variable"]].notnull().any(

        dim=("latitude", "longitude")

    )



    if not bool(valid_months.all().item()):

        missing_times = dataset.time.where(

            ~valid_months,

            drop=True,

        ).values



        raise ValueError(

            f"{input_path.name} has completely missing months: "

            f"{missing_times}"

        )



    finite_values = data_array.values[

        np.isfinite(data_array.values)

    ]



    if finite_values.size == 0:

        raise ValueError(f"{input_path.name} contains no valid data.")



    minimum = float(finite_values.min())

    maximum = float(finite_values.max())

    missing_fraction = float(data_array.isnull().mean().item())



    encoding = {

        config["variable"]: {

            "dtype": "float32",

            "zlib": True,

            "complevel": 4,

            "_FillValue": np.float32(-9999.0),

        }

    }



    dataset.to_netcdf(

        output_path,

        engine="netcdf4",

        encoding=encoding,

    )



    print(f"\nCreated: {output_path.name}")

    print(f"  Shape: {data_array.shape}")

    print(f"  Months: {data_array.sizes['time']}")

    print(f"  Time: {EXPECTED_TIME[0]:%Y-%m} to {EXPECTED_TIME[-1]:%Y-%m}")

    print(f"  Minimum: {minimum:.4f}")

    print(f"  Maximum: {maximum:.4f}")

    print(f"  Missing fraction: {missing_fraction:.4%}")



    return data_array.latitude.values, data_array.longitude.values





# ------------------------------------------------------------

# Convert all five datasets and verify grid equality

# ------------------------------------------------------------



reference_latitude = None

reference_longitude = None



for dataset_config in DATASETS:

    latitude, longitude = convert_dataset(dataset_config)



    if reference_latitude is None:

        reference_latitude = latitude

        reference_longitude = longitude

    else:

        if not np.allclose(latitude, reference_latitude):

            raise ValueError(

                f"Latitude grid mismatch in {dataset_config['input']}"

            )



        if not np.allclose(longitude, reference_longitude):

            raise ValueError(

                f"Longitude grid mismatch in {dataset_config['input']}"

            )



print("\nAll five datasets were converted successfully.")

print("All datasets contain 288 months on the same spatial grid.")
