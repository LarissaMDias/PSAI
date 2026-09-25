#!/usr/bin/env python3
"""Assign broad geographic subregions to observation and model records."""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_BASIN_URL = (
    "https://services.arcgis.com/6lCKYNJLvwTXqrmp/arcgis/rest/services/"
    "Puget_Sound_Basins/FeatureServer/0/query?"
    "where=1=1&outFields=*&outSR=4326&f=geojson"
)


def subregion_creation(
    obs: pd.DataFrame,
    model: pd.DataFrame,
    base_dir: str | Path | None = None,
    *,
    basin_url: str = DEFAULT_BASIN_URL,
    plot: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add geographic region columns to observation and model DataFrames.

    Parameters
    ----------
    obs, model
        DataFrames containing numeric ``lon`` and ``lat`` columns.
    base_dir
        Directory containing the ``N45W125/N45W125.shp`` shapefile.
        If omitted, assumes this module is in ``LiveOcean/Algorithm_Development``
        and uses ``LiveOcean`` as the project root.
    basin_url
        GeoJSON/ArcGIS URL for Puget Sound basin polygons.
    plot
        If True, plot the basin polygons, coastline, and data points.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Copies of ``obs`` and ``model`` with ``region`` and, where available,
        ``basin`` columns.
    """
    required = {"lon", "lat"}
    for label, df in (("obs", obs), ("model", model)):
        missing = required - set(df.columns)
        if missing:
            raise KeyError(f"{label} is missing required columns: {sorted(missing)}")

    obs_out = obs.copy()
    model_out = model.copy()

    # BASE_DIR should be an actual Path object. This default assumes:
    # LiveOcean/Algorithm_Development/subregion_creation.py
    if base_dir is None:
        base_dir = Path(__file__).resolve().parents[1]
    else:
        base_dir = Path(base_dir).expanduser().resolve()

    shapefile = base_dir / "N45W125" / "N45W125.shp"
    if not shapefile.exists():
        raise FileNotFoundError(f"Coastal shapefile not found: {shapefile}")

    print(f"Reading coastal shapefile: {shapefile}")
    coast_gdf = gpd.read_file(shapefile)
    print(f"Coastal layer CRS: {coast_gdf.crs}")
    print(coast_gdf.geometry.geom_type.value_counts())

    if coast_gdf.crs is None:
        raise ValueError("The coastal shapefile has no CRS defined.")
    coast_gdf = coast_gdf.to_crs("EPSG:4326")

    # Make a point layer for each table. WGS84 matches lon/lat and the URL output.
    def make_points(df: pd.DataFrame) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            df.copy(),
            geometry=gpd.points_from_xy(df["lon"], df["lat"]),
            crs="EPSG:4326",
        )

    # Broad preliminary classification. Replace these bounds with bathymetry
    # when a shelf/offshore mask becomes available.
    def add_broad_region(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["region"] = "offshore"
        coastal = (
            out["lon"].between(-126, -122, inclusive="neither")
            & out["lat"].between(44, 52, inclusive="neither")
        )
        out.loc[coastal, "region"] = "coastal"
        return out

    obs_out = add_broad_region(obs_out)
    model_out = add_broad_region(model_out)

    # Load basin polygons. This requires internet access unless basin_url is
    # replaced with a local file path.
    basin_gdf = gpd.read_file(basin_url)
    if basin_gdf.crs is None:
        basin_gdf = basin_gdf.set_crs("EPSG:4326")
    else:
        basin_gdf = basin_gdf.to_crs("EPSG:4326")

    basin_column = "Region1" if "Region1" in basin_gdf.columns else None
    if basin_column is not None:
        basin_fields = [basin_column, "geometry"]

        def assign_basins(df: pd.DataFrame) -> pd.DataFrame:
            points = make_points(df)
            joined = gpd.sjoin(
                points,
                basin_gdf[basin_fields],
                how="left",
                predicate="within",
            )
            joined = joined[~joined.index.duplicated(keep="first")]
            out = df.copy()
            out["basin"] = joined[basin_column].reindex(df.index).to_numpy()
            out.loc[out["basin"].notna(), "region"] = out.loc[
                out["basin"].notna(), "basin"
            ]
            return out

        obs_out = assign_basins(obs_out)
        model_out = assign_basins(model_out)
    else:
        print("Warning: basin layer has no 'Region1' column; basin assignment skipped.")

    if plot:
        fig, ax = plt.subplots(figsize=(10, 8))
        basin_gdf.plot(ax=ax, column=basin_column, alpha=0.6, legend=True) if basin_column else basin_gdf.plot(ax=ax, alpha=0.6)
        coast_gdf.boundary.plot(ax=ax, color="black", linewidth=1)
        ax.scatter(obs_out["lon"], obs_out["lat"], s=5, color="red", label="Observations")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Assigned subregions")
        ax.legend()
        plt.tight_layout()
        plt.show()

    return obs_out, model_out


if __name__ == "__main__":
    print("Import subregion_creation() from this module and call it with obs and model.")
