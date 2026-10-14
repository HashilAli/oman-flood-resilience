"""
Tests for geospatial data acquisition, coordinate transformation, and terrain preparation.
"""
import pytest
import numpy as np
from rasterio.transform import from_origin

from src.geodata.dem import (
    validate_bbox,
    utm_epsg_for,
    copernicus_tile_names,
    DemGrid,
    validate_dem,
    DemError
)


def test_validate_bbox():
    bbox = [58.50, 23.54, 58.58, 23.62]
    w, s, e, n = validate_bbox(bbox)
    assert w == 58.50
    assert s == 23.54
    assert e == 58.58
    assert n == 23.62

    with pytest.raises(DemError):
        validate_bbox([58.58, 23.62, 58.50, 23.54])  # invalid bounds


def test_utm_epsg_for_oman():
    # Muscat, Oman (~58.5° E, ~23.6° N) is in UTM Zone 40 North (EPSG:32640)
    epsg_muscat = utm_epsg_for(58.54, 23.58)
    assert epsg_muscat == 32640

    # Salalah, Dhofar (~54.1° E, ~17.0° N) is in UTM Zone 40 North (EPSG:32640)
    epsg_dhofar = utm_epsg_for(54.10, 17.02)
    assert epsg_dhofar == 32640


def test_copernicus_tile_names():
    bbox = [58.50, 23.54, 58.58, 23.62]
    tiles = copernicus_tile_names(bbox)
    assert len(tiles) == 1
    assert tiles[0] == "Copernicus_DSM_COG_10_N23_00_E058_00_DEM"


def test_dem_grid_validation():
    rows, cols = 50, 50
    res = 30.0
    elev = np.ones((rows, cols), dtype=np.float64) * 150.0
    transform = from_origin(600000.0, 2600000.0, res, res)
    valid = np.ones((rows, cols), dtype=bool)
    bbox = (58.50, 23.54, 58.58, 23.62)

    grid = DemGrid(elev, transform, "EPSG:32640", valid, bbox)
    report = validate_dem(grid)
    assert report["ok"] is True
    assert report["rows"] == 50
    assert report["cols"] == 50
    assert report["elevation_min_m"] == 150.0
    assert report["void_fraction"] == 0.0
