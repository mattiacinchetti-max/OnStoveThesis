import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.request import urlretrieve
from zipfile import ZipFile

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, MultiLineString, MultiPoint, Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union


NATURAL_EARTH_URL = (
    "https://naturalearth.s3.amazonaws.com/10m_cultural/"
    "ne_10m_admin_0_countries.zip"
)
GEOFABRIK_INDEX_URL = "https://download.geofabrik.de/index-v1.json"

DEFAULT_MAJOR_HIGHWAYS = [
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract border crossing points from major OSM roads in Sub-Saharan Africa "
            "and save them as a point layer with country1/country2 fields."
        )
    )
    parser.add_argument(
        "--output",
        default="thesis/script_nostri/border_control_points.gpkg",
        help="Output GeoPackage path.",
    )
    parser.add_argument(
        "--layer-name",
        default="border_control_points",
        help="Output layer name inside the GeoPackage.",
    )
    parser.add_argument(
        "--cache-dir",
        default="thesis/script_nostri/_cache_border_controls",
        help="Directory used for temporary downloads.",
    )
    parser.add_argument(
        "--boundaries-url",
        default=NATURAL_EARTH_URL,
        help="Natural Earth countries zip URL.",
    )
    parser.add_argument(
        "--geofabrik-index-url",
        default=GEOFABRIK_INDEX_URL,
        help="Geofabrik index-v1.json URL.",
    )
    parser.add_argument(
        "--highways",
        nargs="+",
        default=DEFAULT_MAJOR_HIGHWAYS,
        help="Road classes (fclass) to keep as major roads.",
    )
    parser.add_argument(
        "--save-shared-borders",
        action="store_true",
        help="Also write shared borders as a second layer in the same GeoPackage.",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def download_if_missing(url: str, target_path: Path, label: str) -> Path:
    ensure_parent(target_path)
    if target_path.exists():
        print(f"[cache] {label}: {target_path}")
        return target_path
    print(f"[download] {label}: {url}")
    urlretrieve(url, target_path)
    return target_path


def get_subsaharan_countries(boundaries_zip: Path) -> gpd.GeoDataFrame:
    countries = gpd.read_file(boundaries_zip)
    africa = countries[countries["CONTINENT"] == "Africa"].copy()

    subsaharan = africa[
        africa["SUBREGION"].isin(
            ["Eastern Africa", "Middle Africa", "Southern Africa", "Western Africa"]
        )
    ].copy()

    subsaharan = subsaharan[["NAME", "ISO_A2", "geometry"]].copy()
    subsaharan.rename(columns={"NAME": "country", "ISO_A2": "iso_a2"}, inplace=True)
    subsaharan = subsaharan[subsaharan.geometry.notnull()].copy()
    subsaharan = subsaharan[~subsaharan.geometry.is_empty].copy()
    subsaharan = subsaharan.set_crs(4326, allow_override=True)
    return subsaharan


def extract_line_geometries(geom: BaseGeometry) -> List[LineString]:
    if geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, MultiLineString):
        return [g for g in geom.geoms if not g.is_empty]
    if hasattr(geom, "geoms"):
        out: List[LineString] = []
        for part in geom.geoms:
            out.extend(extract_line_geometries(part))
        return out
    return []


def build_shared_borders(countries: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    sindex = countries.sindex
    borders: Dict[Tuple[str, str], List[LineString]] = {}

    print("[compute] Building shared international borders")
    for idx, row in countries.iterrows():
        candidate_idxs = sindex.query(row.geometry, predicate="touches")
        for other_idx in candidate_idxs:
            if other_idx <= idx:
                continue

            other = countries.loc[other_idx]
            shared = row.geometry.boundary.intersection(other.geometry.boundary)
            lines = extract_line_geometries(shared)
            if not lines:
                continue

            pair = tuple(sorted((row["country"], other["country"])))
            borders.setdefault(pair, []).extend(lines)

    records = []
    for (country1, country2), lines in borders.items():
        merged = unary_union(lines)
        if merged.is_empty:
            continue
        records.append(
            {
                "country1": country1,
                "country2": country2,
                "geometry": merged,
            }
        )

    border_gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=4326)
    border_gdf = border_gdf[border_gdf.geometry.notnull()].copy()
    border_gdf = border_gdf[~border_gdf.geometry.is_empty].copy()
    print(f"[compute] Shared borders found: {len(border_gdf)}")
    return border_gdf


def load_geofabrik_index(index_json_path: Path) -> dict:
    with open(index_json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def geofabrik_iso2_to_shp(index_data: dict) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for feature in index_data.get("features", []):
        props = feature.get("properties", {})
        urls = props.get("urls", {})
        shp_url = urls.get("shp")
        iso_list = props.get("iso3166-1:alpha2", [])
        if not shp_url or not iso_list:
            continue
        for iso2 in iso_list:
            if isinstance(iso2, str) and len(iso2) == 2:
                mapping[iso2.upper()] = shp_url
    return mapping


def find_roads_shp_in_zip(zip_path: Path) -> Optional[str]:
    with ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            lower = name.lower()
            if lower.endswith("roads_free_1.shp"):
                return name
    return None


def load_major_roads(
    countries: gpd.GeoDataFrame,
    geofabrik_map: Dict[str, str],
    cache_dir: Path,
    major_classes: Iterable[str],
) -> gpd.GeoDataFrame:
    roads_parts: List[gpd.GeoDataFrame] = []
    major_classes_set = set(major_classes)

    iso_codes = sorted({iso for iso in countries["iso_a2"].dropna().astype(str).str.upper() if iso != "-99"})
    print(f"[download] Countries to process from Geofabrik: {len(iso_codes)}")

    for iso2 in iso_codes:
        shp_url = geofabrik_map.get(iso2)
        if not shp_url:
            print(f"[skip] No Geofabrik shapefile URL for ISO2={iso2}")
            continue

        zip_name = shp_url.rstrip("/").split("/")[-1]
        zip_path = cache_dir / "geofabrik" / zip_name
        download_if_missing(shp_url, zip_path, label=f"{iso2} roads")

        roads_shp_rel = find_roads_shp_in_zip(zip_path)
        if not roads_shp_rel:
            print(f"[skip] roads layer not found in {zip_path.name}")
            continue

        roads_path = f"zip://{zip_path}!{roads_shp_rel}"
        roads = gpd.read_file(roads_path)
        if "fclass" not in roads.columns:
            print(f"[skip] Missing 'fclass' in {zip_path.name}")
            continue

        roads = roads[roads["fclass"].isin(major_classes_set)].copy()
        if roads.empty:
            continue

        keep_cols = [c for c in ["osm_id", "code", "fclass", "name", "ref", "geometry"] if c in roads.columns]
        roads = roads[keep_cols].copy()
        roads["source_iso2"] = iso2
        roads_parts.append(roads)

    if not roads_parts:
        raise RuntimeError("No major roads were loaded. Check internet access and selected classes.")

    roads_all = gpd.GeoDataFrame(pd.concat(roads_parts, ignore_index=True), geometry="geometry")
    roads_all = roads_all.set_crs(4326, allow_override=True)
    roads_all = roads_all[roads_all.geometry.notnull()].copy()
    roads_all = roads_all[~roads_all.geometry.is_empty].copy()
    print(f"[compute] Major road segments loaded: {len(roads_all)}")
    return roads_all


def extract_point_geometries(geom: BaseGeometry) -> List[Point]:
    if geom.is_empty:
        return []
    if isinstance(geom, Point):
        return [geom]
    if isinstance(geom, MultiPoint):
        return [g for g in geom.geoms if not g.is_empty]
    if hasattr(geom, "geoms"):
        points: List[Point] = []
        for part in geom.geoms:
            points.extend(extract_point_geometries(part))
        return points
    return []


def build_border_control_points(
    roads: gpd.GeoDataFrame,
    shared_borders: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    print("[compute] Intersecting roads with shared borders")
    candidates = gpd.sjoin(
        roads,
        shared_borders,
        how="inner",
        predicate="intersects",
        lsuffix="road",
        rsuffix="border",
    )

    if candidates.empty:
        return gpd.GeoDataFrame(
            columns=["country1", "country2", "geometry"],
            geometry="geometry",
            crs=4326,
        )

    points_records = []
    borders_geom = shared_borders.geometry

    for road_idx, row in candidates.iterrows():
        border_idx = row["index_border"]
        road_geom = roads.loc[road_idx, "geometry"]
        border_geom = borders_geom.loc[border_idx]
        inter = road_geom.intersection(border_geom)

        for point in extract_point_geometries(inter):
            record = {
                "country1": row["country1"],
                "country2": row["country2"],
                "geometry": point,
            }
            if "fclass" in roads.columns:
                record["road_class"] = roads.loc[road_idx, "fclass"]
            if "osm_id" in roads.columns:
                record["osm_id"] = roads.loc[road_idx, "osm_id"]
            points_records.append(record)

    points_gdf = gpd.GeoDataFrame(points_records, geometry="geometry", crs=4326)
    if points_gdf.empty:
        return points_gdf

    # Dedupe points by pair and rounded coordinates.
    points_gdf["lon"] = points_gdf.geometry.x.round(6)
    points_gdf["lat"] = points_gdf.geometry.y.round(6)
    points_gdf = points_gdf.drop_duplicates(subset=["country1", "country2", "lon", "lat"]).copy()
    points_gdf.reset_index(drop=True, inplace=True)
    print(f"[compute] Border crossing points found: {len(points_gdf)}")
    return points_gdf


def main() -> None:
    args = parse_args()

    cache_dir = Path(args.cache_dir)
    output_path = Path(args.output)
    ensure_parent(output_path)

    boundaries_zip = download_if_missing(
        args.boundaries_url,
        cache_dir / "boundaries" / "ne_10m_admin_0_countries.zip",
        label="Natural Earth countries",
    )
    geofabrik_index_path = download_if_missing(
        args.geofabrik_index_url,
        cache_dir / "geofabrik" / "index-v1.json",
        label="Geofabrik index",
    )

    countries = get_subsaharan_countries(boundaries_zip)
    print(f"[info] Sub-Saharan countries loaded: {len(countries)}")

    shared_borders = build_shared_borders(countries)
    if shared_borders.empty:
        raise RuntimeError("No shared borders found. Check country boundaries input.")

    geofabrik_index = load_geofabrik_index(geofabrik_index_path)
    geofabrik_map = geofabrik_iso2_to_shp(geofabrik_index)

    roads = load_major_roads(
        countries=countries,
        geofabrik_map=geofabrik_map,
        cache_dir=cache_dir,
        major_classes=args.highways,
    )

    points = build_border_control_points(roads, shared_borders)
    if points.empty:
        print("[warn] No crossing points found with current filters.")

    points.to_file(output_path, layer=args.layer_name, driver="GPKG")
    print(f"[save] Points layer written to: {output_path} (layer={args.layer_name})")

    if args.save_shared_borders:
        shared_borders.to_file(output_path, layer="shared_borders", driver="GPKG")
        print("[save] Shared borders written to layer: shared_borders")


if __name__ == "__main__":
    main()
