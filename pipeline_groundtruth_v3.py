from geopy.geocoders import Nominatim
import osmnx as ox
import random
import csv
import math
import time
from math import ceil
import requests
import pandas as pd

TARGET_POI = {
    "amenity": [
        "cafe",
        "restaurant",
        "hospital",
        "fuel",
        "casino",
        "marketplace"
    ],
    "tourism": [
        "hotel",
        "attraction"
    ],
    "shop": [
        "supermarket"
    ]
}

OUTPUT_FIELDS = [
    "name",
    "category",
    "lat",
    "lon",
    "amenity",
    "addr:housenumber",
    "addr:street",
    "addr:suburb",
    "opening_hours",
    "website",
    "phone"
]

CATEGORY_DISPLAY_NAMES = {
    "cafe": "Cafe",
    "restaurant": "Restaurant",
    "hospital": "Hospital",
    "fuel": "Fuel Stop",
    "casino": "Casino",
    "marketplace": "Marketplace",
    "hotel": "Hotel",
    "attraction": "Attraction",
    "supermarket": "Supermarket",
    "unknown": "Place"
}

AMENITY_CATEGORIES = set(TARGET_POI["amenity"])
CATEGORY_SOURCE_FIELDS = ["amenity", "tourism", "shop"]

OSRM_BASE_URL = "http://router.project-osrm.org/route/v1/driving"

DEFAULT_MIN_CROWFLY_DISTANCE_M = 400
DEFAULT_MAX_CROWFLY_DISTANCE_M = 3200
DEFAULT_MIN_ROUTE_DISTANCE_M = 800
DEFAULT_MAX_ROUTE_DISTANCE_M = 4000

DEFAULT_MAX_DEST_CANDIDATES_PER_ORIGIN = 12
DEFAULT_MAX_VALID_PAIRS = 60
PAIR_REQUEST_SLEEP_SEC = 0.15

PREFERENCE_WORDS = [
    "quiet",
    "cheap",
    "convenient",
    "popular",
    "highly rated",
    "cozy"
]

DEFAULT_BUFFER_OPTIONS_KM = [0.5, 1.0, 1.5]

DEFAULT_GT_MAX_ALONG_ROUTE_DISTANCE_M = 800
DEFAULT_GT_MAX_VIA_WAYPOINT_ROUTE_DISTANCE_M = 1000
DEFAULT_GT_MAX_WAYPOINT_DISTANCE_M = 1500
GT_TOP_K = 5

SUBQUERY_FIELDS = [
    "subquery_group_id",
    "template_id",
    "template_name",
    "query_skeleton",
    "origin_subquery",
    "destination_subquery",
    "route_subquery",
    "waypoint_subquery",
    "buffer_subquery",
    "poi_category_subquery",
    "preference_subquery",
    "origin_id",
    "origin_name",
    "origin_category",
    "origin_lat",
    "origin_lon",
    "dest_id",
    "dest_name",
    "dest_category",
    "dest_lat",
    "dest_lon",
    "waypoint_id",
    "waypoint_name",
    "waypoint_category",
    "waypoint_lat",
    "waypoint_lon",
    "target_category",
    "preference",
    "buffer_km",
    "route_distance_m",
    "route_duration_s"
]

QUERY_FIELDS = [
    "query_id",
    "subquery_group_id",
    "template_id",
    "template_name",
    "query_text",
    "query_skeleton",
    "origin_id",
    "origin_name",
    "origin_category",
    "origin_lat",
    "origin_lon",
    "dest_id",
    "dest_name",
    "dest_category",
    "dest_lat",
    "dest_lon",
    "waypoint_id",
    "waypoint_name",
    "waypoint_category",
    "waypoint_lat",
    "waypoint_lon",
    "target_category",
    "preference",
    "buffer_km",
    "route_distance_m",
    "route_duration_s"
]

GROUND_TRUTH_FIELDS = [
    "ground_truth_id",
    "query_id",
    "subquery_group_id",
    "template_id",
    "template_name",
    "query_text",
    "origin_id",
    "origin_name",
    "origin_lat",
    "origin_lon",
    "dest_id",
    "dest_name",
    "dest_lat",
    "dest_lon",
    "waypoint_id",
    "waypoint_name",
    "waypoint_lat",
    "waypoint_lon",
    "target_category",
    "preference",
    "buffer_km",
    "candidate_count",
    "route_distance_m",
    "route_duration_s"
]

for rank in range(1, GT_TOP_K + 1):
    GROUND_TRUTH_FIELDS.extend([
        f"ground_truth_top{rank}_id",
        f"ground_truth_top{rank}_name",
        f"ground_truth_top{rank}_category",
        f"ground_truth_top{rank}_lat",
        f"ground_truth_top{rank}_lon",
        f"ground_truth_top{rank}_score",
        f"ground_truth_top{rank}_distance_to_route_m",
        f"ground_truth_top{rank}_distance_to_waypoint_m",
        f"ground_truth_top{rank}_preference_score",
        f"ground_truth_top{rank}_metadata_quality_score",
        f"ground_truth_top{rank}_quiet_score",
        f"ground_truth_top{rank}_cheap_score",
        f"ground_truth_top{rank}_convenient_score",
        f"ground_truth_top{rank}_popular_score",
        f"ground_truth_top{rank}_highly_rated_score",
        f"ground_truth_top{rank}_cozy_score",
        f"ground_truth_top{rank}_synthetic_rating",
        f"ground_truth_top{rank}_synthetic_price_level"
    ])




def estimate_bbox_dimensions_m(polygon):
    minx, miny, maxx, maxy = polygon.bounds
    width_m = haversine_meters(miny, minx, miny, maxx)
    height_m = haversine_meters(miny, minx, maxy, minx)
    diagonal_m = haversine_meters(miny, minx, maxy, maxx)
    return width_m, height_m, diagonal_m


def estimate_median_nearest_neighbor_distance(points, max_points=300):
    if len(points) < 2:
        return 500.0

    sampled = points[:max_points]
    nearest = []
    for i, p in enumerate(sampled):
        best = float("inf")
        for j, q in enumerate(sampled):
            if i == j:
                continue
            dist = haversine_meters(p["lat"], p["lon"], q["lat"], q["lon"])
            if dist < best:
                best = dist
        if math.isfinite(best):
            nearest.append(best)

    if not nearest:
        return 500.0
    nearest.sort()
    return nearest[len(nearest) // 2]


def build_adaptive_config(polygon, raw_points_count, cleaned_points_count, requested_sample_size=None):
    width_m, height_m, diagonal_m = estimate_bbox_dimensions_m(polygon)
    bbox_area_km2 = max(0.01, (width_m * height_m) / 1_000_000.0)

    if requested_sample_size is None:
        if bbox_area_km2 <= 25:
            sample_size = min(cleaned_points_count, 120)
        elif bbox_area_km2 <= 250:
            sample_size = min(cleaned_points_count, 220)
        elif bbox_area_km2 <= 1500:
            sample_size = min(cleaned_points_count, 320)
        else:
            sample_size = min(cleaned_points_count, 450)
    else:
        sample_size = min(cleaned_points_count, requested_sample_size)

    density_factor = min(1.8, max(0.8, diagonal_m / 20000.0))

    min_crowfly = max(250, min(1500, int(diagonal_m * 0.01)))
    max_crowfly = max(min_crowfly + 800, min(12000, int(diagonal_m * 0.08)))
    min_route = max(500, int(min_crowfly * 1.4))
    max_route = max(min_route + 1000, min(15000, int(max_crowfly * 1.5)))

    gt_along = min(2000, max(700, int(600 * density_factor)))
    gt_via_route = min(2500, max(900, int(gt_along * 1.25)))
    gt_waypoint = min(3000, max(1200, int(gt_along * 1.7)))

    if diagonal_m <= 12000:
        buffer_options = [0.5, 1.0, 1.5]
    elif diagonal_m <= 40000:
        buffer_options = [1.0, 1.5, 2.0]
    else:
        buffer_options = [1.5, 2.0, 3.0]

    max_dest_candidates_per_origin = min(25, max(10, int(12 * density_factor)))
    max_valid_pairs = min(180, max(60, int(sample_size * 0.6)))

    return {
        "bbox_width_m": round(width_m, 2),
        "bbox_height_m": round(height_m, 2),
        "bbox_diagonal_m": round(diagonal_m, 2),
        "bbox_area_km2": round(bbox_area_km2, 2),
        "sample_size": int(sample_size),
        "min_crowfly_distance_m": int(min_crowfly),
        "max_crowfly_distance_m": int(max_crowfly),
        "min_route_distance_m": int(min_route),
        "max_route_distance_m": int(max_route),
        "gt_max_along_route_distance_m": int(gt_along),
        "gt_max_via_waypoint_route_distance_m": int(gt_via_route),
        "gt_max_waypoint_distance_m": int(gt_waypoint),
        "buffer_options_km": buffer_options,
        "max_dest_candidates_per_origin": int(max_dest_candidates_per_origin),
        "max_valid_pairs": int(max_valid_pairs),
    }


def estimate_required_valid_pairs(target_query_count, templates_per_pair=3):
    if target_query_count is None:
        return None
    return max(1, int(ceil(float(target_query_count) / max(1, templates_per_pair))))


def maybe_expand_sampling_budget(config, target_query_count, cleaned_points_count):
    if target_query_count is None:
        return config

    required_pairs = estimate_required_valid_pairs(target_query_count, templates_per_pair=3)

    # 给更大的 query 目标更多点位与 pair 配额，避免采样过小导致 pair 不够。
    if target_query_count <= 600:
        min_sample_size = 220
    elif target_query_count <= 1200:
        min_sample_size = 380
    elif target_query_count <= 2000:
        min_sample_size = 520
    else:
        min_sample_size = 700

    config = config.copy()
    config["sample_size"] = int(min(cleaned_points_count, max(config["sample_size"], min_sample_size)))
    config["max_valid_pairs"] = int(max(config["max_valid_pairs"], required_pairs))
    config["max_dest_candidates_per_origin"] = int(max(config["max_dest_candidates_per_origin"], min(40, required_pairs // 8 + 12)))
    return config


def corridor_category_pool(points, excluded_ids, route_geometry, route_limit_m, waypoint=None, waypoint_limit_m=None):
    category_counts = {}

    for point in points:
        if point["point_id"] in excluded_ids:
            continue

        category = normalize_value(point.get("category", ""))
        if not category:
            continue

        distance_to_route_m = min_distance_to_route_m(route_geometry, point["lat"], point["lon"])
        if math.isinf(distance_to_route_m) or distance_to_route_m > route_limit_m:
            continue

        if waypoint is not None and waypoint_limit_m is not None:
            distance_to_waypoint_m = haversine_meters(point["lat"], point["lon"], waypoint["lat"], waypoint["lon"])
            if distance_to_waypoint_m > waypoint_limit_m:
                continue

        category_counts[category] = category_counts.get(category, 0) + 1

    return category_counts


def choose_target_category_for_pair(points, excluded_ids, route_geometry, config, rng, waypoint=None, buffer_km=None):
    if waypoint is not None:
        category_counts = corridor_category_pool(
            points,
            excluded_ids=excluded_ids,
            route_geometry=route_geometry,
            route_limit_m=config["gt_max_via_waypoint_route_distance_m"],
            waypoint=waypoint,
            waypoint_limit_m=config["gt_max_waypoint_distance_m"]
        )
    elif buffer_km is not None:
        category_counts = corridor_category_pool(
            points,
            excluded_ids=excluded_ids,
            route_geometry=route_geometry,
            route_limit_m=max(500.0, float(buffer_km) * 1000.0)
        )
    else:
        category_counts = corridor_category_pool(
            points,
            excluded_ids=excluded_ids,
            route_geometry=route_geometry,
            route_limit_m=config["gt_max_along_route_distance_m"]
        )

    if category_counts:
        sorted_items = sorted(category_counts.items(), key=lambda x: (-x[1], x[0]))
        top_pool = [cat for cat, _ in sorted_items[: min(5, len(sorted_items))]]
        return rng.choice(top_pool)

    return choose_target_category(points, excluded_ids=excluded_ids, seed=rng.randint(1, 10**9))

def category_to_generic_phrase(category):
    category = normalize_value(category).lower()
    if not category or category == "unknown":
        return "a place"

    article_map = {
        "attraction": "an attraction"
    }
    return article_map.get(category, f"a {category}")


def resolve_place_reference(name, category, generic_place_references=False):
    if generic_place_references:
        return category_to_generic_phrase(category)

    name = normalize_value(name)
    if name:
        return name

    return category_to_generic_phrase(category)


def get_region_boundary(place):
    geolocator = Nominatim(user_agent="geo_dataset")
    location = geolocator.geocode(place)

    if location is None:
        raise ValueError(f"Cannot find place: {place}")

    gdf = ox.geocode_to_gdf(place)
    if gdf.empty:
        raise ValueError(f"Cannot get boundary for place: {place}")

    return gdf.geometry.iloc[0]


def get_road_network(polygon):
    G = ox.graph_from_polygon(polygon, network_type="drive")
    return G


def get_poi(polygon):
    poi = ox.features_from_polygon(polygon, tags=TARGET_POI)
    if poi.empty:
        raise ValueError("No POIs found in this region.")
    return poi


def normalize_value(value):
    if pd.isna(value):
        return ""
    if isinstance(value, str):
        value = value.strip()
        if value.lower() in {"nan", "none", "null"}:
            return ""
    return value


def extract_points_with_meta(poi):
    results = []

    for _, row in poi.iterrows():
        geom = row.geometry
        if geom is None:
            continue

        if geom.geom_type == "Point":
            lat, lon = geom.y, geom.x
        elif geom.geom_type in ["Polygon", "MultiPolygon"]:
            c = geom.centroid
            lat, lon = c.y, c.x
        else:
            continue

        category = "unknown"
        for key in TARGET_POI.keys():
            if key in row.index and pd.notna(row[key]):
                category = normalize_value(row[key])
                break

        item = {
            "name": normalize_value(row.get("name", "")),
            "category": normalize_value(category),
            "lat": lat,
            "lon": lon,
            "amenity": normalize_value(row.get("amenity", "")),
            "tourism": normalize_value(row.get("tourism", "")),
            "shop": normalize_value(row.get("shop", "")),
            "addr:housenumber": normalize_value(row.get("addr:housenumber", "")),
            "addr:street": normalize_value(row.get("addr:street", "")),
            "addr:suburb": normalize_value(row.get("addr:suburb", "")),
            "opening_hours": normalize_value(row.get("opening_hours", "")),
            "website": normalize_value(row.get("website", "")),
            "phone": normalize_value(row.get("phone", ""))
        }
        results.append(item)

    return results


def choose_better_text(v1, v2):
    v1 = normalize_value(v1)
    v2 = normalize_value(v2)
    if not v1:
        return v2
    if not v2:
        return v1
    return v1 if len(str(v1)) >= len(str(v2)) else v2


def merge_duplicate_group(group):
    merged = group[0].copy()

    for point in group[1:]:
        for field in [
            "name", "category", "amenity", "tourism", "shop",
            "addr:housenumber", "addr:street", "addr:suburb",
            "opening_hours", "website", "phone"
        ]:
            merged[field] = choose_better_text(merged.get(field, ""), point.get(field, ""))

    merged["lat"] = sum(p["lat"] for p in group) / len(group)
    merged["lon"] = sum(p["lon"] for p in group) / len(group)
    return merged


def deduplicate_points(points, precision=6):
    grouped = {}
    for p in points:
        key = (round(p["lat"], precision), round(p["lon"], precision))
        grouped.setdefault(key, []).append(p)

    unique_points = [merge_duplicate_group(group) for group in grouped.values()]
    return unique_points


def infer_category(point):
    current_category = normalize_value(point.get("category", ""))
    if current_category and current_category != "unknown":
        return current_category

    for source_field in CATEGORY_SOURCE_FIELDS:
        source_value = normalize_value(point.get(source_field, ""))
        if source_value:
            return source_value

    return "unknown"


def sync_category_and_amenity(point):
    point["category"] = infer_category(point)

    if not normalize_value(point.get("amenity", "")) and point["category"] in AMENITY_CATEGORIES:
        point["amenity"] = point["category"]

    return point


def haversine_meters(lat1, lon1, lat2, lon2):
    r = 6371000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def fill_address_by_neighbors(points, target_field, radius_meters=120, k=5, min_votes=2):
    for point in points:
        if normalize_value(point.get(target_field, "")):
            continue

        candidates = []
        for other in points:
            if other is point:
                continue
            field_value = normalize_value(other.get(target_field, ""))
            if not field_value:
                continue

            dist = haversine_meters(point["lat"], point["lon"], other["lat"], other["lon"])
            if dist <= radius_meters:
                candidates.append((dist, field_value))

        if not candidates:
            continue

        candidates.sort(key=lambda x: x[0])
        nearest_values = [value for _, value in candidates[:k]]

        counts = {}
        for value in nearest_values:
            counts[value] = counts.get(value, 0) + 1

        best_value, best_count = max(counts.items(), key=lambda x: x[1])
        if best_count >= min_votes:
            point[target_field] = best_value

    return points


def build_dynamic_virtual_name(point, category_counters):
    category = normalize_value(point.get("category", "unknown")).lower() or "unknown"
    category_label = CATEGORY_DISPLAY_NAMES.get(category, category.replace("_", " ").title() if category else "Place")

    suburb = normalize_value(point.get("addr:suburb", ""))
    street = normalize_value(point.get("addr:street", ""))

    category_counters[category] = category_counters.get(category, 0) + 1
    suffix = category_counters[category]

    if suburb:
        return f"{suburb} {category_label} {suffix}"
    if street:
        return f"{street} {category_label} {suffix}"
    return f"Virtual {category_label} {suffix}"


def fill_missing_names(points, seed=42):
    category_counters = {}

    for point in points:
        name = normalize_value(point.get("name", ""))
        if name:
            continue
        point["name"] = build_dynamic_virtual_name(point, category_counters)

    return points


def add_synthetic_semantic_scores(points, seed=42):
    category_bias = {
        "cafe": {"quiet_score": 0.70, "cheap_score": 0.55, "convenient_score": 0.65, "popular_score": 0.62, "highly_rated_score": 0.72, "cozy_score": 0.82},
        "restaurant": {"quiet_score": 0.45, "cheap_score": 0.42, "convenient_score": 0.60, "popular_score": 0.74, "highly_rated_score": 0.70, "cozy_score": 0.63},
        "hospital": {"quiet_score": 0.35, "cheap_score": 0.20, "convenient_score": 0.88, "popular_score": 0.58, "highly_rated_score": 0.66, "cozy_score": 0.20},
        "fuel": {"quiet_score": 0.20, "cheap_score": 0.62, "convenient_score": 0.90, "popular_score": 0.60, "highly_rated_score": 0.52, "cozy_score": 0.10},
        "casino": {"quiet_score": 0.10, "cheap_score": 0.18, "convenient_score": 0.48, "popular_score": 0.72, "highly_rated_score": 0.50, "cozy_score": 0.28},
        "marketplace": {"quiet_score": 0.25, "cheap_score": 0.80, "convenient_score": 0.68, "popular_score": 0.70, "highly_rated_score": 0.58, "cozy_score": 0.36},
        "hotel": {"quiet_score": 0.66, "cheap_score": 0.26, "convenient_score": 0.72, "popular_score": 0.64, "highly_rated_score": 0.74, "cozy_score": 0.78},
        "attraction": {"quiet_score": 0.32, "cheap_score": 0.38, "convenient_score": 0.56, "popular_score": 0.82, "highly_rated_score": 0.68, "cozy_score": 0.42},
        "supermarket": {"quiet_score": 0.22, "cheap_score": 0.78, "convenient_score": 0.86, "popular_score": 0.68, "highly_rated_score": 0.60, "cozy_score": 0.20},
        "unknown": {"quiet_score": 0.50, "cheap_score": 0.50, "convenient_score": 0.50, "popular_score": 0.50, "highly_rated_score": 0.50, "cozy_score": 0.50},
    }

    for idx, point in enumerate(points, start=1):
        category = normalize_value(point.get("category", "unknown")) or "unknown"
        base = category_bias.get(category, category_bias["unknown"])
        rng = random.Random(seed * 100003 + idx * 97)

        for field, center in base.items():
            noise = rng.uniform(-0.15, 0.15)
            point[field] = round(min(1.0, max(0.0, center + noise)), 4)

        point["synthetic_rating"] = round(2.8 + 2.1 * point["highly_rated_score"], 2)
        point["synthetic_price_level"] = max(1, min(4, int(round(4.5 - 3.5 * point["cheap_score"]))))
        point["metadata_quality_score"] = round(
            0.25 * (1.0 if normalize_value(point.get("opening_hours", "")) else 0.0) +
            0.20 * (1.0 if normalize_value(point.get("website", "")) else 0.0) +
            0.20 * (1.0 if normalize_value(point.get("phone", "")) else 0.0) +
            0.15 * (1.0 if normalize_value(point.get("addr:street", "")) else 0.0) +
            0.20 * (1.0 if normalize_value(point.get("addr:suburb", "")) else 0.0),
            4
        )

    return points


def clean_points(points, seed=42):
    print("[INFO] Deduplicating and merging duplicate points...")
    points = deduplicate_points(points)
    print(f"[INFO] Unique points after merge: {len(points)}")

    print("[INFO] Syncing category-related fields...")
    points = [sync_category_and_amenity(point) for point in points]

    print("[INFO] Filling missing suburb from neighbors...")
    points = fill_address_by_neighbors(points, target_field="addr:suburb", radius_meters=200, k=5, min_votes=2)

    print("[INFO] Filling missing street from neighbors...")
    points = fill_address_by_neighbors(points, target_field="addr:street", radius_meters=120, k=5, min_votes=2)

    print("[INFO] Filling missing names from virtual name pool...")
    points = fill_missing_names(points, seed=seed)

    print("[INFO] Adding synthetic semantic scores for downstream ground-truth ranking...")
    points = add_synthetic_semantic_scores(points, seed=seed)

    return points


def sample_points(points, n=100, seed=42):
    random.seed(seed)
    if len(points) <= n:
        return points
    return random.sample(points, n)


def prepare_points_for_csv(points):
    cleaned_for_csv = []
    for p in points:
        row = {field: p.get(field, "") for field in OUTPUT_FIELDS}
        cleaned_for_csv.append(row)
    return cleaned_for_csv


def save_points_to_csv(points, output_file):
    rows = prepare_points_for_csv(points)
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def add_point_ids(points):
    for idx, point in enumerate(points, start=1):
        point["point_id"] = idx
    return points


def get_osrm_route_info(origin_lon, origin_lat, dest_lon, dest_lat, timeout=10, overview="false", annotations="false"):
    url = (
        f"{OSRM_BASE_URL}/"
        f"{origin_lon},{origin_lat};{dest_lon},{dest_lat}"
        f"?overview={overview}&geometries=geojson&annotations={annotations}"
    )

    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        data = r.json()

        if data.get("code") != "Ok":
            return None

        routes = data.get("routes", [])
        if not routes:
            return None

        route = routes[0]
        geometry = route.get("geometry", {})
        coordinates = geometry.get("coordinates", []) if isinstance(geometry, dict) else []
        return {
            "route_distance_m": route.get("distance"),
            "route_duration_s": route.get("duration"),
            "route_geometry": coordinates
        }
    except Exception:
        return None


def generate_candidate_pairs(points, config, seed=42, max_dest_candidates_per_origin=None):
    rng = random.Random(seed)
    pair_rows = []
    if max_dest_candidates_per_origin is None:
        max_dest_candidates_per_origin = config["max_dest_candidates_per_origin"]


    for i, origin in enumerate(points):
        local_candidates = []

        for j, dest in enumerate(points):
            if i == j:
                continue

            crowfly_m = haversine_meters(origin["lat"], origin["lon"], dest["lat"], dest["lon"])
            if config["min_crowfly_distance_m"] <= crowfly_m <= config["max_crowfly_distance_m"]:
                local_candidates.append((dest, crowfly_m))

        if len(local_candidates) > max_dest_candidates_per_origin:
            local_candidates = rng.sample(local_candidates, max_dest_candidates_per_origin)

        for dest, crowfly_m in local_candidates:
            pair_rows.append({
                "origin_id": origin["point_id"],
                "origin_name": origin["name"],
                "origin_category": origin["category"],
                "origin_lat": origin["lat"],
                "origin_lon": origin["lon"],
                "dest_id": dest["point_id"],
                "dest_name": dest["name"],
                "dest_category": dest["category"],
                "dest_lat": dest["lat"],
                "dest_lon": dest["lon"],
                "crowfly_distance_m": round(crowfly_m, 2)
            })

    seen = set()
    unique_rows = []
    for row in pair_rows:
        key = (row["origin_id"], row["dest_id"])
        if key not in seen:
            seen.add(key)
            unique_rows.append(row)

    return unique_rows


def filter_valid_pairs_by_osrm(candidate_pairs, config, max_valid_pairs=None):
    valid_rows = []
    if max_valid_pairs is None:
        max_valid_pairs = config["max_valid_pairs"]

    for row in candidate_pairs:
        route_info = get_osrm_route_info(
            row["origin_lon"], row["origin_lat"],
            row["dest_lon"], row["dest_lat"],
            overview="full"
        )

        if route_info is None:
            continue

        route_distance_m = route_info["route_distance_m"]
        route_duration_s = route_info["route_duration_s"]

        if route_distance_m is None or route_duration_s is None:
            continue

        if config["min_route_distance_m"] <= route_distance_m <= config["max_route_distance_m"]:
            enriched = row.copy()
            enriched["route_distance_m"] = round(route_distance_m, 2)
            enriched["route_duration_s"] = round(route_duration_s, 2)
            enriched["route_geometry"] = route_info.get("route_geometry", [])
            valid_rows.append(enriched)

        if len(valid_rows) >= max_valid_pairs:
            break

        time.sleep(PAIR_REQUEST_SLEEP_SEC)

    return valid_rows


def choose_waypoint(points, excluded_ids, seed=None):
    candidates = [p for p in points if p["point_id"] not in excluded_ids]
    if not candidates:
        return None

    rng = random.Random(seed)
    return rng.choice(candidates)


def choose_target_category(points, excluded_ids, seed=None):
    candidates = [p for p in points if p["point_id"] not in excluded_ids]
    if not candidates:
        candidates = points

    categories = [normalize_value(p.get("category", "")) for p in candidates]
    categories = [c for c in categories if c]
    if not categories:
        return "place"

    rng = random.Random(seed)
    return rng.choice(categories)


def build_template_1(origin_name, dest_name, target_category, preference):
    return f"Find me a {preference} {target_category} along the way from {origin_name} to {dest_name}."


def build_template_2(origin_name, dest_name, waypoint_name, target_category, preference):
    return (
        f"Find me a {preference} {target_category} along the way from "
        f"{origin_name} to {dest_name}, passing through {waypoint_name}."
    )


def build_template_3(origin_name, dest_name, target_category, preference, buffer_km):
    return (
        f"Find me a {preference} {target_category} along the way from "
        f"{origin_name} to {dest_name}, within {buffer_km}km of the route."
    )


def make_subquery_record(
    subquery_group_id,
    template_id,
    template_name,
    origin,
    dest,
    target_category,
    preference,
    route_distance_m,
    route_duration_s,
    waypoint=None,
    buffer_km=""
):
    query_skeleton = (
        f"origin={origin['point_id']} | destination={dest['point_id']} | "
        f"poi_type={target_category} | preference={preference}"
    )

    waypoint_subquery = ""
    waypoint_id = ""
    waypoint_name = ""
    waypoint_category = ""
    waypoint_lat = ""
    waypoint_lon = ""

    if waypoint is not None:
        waypoint_subquery = f"waypoint = point_{waypoint['point_id']}"
        waypoint_id = waypoint["point_id"]
        waypoint_name = waypoint["name"]
        waypoint_category = waypoint.get("category", "")
        waypoint_lat = waypoint["lat"]
        waypoint_lon = waypoint["lon"]
        query_skeleton += f" | waypoint={waypoint['point_id']}"

    buffer_subquery = ""
    if buffer_km != "":
        buffer_subquery = f"buffer = {buffer_km} km"
        query_skeleton += f" | buffer={buffer_km}km"

    return {
        "subquery_group_id": subquery_group_id,
        "template_id": template_id,
        "template_name": template_name,
        "query_skeleton": query_skeleton,
        "origin_subquery": f"origin = point_{origin['point_id']}",
        "destination_subquery": f"destination = point_{dest['point_id']}",
        "route_subquery": f"route = route(point_{origin['point_id']}, point_{dest['point_id']})",
        "waypoint_subquery": waypoint_subquery,
        "buffer_subquery": buffer_subquery,
        "poi_category_subquery": f"poi_type = {target_category}",
        "preference_subquery": f"preference = {preference}",
        "origin_id": origin["point_id"],
        "origin_name": origin["name"],
        "origin_category": origin.get("category", ""),
        "origin_lat": origin["lat"],
        "origin_lon": origin["lon"],
        "dest_id": dest["point_id"],
        "dest_name": dest["name"],
        "dest_category": dest.get("category", ""),
        "dest_lat": dest["lat"],
        "dest_lon": dest["lon"],
        "waypoint_id": waypoint_id,
        "waypoint_name": waypoint_name,
        "waypoint_category": waypoint_category,
        "waypoint_lat": waypoint_lat,
        "waypoint_lon": waypoint_lon,
        "target_category": target_category,
        "preference": preference,
        "buffer_km": buffer_km,
        "route_distance_m": route_distance_m,
        "route_duration_s": route_duration_s
    }


def generate_subquery_records_from_pairs(valid_pairs, points, config, seed=42):
    rng = random.Random(seed)
    point_lookup = {p["point_id"]: p for p in points}
    subquery_rows = []
    subquery_group_id = 1

    for row in valid_pairs:
        origin = point_lookup[row["origin_id"]]
        dest = point_lookup[row["dest_id"]]
        route_geometry = row.get("route_geometry", [])
        excluded = {origin["point_id"], dest["point_id"]}
        preference = rng.choice(PREFERENCE_WORDS)

        target_category_1 = choose_target_category_for_pair(
            points,
            excluded_ids=excluded,
            route_geometry=route_geometry,
            config=config,
            rng=rng
        )
        subquery_rows.append(
            make_subquery_record(
                subquery_group_id=subquery_group_id,
                template_id=1,
                template_name="along_route",
                origin=origin,
                dest=dest,
                target_category=target_category_1,
                preference=preference,
                route_distance_m=row["route_distance_m"],
                route_duration_s=row["route_duration_s"]
            )
        )
        subquery_group_id += 1

        waypoint = choose_waypoint(points, excluded_ids=excluded, seed=rng.randint(1, 10**9))
        if waypoint is not None:
            target_category_2 = choose_target_category_for_pair(
                points,
                excluded_ids=excluded | {waypoint["point_id"]},
                route_geometry=route_geometry,
                config=config,
                rng=rng,
                waypoint=waypoint
            )
            subquery_rows.append(
                make_subquery_record(
                    subquery_group_id=subquery_group_id,
                    template_id=2,
                    template_name="via_waypoint",
                    origin=origin,
                    dest=dest,
                    target_category=target_category_2,
                    preference=preference,
                    route_distance_m=row["route_distance_m"],
                    route_duration_s=row["route_duration_s"],
                    waypoint=waypoint
                )
            )
            subquery_group_id += 1

        buffer_km = rng.choice(config["buffer_options_km"])
        target_category_3 = choose_target_category_for_pair(
            points,
            excluded_ids=excluded,
            route_geometry=route_geometry,
            config=config,
            rng=rng,
            buffer_km=buffer_km
        )
        subquery_rows.append(
            make_subquery_record(
                subquery_group_id=subquery_group_id,
                template_id=3,
                template_name="route_buffer",
                origin=origin,
                dest=dest,
                target_category=target_category_3,
                preference=preference,
                route_distance_m=row["route_distance_m"],
                route_duration_s=row["route_duration_s"],
                buffer_km=buffer_km
            )
        )
        subquery_group_id += 1

    return subquery_rows


def render_queries_from_subqueries(subquery_rows, generic_place_references=False):
    query_rows = []
    query_id = 1

    for row in subquery_rows:
        origin_ref = resolve_place_reference(
            row["origin_name"],
            row.get("origin_category", ""),
            generic_place_references=generic_place_references
        )
        dest_ref = resolve_place_reference(
            row["dest_name"],
            row.get("dest_category", ""),
            generic_place_references=generic_place_references
        )

        if row["template_name"] == "along_route":
            query_text = build_template_1(
                origin_ref,
                dest_ref,
                row["target_category"],
                row["preference"]
            )
        elif row["template_name"] == "via_waypoint":
            waypoint_ref = resolve_place_reference(
                row["waypoint_name"],
                row.get("waypoint_category", ""),
                generic_place_references=generic_place_references
            )
            query_text = build_template_2(
                origin_ref,
                dest_ref,
                waypoint_ref,
                row["target_category"],
                row["preference"]
            )
        elif row["template_name"] == "route_buffer":
            query_text = build_template_3(
                origin_ref,
                dest_ref,
                row["target_category"],
                row["preference"],
                row["buffer_km"]
            )
        else:
            raise ValueError(f"Unknown template_name: {row['template_name']}")

        query_row = {field: row.get(field, "") for field in QUERY_FIELDS if field not in {"query_id", "query_text"}}
        query_row["query_id"] = query_id
        query_row["query_text"] = query_text
        query_rows.append(query_row)
        query_id += 1

    return query_rows


def equirectangular_xy_m(lat, lon, ref_lat):
    x = math.radians(lon) * 6371000 * math.cos(math.radians(ref_lat))
    y = math.radians(lat) * 6371000
    return x, y


def point_to_segment_distance_m(point_lat, point_lon, a_lat, a_lon, b_lat, b_lon, ref_lat):
    px, py = equirectangular_xy_m(point_lat, point_lon, ref_lat)
    ax, ay = equirectangular_xy_m(a_lat, a_lon, ref_lat)
    bx, by = equirectangular_xy_m(b_lat, b_lon, ref_lat)

    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    denom = abx * abx + aby * aby

    if denom == 0:
        return math.hypot(px - ax, py - ay)

    t = max(0.0, min(1.0, (apx * abx + apy * aby) / denom))
    cx = ax + t * abx
    cy = ay + t * aby
    return math.hypot(px - cx, py - cy)


def min_distance_to_route_m(route_geometry, point_lat, point_lon):
    if not route_geometry or len(route_geometry) < 2:
        return float("inf")

    ref_lat = point_lat
    min_dist = float("inf")

    for i in range(len(route_geometry) - 1):
        a_lon, a_lat = route_geometry[i]
        b_lon, b_lat = route_geometry[i + 1]
        dist = point_to_segment_distance_m(point_lat, point_lon, a_lat, a_lon, b_lat, b_lon, ref_lat)
        if dist < min_dist:
            min_dist = dist

    return min_dist


def preference_to_score(point, preference):
    preference_map = {
        "quiet": point.get("quiet_score", 0.0),
        "cheap": point.get("cheap_score", 0.0),
        "convenient": point.get("convenient_score", 0.0),
        "popular": point.get("popular_score", 0.0),
        "highly rated": point.get("highly_rated_score", 0.0),
        "cozy": point.get("cozy_score", 0.0)
    }
    return float(preference_map.get(preference, 0.0))


def get_candidate_points_for_subquery(subquery, points, pair_lookup, config):
    pair_key = (subquery["origin_id"], subquery["dest_id"])
    pair_row = pair_lookup.get(pair_key)
    if pair_row is None:
        return []

    route_geometry = pair_row.get("route_geometry", [])
    target_category = normalize_value(subquery.get("target_category", ""))
    template_name = subquery.get("template_name", "")
    excluded_ids = {subquery["origin_id"], subquery["dest_id"]}
    if subquery.get("waypoint_id", "") != "":
        excluded_ids.add(subquery["waypoint_id"])

    candidates = []

    for point in points:
        if point["point_id"] in excluded_ids:
            continue
        if normalize_value(point.get("category", "")) != target_category:
            continue

        distance_to_route_m = min_distance_to_route_m(route_geometry, point["lat"], point["lon"])
        if math.isinf(distance_to_route_m):
            continue

        distance_to_waypoint_m = ""
        allowed = False

        if template_name == "along_route":
            allowed = distance_to_route_m <= config["gt_max_along_route_distance_m"]

        elif template_name == "via_waypoint":
            if subquery.get("waypoint_lat", "") != "" and subquery.get("waypoint_lon", "") != "":
                distance_to_waypoint_m = haversine_meters(
                    point["lat"], point["lon"],
                    float(subquery["waypoint_lat"]), float(subquery["waypoint_lon"])
                )
                allowed = (
                    distance_to_route_m <= config["gt_max_via_waypoint_route_distance_m"] and
                    distance_to_waypoint_m <= config["gt_max_waypoint_distance_m"]
                )

        elif template_name == "route_buffer":
            buffer_limit_m = float(subquery.get("buffer_km", 0)) * 1000.0
            allowed = distance_to_route_m <= buffer_limit_m

        if not allowed:
            continue

        candidate = point.copy()
        candidate["distance_to_route_m"] = round(distance_to_route_m, 2)
        candidate["distance_to_waypoint_m"] = "" if distance_to_waypoint_m == "" else round(distance_to_waypoint_m, 2)
        candidates.append(candidate)

    return candidates


def enrich_candidate_with_scoring_details(subquery, candidate, config):
    candidate = candidate.copy()
    candidate["preference_score"] = round(preference_to_score(candidate, subquery["preference"]), 6)
    candidate["metadata_quality_score"] = round(float(candidate.get("metadata_quality_score", 0.0)), 6)
    candidate["ground_truth_score"] = score_candidate_for_subquery(subquery, candidate, config)
    return candidate


def flatten_topk_candidates(top_candidates, top_k=GT_TOP_K):
    flattened = {}

    for rank in range(1, top_k + 1):
        candidate = top_candidates[rank - 1] if rank <= len(top_candidates) else None

        flattened[f"ground_truth_top{rank}_id"] = "" if candidate is None else candidate.get("point_id", "")
        flattened[f"ground_truth_top{rank}_name"] = "" if candidate is None else candidate.get("name", "")
        flattened[f"ground_truth_top{rank}_category"] = "" if candidate is None else candidate.get("category", "")
        flattened[f"ground_truth_top{rank}_lat"] = "" if candidate is None else candidate.get("lat", "")
        flattened[f"ground_truth_top{rank}_lon"] = "" if candidate is None else candidate.get("lon", "")
        flattened[f"ground_truth_top{rank}_score"] = "" if candidate is None else round(candidate.get("ground_truth_score", 0.0), 6)
        flattened[f"ground_truth_top{rank}_distance_to_route_m"] = "" if candidate is None else candidate.get("distance_to_route_m", "")
        flattened[f"ground_truth_top{rank}_distance_to_waypoint_m"] = "" if candidate is None else candidate.get("distance_to_waypoint_m", "")
        flattened[f"ground_truth_top{rank}_preference_score"] = "" if candidate is None else candidate.get("preference_score", "")
        flattened[f"ground_truth_top{rank}_metadata_quality_score"] = "" if candidate is None else candidate.get("metadata_quality_score", "")
        flattened[f"ground_truth_top{rank}_quiet_score"] = "" if candidate is None else candidate.get("quiet_score", "")
        flattened[f"ground_truth_top{rank}_cheap_score"] = "" if candidate is None else candidate.get("cheap_score", "")
        flattened[f"ground_truth_top{rank}_convenient_score"] = "" if candidate is None else candidate.get("convenient_score", "")
        flattened[f"ground_truth_top{rank}_popular_score"] = "" if candidate is None else candidate.get("popular_score", "")
        flattened[f"ground_truth_top{rank}_highly_rated_score"] = "" if candidate is None else candidate.get("highly_rated_score", "")
        flattened[f"ground_truth_top{rank}_cozy_score"] = "" if candidate is None else candidate.get("cozy_score", "")
        flattened[f"ground_truth_top{rank}_synthetic_rating"] = "" if candidate is None else candidate.get("synthetic_rating", "")
        flattened[f"ground_truth_top{rank}_synthetic_price_level"] = "" if candidate is None else candidate.get("synthetic_price_level", "")

    return flattened


def score_candidate_for_subquery(subquery, candidate, config):
    preference_score = preference_to_score(candidate, subquery["preference"])
    metadata_quality = float(candidate.get("metadata_quality_score", 0.0))

    if subquery["template_name"] == "along_route":
        max_dist = config["gt_max_along_route_distance_m"]
        route_closeness = max(0.0, 1.0 - candidate["distance_to_route_m"] / max_dist)
        score = 0.50 * preference_score + 0.35 * route_closeness + 0.15 * metadata_quality
        return round(score, 6)

    if subquery["template_name"] == "via_waypoint":
        route_closeness = max(0.0, 1.0 - candidate["distance_to_route_m"] / config["gt_max_via_waypoint_route_distance_m"])
        waypoint_dist = candidate.get("distance_to_waypoint_m", config["gt_max_waypoint_distance_m"])
        waypoint_closeness = max(0.0, 1.0 - float(waypoint_dist) / config["gt_max_waypoint_distance_m"])
        score = 0.45 * preference_score + 0.25 * route_closeness + 0.20 * waypoint_closeness + 0.10 * metadata_quality
        return round(score, 6)

    if subquery["template_name"] == "route_buffer":
        buffer_limit_m = max(1.0, float(subquery.get("buffer_km", 0.5)) * 1000.0)
        buffer_closeness = max(0.0, 1.0 - candidate["distance_to_route_m"] / buffer_limit_m)
        score = 0.60 * preference_score + 0.30 * buffer_closeness + 0.10 * metadata_quality
        return round(score, 6)

    return round(0.70 * preference_score + 0.30 * metadata_quality, 6)


def generate_ground_truth_from_queries(queries, points, valid_pairs, config, top_k=GT_TOP_K):
    pair_lookup = {(row["origin_id"], row["dest_id"]): row for row in valid_pairs}
    ground_truth_rows = []

    for gt_id, query in enumerate(queries, start=1):
        candidates = get_candidate_points_for_subquery(query, points, pair_lookup, config)

        candidates = [enrich_candidate_with_scoring_details(query, candidate, config) for candidate in candidates]
        candidates.sort(key=lambda x: (-x["ground_truth_score"], x["distance_to_route_m"], x["point_id"]))
        top_candidates = candidates[:top_k]

        row = {
            "ground_truth_id": gt_id,
            "query_id": query["query_id"],
            "subquery_group_id": query["subquery_group_id"],
            "template_id": query["template_id"],
            "template_name": query["template_name"],
            "query_text": query["query_text"],
            "origin_id": query["origin_id"],
            "origin_name": query["origin_name"],
            "origin_lat": query["origin_lat"],
            "origin_lon": query["origin_lon"],
            "dest_id": query["dest_id"],
            "dest_name": query["dest_name"],
            "dest_lat": query["dest_lat"],
            "dest_lon": query["dest_lon"],
            "waypoint_id": query.get("waypoint_id", ""),
            "waypoint_name": query.get("waypoint_name", ""),
            "waypoint_lat": query.get("waypoint_lat", ""),
            "waypoint_lon": query.get("waypoint_lon", ""),
            "target_category": query["target_category"],
            "preference": query["preference"],
            "buffer_km": query["buffer_km"],
            "candidate_count": len(candidates),
            "route_distance_m": query["route_distance_m"],
            "route_duration_s": query["route_duration_s"]
        }
        row.update(flatten_topk_candidates(top_candidates, top_k=top_k))
        ground_truth_rows.append(row)

    return ground_truth_rows


def save_pairs_to_csv(pairs, output_file):
    base_cols = [
        "origin_id", "origin_name", "origin_category", "origin_lat", "origin_lon",
        "dest_id", "dest_name", "dest_category", "dest_lat", "dest_lon",
        "crowfly_distance_m", "route_distance_m", "route_duration_s", "route_geometry"
    ]
    if not pairs:
        pd.DataFrame(columns=base_cols).to_csv(output_file, index=False, encoding="utf-8-sig")
        return

    pd.DataFrame(pairs).to_csv(output_file, index=False, encoding="utf-8-sig")


def save_subqueries_to_csv(subqueries, output_file):
    if not subqueries:
        pd.DataFrame(columns=SUBQUERY_FIELDS).to_csv(output_file, index=False, encoding="utf-8-sig")
        return

    pd.DataFrame(subqueries).to_csv(output_file, index=False, encoding="utf-8-sig")


def save_queries_to_csv(queries, output_file):
    if not queries:
        pd.DataFrame(columns=QUERY_FIELDS).to_csv(output_file, index=False, encoding="utf-8-sig")
        return

    pd.DataFrame(queries).to_csv(output_file, index=False, encoding="utf-8-sig")


def save_ground_truth_to_csv(ground_truth_rows, output_file):
    if not ground_truth_rows:
        pd.DataFrame(columns=GROUND_TRUTH_FIELDS).to_csv(output_file, index=False, encoding="utf-8-sig")
        return

    pd.DataFrame(ground_truth_rows).to_csv(output_file, index=False, encoding="utf-8-sig")


def build_dataset(
    place,
    sample_size=None,
    target_query_count=1500,
    output_file="general_points_cleaned.csv",
    use_road_network=False,
    seed=42,
    pair_output_file="general_valid_pairs.csv",
    subquery_output_file="general_subqueries.csv",
    query_output_file="general_queries.csv",
    ground_truth_output_file="general_ground_truth.csv",
    generic_place_references=False
):
    print(f"[INFO] Processing region: {place}")

    print("[INFO] Getting region boundary...")
    polygon = get_region_boundary(place)
    print("[INFO] Region boundary loaded.")

    if use_road_network:
        print("[INFO] Loading road network...")
        G = get_road_network(polygon)
        print(f"[INFO] Road network loaded: {len(G.nodes)} nodes, {len(G.edges)} edges")

    print("[INFO] Fetching POIs...")
    poi = get_poi(polygon)
    print(f"[INFO] Raw POIs fetched: {len(poi)}")

    print("[INFO] Extracting points...")
    points = extract_points_with_meta(poi)
    print(f"[INFO] Extracted points: {len(points)}")

    points = clean_points(points, seed=seed)
    config = build_adaptive_config(
        polygon=polygon,
        raw_points_count=len(poi),
        cleaned_points_count=len(points),
        requested_sample_size=sample_size
    )
    config = maybe_expand_sampling_budget(
        config=config,
        target_query_count=target_query_count,
        cleaned_points_count=len(points)
    )
    required_pairs = estimate_required_valid_pairs(target_query_count, templates_per_pair=3)
    print("[INFO] Adaptive region config:")
    print({
        "bbox_area_km2": config["bbox_area_km2"],
        "bbox_diagonal_m": config["bbox_diagonal_m"],
        "sample_size": config["sample_size"],
        "crowfly_range_m": [config["min_crowfly_distance_m"], config["max_crowfly_distance_m"]],
        "route_range_m": [config["min_route_distance_m"], config["max_route_distance_m"]],
        "gt_along_route_m": config["gt_max_along_route_distance_m"],
        "gt_via_route_m": config["gt_max_via_waypoint_route_distance_m"],
        "gt_waypoint_m": config["gt_max_waypoint_distance_m"],
        "buffer_options_km": config["buffer_options_km"],
        "target_query_count": target_query_count,
        "required_valid_pairs": required_pairs,
        "max_valid_pairs": config["max_valid_pairs"]
    })

    sampled_points = sample_points(points, n=config["sample_size"], seed=seed)
    sampled_points = add_point_ids(sampled_points)
    print(f"[INFO] Sampled points: {len(sampled_points)}")

    save_points_to_csv(sampled_points, output_file)
    print(f"[INFO] Saved points to: {output_file}")

    print("[INFO] Generating OD candidate pairs...")
    candidate_pairs = generate_candidate_pairs(sampled_points, config=config, seed=seed)
    print(f"[INFO] Candidate pair count: {len(candidate_pairs)}")

    print("[INFO] Filtering OD pairs with OSRM...")
    valid_pairs = filter_valid_pairs_by_osrm(
        candidate_pairs,
        config=config,
        max_valid_pairs=config["max_valid_pairs"]
    )
    print(f"[INFO] Valid pair count: {len(valid_pairs)}")

    save_pairs_to_csv(valid_pairs, pair_output_file)
    print(f"[INFO] Saved valid pairs to: {pair_output_file}")

    print("[INFO] Generating structured subquery records...")
    subqueries = generate_subquery_records_from_pairs(valid_pairs, sampled_points, config=config, seed=seed)
    print(f"[INFO] Subquery record count: {len(subqueries)}")

    print("[INFO] Rendering final template-based queries from subqueries...")
    queries = render_queries_from_subqueries(
        subqueries,
        generic_place_references=generic_place_references
    )

    if target_query_count is not None and len(queries) > target_query_count:
        queries = queries[:target_query_count]
        valid_group_ids = {row["subquery_group_id"] for row in queries}
        subqueries = [row for row in subqueries if row["subquery_group_id"] in valid_group_ids]
        print(f"[INFO] Query count truncated to target: {len(queries)}")
    else:
        print(f"[INFO] Query count: {len(queries)}")

    save_subqueries_to_csv(subqueries, subquery_output_file)
    print(f"[INFO] Saved subqueries to: {subquery_output_file}")

    save_queries_to_csv(queries, query_output_file)
    print(f"[INFO] Saved queries to: {query_output_file}")

    print("[INFO] Generating proxy ground-truth results from structured subqueries...")
    ground_truth_rows = generate_ground_truth_from_queries(
        queries,
        sampled_points,
        valid_pairs,
        config=config,
        top_k=GT_TOP_K
    )
    print(f"[INFO] Ground-truth row count: {len(ground_truth_rows)}")

    save_ground_truth_to_csv(ground_truth_rows, ground_truth_output_file)
    print(f"[INFO] Saved ground truth to: {ground_truth_output_file}")

    return sampled_points, valid_pairs, subqueries, queries, ground_truth_rows


def run_with_user_settings():
    place = "Sydney, Australia"
    target_query_count = 1500
    sampled_points, valid_pairs, subqueries, queries, ground_truth_rows = build_dataset(
        place=place,
        sample_size=None,
        target_query_count=target_query_count,
        output_file="general_points_cleaned.csv",
        use_road_network=False,
        seed=42,
        pair_output_file="general_valid_pairs.csv",
        subquery_output_file="general_subqueries.csv",
        query_output_file="general_queries.csv",
        ground_truth_output_file="general_ground_truth.csv",
        generic_place_references=False
    )
    return sampled_points, valid_pairs, subqueries, queries, ground_truth_rows


if __name__ == "__main__":
    place = "Melbourne, Australia"
    sampled_points, valid_pairs, subqueries, queries, ground_truth_rows = build_dataset(
        place=place,
        sample_size=None,
        target_query_count=1500,
        output_file="general_points_cleaned.csv",
        use_road_network=False,
        seed=42,
        pair_output_file="general_valid_pairs.csv",
        subquery_output_file="general_subqueries.csv",
        query_output_file="general_queries.csv",
        ground_truth_output_file="general_ground_truth.csv",
        generic_place_references=False
    )

    print("\n[INFO] First 5 sampled points:")
    for i, p in enumerate(sampled_points[:5], start=1):
        print(i, prepare_points_for_csv([p])[0])

    print("\n[INFO] First 5 valid pairs:")
    for i, row in enumerate(valid_pairs[:5], start=1):
        print(i, {k: row[k] for k in row if k != "route_geometry"})

    print("\n[INFO] First 5 subquery records:")
    for i, row in enumerate(subqueries[:5], start=1):
        print(i, row["query_skeleton"])

    print("\n[INFO] First 5 queries:")
    for i, row in enumerate(queries[:5], start=1):
        print(i, row["query_text"])

    print("\n[INFO] First 5 ground-truth rows:")
    for i, row in enumerate(ground_truth_rows[:5], start=1):
        print(f"GT Row {i}")
        for k in range(1, 6):
            print(
                f"  top{k}:",
                {
                    "id": row.get(f"ground_truth_top{k}_id", ""),
                    "name": row.get(f"ground_truth_top{k}_name", ""),
                    "lat": row.get(f"ground_truth_top{k}_lat", ""),
                    "lon": row.get(f"ground_truth_top{k}_lon", ""),
                    "score": row.get(f"ground_truth_top{k}_score", ""),
                    "distance_to_route_m": row.get(f"ground_truth_top{k}_distance_to_route_m", ""),
                    "distance_to_waypoint_m": row.get(f"ground_truth_top{k}_distance_to_waypoint_m", ""),
                    "preference_score": row.get(f"ground_truth_top{k}_preference_score", "")
                }
            )
