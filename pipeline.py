from geopy.geocoders import Nominatim
import osmnx as ox
import random
import csv
import math
import time
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

# 你可以继续自己扩充这些名字池
NAME_POOL = {
    "cafe": [
        "Maple Cafe", "Lantern Coffee", "Morning Bean", "Quiet Corner Cafe",
        "Blue Cup", "Clover Cafe", "Oak Brew", "Little Lane Coffee"
    ],
    "restaurant": [
        "Golden Table", "City Spoon", "Harbour Kitchen", "Olive House",
        "Sunset Diner", "Riverstone Grill", "Urban Taste", "Garden Plate"
    ],
    "hospital": [
        "Central Health Clinic", "Northside Medical Centre", "CityCare Hospital",
        "Green Cross Health", "Riverside Medical Hub"
    ],
    "fuel": [
        "Metro Fuel", "Rapid Gas", "City Petrol Stop", "Northway Fuel"
    ],
    "casino": [
        "Grand Crown Casino", "Aurora Gaming Hall", "Silver Star Casino"
    ],
    "marketplace": [
        "Fresh Square Market", "Central Marketplace", "Urban Harvest Market"
    ],
    "hotel": [
        "Grand Stay Hotel", "Parkview Suites", "Laneway Hotel", "City Rest Inn"
    ],
    "attraction": [
        "Heritage Plaza", "Skyline Point", "Riverwalk Landmark", "Victoria Gallery"
    ],
    "supermarket": [
        "FreshMart", "Daily Choice Market", "Green Basket Supermarket", "UrbanGrocer"
    ],
    "unknown": [
        "Urban Place", "City Spot", "Local Point", "Central Venue"
    ]
}

AMENITY_CATEGORIES = set(TARGET_POI["amenity"])
CATEGORY_SOURCE_FIELDS = ["amenity", "tourism", "shop"]

OSRM_BASE_URL = "http://router.project-osrm.org/route/v1/driving"

# Carlton demo 用的距离阈值（可后续再调）
MIN_CROWFLY_DISTANCE_M = 500
MAX_CROWFLY_DISTANCE_M = 3200
MIN_ROUTE_DISTANCE_M = 800
MAX_ROUTE_DISTANCE_M = 4000

MAX_DEST_CANDIDATES_PER_ORIGIN = 12
MAX_VALID_PAIRS = 60
PAIR_REQUEST_SLEEP_SEC = 0.15

PREFERENCE_WORDS = [
    "quiet",
    "cheap",
    "convenient",
    "popular",
    "highly rated",
    "cozy"
]

BUFFER_OPTIONS_KM = [0.5, 1.0, 1.5]


def category_to_generic_phrase(category):
    category = normalize_value(category).lower()
    if not category or category == "unknown":
        return "a place"

    article_map = {
        "attraction": "an attraction"
    }
    return article_map.get(category, f"a {category}")


def resolve_place_reference(name, category, generic_place_references=False):
    """
    控制 query 里地点是显示具体名字，还是显示更泛化的类别表达。
    """
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
    """
    返回包含更多 OSM 属性的 POI 列表，同时保留原始标签，方便后续补全 category
    """
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
    """
    合并同坐标/近似同坐标的重复点，优先保留非空、信息更完整的字段
    """
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
    """
    按坐标去重，但不是简单保留第一个，而是做字段合并
    """
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
    """
    用邻近点补 suburb / street。
    只有在近邻里多数一致时才补，避免乱填。
    """
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


def fill_missing_names(points, seed=42):
    rng = random.Random(seed)

    for point in points:
        name = normalize_value(point.get("name", ""))
        if name:
            continue

        category = normalize_value(point.get("category", "unknown")) or "unknown"
        pool = NAME_POOL.get(category, NAME_POOL["unknown"])
        point["name"] = rng.choice(pool)

    return points


def clean_points(points, seed=42):
    """
    执行你当前认可的数据清理逻辑：
    1. 去重并合并重复字段
    2. category / amenity 互相补齐
    3. 用邻近点补 suburb / street
    4. 缺失 name 从名字池随机补
    5. housenumber / opening_hours / website / phone 不激进补值
    """
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


def get_osrm_route_info(origin_lon, origin_lat, dest_lon, dest_lat, timeout=10):
    url = (
        f"{OSRM_BASE_URL}/"
        f"{origin_lon},{origin_lat};{dest_lon},{dest_lat}"
        f"?overview=false"
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
        return {
            "route_distance_m": route.get("distance"),
            "route_duration_s": route.get("duration")
        }
    except Exception:
        return None


def generate_candidate_pairs(points, seed=42, max_dest_candidates_per_origin=MAX_DEST_CANDIDATES_PER_ORIGIN):
    rng = random.Random(seed)
    pair_rows = []

    for i, origin in enumerate(points):
        local_candidates = []

        for j, dest in enumerate(points):
            if i == j:
                continue

            crowfly_m = haversine_meters(origin["lat"], origin["lon"], dest["lat"], dest["lon"])
            if MIN_CROWFLY_DISTANCE_M <= crowfly_m <= MAX_CROWFLY_DISTANCE_M:
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


def filter_valid_pairs_by_osrm(candidate_pairs, max_valid_pairs=MAX_VALID_PAIRS):
    valid_rows = []

    for row in candidate_pairs:
        route_info = get_osrm_route_info(
            row["origin_lon"], row["origin_lat"],
            row["dest_lon"], row["dest_lat"]
        )

        if route_info is None:
            continue

        route_distance_m = route_info["route_distance_m"]
        route_duration_s = route_info["route_duration_s"]

        if route_distance_m is None or route_duration_s is None:
            continue

        if MIN_ROUTE_DISTANCE_M <= route_distance_m <= MAX_ROUTE_DISTANCE_M:
            enriched = row.copy()
            enriched["route_distance_m"] = round(route_distance_m, 2)
            enriched["route_duration_s"] = round(route_duration_s, 2)
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


def generate_queries_from_pairs(valid_pairs, points, seed=42, generic_place_references=False):
    rng = random.Random(seed)
    query_rows = []
    query_id = 1

    for row in valid_pairs:
        origin_id = row["origin_id"]
        dest_id = row["dest_id"]
        origin_name = row["origin_name"]
        dest_name = row["dest_name"]
        origin_category = row.get("origin_category", "")
        dest_category = row.get("dest_category", "")

        origin_ref = resolve_place_reference(
            origin_name,
            origin_category,
            generic_place_references=generic_place_references
        )
        dest_ref = resolve_place_reference(
            dest_name,
            dest_category,
            generic_place_references=generic_place_references
        )

        excluded = {origin_id, dest_id}
        preference = rng.choice(PREFERENCE_WORDS)

        # Template 1
        target_category_1 = choose_target_category(points, excluded_ids=excluded, seed=rng.randint(1, 10**9))
        query_rows.append({
            "query_id": query_id,
            "template_id": 1,
            "template_name": "along_route",
            "query_text": build_template_1(origin_ref, dest_ref, target_category_1, preference),
            "origin_id": origin_id,
            "origin_name": origin_name,
            "dest_id": dest_id,
            "dest_name": dest_name,
            "waypoint_id": "",
            "waypoint_name": "",
            "target_category": target_category_1,
            "preference": preference,
            "buffer_km": "",
            "route_distance_m": row["route_distance_m"],
            "route_duration_s": row["route_duration_s"]
        })
        query_id += 1

        # Template 2
        waypoint = choose_waypoint(points, excluded_ids=excluded, seed=rng.randint(1, 10**9))
        if waypoint is not None:
            waypoint_ref = resolve_place_reference(
                waypoint.get("name", ""),
                waypoint.get("category", ""),
                generic_place_references=generic_place_references
            )
            target_category_2 = choose_target_category(points, excluded_ids=excluded | {waypoint["point_id"]}, seed=rng.randint(1, 10**9))
            query_rows.append({
                "query_id": query_id,
                "template_id": 2,
                "template_name": "via_waypoint",
                "query_text": build_template_2(origin_ref, dest_ref, waypoint_ref, target_category_2, preference),
                "origin_id": origin_id,
                "origin_name": origin_name,
                "dest_id": dest_id,
                "dest_name": dest_name,
                "waypoint_id": waypoint["point_id"],
                "waypoint_name": waypoint["name"],
                "target_category": target_category_2,
                "preference": preference,
                "buffer_km": "",
                "route_distance_m": row["route_distance_m"],
                "route_duration_s": row["route_duration_s"]
            })
            query_id += 1

        # Template 3
        target_category_3 = choose_target_category(points, excluded_ids=excluded, seed=rng.randint(1, 10**9))
        buffer_km = rng.choice(BUFFER_OPTIONS_KM)
        query_rows.append({
            "query_id": query_id,
            "template_id": 3,
            "template_name": "route_buffer",
            "query_text": build_template_3(origin_ref, dest_ref, target_category_3, preference, buffer_km),
            "origin_id": origin_id,
            "origin_name": origin_name,
            "dest_id": dest_id,
            "dest_name": dest_name,
            "waypoint_id": "",
            "waypoint_name": "",
            "target_category": target_category_3,
            "preference": preference,
            "buffer_km": buffer_km,
            "route_distance_m": row["route_distance_m"],
            "route_duration_s": row["route_duration_s"]
        })
        query_id += 1

    return query_rows


def save_pairs_to_csv(pairs, output_file):
    if not pairs:
        pd.DataFrame(columns=[
            "origin_id", "origin_name", "origin_category", "origin_lat", "origin_lon",
            "dest_id", "dest_name", "dest_category", "dest_lat", "dest_lon",
            "crowfly_distance_m", "route_distance_m", "route_duration_s"
        ]).to_csv(output_file, index=False, encoding="utf-8-sig")
        return

    pd.DataFrame(pairs).to_csv(output_file, index=False, encoding="utf-8-sig")


def save_queries_to_csv(queries, output_file):
    if not queries:
        pd.DataFrame(columns=[
            "query_id", "template_id", "template_name", "query_text",
            "origin_id", "origin_name", "dest_id", "dest_name",
            "waypoint_id", "waypoint_name", "target_category", "preference",
            "buffer_km", "route_distance_m", "route_duration_s"
        ]).to_csv(output_file, index=False, encoding="utf-8-sig")
        return

    pd.DataFrame(queries).to_csv(output_file, index=False, encoding="utf-8-sig")


def build_dataset(
    place,
    sample_size=100,
    output_file="output_points.csv",
    use_road_network=False,
    seed=42,
    pair_output_file="carlton_valid_pairs.csv",
    query_output_file="carlton_queries.csv",
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

    sampled_points = sample_points(points, n=sample_size, seed=seed)
    sampled_points = add_point_ids(sampled_points)
    print(f"[INFO] Sampled points: {len(sampled_points)}")

    save_points_to_csv(sampled_points, output_file)
    print(f"[INFO] Saved points to: {output_file}")

    print("[INFO] Generating OD candidate pairs...")
    candidate_pairs = generate_candidate_pairs(sampled_points, seed=seed)
    print(f"[INFO] Candidate pair count: {len(candidate_pairs)}")

    print("[INFO] Filtering OD pairs with OSRM...")
    valid_pairs = filter_valid_pairs_by_osrm(candidate_pairs, max_valid_pairs=MAX_VALID_PAIRS)
    print(f"[INFO] Valid pair count: {len(valid_pairs)}")

    save_pairs_to_csv(valid_pairs, pair_output_file)
    print(f"[INFO] Saved valid pairs to: {pair_output_file}")

    print("[INFO] Generating template-based queries...")
    queries = generate_queries_from_pairs(
        valid_pairs,
        sampled_points,
        seed=seed,
        generic_place_references=generic_place_references
    )
    print(f"[INFO] Query count: {len(queries)}")

    save_queries_to_csv(queries, query_output_file)
    print(f"[INFO] Saved queries to: {query_output_file}")

    return sampled_points, valid_pairs, queries


if __name__ == "__main__":
    place = "Carlton, Melbourne, Australia"
    sampled_points, valid_pairs, queries = build_dataset(
        place=place,
        sample_size=100,
        output_file="melbourne_100_points_cleaned.csv",
        use_road_network=False,
        seed=42,
        pair_output_file="carlton_valid_pairs.csv",
        query_output_file="carlton_queries.csv",
        generic_place_references=True
    )

    print("\n[INFO] First 5 sampled points:")
    for i, p in enumerate(sampled_points[:5], start=1):
        print(i, prepare_points_for_csv([p])[0])

    print("\n[INFO] First 5 valid pairs:")
    for i, row in enumerate(valid_pairs[:5], start=1):
        print(i, row)

    print("\n[INFO] First 5 queries:")
    for i, row in enumerate(queries[:5], start=1):
        print(i, row["query_text"])
