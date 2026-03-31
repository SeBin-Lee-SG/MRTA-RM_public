import json
import os

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MAPS_DIR = os.path.join(_BASE_DIR, "maps")

# test_set integer -> JSON filename mapping
MAP_REGISTRY = {
    1: "random",
    2: "department_store",
    3: "warehouse",
    4: "demo",
    11: "random_mini",
    22: "department_store_mini",
    33: "warehouse_mini",
}


def load_map(test_set):
    """Load obstacle polygons and metadata from a JSON map file.

    Args:
        test_set: Integer test_set ID (1, 2, 3, 4, 11, 22, 33) or
                  a string map name (e.g., "warehouse").

    Returns:
        Tuple of (polygons, map_width, map_height, max_robots, robot_size).
        polygons is a list of polygon vertex lists.
    """
    if isinstance(test_set, int):
        map_name = MAP_REGISTRY.get(test_set)
        if map_name is None:
            raise ValueError(f"Unknown test_set: {test_set}. "
                             f"Available: {list(MAP_REGISTRY.keys())}")
    else:
        map_name = test_set

    file_path = os.path.join(_MAPS_DIR, f"{map_name}.json")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Map file not found: {file_path}")

    with open(file_path, "r") as f:
        data = json.load(f)

    # Convert polygon vertex lists to tuples for compatibility
    polygons = []
    for poly in data["polygons"]:
        polygons.append([tuple(v) for v in poly])

    return (
        polygons,
        data["map_width"],
        data["map_height"],
        data["max_robots"],
        data["robot_size"],
    )
