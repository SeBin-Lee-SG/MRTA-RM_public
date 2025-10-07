# func/func.py
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import networkx as nx

__all__ = [
    "euclidean_distance",
    "dist",
    "a_sub_b",
    "list_summation",
    "find_index_with_value",
    "find_index_with_key",
    "calc_path_length",
    "make_simple_path",
    "extract_path2section",
    "make_two_list",
    "find_and_delete_allocate",
    "calc_dist_in_section",
    "normalize",
]


# ----------------------------
# Geometry & basic distances
# ----------------------------
def euclidean_distance(p1: Sequence[float], p2: Sequence[float]) -> float:
    """Return Euclidean distance between 2D points p1 and p2."""
    return float(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2))


def dist(a: Any, b: Any, pos_dict: Dict[Any, Sequence[float]]) -> float:
    """Return Euclidean distance between two nodes using a position dictionary."""
    return euclidean_distance(pos_dict[a], pos_dict[b])


# ----------------------------
# Small list utilities
# ----------------------------
def a_sub_b(a: Sequence[Any], b: Sequence[Any]) -> List[Any]:
    """Return list A without items that appear in list B (order-preserving)."""
    b_set = set(b)
    return [x for x in a if x not in b_set]


def list_summation(l: Sequence[Sequence[Any]]) -> List[List[Any]]:
    """
    Transpose a list of equal-length lists.
    Example: [[1,2,3],[4,5,6]] -> [[1,4],[2,5],[3,6]]
    """
    if not l:
        return []
    length = len(l[0])
    # Validate equal-length input to avoid silent mistakes
    for sub in l:
        if len(sub) != length:
            raise ValueError("All sublists must have the same length.")
    return [list(col) for col in zip(*l)]


def find_index_with_value(l: Sequence[Sequence[Any]], value: Any) -> Optional[int]:
    """Return first index i where l[i][1] == value, else None."""
    for i, sublist in enumerate(l):
        if len(sublist) > 1 and sublist[1] == value:
            return i
    return None


def find_index_with_key(l: Sequence[Sequence[Any]], key: Any) -> Optional[int]:
    """Return first index i where l[i][0] == key, else None."""
    for i, sublist in enumerate(l):
        if len(sublist) > 0 and sublist[0] == key:
            return i
    return None


# ----------------------------
# Paths and sections
# ----------------------------
def calc_path_length(path: Sequence[Any], graph: nx.Graph) -> float:
    """Calculate the total length of a path on a weighted graph."""
    if not path or len(path) == 1:
        return 0.0
    return float(
        sum(graph.get_edge_data(path[i], path[i + 1])["weight"] for i in range(len(path) - 1))
    )


def make_simple_path(
    path: Sequence[Any],
    uniform_JC_node_list: Sequence[int],
    uniform_additional_JC_node_list: Sequence[int],
) -> List[Any]:
    """
    Keep only endpoints and JC nodes in the path.
    """
    if not path:
        return []
    jc_set = set(uniform_JC_node_list) | set(uniform_additional_JC_node_list)
    n = len(path)
    return [path[i] for i in range(n) if i in (0, n - 1) or path[i] in jc_set]


def extract_path2section(
    path: Sequence[Any],
    JC_nodes: Sequence[int],
    full_section_number: int,
) -> List[int]:
    """
    Convert a simple-graph path (S# nodes and/or JC nodes) to section indices.

    - "S{k}" -> k
    - JC node -> full_section_number + JC_index + 1
    """
    out: List[int] = []
    for node in path:
        if isinstance(node, str) and node.startswith("S") and len(node) > 1:
            # S# node
            try:
                out.append(int(node[1:]))
            except ValueError:
                # Ignore malformed S node
                pass
        elif node in JC_nodes:
            out.append(full_section_number + JC_nodes.index(node) + 1)
    return out


def make_two_list(
    list1: Sequence[Sequence[int]],
    list2: Sequence[Sequence[int]],
) -> Tuple[List[int], List[int]]:
    """
    Split pairs into two flat lists.

    Example:
      list1 = [[r1, g1], [r2, g2]]
      list2 = [[r3, g3]]
      -> ([r1, r2, r3], [g1, g2, g3])
    """
    robots: List[int] = []
    goals: List[int] = []
    for sub in list1:
        robots.append(sub[0])
        goals.append(sub[1])
    for sub in list2:
        robots.append(sub[0])
        goals.append(sub[1])
    return robots, goals


def find_and_delete_allocate(
    init_allocate: List[List[int]],
    init_path_list: List[List[Any]],
    init_path_dist_list: List[float],
    using_section_list: List[List[int]],
    allocation: List[int],
) -> None:
    """
    Remove an allocation and its aligned entries from parallel lists.

    Args
    ----
    allocation : [robot_index, goal_index]

    Side effects
    ------------
    Modifies the four lists in-place. Prints an error if allocation is absent.
    """
    if allocation not in init_allocate:
        print("error in find_and_delete_allocate")
        return
    idx = init_allocate.index(allocation)
    del init_allocate[idx]
    del init_path_list[idx]
    del init_path_dist_list[idx]
    del using_section_list[idx]


def calc_dist_in_section(
    graph: nx.Graph,
    section_class_list: Sequence[Any],
    section_index: int,
    a: Any,
    b: Any,
) -> float:
    """
    Calculate distance between two nodes a and b within a given section.

    Rules
    -----
    1) If a == b -> 0
    2) If (a,b) are the endpoints of the section -> section.length
    3) If both a and b are in section.way_point -> sum of edge weights along the slice
       (respects actual graph weights, not uniform proportion)
    4) Otherwise -> shortest path distance restricted to this section's waypoints

    This improves numerical consistency over a simple index-based proportion,
    while preserving the original intent.
    """
    section = section_class_list[section_index]
    if a == b:
        return 0.0

    start, end = section.start, section.end
    if (a == start and b == end) or (a == end and b == start):
        return float(section.length)

    wp: List[Any] = list(section.way_point)

    # If both nodes are in way_point, use the contiguous slice based on indices.
    if a in wp and b in wp:
        ia, ib = wp.index(a), wp.index(b)
        if ia <= ib:
            sub = wp[ia:ib]
            return calc_path_length(sub + [b], graph)
        else:
            sub = wp[ia:ib:-1]
            return calc_path_length(sub + [b], graph)

    # Fallback: shortest path restricted to the section's waypoints
    subG = graph.subgraph(wp).copy()
    try:
        length = nx.shortest_path_length(subG, a, b, weight="weight")
        return float(length)
    except Exception:
        # As a last resort, use global graph (should rarely happen)
        try:
            return float(nx.shortest_path_length(graph, a, b, weight="weight"))
        except Exception:
            return float("inf")


# ----------------------------
# Misc
# ----------------------------
def normalize(value: float, min_val: float, max_val: float, new_min: float = 0, new_max: float = 1000) -> float:
    """Linearly map value from [min_val, max_val] to [new_min, new_max]."""
    if max_val == min_val:
        # Avoid division by zero; map to midpoint
        return (new_min + new_max) / 2.0
    return (value - min_val) / (max_val - min_val) * (new_max - new_min) + new_min
