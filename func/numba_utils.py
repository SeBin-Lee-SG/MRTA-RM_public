"""Numba-accelerated utility functions for MRTA-RM."""

import numpy as np
import numba as nb


@nb.njit(cache=True)
def euclidean_distance(p1_x, p1_y, p2_x, p2_y):
    """Fast scalar Euclidean distance."""
    dx = p1_x - p2_x
    dy = p1_y - p2_y
    return np.sqrt(dx * dx + dy * dy)


@nb.njit(cache=True)
def euclidean_distance_tuple(p1, p2):
    """Euclidean distance accepting tuple-like inputs (array of 2)."""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return np.sqrt(dx * dx + dy * dy)


@nb.njit(cache=True)
def pairwise_distances(points_a, points_b):
    """Compute pairwise distance matrix between two sets of 2D points.

    Args:
        points_a: (N, 2) array
        points_b: (M, 2) array

    Returns:
        (N, M) distance matrix
    """
    n = points_a.shape[0]
    m = points_b.shape[0]
    result = np.empty((n, m), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            dx = points_a[i, 0] - points_b[j, 0]
            dy = points_a[i, 1] - points_b[j, 1]
            result[i, j] = np.sqrt(dx * dx + dy * dy)
    return result


@nb.njit(cache=True)
def batch_distances_from_point(point_x, point_y, targets):
    """Compute distances from a single point to an array of targets.

    Args:
        point_x, point_y: scalar coordinates
        targets: (N, 2) array of target points

    Returns:
        (N,) array of distances
    """
    n = targets.shape[0]
    result = np.empty(n, dtype=np.float64)
    for i in range(n):
        dx = point_x - targets[i, 0]
        dy = point_y - targets[i, 1]
        result[i] = np.sqrt(dx * dx + dy * dy)
    return result


@nb.njit(cache=True)
def sample_boundary_edges(vertices, sampling_dist):
    """Sample points along polygon edges.

    Args:
        vertices: (N, 2) array of polygon vertices (closed: last == first)
        sampling_dist: distance between samples

    Returns:
        (M, 2) array of sampled boundary points
    """
    n_edges = vertices.shape[0] - 1
    # First pass: count total samples
    total = 0
    for e in range(n_edges):
        dx = vertices[e + 1, 0] - vertices[e, 0]
        dy = vertices[e + 1, 1] - vertices[e, 1]
        edge_len = np.sqrt(dx * dx + dy * dy)
        total += int(edge_len / sampling_dist) + 1

    result = np.empty((total, 2), dtype=np.float64)
    idx = 0
    for e in range(n_edges):
        x0, y0 = vertices[e, 0], vertices[e, 1]
        x1, y1 = vertices[e + 1, 0], vertices[e + 1, 1]
        dx = x1 - x0
        dy = y1 - y0
        edge_len = np.sqrt(dx * dx + dy * dy)
        how_many = int(edge_len / sampling_dist) + 1
        for i in range(how_many):
            t = i / how_many
            result[idx, 0] = x0 + dx * t
            result[idx, 1] = y0 + dy * t
            idx += 1

    return result[:idx]


@nb.njit(cache=True)
def find_uniform_node_indices(accumulate_length_list, how_many_node, uniform_dist):
    """Find indices where cumulative length exceeds uniform spacing thresholds.

    Args:
        accumulate_length_list: (N,) array of cumulative edge lengths
        how_many_node: number of nodes to place
        uniform_dist: spacing between nodes

    Returns:
        array of edge indices
    """
    ind = np.empty(how_many_node, dtype=np.int64)
    count = 0
    for i in range(how_many_node):
        threshold = uniform_dist * (i + 1)
        for j in range(len(accumulate_length_list)):
            if accumulate_length_list[j] >= threshold:
                ind[count] = j
                count += 1
                break
    return ind[:count]
