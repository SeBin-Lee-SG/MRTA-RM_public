# func/my_class.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class obj_node_class:
    """
    Container for Robot/Goal node attributes.

    For robots:
      - traveled_distance: accumulated scalar
      - travel_way_point: concatenated node sequence along the route
      - priority, time_offset: optional scheduling hints
    """
    pos: Optional[tuple[int, int]] = None
    nearest_valid_vertex: Optional[Any] = None
    dist_to_valid_vertex: Optional[float] = None
    nearest_JC_node: Optional[Any] = None
    dist_to_JC_node: Optional[float] = None
    section: List[int] = field(default_factory=list)

    # Robot-specific runtime fields
    traveled_distance: float = 0.0
    travel_way_point: List[Any] = field(default_factory=list)
    priority: int = 0
    time_offset: int = 0


@dataclass
class temp_section_class:
    """Temporary section built from a path between two JC nodes."""
    start: Optional[int] = None
    end: Optional[int] = None
    way_point: List[Any] = field(default_factory=list)
    length: Optional[float] = None


@dataclass
class section_class:
    """
    A section between two (possibly same) JC nodes.

    Note
    ----
    - start == end means JC-only section (degenerate)
    - way_point includes both endpoints
    - transfer/receive dictionaries use keys {"D","R","E"}
    """
    start: Optional[int] = None
    end: Optional[int] = None
    way_point: List[Any] = field(default_factory=list)
    length: Optional[float] = None
    width: float = 0.0
    robot_size: Optional[int] = None

    # Population
    robot_num: int = 0
    robot_list: List[int] = field(default_factory=list)
    robot_vertex_list: List[Any] = field(default_factory=list)
    goal_num: int = 0
    goal_list: List[int] = field(default_factory=list)
    goal_vertex_list: List[Any] = field(default_factory=list)

    # Remaining after in-section balancing
    remain_robot_num: int = 0
    remain_goal_num: int = 0

    # Transfer / receive accounting (direction from source/dest perspective)
    transfer_section_index_dict: Dict[int, List[Any]] = field(default_factory=dict)
    receive_section_index_dict: Dict[int, List[Any]] = field(default_factory=dict)

    num_of_transfer: Dict[str, int] = field(default_factory=lambda: {"D": 0, "R": 0, "E": 0})
    num_of_receive: Dict[str, int] = field(default_factory=lambda: {"D": 0, "R": 0, "E": 0})

    # Robot queues for scheduling; dictionaries store lists of robot indices
    transfer_robot_dict: Dict[str, List[int]] = field(default_factory=lambda: {"D": [], "R": [], "E": []})
    receive_robot_dict: Dict[str, List[int]] = field(default_factory=lambda: {"D": [], "R": [], "E": []})

    robot_queue: List[int] = field(default_factory=list)

    # Optional: mapping from receive direction to goal list (if planner uses it)
    goal_dict_for_received_robot: Dict[str, List[int]] = field(default_factory=lambda: {"D": [], "R": [], "E": []})


@dataclass
class allocation_set_class:
    """A (robot, goal) pair with travel cost/path/section description."""
    robot_index: int = 0
    goal_index: int = 0
    travel_dist: float = 0.0
    travel_path: List[Any] = field(default_factory=list)
    travel_section: List[int] = field(default_factory=list)


@dataclass
class set_class:
    """
    A container for intermediate allocation sets and summaries.
    Most fields are optional and may be filled progressively by the pipeline.
    """
    allocation_result: Optional[List[List[int]]] = None

    init_allocation: Optional[List[List[int]]] = None
    init_path_list: Optional[List[List[Any]]] = None
    init_path_dist_list: Optional[List[float]] = None

    same_seg_allocation: Optional[List[List[int]]] = None
    same_seg_path_list: Optional[List[List[Any]]] = None
    same_seg_path_dist_list: Optional[List[float]] = None

    diff_seg_allocation: Optional[List[List[int]]] = None
    diff_seg_path_list: Optional[List[List[Any]]] = None
    diff_seg_path_dist_list: Optional[List[float]] = None

    cost_sum: Optional[float] = None

    using_section: Optional[List[int]] = None
    case1_section: Optional[List[int]] = None
    case2_section: Optional[List[int]] = None
    case3_section: Optional[List[int]] = None
