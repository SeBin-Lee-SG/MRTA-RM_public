# src/initial_allocator.py
from __future__ import annotations

from typing import List, Dict, Tuple, Sequence, Optional
import time

import numpy as np
import networkx as nx
from scipy.optimize import linear_sum_assignment

# Import only what we use (no wildcard)
from func.func import dist, calc_path_length, extract_path2section


class InitialAllocator:
    """
    Initial assignment phase:
    - Balance robots/goals within sections
    - Build a coarse "section graph" for remaining (unbalanced) demand
    - Solve remaining transfers via shortest paths + Hungarian
    """

    # ---------------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------------
    def __init__(
        self,
        graph: nx.Graph,
        pos_dict: Dict,                       # pos for uniform_G nodes (including 'R{i}', 'G{i}')
        robot_class_list: Sequence,           # obj_node_class list
        goal_class_list: Sequence,            # obj_node_class list
        section_class_list: Sequence,         # section_class list
        robot_num: int,
        goal_num: int,
        uniform_JC_node_list: Sequence[int],
        uniform_additional_JC_node_list: Sequence[int],
        full_section_number: int,
    ) -> None:
        self.uniform_G: nx.Graph = graph
        self.uniform_pos_dict: Dict = pos_dict
        self.robot_class_list = list(robot_class_list)
        self.goal_class_list = list(goal_class_list)
        self.section_class_list = list(section_class_list)
        self.robot_num = int(robot_num)
        self.goal_num = int(goal_num)

        self.uniform_JC_node_list = list(uniform_JC_node_list)
        self.uniform_additional_JC_node_list = list(uniform_additional_JC_node_list)
        self.full_section_number = int(full_section_number)

        # Simple-graph (section graph) built on-demand
        self.simple_G: Optional[nx.Graph] = None
        self.simple_G_pos_dict: Dict = {}

        # Remaining demand holders after in-section balancing
        self.remain_robot_list: List = []
        self.remain_goal_list: List = []

        # Cache for path queries (A* not used by default, but keep cache)
        self._path_and_length_cache: Dict[Tuple, Tuple[List, float]] = {}

        # Timing
        self.time_for_cost_matrix: float = 0.0

    # ---------------------------------------------------------------------
    # Phase 1: balance within sections
    # ---------------------------------------------------------------------
    def allocate_within_sections(self) -> None:
        """
        Compute remaining deficits/excess per section.
        Sets:
          section.remain_robot_num, section.remain_goal_num
        """
        for section in self.section_class_list:
            # ensure fields exist (avoid stale values)
            section.remain_robot_num = 0
            section.remain_goal_num = 0

            if section.robot_num > section.goal_num:
                section.remain_robot_num = section.robot_num - section.goal_num
            elif section.robot_num < section.goal_num:
                section.remain_goal_num = section.goal_num - section.robot_num
            # equal -> both remain_* stay 0

    # ---------------------------------------------------------------------
    # Phase 2: build a coarse "section graph" for remaining demand
    # ---------------------------------------------------------------------
    def make_simple_graph(self) -> None:
        """
        Build a graph with:
          - One Steiner-like node 'S{k}' per section (0..full_section_number)
          - Edges: (start -- S{k}) and (end -- S{k}) with weight = section.length / 2
          - JC-only sections (after full_section_number) use their start node directly
        Populate:
          self.remain_robot_list, self.remain_goal_list with sources/targets in the simple graph.
        """
        self.simple_G = nx.Graph()
        self.simple_G_pos_dict = {}
        self.remain_robot_list = []
        self.remain_goal_list = []

        # For real path distance on simple_G, we need coordinates for start/end/S{k}
        def _mid(a: List[float], b: List[float]) -> List[float]:
            return [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0]

        # Regular sections
        for s_idx, section in enumerate(self.section_class_list[: self.full_section_number + 1]):
            skey = f"S{s_idx}"
            # connect endpoints to S
            self.simple_G.add_edge(section.start, skey, weight=section.length / 2.0)
            self.simple_G.add_edge(section.end, skey, weight=section.length / 2.0)

            # positions
            self.simple_G_pos_dict[section.start] = self.uniform_pos_dict[section.start]
            self.simple_G_pos_dict[section.end] = self.uniform_pos_dict[section.end]
            self.simple_G_pos_dict[skey] = _mid(
                self.uniform_pos_dict[section.start],
                self.uniform_pos_dict[section.end],
            )

            # remaining demand mapped to S
            for _ in range(getattr(section, "remain_robot_num", 0)):
                self.remain_robot_list.append(skey)
            for _ in range(getattr(section, "remain_goal_num", 0)):
                self.remain_goal_list.append(skey)

        # JC-only sections
        for section in self.section_class_list[self.full_section_number + 1 :]:
            for _ in range(getattr(section, "remain_robot_num", 0)):
                self.remain_robot_list.append(section.start)
            for _ in range(getattr(section, "remain_goal_num", 0)):
                self.remain_goal_list.append(section.start)

    # ---------------------------------------------------------------------
    # Phase 3: solve remaining via shortest paths + Hungarian
    # ---------------------------------------------------------------------
    def _hungarian_from_costs(self, cost_rows: List[List[float]]) -> List[Tuple[int, int]]:
        """Return list of (row, col) indices selected by Hungarian."""
        cost_mat = np.array(cost_rows, dtype=float)
        cost_mat[~np.isfinite(cost_mat)] = 1e10  # large but not inf
        row_ind, col_ind = linear_sum_assignment(cost_mat)
        return list(zip(row_ind.tolist(), col_ind.tolist()))

    def allocate_remaining_dijkstra(self) -> List[List[int]]:
        """
        Use single-source Dijkstra on the simple graph from each remaining source to all targets.
        Then solve the assignment with Hungarian. Finally, convert paths to section sequences.
        Returns: list of section-id sequences (for Arbiter).
        """
        if not self.remain_robot_list:
            return []

        all_cost_rows: List[List[float]] = []
        all_path_rows: List[List[List]] = []

        # For each remaining robot source, get shortest paths to all goal sinks
        for src in self.remain_robot_list:
            length_dict, path_dict = nx.single_source_dijkstra(self.simple_G, src)

            row_costs: List[float] = []
            row_paths: List[List] = []
            for tgt in self.remain_goal_list:
                if tgt in path_dict:
                    row_costs.append(length_dict[tgt])
                    row_paths.append(path_dict[tgt])
                else:
                    row_costs.append(float("inf"))
                    row_paths.append([])

            all_cost_rows.append(row_costs)
            all_path_rows.append(row_paths)

        # Hungarian on the cost matrix
        matching = self._hungarian_from_costs(all_cost_rows)

        # Convert the chosen simple-graph paths into "section sequences"
        JC_nodes = self.uniform_JC_node_list + self.uniform_additional_JC_node_list
        result: List[List[int]] = []
        for r, c in matching:
            result.append(
                extract_path2section(all_path_rows[r][c], JC_nodes, self.full_section_number)
            )
        return result

    # ----- Public surface -------------------------------------------------
    def allocation(self) -> List[List[int]]:
        """Convenience: run the full initial allocation pipeline and return section sequences."""
        self.allocate_within_sections()
        self.make_simple_graph()
        return self.allocate_remaining_dijkstra()

    # ---------------------------------------------------------------------
    def on_init(self):
        """Back-compat: same as allocation()."""
        return self.allocation()

