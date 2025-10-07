# src/transfer_planner.py
from __future__ import annotations

from collections import defaultdict
from typing import List, Dict, Tuple, Sequence, Any


class TransferPlanner:
    """
    Analyze inter-section transfers implied by section sequences produced in the
    initial allocation phase.

    Inputs
    ------
    section_class_list : Sequence[Any]
        List of section objects. Each section is expected to have at least:
            - start (int): start node id
            - end (int): end node id
            - length (float): section length
            - robot_num, goal_num (int)
            - robot_list, goal_list (List[int])
            - robot_vertex_list, goal_vertex_list (List[int])
        And will be (re)populated with:
            - transfer_section_index_dict: Dict[int, [str, int]]
            - receive_section_index_dict: Dict[int, [str, int]]
            - num_of_transfer: Dict[str, int] with keys {"D","R","E"}
            - num_of_receive: Dict[str, int] with keys {"D","R","E"}

    diff_seg_section_list : Sequence[List[int]]
        For each unbalanced robot, a list of section indices describing the
        sequence of sections it must traverse. Example: [3, 7, 12].
    """

    def __init__(self, section_class_list: Sequence[Any], diff_seg_section_list: Sequence[List[int]]):
        self.section_class_list: List[Any] = list(section_class_list)
        self.diff_seg_section_list: List[List[int]] = [list(p) for p in diff_seg_section_list]

        # Outputs
        self.section2section_transfer_dict: Dict[Tuple[int, int], int] = defaultdict(int)
        self.section2section_transfer_dict_key_list: List[Tuple[int, int]] = []

        self.case1_section_list: List[int] = []  # no incoming, no outgoing
        self.case2_section_list: List[int] = []  # outgoing only
        self.case3_section_list: List[int] = []  # incoming only
        self.case4_section_list: List[int] = []  # both incoming and outgoing

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _is_jc(section: Any) -> bool:
        """Return True if the section represents a JC-only node (start == end)."""
        return getattr(section, "start") == getattr(section, "end")

    @staticmethod
    def _direction_from_source_to_dest(src: Any, dst: Any) -> str:
        """
        Direction code from source section to destination section.
        'E' : source is a JC-only section
        'D' : moving forward with the 'end' side of source
        'R' : moving backward with the 'start' side of source
        """
        if TransferPlanner._is_jc(src):
            return "E"
        if src.end == dst.start or src.end == dst.end:
            return "D"
        if src.start == dst.start or src.start == dst.end:
            return "R"
        # Fallback: default to 'D' (should not happen in a consistent section graph)
        return "D"

    @staticmethod
    def _direction_from_dest_perspective(src: Any, dst: Any) -> str:
        """
        Direction code from destination's perspective about the source:
        'E' : destination is a JC-only section
        'D' : arrival at destination's start side
        'R' : arrival at destination's end side
        """
        if TransferPlanner._is_jc(dst):
            return "E"
        if src.end == dst.start or src.start == dst.start:
            return "D"
        if src.end == dst.end or src.start == dst.end:
            return "R"
        # Fallback
        return "D"

    def _reset_section_counters(self) -> None:
        """Ensure all sections start from a clean state (no stale counts)."""
        for s in self.section_class_list:
            s.transfer_section_index_dict = {}
            s.receive_section_index_dict = {}
            s.num_of_transfer = {"D": 0, "R": 0, "E": 0}
            s.num_of_receive = {"D": 0, "R": 0, "E": 0}

    def _accumulate_pair(self, src_idx: int, dst_idx: int) -> None:
        """Update dictionaries/counters for one (source -> destination) hop."""
        src = self.section_class_list[src_idx]
        dst = self.section_class_list[dst_idx]

        # Global pair counter & stable key order
        key = (src_idx, dst_idx)
        if key not in self.section2section_transfer_dict:
            # Keep first-seen order of keys
            self.section2section_transfer_dict_key_list.append(key)
        self.section2section_transfer_dict[key] += 1

        # Source perspective
        src_dir = self._direction_from_source_to_dest(src, dst)
        if dst_idx in src.transfer_section_index_dict:
            src.transfer_section_index_dict[dst_idx][1] += 1
        else:
            src.transfer_section_index_dict[dst_idx] = [src_dir, 1]

        # Destination perspective
        dst_dir = self._direction_from_dest_perspective(src, dst)
        if src_idx in dst.receive_section_index_dict:
            dst.receive_section_index_dict[src_idx][1] += 1
        else:
            dst.receive_section_index_dict[src_idx] = [dst_dir, 1]

        # Aggregate counts (per section), respecting endpoint configuration
        if src.end == dst.start:
            src.num_of_transfer["E" if self._is_jc(src) else "D"] += 1
            dst.num_of_receive["E" if self._is_jc(dst) else "D"] += 1
        elif src.end == dst.end:
            src.num_of_transfer["E" if self._is_jc(src) else "D"] += 1
            dst.num_of_receive["E" if self._is_jc(dst) else "R"] += 1
        elif src.start == dst.end:
            src.num_of_transfer["E" if self._is_jc(src) else "R"] += 1
            dst.num_of_receive["E" if self._is_jc(dst) else "R"] += 1
        elif src.start == dst.start:
            src.num_of_transfer["E" if self._is_jc(src) else "R"] += 1
            dst.num_of_receive["E" if self._is_jc(dst) else "D"] += 1

    def _classify_sections(self) -> None:
        """Populate case1..case4 lists based on transfer/receive counts."""
        self.case1_section_list.clear()
        self.case2_section_list.clear()
        self.case3_section_list.clear()
        self.case4_section_list.clear()

        for idx, s in enumerate(self.section_class_list):
            t = s.num_of_transfer["D"] + s.num_of_transfer["R"] + s.num_of_transfer["E"]
            r = s.num_of_receive["D"] + s.num_of_receive["R"] + s.num_of_receive["E"]
            if t == 0 and r == 0:
                self.case1_section_list.append(idx)
            elif t != 0 and r == 0:
                self.case2_section_list.append(idx)
            elif t == 0 and r != 0:
                self.case3_section_list.append(idx)
            elif t != 0 and r != 0:
                self.case4_section_list.append(idx)

    def _sort_case4_sections(self) -> None:
        """
        Sort case4 sections by the latest appearance index across all paths in
        diff_seg_section_list (stable with original behavior).
        """
        appear: Dict[int, int] = {}
        for sec_idx in self.case4_section_list:
            for path in self.diff_seg_section_list:
                # find latest position of sec_idx in this path
                for i, v in enumerate(path):
                    if v == sec_idx:
                        if sec_idx not in appear or i > appear[sec_idx]:
                            appear[sec_idx] = i
        self.case4_section_list.sort(key=lambda x: appear.get(x, -1))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze(self):
        """
        Run the full analysis pipeline.

        Returns
        -------
        (case1, case2, case3, case4, key_list, pair_count_dict)
          - case{1..4}_section_list : List[int]
          - key_list : List[Tuple[int,int]]
          - pair_count_dict : Dict[Tuple[int,int], int]
        """
        self._reset_section_counters()

        # Build per-hop statistics
        for path in self.diff_seg_section_list:
            for i in range(len(path) - 1):
                self._accumulate_pair(path[i], path[i + 1])

        # Classify sections
        self._classify_sections()
        self._sort_case4_sections()

        return (
            self.case1_section_list,
            self.case2_section_list,
            self.case3_section_list,
            self.case4_section_list,
            self.section2section_transfer_dict_key_list,
            self.section2section_transfer_dict,
        )

    # Backward-compat method name (old code may call .analyzer())
    def analyzer(self):
        return self.analyze()
