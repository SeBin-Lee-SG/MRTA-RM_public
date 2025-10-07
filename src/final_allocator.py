# src/final_allocator.py  (you can name it final_allocator.py if you prefer)
from __future__ import annotations

from typing import List, Dict, Any
import copy

# Keep explicit imports (avoid wildcard) – but we keep legacy helpers for compatibility
from func.func import (
    a_sub_b,
    dist,
    euclidean_distance,
    calc_dist_in_section,
)

class FinalAllocator:
    """
    Final allocator:
    - Equalization and allocation within/between sections using planner output.
    - Preserves original behavior; adds safety initializers and removes module-level globals.
    """

    def __init__(
        self,
        uniform_G,
        uniform_pos_dict,
        section_class_list: List[Any],
        robot_class_list: List[Any],
        goal_class_list: List[Any],
        case1_section_list: List[int],
        case2_section_list: List[int],
        case3_section_list: List[int],
        case4_section_list: List[int],
        realistic: bool,
        full_section_number: int,
        *,
        ver2: bool = True,
        wp_ver2: bool = False,
    ):
        self.uniform_G = uniform_G
        self.uniform_pos_dict = uniform_pos_dict

        self.section_class_list = section_class_list
        self.robot_class_list = robot_class_list
        self.goal_class_list = goal_class_list

        self.case1_section_list = list(case1_section_list)
        self.case2_section_list = list(case2_section_list)
        self.case3_section_list = list(case3_section_list)
        self.case4_section_list = list(case4_section_list)

        self.realistic = bool(realistic)
        self.full_section_number = int(full_section_number)

        # formerly module-level globals; now instance flags
        self.ver2 = bool(ver2)
        self.wp_ver2 = bool(wp_ver2)

        self.final_allocation_result: List[List[int]] = []

        # --- SAFETY: ensure runtime fields exist on every section ---
        for s in self.section_class_list:
            # transfer/receive dicts with D/R/E keys
            if not hasattr(s, "transfer_robot_dict") or not isinstance(getattr(s, "transfer_robot_dict"), dict):
                s.transfer_robot_dict = {"D": [], "R": [], "E": []}
            else:
                for k in ("D", "R", "E"):
                    s.transfer_robot_dict.setdefault(k, [])
            if not hasattr(s, "receive_robot_dict") or not isinstance(getattr(s, "receive_robot_dict"), dict):
                s.receive_robot_dict = {"D": [], "R": [], "E": []}
            else:
                for k in ("D", "R", "E"):
                    s.receive_robot_dict.setdefault(k, [])
            # robot_queue defaults to robot_list
            if not hasattr(s, "robot_queue") or not isinstance(getattr(s, "robot_queue"), list):
                s.robot_queue = list(getattr(s, "robot_list", []))

        # --- SAFETY: ensure robot runtime fields exist ---
        for i, rc in enumerate(self.robot_class_list):
            if not hasattr(rc, "travel_way_point"):
                rc.travel_way_point = []
            if not hasattr(rc, "traveled_distance"):
                rc.traveled_distance = 0.0

    # kept for possible A* heuristic usage (unchanged behavior)
    def dist_for_uniform_astar(self, a, b):
        return dist(a, b, self.uniform_pos_dict)

    # ---------------- Case 1 ----------------
    def allocate_case1(self):
        """Sections with no incoming and no outgoing transfers: assign within the section."""
        for section_idx in self.case1_section_list:
            section = self.section_class_list[section_idx]
            if not section.robot_num:
                section.robot_queue = []
                continue

            for i in range(min(section.robot_num, section.goal_num)):
                r_idx = section.robot_list[i]
                g_idx = section.goal_list[i]
                self.final_allocation_result.append([r_idx, g_idx])

                r_v = self.robot_class_list[r_idx].nearest_valid_vertex
                g_v = self.goal_class_list[g_idx].nearest_valid_vertex

                if r_v == g_v:
                    self.robot_class_list[r_idx].travel_way_point.append(g_v)
                elif section.way_point.index(r_v) < section.way_point.index(g_v):
                    route = section.way_point[
                        section.way_point.index(r_v): section.way_point.index(g_v)
                    ] + [g_v]
                    self.robot_class_list[r_idx].travel_way_point += route
                else:
                    route = section.way_point[
                        section.way_point.index(r_v): section.way_point.index(g_v): -1
                    ] + [g_v]
                    self.robot_class_list[r_idx].travel_way_point += route

            section.robot_queue = []

    # ---------------- Case 2 ----------------
    def allocate_case2(self):
        """
        Sections with outgoing only (no incoming): equalize (decide who to transfer) and allocate.
        This mirrors the original logic, but avoids globals and adds minor safety.
        """
        for section_idx in self.case2_section_list:
            section = self.section_class_list[section_idx]
            transfer_D = section.num_of_transfer["D"]
            transfer_R = section.num_of_transfer["R"]
            transfer_E = section.num_of_transfer["E"]

            # JC-only section
            if transfer_E:
                section.transfer_robot_dict["E"] = section.robot_list[:transfer_E]

                # in-section assignment for the rest
                for i in range(transfer_E, len(section.robot_list)):
                    self.final_allocation_result.append([section.robot_list[i], section.goal_list[i - transfer_E]])

                # prefer case4, then case3
                transfer_section_index_list = list(section.transfer_section_index_dict.keys())
                transfer_section_index_list = (
                    [idx for idx in transfer_section_index_list if idx in self.case4_section_list] +
                    [idx for idx in transfer_section_index_list if idx in self.case3_section_list]
                )

                for transfer_section_idx in transfer_section_index_list:
                    transfer_section = self.section_class_list[transfer_section_idx]
                    how_many = section.transfer_section_index_dict[transfer_section_idx][1]
                    dir_in_transfer_section = transfer_section.receive_section_index_dict[section_idx][0]

                    transfer_robot_list = section.transfer_robot_dict["E"][:how_many]
                    section.transfer_robot_dict["E"] = a_sub_b(section.transfer_robot_dict["E"], transfer_robot_list)

                    # push to start of this JC-section
                    for robot_idx in transfer_robot_list:
                        route = section.way_point[
                            section.way_point.index(self.robot_class_list[robot_idx].nearest_valid_vertex):0:-1
                        ] + [section.start]
                        self.robot_class_list[robot_idx].travel_way_point += route

                    section.robot_queue = a_sub_b(section.robot_queue, transfer_robot_list)

                    # enqueue into neighbor per direction
                    if dir_in_transfer_section == "D":
                        transfer_section.receive_robot_dict["D"].extend(transfer_robot_list)
                        transfer_section.robot_queue = transfer_robot_list + transfer_section.robot_queue
                    elif dir_in_transfer_section == "R":
                        transfer_section.receive_robot_dict["R"].extend(transfer_robot_list)
                        transfer_section.robot_queue = transfer_section.robot_queue + transfer_robot_list
                    else:  # "E"
                        transfer_section.receive_robot_dict["E"].extend(transfer_robot_list)
                        transfer_section.robot_queue = transfer_section.robot_queue + transfer_robot_list

            # Non-JC section
            else:
                # only D direction
                if transfer_D != 0 and transfer_R == 0:
                    section.transfer_robot_dict["D"] = list(
                        reversed(section.robot_list[len(section.robot_list) - transfer_D:])
                    )
                    # pair up remaining inside
                    for i in range(len(section.robot_list) - transfer_D):
                        self.final_allocation_result.append([section.robot_list[i], section.goal_list[i]])

                    transfer_section_index_list = list(section.transfer_section_index_dict.keys())
                    transfer_section_index_list = (
                        [idx for idx in transfer_section_index_list if idx in self.case4_section_list] +
                        [idx for idx in transfer_section_index_list if idx in self.case3_section_list]
                    )

                    for transfer_section_idx in transfer_section_index_list:
                        transfer_section = self.section_class_list[transfer_section_idx]
                        how_many = section.transfer_section_index_dict[transfer_section_idx][1]
                        dir_in_transfer_section = transfer_section.receive_section_index_dict[section_idx][0]

                        transfer_robot_list = section.transfer_robot_dict["D"][:how_many]
                        section.transfer_robot_dict["D"] = a_sub_b(section.transfer_robot_dict["D"], transfer_robot_list)

                        for robot_idx in transfer_robot_list:
                            if self.ver2 and section_idx <= self.full_section_number:
                                rnode = self.robot_class_list[robot_idx].nearest_valid_vertex
                                nxt = section.way_point[section.way_point.index(rnode) + 1]
                                self.robot_class_list[robot_idx].traveled_distance += calc_dist_in_section(
                                    self.uniform_G, self.section_class_list, section_idx, nxt, section.end
                                )
                                self.robot_class_list[robot_idx].traveled_distance += euclidean_distance(
                                    self.uniform_pos_dict[nxt], self.uniform_pos_dict['R' + str(robot_idx)]
                                )
                                route = section.way_point[section.way_point.index(rnode):-1]
                                self.robot_class_list[robot_idx].travel_way_point += route
                            self.robot_class_list[robot_idx].travel_way_point.append(section.end)

                        section.robot_queue = a_sub_b(section.robot_queue, transfer_robot_list)

                        if dir_in_transfer_section == "D":
                            transfer_section.receive_robot_dict["D"].extend(transfer_robot_list)
                            transfer_section.robot_queue = transfer_robot_list + transfer_section.robot_queue
                        elif dir_in_transfer_section == "R":
                            transfer_section.receive_robot_dict["R"].extend(transfer_robot_list)
                            transfer_section.robot_queue = transfer_section.robot_queue + transfer_robot_list
                        else:
                            transfer_section.receive_robot_dict["E"].extend(transfer_robot_list)
                            transfer_section.robot_queue = transfer_section.robot_queue + transfer_robot_list

                # only R direction
                elif transfer_D == 0 and transfer_R != 0:
                    section.transfer_robot_dict["R"] = section.robot_list[:transfer_R]

                    # in-section assignment for the rest
                    for i in range(transfer_R, len(section.robot_list)):
                        self.final_allocation_result.append([section.robot_list[i], section.goal_list[i - transfer_R]])

                    transfer_section_index_list = list(section.transfer_section_index_dict.keys())
                    transfer_section_index_list = (
                        [idx for idx in transfer_section_index_list if idx in self.case4_section_list] +
                        [idx for idx in transfer_section_index_list if idx in self.case3_section_list]
                    )

                    for transfer_section_idx in transfer_section_index_list:
                        transfer_section = self.section_class_list[transfer_section_idx]
                        how_many = section.transfer_section_index_dict[transfer_section_idx][1]
                        dir_in_transfer_section = transfer_section.receive_section_index_dict[section_idx][0]

                        transfer_robot_list = section.transfer_robot_dict["R"][:how_many]
                        section.transfer_robot_dict["R"] = a_sub_b(section.transfer_robot_dict["R"], transfer_robot_list)

                        for robot_idx in transfer_robot_list:
                            if self.ver2 and section_idx <= self.full_section_number:
                                rnode = self.robot_class_list[robot_idx].nearest_valid_vertex
                                prv = section.way_point[section.way_point.index(rnode) - 1]
                                self.robot_class_list[robot_idx].traveled_distance += calc_dist_in_section(
                                    self.uniform_G, self.section_class_list, section_idx, prv, section.start
                                )
                                self.robot_class_list[robot_idx].traveled_distance += euclidean_distance(
                                    self.uniform_pos_dict[prv], self.uniform_pos_dict['R' + str(robot_idx)]
                                )
                                route = section.way_point[section.way_point.index(rnode):0:-1]
                                self.robot_class_list[robot_idx].travel_way_point += route
                            self.robot_class_list[robot_idx].travel_way_point.append(section.start)

                        section.robot_queue = a_sub_b(section.robot_queue, transfer_robot_list)

                        if dir_in_transfer_section == "D":
                            transfer_section.receive_robot_dict["D"].extend(transfer_robot_list)
                            transfer_section.robot_queue = transfer_robot_list + transfer_section.robot_queue
                        elif dir_in_transfer_section == "R":
                            transfer_section.receive_robot_dict["R"].extend(transfer_robot_list)
                            transfer_section.robot_queue = transfer_section.robot_queue + transfer_robot_list
                        else:
                            transfer_section.receive_robot_dict["E"].extend(transfer_robot_list)
                            transfer_section.robot_queue = transfer_section.robot_queue + transfer_robot_list

                # both directions
                else:
                    section.transfer_robot_dict["D"] = list(reversed(section.robot_list[len(section.robot_list) - transfer_D:]))
                    section.transfer_robot_dict["R"] = section.robot_list[:transfer_R]

                    for i in range(transfer_R, len(section.robot_list) - transfer_D):
                        self.final_allocation_result.append([section.robot_list[i], section.goal_list[i - transfer_R]])

                    transfer_section_index_list = list(section.transfer_section_index_dict.keys())
                    transfer_section_index_list = (
                        [idx for idx in transfer_section_index_list if idx in self.case4_section_list] +
                        [idx for idx in transfer_section_index_list if idx in self.case3_section_list]
                    )

                    for transfer_section_idx in transfer_section_index_list:
                        transfer_section = self.section_class_list[transfer_section_idx]
                        how_many = section.transfer_section_index_dict[transfer_section_idx][1]
                        dir_in_transfer_section = transfer_section.receive_section_index_dict[section_idx][0]
                        dir_in_this_section = section.transfer_section_index_dict[transfer_section_idx][0]

                        if dir_in_this_section == "D":
                            transfer_robot_list = section.transfer_robot_dict["D"][:how_many]
                            section.transfer_robot_dict["D"] = a_sub_b(section.transfer_robot_dict["D"], transfer_robot_list)

                            for robot_idx in transfer_robot_list:
                                if self.ver2 and section_idx <= self.full_section_number:
                                    rnode = self.robot_class_list[robot_idx].nearest_valid_vertex
                                    nxt = section.way_point[section.way_point.index(rnode) + 1]
                                    self.robot_class_list[robot_idx].traveled_distance += calc_dist_in_section(
                                        self.uniform_G, self.section_class_list, section_idx, nxt, section.end
                                    )
                                    self.robot_class_list[robot_idx].traveled_distance += euclidean_distance(
                                        self.uniform_pos_dict[nxt], self.uniform_pos_dict['R' + str(robot_idx)]
                                    )
                                    route = section.way_point[section.way_point.index(rnode):-1]
                                    self.robot_class_list[robot_idx].travel_way_point += route
                                self.robot_class_list[robot_idx].travel_way_point.append(section.end)

                        elif dir_in_this_section == "R":
                            transfer_robot_list = section.transfer_robot_dict["R"][:how_many]
                            section.transfer_robot_dict["R"] = a_sub_b(section.transfer_robot_dict["R"], transfer_robot_list)

                            for robot_idx in transfer_robot_list:
                                if self.ver2 and section_idx <= self.full_section_number:
                                    rnode = self.robot_class_list[robot_idx].nearest_valid_vertex
                                    prv = section.way_point[section.way_point.index(rnode) - 1]
                                    self.robot_class_list[robot_idx].traveled_distance += calc_dist_in_section(
                                        self.uniform_G, self.section_class_list, section_idx, prv, section.start
                                    )
                                    self.robot_class_list[robot_idx].traveled_distance += euclidean_distance(
                                        self.uniform_pos_dict[prv], self.uniform_pos_dict['R' + str(robot_idx)]
                                    )
                                    route = section.way_point[section.way_point.index(rnode):0:-1]
                                    self.robot_class_list[robot_idx].travel_way_point += route
                                self.robot_class_list[robot_idx].travel_way_point.append(section.start)

                        section.robot_queue = a_sub_b(section.robot_queue, transfer_robot_list)

                        if dir_in_transfer_section == "D":
                            transfer_section.receive_robot_dict["D"].extend(transfer_robot_list)
                            transfer_section.robot_queue = transfer_robot_list + transfer_section.robot_queue
                        elif dir_in_transfer_section == "R":
                            transfer_section.receive_robot_dict["R"].extend(transfer_robot_list)
                            transfer_section.robot_queue = transfer_section.robot_queue + transfer_robot_list
                        else:
                            transfer_section.receive_robot_dict["E"].extend(transfer_robot_list)
                            transfer_section.robot_queue = transfer_section.robot_queue + transfer_robot_list

    # ---------------- Case 4: receive then transfer ----------------
    def receive_and_transfer_case4(self):
        case4_section_queue = copy.deepcopy(self.case4_section_list)

        while case4_section_queue:
            received_done_section_idx = []
            for section_idx in case4_section_queue:
                section = self.section_class_list[section_idx]
                transfer_D = section.num_of_transfer["D"]
                transfer_R = section.num_of_transfer["R"]
                transfer_E = section.num_of_transfer["E"]
                receive_D = section.num_of_receive["D"]
                receive_R = section.num_of_receive["R"]
                receive_E = section.num_of_receive["E"]

                goal_num = section.goal_num

                # If all expected robots arrived, we can sort/transfer.
                if len(section.robot_queue) == goal_num + transfer_D + transfer_R + transfer_E:
                    # sort by traveled_distance depending on receive side
                    if receive_D:
                        recv = section.receive_robot_dict["D"]
                        pairs = sorted((self.robot_class_list[r].traveled_distance, r) for r in recv)
                        section.robot_queue = a_sub_b(section.robot_queue, recv)
                        for _, r in pairs:
                            section.robot_queue.insert(0, r)
                    elif receive_R:
                        recv = section.receive_robot_dict["R"]
                        pairs = sorted((self.robot_class_list[r].traveled_distance, r) for r in recv)
                        section.robot_queue = a_sub_b(section.robot_queue, recv)
                        for _, r in pairs:
                            section.robot_queue.append(r)
                    elif receive_E:
                        recv = section.receive_robot_dict["E"]
                        pairs = sorted((self.robot_class_list[r].traveled_distance, r) for r in recv)
                        section.robot_queue = a_sub_b(section.robot_queue, recv)
                        for _, r in pairs:
                            section.robot_queue.append(r)

                    # choose whom to transfer
                    if transfer_D:
                        section.transfer_robot_dict["D"] = list(reversed(section.robot_queue[len(section.robot_queue) - transfer_D:]))
                    elif transfer_R:
                        section.transfer_robot_dict["R"] = section.robot_queue[:transfer_R]
                    elif transfer_E:
                        section.transfer_robot_dict["E"] = section.robot_queue[:transfer_E]

                    transfer_section_index_list = list(section.transfer_section_index_dict.keys())
                    transfer_section_index_list = (
                        [idx for idx in transfer_section_index_list if idx in self.case4_section_list] +
                        [idx for idx in transfer_section_index_list if idx in self.case3_section_list]
                    )

                    for transfer_section_idx in transfer_section_index_list:
                        transfer_section = self.section_class_list[transfer_section_idx]
                        how_many = section.transfer_section_index_dict[transfer_section_idx][1]
                        dir_in_transfer_section = transfer_section.receive_section_index_dict[section_idx][0]
                        dir_in_this_section = section.transfer_section_index_dict[transfer_section_idx][0]

                        # D side transfer
                        if dir_in_this_section == "D":
                            transfer_robot_list = section.transfer_robot_dict["D"][:how_many]
                            section.transfer_robot_dict["D"] = a_sub_b(section.transfer_robot_dict["D"], transfer_robot_list)

                            for robot_idx in transfer_robot_list:
                                if (robot_idx in section.receive_robot_dict["D"]) or (robot_idx in section.receive_robot_dict["R"]):
                                    # crossed through this section already
                                    self.robot_class_list[robot_idx].traveled_distance += section.length
                                    route = section.way_point[1:-1]
                                    self.robot_class_list[robot_idx].travel_way_point += route
                                    self.robot_class_list[robot_idx].travel_way_point.append(section.end)
                                else:
                                    if self.ver2 and section_idx <= self.full_section_number:
                                        rnode = self.robot_class_list[robot_idx].nearest_valid_vertex
                                        nxt = section.way_point[section.way_point.index(rnode) + 1]
                                        self.robot_class_list[robot_idx].traveled_distance += calc_dist_in_section(
                                            self.uniform_G, self.section_class_list, section_idx, nxt, section.end
                                        )
                                        self.robot_class_list[robot_idx].traveled_distance += euclidean_distance(
                                            self.uniform_pos_dict[nxt], self.uniform_pos_dict['R' + str(robot_idx)]
                                        )
                                        route = section.way_point[section.way_point.index(rnode):-1]
                                        self.robot_class_list[robot_idx].travel_way_point += route
                                    self.robot_class_list[robot_idx].travel_way_point.append(section.end)

                        # R side transfer
                        elif dir_in_this_section == "R":
                            transfer_robot_list = section.transfer_robot_dict["R"][:how_many]
                            section.transfer_robot_dict["R"] = a_sub_b(section.transfer_robot_dict["R"], transfer_robot_list)

                            for robot_idx in transfer_robot_list:
                                if (robot_idx in section.receive_robot_dict["D"]) or (robot_idx in section.receive_robot_dict["R"]):
                                    self.robot_class_list[robot_idx].traveled_distance += section.length
                                    route = section.way_point[-2:0:-1]
                                    self.robot_class_list[robot_idx].travel_way_point += route
                                    self.robot_class_list[robot_idx].travel_way_point.append(section.start)
                                else:
                                    if self.ver2 and section_idx <= self.full_section_number:
                                        rnode = self.robot_class_list[robot_idx].nearest_valid_vertex
                                        prv = section.way_point[section.way_point.index(rnode) - 1]
                                        self.robot_class_list[robot_idx].traveled_distance += calc_dist_in_section(
                                            self.uniform_G, self.section_class_list, section_idx, prv, section.start
                                        )
                                        self.robot_class_list[robot_idx].traveled_distance += euclidean_distance(
                                            self.uniform_pos_dict[prv], self.uniform_pos_dict['R' + str(robot_idx)]
                                        )
                                        route = section.way_point[section.way_point.index(rnode):0:-1]
                                        self.robot_class_list[robot_idx].travel_way_point += route
                                    self.robot_class_list[robot_idx].travel_way_point.append(section.start)

                        # E (JC) transfer
                        elif dir_in_this_section == "E":
                            transfer_robot_list = section.transfer_robot_dict["E"][:how_many]
                            section.transfer_robot_dict["E"] = a_sub_b(section.transfer_robot_dict["E"], transfer_robot_list)

                            for robot_idx in transfer_robot_list:
                                self.robot_class_list[robot_idx].travel_way_point.append(section.start)

                        # dequeue from current section
                        section.robot_queue = a_sub_b(section.robot_queue, transfer_robot_list)

                        # enqueue to neighbor
                        if dir_in_transfer_section == "D":
                            transfer_section.receive_robot_dict["D"].extend(transfer_robot_list)
                            transfer_section.robot_queue = transfer_robot_list + transfer_section.robot_queue
                        elif dir_in_transfer_section == "R":
                            transfer_section.receive_robot_dict["R"].extend(transfer_robot_list)
                            transfer_section.robot_queue = transfer_section.robot_queue + transfer_robot_list
                        else:
                            transfer_section.receive_robot_dict["E"].extend(transfer_robot_list)
                            transfer_section.robot_queue = transfer_section.robot_queue + transfer_robot_list

                    received_done_section_idx.append(section_idx)

                else:
                    continue

            case4_section_queue = a_sub_b(case4_section_queue, received_done_section_idx)

    # ---------------- Case 3 & 4 allocation ----------------
    def allocate_case3_4(self):
        # case3 only receive
        for section_idx in self.case3_section_list:
            section = self.section_class_list[section_idx]
            D_receive_num = section.num_of_receive["D"]
            R_receive_num = section.num_of_receive["R"]
            E_receive_num = section.num_of_receive["E"]

            if D_receive_num > 0 and R_receive_num > 0:
                D_robot_list = section.receive_robot_dict["D"]
                R_robot_list = section.receive_robot_dict["R"]
                section.robot_queue = section.robot_queue[D_receive_num:-R_receive_num]

                pairs = sorted((self.robot_class_list[r].traveled_distance, r) for r in D_robot_list)
                for _, r in pairs:
                    section.robot_queue.insert(0, r)

                pairs = sorted((self.robot_class_list[r].traveled_distance, r) for r in R_robot_list)
                for _, r in pairs:
                    section.robot_queue.append(r)

                for i in range(section.goal_num):
                    self.final_allocation_result.append([section.robot_queue[i], section.goal_list[i]])
                    # route stitching
                    if self.robot_class_list[section.robot_queue[i]].travel_way_point[-1] == section.start:
                        route = section.way_point[1: section.way_point.index(self.goal_class_list[section.goal_list[i]].nearest_valid_vertex)]
                        route += [self.goal_class_list[section.goal_list[i]].nearest_valid_vertex]
                        self.robot_class_list[section.robot_queue[i]].travel_way_point += route
                    elif self.robot_class_list[section.robot_queue[i]].travel_way_point[-1] == section.end:
                        route = section.way_point[-2: section.way_point.index(self.goal_class_list[section.goal_list[i]].nearest_valid_vertex): -1]
                        route += [self.goal_class_list[section.goal_list[i]].nearest_valid_vertex]
                        self.robot_class_list[section.robot_queue[i]].travel_way_point += route
                    else:
                        rv = self.robot_class_list[section.robot_queue[i]].nearest_valid_vertex
                        gv = self.goal_class_list[section.goal_list[i]].nearest_valid_vertex
                        if section.way_point.index(rv) <= section.way_point.index(gv):
                            route = section.way_point[section.way_point.index(rv): section.way_point.index(gv)] + [gv]
                        else:
                            route = section.way_point[section.way_point.index(rv): section.way_point.index(gv): -1] + [gv]
                        self.robot_class_list[section.robot_queue[i]].travel_way_point += route

            elif D_receive_num > 0:
                D_robot_list = section.receive_robot_dict["D"]
                section.robot_queue = section.robot_queue[D_receive_num:]
                pairs = sorted((self.robot_class_list[r].traveled_distance, r) for r in D_robot_list)
                for _, r in pairs:
                    section.robot_queue.insert(0, r)

                for i in range(section.goal_num):
                    self.final_allocation_result.append([section.robot_queue[i], section.goal_list[i]])
                    if self.robot_class_list[section.robot_queue[i]].travel_way_point:
                        if self.robot_class_list[section.robot_queue[i]].travel_way_point[-1] == section.start:
                            route = section.way_point[1: section.way_point.index(self.goal_class_list[section.goal_list[i]].nearest_valid_vertex)]
                            route += [self.goal_class_list[section.goal_list[i]].nearest_valid_vertex]
                            self.robot_class_list[section.robot_queue[i]].travel_way_point += route
                        elif self.robot_class_list[section.robot_queue[i]].travel_way_point[-1] == section.end:
                            route = section.way_point[-2: section.way_point.index(self.goal_class_list[section.goal_list[i]].nearest_valid_vertex): -1]
                            route += [self.goal_class_list[section.goal_list[i]].nearest_valid_vertex]
                            self.robot_class_list[section.robot_queue[i]].travel_way_point += route
                    else:
                        rv = self.robot_class_list[section.robot_queue[i]].nearest_valid_vertex
                        gv = self.goal_class_list[section.goal_list[i]].nearest_valid_vertex
                        if section.way_point.index(rv) <= section.way_point.index(gv):
                            route = section.way_point[section.way_point.index(rv): section.way_point.index(gv)] + [gv]
                        else:
                            route = section.way_point[section.way_point.index(rv): section.way_point.index(gv): -1] + [gv]
                        self.robot_class_list[section.robot_queue[i]].travel_way_point += route

            elif R_receive_num > 0:
                R_robot_list = section.receive_robot_dict["R"]
                section.robot_queue = section.robot_queue[:-R_receive_num]
                pairs = sorted((self.robot_class_list[r].traveled_distance, r) for r in R_robot_list)
                for _, r in pairs:
                    section.robot_queue.append(r)

                for i in range(section.goal_num):
                    self.final_allocation_result.append([section.robot_queue[i], section.goal_list[i]])
                    if self.robot_class_list[section.robot_queue[i]].travel_way_point:
                        if self.robot_class_list[section.robot_queue[i]].travel_way_point[-1] == section.start:
                            route = section.way_point[1: section.way_point.index(self.goal_class_list[section.goal_list[i]].nearest_valid_vertex)]
                            route += [self.goal_class_list[section.goal_list[i]].nearest_valid_vertex]
                            self.robot_class_list[section.robot_queue[i]].travel_way_point += route
                        elif self.robot_class_list[section.robot_queue[i]].travel_way_point[-1] == section.end:
                            route = section.way_point[-2: section.way_point.index(self.goal_class_list[section.goal_list[i]].nearest_valid_vertex): -1]
                            route += [self.goal_class_list[section.goal_list[i]].nearest_valid_vertex]
                            self.robot_class_list[section.robot_queue[i]].travel_way_point += route
                    else:
                        rv = self.robot_class_list[section.robot_queue[i]].nearest_valid_vertex
                        gv = self.goal_class_list[section.goal_list[i]].nearest_valid_vertex
                        if section.way_point.index(rv) <= section.way_point.index(gv):
                            route = section.way_point[section.way_point.index(rv): section.way_point.index(gv)] + [gv]
                        else:
                            route = section.way_point[section.way_point.index(rv): section.way_point.index(gv): -1] + [gv]
                        self.robot_class_list[section.robot_queue[i]].travel_way_point += route

            else:
                for i in range(section.goal_num):
                    self.final_allocation_result.append([section.robot_queue[i], section.goal_list[i]])
                    if self.robot_class_list[section.robot_queue[i]].travel_way_point[-1] == section.start:
                        route = section.way_point[1: section.way_point.index(self.goal_class_list[section.goal_list[i]].nearest_valid_vertex)]
                        route += [self.goal_class_list[section.goal_list[i]].nearest_valid_vertex]
                        self.robot_class_list[section.robot_queue[i]].travel_way_point += route
                    elif self.robot_class_list[section.robot_queue[i]].travel_way_point[-1] == section.end:
                        route = section.way_point[-2: section.way_point.index(self.goal_class_list[section.goal_list[i]].nearest_valid_vertex): -1]
                        route += [self.goal_class_list[section.goal_list[i]].nearest_valid_vertex]
                        self.robot_class_list[section.robot_queue[i]].travel_way_point += route
                    else:
                        rv = self.robot_class_list[section.robot_queue[i]].nearest_valid_vertex
                        gv = self.goal_class_list[section.goal_list[i]].nearest_valid_vertex
                        if section.way_point.index(rv) <= section.way_point.index(gv):
                            route = section.way_point[section.way_point.index(rv): section.way_point.index(gv)] + [gv]
                        else:
                            route = section.way_point[section.way_point.index(rv): section.way_point.index(gv): -1] + [gv]
                        self.robot_class_list[section.robot_queue[i]].travel_way_point += route

        # case4: final top-up pass (same as original intent)
        for section_idx in self.case4_section_list:
            section = self.section_class_list[section_idx]
            if section.goal_num == 0:
                continue

            robot_idx_need_to_search = a_sub_b(section.robot_queue, section.robot_list)
            if section.num_of_receive["D"]:
                section.robot_queue = section.robot_queue[len(robot_idx_need_to_search):]
                direction = "D"
            else:
                section.robot_queue = section.robot_queue[:-len(robot_idx_need_to_search)]
                direction = "R"

            pairs = sorted((self.robot_class_list[r].traveled_distance, r) for r in robot_idx_need_to_search)
            if direction == "D":
                for _, r in pairs:
                    section.robot_queue.insert(0, r)
            else:
                for _, r in pairs:
                    section.robot_queue.append(r)

            for i in range(section.goal_num):
                self.final_allocation_result.append([section.robot_queue[i], section.goal_list[i]])

    # ---------------- Entry ----------------
    def on_init(self):
        """Backward-compatible entry (same name as before)."""
        self.allocate_case1()
        self.allocate_case2()
        self.receive_and_transfer_case4()
        self.allocate_case3_4()
        return self.final_allocation_result

    # modern alias
    def run(self):
        return self.on_init()

