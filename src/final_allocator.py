import copy
from func.func import euclidean_distance, calc_dist_in_section, a_sub_b


class FinalAllocator:
    """Performs the final robot-goal allocation based on section case analysis.

    Handles robot redistribution (transfer/receive) between sections and
    produces the final allocation pairs."""

    def __init__(self, uniform_G, uniform_pos_dict, section_class_list,
                 robot_class_list, goal_class_list,
                 case1_section_list, case2_section_list,
                 case3_section_list, case4_section_list,
                 full_section_number):
        self.uniform_G = uniform_G
        self.uniform_pos_dict = uniform_pos_dict
        self.section_class_list = section_class_list
        self.robot_class_list = robot_class_list
        self.goal_class_list = goal_class_list
        self.case1_section_list = case1_section_list
        self.case2_section_list = case2_section_list
        self.case3_section_list = case3_section_list
        self.case4_section_list = case4_section_list
        self.full_section_number = full_section_number
        self.final_allocation_result = []

    def _sorted_transfer_targets(self, section):
        """Sort transfer targets: case4 first, then case3."""
        targets = list(section.transfer_section_index_dict.keys())
        sorted_targets = []
        sorted_targets += [idx for idx in targets if idx in self.case4_section_list]
        sorted_targets += [idx for idx in targets if idx in self.case3_section_list]
        return sorted_targets

    def _transfer_robots_to_section(self, transfer_robot_list, section_idx,
                                     transfer_section_idx, section):
        """Move robots from one section to another, updating queues."""
        transfer_section = self.section_class_list[transfer_section_idx]
        dir_in_transfer_section = transfer_section.receive_section_index_dict[section_idx][0]

        section.robot_queue = a_sub_b(section.robot_queue, transfer_robot_list)

        if dir_in_transfer_section == "D":
            transfer_section.receive_robot_dict["D"].extend(transfer_robot_list)
            transfer_section.robot_queue = transfer_robot_list + transfer_section.robot_queue
        elif dir_in_transfer_section == "R":
            transfer_section.receive_robot_dict["R"].extend(transfer_robot_list)
            transfer_section.robot_queue = transfer_section.robot_queue + transfer_robot_list
        elif dir_in_transfer_section == "E":
            transfer_section.receive_robot_dict["E"].extend(transfer_robot_list)
            transfer_section.robot_queue = transfer_section.robot_queue + transfer_robot_list

    def _update_robot_travel_D(self, robot_idx, section, section_idx):
        """Update travel distance/waypoints for a robot moving toward section end (D)."""
        if section_idx <= self.full_section_number:
            which_node = section.way_point[
                section.way_point.index(self.robot_class_list[robot_idx].nearest_valid_vertex) + 1]
            self.robot_class_list[robot_idx].traveled_distance += calc_dist_in_section(
                self.uniform_G, self.section_class_list, section_idx, which_node, section.end)
            self.robot_class_list[robot_idx].traveled_distance += euclidean_distance(
                self.uniform_pos_dict[which_node], self.uniform_pos_dict['R' + str(robot_idx)])
            self.robot_class_list[robot_idx].travel_way_point.append(which_node)
        self.robot_class_list[robot_idx].travel_way_point.append(section.end)

    def _update_robot_travel_R(self, robot_idx, section, section_idx):
        """Update travel distance/waypoints for a robot moving toward section start (R)."""
        if section_idx <= self.full_section_number:
            which_node = section.way_point[
                section.way_point.index(self.robot_class_list[robot_idx].nearest_valid_vertex) - 1]
            self.robot_class_list[robot_idx].traveled_distance += calc_dist_in_section(
                self.uniform_G, self.section_class_list, section_idx, which_node, section.start)
            self.robot_class_list[robot_idx].traveled_distance += euclidean_distance(
                self.uniform_pos_dict[which_node], self.uniform_pos_dict['R' + str(robot_idx)])
            self.robot_class_list[robot_idx].travel_way_point.append(which_node)
        self.robot_class_list[robot_idx].travel_way_point.append(section.start)

    # Case 1: No transfer, no receive -> direct allocation
    def case1_allocator(self):
        for section_idx in self.case1_section_list:
            section = self.section_class_list[section_idx]
            if section.robot_num:
                for i in range(min(section.robot_num, section.goal_num)):
                    self.final_allocation_result.append([section.robot_list[i], section.goal_list[i]])
            section.robot_queue = []

    # Case 2: Transfer only (no receive) -> equalize then allocate
    def case2_allocator(self):
        for section_idx in self.case2_section_list:
            section = self.section_class_list[section_idx]
            transfer_D = section.num_of_transfer["D"]
            transfer_R = section.num_of_transfer["R"]
            transfer_E = section.num_of_transfer["E"]

            # JC node section
            if transfer_E:
                section.transfer_robot_dict["E"] = section.robot_list[:transfer_E]
                for i in range(transfer_E, len(section.robot_list)):
                    self.final_allocation_result.append([section.robot_list[i], section.goal_list[i - transfer_E]])

                for transfer_section_idx in self._sorted_transfer_targets(section):
                    how_many = section.transfer_section_index_dict[transfer_section_idx][1]
                    transfer_robot_list = section.transfer_robot_dict["E"][:how_many]
                    section.transfer_robot_dict["E"] = a_sub_b(section.transfer_robot_dict["E"], transfer_robot_list)
                    for robot_idx in transfer_robot_list:
                        self.robot_class_list[robot_idx].travel_way_point.append(section.start)
                    self._transfer_robots_to_section(transfer_robot_list, section_idx, transfer_section_idx, section)

            # D direction only
            elif transfer_D != 0 and transfer_R == 0:
                section.transfer_robot_dict["D"] = list(reversed(section.robot_list[len(section.robot_list) - transfer_D:]))
                for i in range(len(section.robot_list) - transfer_D):
                    self.final_allocation_result.append([section.robot_list[i], section.goal_list[i]])

                for transfer_section_idx in self._sorted_transfer_targets(section):
                    how_many = section.transfer_section_index_dict[transfer_section_idx][1]
                    transfer_robot_list = section.transfer_robot_dict["D"][:how_many]
                    section.transfer_robot_dict["D"] = a_sub_b(section.transfer_robot_dict["D"], transfer_robot_list)
                    for robot_idx in transfer_robot_list:
                        self._update_robot_travel_D(robot_idx, section, section_idx)
                    self._transfer_robots_to_section(transfer_robot_list, section_idx, transfer_section_idx, section)

            # R direction only
            elif transfer_D == 0 and transfer_R != 0:
                section.transfer_robot_dict["R"] = section.robot_list[:transfer_R]
                for i in range(transfer_R, len(section.robot_list)):
                    self.final_allocation_result.append([section.robot_list[i], section.goal_list[i - transfer_R]])

                for transfer_section_idx in self._sorted_transfer_targets(section):
                    how_many = section.transfer_section_index_dict[transfer_section_idx][1]
                    transfer_robot_list = section.transfer_robot_dict["R"][:how_many]
                    section.transfer_robot_dict["R"] = a_sub_b(section.transfer_robot_dict["R"], transfer_robot_list)
                    for robot_idx in transfer_robot_list:
                        self._update_robot_travel_R(robot_idx, section, section_idx)
                    self._transfer_robots_to_section(transfer_robot_list, section_idx, transfer_section_idx, section)

            # Both D and R directions
            elif transfer_D != 0 and transfer_R != 0:
                section.transfer_robot_dict["D"] = list(reversed(section.robot_list[len(section.robot_list) - transfer_D:]))
                section.transfer_robot_dict["R"] = section.robot_list[:transfer_R]
                for i in range(transfer_R, len(section.robot_list) - transfer_D):
                    self.final_allocation_result.append([section.robot_list[i], section.goal_list[i - transfer_R]])

                for transfer_section_idx in self._sorted_transfer_targets(section):
                    how_many = section.transfer_section_index_dict[transfer_section_idx][1]
                    dir_in_this_section = section.transfer_section_index_dict[transfer_section_idx][0]

                    if dir_in_this_section == "D":
                        transfer_robot_list = section.transfer_robot_dict["D"][:how_many]
                        section.transfer_robot_dict["D"] = a_sub_b(section.transfer_robot_dict["D"], transfer_robot_list)
                        for robot_idx in transfer_robot_list:
                            self._update_robot_travel_D(robot_idx, section, section_idx)
                    elif dir_in_this_section == "R":
                        transfer_robot_list = section.transfer_robot_dict["R"][:how_many]
                        section.transfer_robot_dict["R"] = a_sub_b(section.transfer_robot_dict["R"], transfer_robot_list)
                        for robot_idx in transfer_robot_list:
                            self._update_robot_travel_R(robot_idx, section, section_idx)

                    self._transfer_robots_to_section(transfer_robot_list, section_idx, transfer_section_idx, section)

    # Case 4: Both transfer and receive
    def case4_receive_n_transfer(self):
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

                # Check if all expected robots have been received
                if len(section.robot_queue) == goal_num + transfer_D + transfer_R + transfer_E:
                    # Sort received robots by traveled distance
                    if receive_D:
                        receive_robot_idx = section.receive_robot_dict["D"]
                        length_list = [self.robot_class_list[r].traveled_distance for r in receive_robot_idx]
                        pairs = sorted(zip(length_list, receive_robot_idx))
                        section.robot_queue = a_sub_b(section.robot_queue, receive_robot_idx)
                        for _, robot_idx in pairs:
                            section.robot_queue.insert(0, robot_idx)
                    elif receive_R:
                        receive_robot_idx = section.receive_robot_dict["R"]
                        length_list = [self.robot_class_list[r].traveled_distance for r in receive_robot_idx]
                        pairs = sorted(zip(length_list, receive_robot_idx))
                        section.robot_queue = a_sub_b(section.robot_queue, receive_robot_idx)
                        for _, robot_idx in pairs:
                            section.robot_queue.append(robot_idx)
                    elif receive_E:
                        receive_robot_idx = section.receive_robot_dict["E"]
                        length_list = [self.robot_class_list[r].traveled_distance for r in receive_robot_idx]
                        pairs = sorted(zip(length_list, receive_robot_idx))
                        section.robot_queue = a_sub_b(section.robot_queue, receive_robot_idx)
                        for _, robot_idx in pairs:
                            section.robot_queue.append(robot_idx)

                    # Set up transfer lists
                    if transfer_D:
                        section.transfer_robot_dict["D"] = list(reversed(
                            section.robot_queue[len(section.robot_queue) - transfer_D:]))
                    elif transfer_R:
                        section.transfer_robot_dict["R"] = section.robot_queue[:transfer_R]
                    elif transfer_E:
                        section.transfer_robot_dict["E"] = section.robot_queue[:transfer_E]

                    for transfer_section_idx in self._sorted_transfer_targets(section):
                        transfer_section = self.section_class_list[transfer_section_idx]
                        how_many = section.transfer_section_index_dict[transfer_section_idx][1]
                        dir_in_transfer_section = transfer_section.receive_section_index_dict[section_idx][0]
                        dir_in_this_section = section.transfer_section_index_dict[transfer_section_idx][0]

                        if dir_in_this_section == "D":
                            transfer_robot_list = section.transfer_robot_dict["D"][:how_many]
                            section.transfer_robot_dict["D"] = a_sub_b(section.transfer_robot_dict["D"], transfer_robot_list)
                            for robot_idx in transfer_robot_list:
                                if robot_idx in section.receive_robot_dict["D"] or robot_idx in section.receive_robot_dict["R"]:
                                    self.robot_class_list[robot_idx].traveled_distance += section.length
                                    self.robot_class_list[robot_idx].travel_way_point.append(section.end)
                                else:
                                    self._update_robot_travel_D(robot_idx, section, section_idx)

                        elif dir_in_this_section == "R":
                            transfer_robot_list = section.transfer_robot_dict["R"][:how_many]
                            section.transfer_robot_dict["R"] = a_sub_b(section.transfer_robot_dict["R"], transfer_robot_list)
                            for robot_idx in transfer_robot_list:
                                if robot_idx in section.receive_robot_dict["D"] or robot_idx in section.receive_robot_dict["R"]:
                                    self.robot_class_list[robot_idx].traveled_distance += section.length
                                    self.robot_class_list[robot_idx].travel_way_point.append(section.start)
                                else:
                                    self._update_robot_travel_R(robot_idx, section, section_idx)

                        elif dir_in_this_section == "E":
                            transfer_robot_list = section.transfer_robot_dict["E"][:how_many]
                            section.transfer_robot_dict["E"] = a_sub_b(section.transfer_robot_dict["E"], transfer_robot_list)
                            for robot_idx in transfer_robot_list:
                                self.robot_class_list[robot_idx].travel_way_point.append(section.start)

                        section.robot_queue = a_sub_b(section.robot_queue, transfer_robot_list)

                        if dir_in_transfer_section == "D":
                            transfer_section.receive_robot_dict["D"].extend(transfer_robot_list)
                            transfer_section.robot_queue = transfer_robot_list + transfer_section.robot_queue
                        elif dir_in_transfer_section == "R":
                            transfer_section.receive_robot_dict["R"].extend(transfer_robot_list)
                            transfer_section.robot_queue = transfer_section.robot_queue + transfer_robot_list
                        elif dir_in_transfer_section == "E":
                            transfer_section.receive_robot_dict["E"].extend(transfer_robot_list)
                            transfer_section.robot_queue = transfer_section.robot_queue + transfer_robot_list

                    received_done_section_idx.append(section_idx)
                else:
                    continue

            case4_section_queue = a_sub_b(case4_section_queue, received_done_section_idx)

    # Case 3 & 4: Final allocation after redistribution
    def case3_4_allocator(self):
        # Case 3 sections
        for section_idx in self.case3_section_list:
            section = self.section_class_list[section_idx]
            D_receive_num = section.num_of_receive["D"]
            R_receive_num = section.num_of_receive["R"]

            if D_receive_num > 0 and R_receive_num > 0:
                D_robot_list = section.receive_robot_dict["D"]
                R_robot_list = section.receive_robot_dict["R"]
                section.robot_queue = section.robot_queue[D_receive_num:-R_receive_num]

                pairs = sorted(zip(
                    [self.robot_class_list[r].traveled_distance for r in D_robot_list], D_robot_list))
                for _, robot_idx in pairs:
                    section.robot_queue.insert(0, robot_idx)

                pairs = sorted(zip(
                    [self.robot_class_list[r].traveled_distance for r in R_robot_list], R_robot_list))
                for _, robot_idx in pairs:
                    section.robot_queue.append(robot_idx)

            elif D_receive_num > 0:
                D_robot_list = section.receive_robot_dict["D"]
                section.robot_queue = section.robot_queue[D_receive_num:]
                pairs = sorted(zip(
                    [self.robot_class_list[r].traveled_distance for r in D_robot_list], D_robot_list))
                for _, robot_idx in pairs:
                    section.robot_queue.insert(0, robot_idx)

            elif R_receive_num > 0:
                R_robot_list = section.receive_robot_dict["R"]
                section.robot_queue = section.robot_queue[:-R_receive_num]
                pairs = sorted(zip(
                    [self.robot_class_list[r].traveled_distance for r in R_robot_list], R_robot_list))
                for _, robot_idx in pairs:
                    section.robot_queue.append(robot_idx)

            for i in range(section.goal_num):
                self.final_allocation_result.append([section.robot_queue[i], section.goal_list[i]])

        # Case 4 sections
        for section_idx in self.case4_section_list:
            section = self.section_class_list[section_idx]
            if section.goal_num != 0:
                robot_idx_need_to_search = a_sub_b(section.robot_queue, section.robot_list)

                if section.num_of_receive["D"]:
                    section.robot_queue = section.robot_queue[len(robot_idx_need_to_search):]
                    direction = "D"
                else:
                    section.robot_queue = section.robot_queue[:-len(robot_idx_need_to_search)]
                    direction = "R"

                pairs = sorted(zip(
                    [self.robot_class_list[r].traveled_distance for r in robot_idx_need_to_search],
                    robot_idx_need_to_search))

                if direction == "D":
                    for _, robot_idx in pairs:
                        section.robot_queue.insert(0, robot_idx)
                elif direction == "R":
                    for _, robot_idx in pairs:
                        section.robot_queue.append(robot_idx)

                for i in range(section.goal_num):
                    self.final_allocation_result.append([section.robot_queue[i], section.goal_list[i]])

    def on_init(self):
        self.case1_allocator()
        self.case2_allocator()
        self.case4_receive_n_transfer()
        self.case3_4_allocator()
        return self.final_allocation_result
