from scipy.optimize import linear_sum_assignment
from func.func import calc_path_length, dist, extract_path2section
import numpy as np
import networkx as nx


class InitialAllocation:
    """Performs initial robot-to-goal allocation using section-level balancing
    followed by Dijkstra-based Hungarian matching on a coarse section graph."""

    def __init__(self, graph, pos_dict, robot_class_list, goal_class_list,
                 section_class_list, robot_num, goal_num,
                 uniform_JC_node_list, uniform_additional_JC_node_list,
                 full_section_number):
        self.uniform_G = graph
        self.uniform_pos_dict = pos_dict
        self.robot_class_list = robot_class_list
        self.goal_class_list = goal_class_list
        self.section_class_list = section_class_list
        self.robot_num = robot_num
        self.goal_num = goal_num
        self.uniform_JC_node_list = uniform_JC_node_list
        self.uniform_additional_JC_node_list = uniform_additional_JC_node_list
        self.full_section_number = full_section_number
        self._path_and_length_cache = {}

    def dist_for_simple_astar(self, a, b):
        return dist(a, b, self.simple_G_pos_dict)

    def allocate_within_sections(self):
        """Count surplus robots/goals in each section."""
        self.remain_robot_list = []
        self.remain_goal_list = []
        for section in self.section_class_list:
            if section.robot_num > section.goal_num:
                section.remain_robot_num = section.robot_num - section.goal_num
            elif section.robot_num < section.goal_num:
                section.remain_goal_num = section.goal_num - section.robot_num

    def make_simple_graph(self):
        """Build a coarse section graph for cross-section allocation."""
        self.simple_G = nx.Graph()
        self.remain_robot_list = []
        self.remain_goal_list = []
        self.simple_G_pos_dict = {}
        for section in self.section_class_list[:self.full_section_number + 1]:
            section_label = "S" + str(self.section_class_list.index(section))
            self.simple_G.add_edge(section.start, section_label, weight=section.length / 2)
            self.simple_G.add_edge(section.end, section_label, weight=section.length / 2)
            self.simple_G_pos_dict[section.start] = self.uniform_pos_dict[section.start]
            self.simple_G_pos_dict[section.end] = self.uniform_pos_dict[section.end]
            self.simple_G_pos_dict[section_label] = [
                (self.uniform_pos_dict[section.start][0] + self.uniform_pos_dict[section.end][0]) / 2,
                (self.uniform_pos_dict[section.start][1] + self.uniform_pos_dict[section.end][1]) / 2,
            ]
            for _ in range(section.remain_robot_num):
                self.remain_robot_list.append(section_label)
            for _ in range(section.remain_goal_num):
                self.remain_goal_list.append(section_label)

        for section in self.section_class_list[self.full_section_number + 1:]:
            for _ in range(section.remain_robot_num):
                self.remain_robot_list.append(section.start)
            for _ in range(section.remain_goal_num):
                self.remain_goal_list.append(section.start)

    def allocate_remaining(self):
        """Allocate remaining robots via Dijkstra + Hungarian on the section graph."""
        if len(self.remain_robot_list) == 0:
            return []

        all_cost_list = []
        all_path_list = []

        for robot in self.remain_robot_list:
            length_dict, path_dict = nx.single_source_dijkstra(self.simple_G, robot)
            cost_list = []
            path_list = []
            for gp in self.remain_goal_list:
                if gp in path_dict:
                    cost_list.append(length_dict[gp])
                    path_list.append(path_dict[gp])
                else:
                    cost_list.append(float('inf'))
                    path_list.append([])
            all_cost_list.append(cost_list)
            all_path_list.append(path_list)

        path_dist_matrix = np.array(all_cost_list)
        row_ind, col_ind = linear_sum_assignment(path_dist_matrix)

        JC_nodes = self.uniform_JC_node_list + self.uniform_additional_JC_node_list
        return [
            extract_path2section(all_path_list[row][col], JC_nodes, self.full_section_number)
            for row, col in zip(row_ind, col_ind)
        ]

    def allocation(self):
        self.allocate_within_sections()
        self.make_simple_graph()
        return self.allocate_remaining()

    def on_init(self):
        return self.allocation()
