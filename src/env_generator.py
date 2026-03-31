import copy
import time
from tqdm import tqdm
import shapely.geometry as sh
import numpy as np

from func.my_class import obj_node_class
from func.func import euclidean_distance, dist


class env:
    """Generates robot/goal positions and connects them to the roadmap graph."""

    def __init__(self, mapw, maph, robot_num, goal_num, robot_radius, test_set,
                 polygon_list, uniform_G, uniform_pos_dict, uniform_color_list,
                 uniform_size_list, uniform_JC_node_list, uniform_additional_JC_node_list,
                 uniform_node_list, section_class_list, separate, full_section_number,
                 robot_start_pos=None, robot_goal_pos=None):
        self.mapw = mapw
        self.maph = maph
        self.test_set = test_set
        self.robot_num = robot_num
        self.goal_num = goal_num
        self.robot_radius = robot_radius
        self.goal_radius = robot_radius

        self.polygon_list = polygon_list
        self.uniform_G = uniform_G
        self.uniform_pos_dict = uniform_pos_dict
        self.uniform_color_list = uniform_color_list
        self.uniform_size_list = uniform_size_list
        self.uniform_JC_node_list = uniform_JC_node_list
        self.uniform_additional_JC_node_list = uniform_additional_JC_node_list
        self.uniform_node_list = uniform_node_list
        self.section_class_list = section_class_list
        self.separate = separate
        self.full_section_number = full_section_number

        self.robot_start_pos = robot_start_pos if robot_start_pos is not None else []
        self.robot_goal_pos = robot_goal_pos if robot_goal_pos is not None else []

    def _random_pos(self, is_goal=False):
        """Generate a random position, respecting the 'separate' flag."""
        if self.separate and not is_goal:
            x = np.random.randint(self.robot_radius, self.mapw // 2 - self.robot_radius)
        elif self.separate and is_goal:
            x = np.random.randint(self.mapw // 2 + self.robot_radius, self.mapw - self.robot_radius)
        else:
            x = np.random.randint(self.robot_radius, self.mapw - self.robot_radius)
        y = np.random.randint(self.robot_radius, self.maph - self.robot_radius)
        return (x, y)

    def _collides_with_obstacles(self, pos, radius):
        point_buf = sh.Point(pos).buffer(radius)
        return any(poly.intersects(point_buf) for poly in self.polygon_list)

    def _too_close_to_existing(self, pos, existing_positions, min_dist):
        return any(euclidean_distance(pos, p) < min_dist for p in existing_positions)

    def set_sp_gp(self):
        """Generate start/goal positions if not already provided."""
        if self.robot_num == len(self.robot_start_pos) and self.goal_num == len(self.robot_goal_pos):
            pass
        else:
            min_dist = 2 * self.robot_radius + 1
            with tqdm(total=self.robot_num, desc="Generating robot start positions") as pbar:
                while len(self.robot_start_pos) < self.robot_num:
                    pos = self._random_pos(is_goal=False)
                    if self._collides_with_obstacles(pos, self.robot_radius):
                        continue
                    if self._too_close_to_existing(pos, self.robot_start_pos, min_dist):
                        continue
                    self.robot_start_pos.append(pos)
                    pbar.update(1)

            with tqdm(total=self.goal_num, desc="Generating robot goal positions ") as pbar:
                while len(self.robot_goal_pos) < self.goal_num:
                    pos = self._random_pos(is_goal=True)
                    if self._collides_with_obstacles(pos, self.goal_radius):
                        continue
                    if self._too_close_to_existing(pos, self.robot_goal_pos, min_dist):
                        continue
                    self.robot_goal_pos.append(pos)
                    pbar.update(1)

        print("robot_start_pos =", self.robot_start_pos)
        print("robot_goal_pos =", self.robot_goal_pos)

    def make_robot_and_goal_class_list(self):
        self.robot_class_list = []
        self.goal_class_list = []
        for i in range(self.robot_num):
            self.robot_class_list.append(obj_node_class(pos=self.robot_start_pos[i]))
        for i in range(self.goal_num):
            self.goal_class_list.append(obj_node_class(pos=self.robot_goal_pos[i]))

    def make_uniform_graph(self):
        for i in range(self.robot_num):
            self.uniform_G.add_node("R" + str(i))
            self.uniform_pos_dict["R" + str(i)] = [
                self.robot_start_pos[i][0], self.maph - self.robot_start_pos[i][1]]
            self.uniform_color_list.append("green")
            self.uniform_size_list.append(200)
        for i in range(self.goal_num):
            self.uniform_G.add_node("G" + str(i))
            self.uniform_pos_dict["G" + str(i)] = [
                self.robot_goal_pos[i][0], self.maph - self.robot_goal_pos[i][1]]
            self.uniform_color_list.append("yellow")
            self.uniform_size_list.append(200)

    def generate_key(self, prefix, index):
        return prefix + str(index)

    def process_class_list(self, class_list, prefix, valid_vertex_list):
        for entity_class in class_list:
            key_distance_list = [
                (vertex, dist(self.generate_key(prefix, class_list.index(entity_class)),
                              vertex, self.uniform_pos_dict))
                for vertex in valid_vertex_list
            ]
            sorted_key_distance_list = sorted(key_distance_list, key=lambda x: x[1])
            sorted_keys = [item[0] for item in sorted_key_distance_list]
            sorted_distances = [item[1] for item in sorted_key_distance_list]

            for i in range(len(sorted_keys)):
                if self.check_intersection(
                    sorted_keys[i], class_list.index(entity_class),
                    prefix == "R", prefix == "G"):
                    continue
                entity_class.dist_to_valid_vertex = sorted_distances[i]
                entity_class.nearest_valid_vertex = sorted_keys[i]
                break

    def check_intersection(self, node_index, obj_index, robot, goal):
        if robot:
            obj_pos = self.uniform_pos_dict["R" + str(obj_index)]
        elif goal:
            obj_pos = self.uniform_pos_dict["G" + str(obj_index)]
        else:
            return False
        obj_pos = [obj_pos[0], self.maph - obj_pos[1]]

        node_pos = self.uniform_pos_dict[node_index]
        node_pos = [node_pos[0], self.maph - node_pos[1]]

        line = sh.LineString([node_pos, obj_pos])
        return any(line.intersects(polygon) for polygon in self.polygon_list)

    def find_nearest_valid_vertex(self):
        JC_nodes = self.uniform_JC_node_list + self.uniform_additional_JC_node_list

        self.process_class_list(self.robot_class_list, "R", self.uniform_node_list)
        self.process_class_list(self.goal_class_list, "G", self.uniform_node_list)

        for i in range(self.robot_num):
            self.uniform_G.add_edge(
                "R" + str(i), self.robot_class_list[i].nearest_valid_vertex,
                weight=self.robot_class_list[i].dist_to_valid_vertex, color="black")
        for i in range(self.goal_num):
            self.uniform_G.add_edge(
                "G" + str(i), self.goal_class_list[i].nearest_valid_vertex,
                weight=self.goal_class_list[i].dist_to_valid_vertex, color="black")

        for robot_class in self.robot_class_list:
            nearest_valid_vertex = robot_class.nearest_valid_vertex
            if nearest_valid_vertex in JC_nodes:
                section = self.section_class_list[self.full_section_number + JC_nodes.index(nearest_valid_vertex) + 1]
                section.robot_num += 1
                section.robot_list.append(self.robot_class_list.index(robot_class))
                section.robot_vertex_list.append(nearest_valid_vertex)
                robot_class.section.append(self.section_class_list.index(section))
            else:
                for section in self.section_class_list:
                    if nearest_valid_vertex in section.way_point:
                        section.robot_num += 1
                        section.robot_list.append(self.robot_class_list.index(robot_class))
                        section.robot_vertex_list.append(nearest_valid_vertex)
                        robot_class.section.append(self.section_class_list.index(section))
                        break

        for goal_class in self.goal_class_list:
            nearest_valid_vertex = goal_class.nearest_valid_vertex
            if nearest_valid_vertex in JC_nodes:
                section = self.section_class_list[self.full_section_number + JC_nodes.index(nearest_valid_vertex) + 1]
                section.goal_num += 1
                section.goal_list.append(self.goal_class_list.index(goal_class))
                section.goal_vertex_list.append(nearest_valid_vertex)
                goal_class.section.append(self.section_class_list.index(section))
            else:
                for section in self.section_class_list:
                    if nearest_valid_vertex in section.way_point:
                        section.goal_num += 1
                        section.goal_list.append(self.goal_class_list.index(goal_class))
                        section.goal_vertex_list.append(nearest_valid_vertex)
                        goal_class.section.append(self.section_class_list.index(section))
                        break

    def sort_section(self):
        for section in self.section_class_list[:self.full_section_number + 1]:
            robot_vertex_list = section.robot_vertex_list[:]
            goal_vertex_list = section.goal_vertex_list[:]
            robot_index_list = section.robot_list[:]
            goal_index_list = section.goal_list[:]

            tmp_robot_list = []
            tmp_goal_list = []
            tmp_robot_vertex_list = []
            tmp_goal_vertex_list = []

            start_point = self.uniform_pos_dict[section.start]

            while len(robot_vertex_list) > 0:
                min_vertex = min(robot_vertex_list)
                min_indices = [i for i, v in enumerate(robot_vertex_list) if v == min_vertex]
                if len(min_indices) > 1:
                    distances = [
                        euclidean_distance(start_point, self.uniform_pos_dict["R" + str(robot_index_list[i])])
                        for i in min_indices]
                    min_distance_index = min_indices[distances.index(min(distances))]
                else:
                    min_distance_index = robot_vertex_list.index(min_vertex)
                tmp_robot_list.append(robot_index_list[min_distance_index])
                tmp_robot_vertex_list.append(robot_vertex_list[min_distance_index])
                del robot_index_list[min_distance_index]
                del robot_vertex_list[min_distance_index]

            while len(goal_vertex_list) > 0:
                min_vertex = min(goal_vertex_list)
                min_indices = [i for i, v in enumerate(goal_vertex_list) if v == min_vertex]
                if len(min_indices) > 1:
                    distances = [
                        euclidean_distance(start_point, self.uniform_pos_dict["G" + str(goal_index_list[i])])
                        for i in min_indices]
                    min_distance_index = min_indices[distances.index(min(distances))]
                else:
                    min_distance_index = goal_vertex_list.index(min_vertex)
                tmp_goal_list.append(goal_index_list[min_distance_index])
                tmp_goal_vertex_list.append(goal_vertex_list[min_distance_index])
                del goal_index_list[min_distance_index]
                del goal_vertex_list[min_distance_index]

            section.robot_list = tmp_robot_list
            section.robot_queue = tmp_robot_list[:]
            section.goal_list = tmp_goal_list
            section.robot_vertex_list = tmp_robot_vertex_list
            section.goal_vertex_list = tmp_goal_vertex_list

        for section in self.section_class_list[self.full_section_number + 1:]:
            section.robot_queue = copy.deepcopy(section.robot_list)

    def on_init(self):
        self.set_sp_gp()
        time_wo_set_sp_gp = time.time()
        self.make_robot_and_goal_class_list()
        self.make_uniform_graph()
        self.find_nearest_valid_vertex()
        self.sort_section()
        time_wo_set_sp_gp = time.time() - time_wo_set_sp_gp

        return (self.uniform_G, self.uniform_size_list, self.uniform_color_list,
                self.uniform_pos_dict, self.robot_class_list, self.goal_class_list,
                self.section_class_list, self.robot_start_pos, self.robot_goal_pos,
                time_wo_set_sp_gp)
