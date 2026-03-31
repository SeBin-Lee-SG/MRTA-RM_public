import copy
import json
import os
import time

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx

from src.GVD_generator import VBRM
from src.env_generator import env
from src.initial_allocator import InitialAllocation
from src.transfer_planner import Arbiter
from src.final_allocator import FinalAllocator
from func.my_map import load_map, MAP_REGISTRY


class MRTA_RM:
    """Multi-Robot Task Allocation via Robot Redistribution Mechanism.

    Pipeline: Roadmap -> Environment -> Initial Allocation -> Transfer Analysis -> Final Allocation
    """

    def __init__(self, test_set=4, robot_num=10, robot_size=8):
        self.test_set = test_set
        self.robot_num = robot_num
        self.goal_num = robot_num
        self.robot_size = robot_size

        # Load map metadata
        _, mapw, maph, max_robots, _ = load_map(test_set)
        self.mapw = mapw
        self.maph = maph

        if self.robot_num > max_robots:
            print(f"robot num limit = {max_robots}")
            self.robot_num = max_robots
            self.goal_num = max_robots

        self.separate = False

    # ---- Roadmap ----

    def make_roadmap(self):
        self.roadmap = VBRM(self.mapw, self.maph, self.test_set, self.robot_size)
        self.reusable_roadmap_result = self.roadmap.on_init()
        self._unpack_roadmap(copy.deepcopy(self.reusable_roadmap_result))

    def roadmap_reuse(self):
        self._unpack_roadmap(copy.deepcopy(self.reusable_roadmap_result))

    def _unpack_roadmap(self, result):
        self.uniform_G = result[0]
        self.uniform_node_size_list = result[1]
        self.uniform_node_color_list = result[2]
        self.uniform_pos_dict = result[3]
        self.section_class_list = result[4]
        self.section_class_dict = result[11]
        self.uniform_JC_node_list = result[5]
        self.uniform_additional_JC_node_list = result[6]
        self.uniform_edge_list = result[7]
        self.uniform_node_list = result[8]
        self.polygon_list = result[9]
        self.polygon_vertex_list = result[10]
        self.roadmap_time = result[12]
        self.full_section_number = result[13]

    # ---- Environment ----

    def make_env(self):
        self.robot_start_pos = []
        self.robot_goal_pos = []

        self.env = env(
            self.mapw, self.maph, self.robot_num, self.goal_num,
            self.robot_size, self.test_set, self.polygon_list,
            self.uniform_G, self.uniform_pos_dict,
            self.uniform_node_color_list, self.uniform_node_size_list,
            self.uniform_JC_node_list, self.uniform_additional_JC_node_list,
            self.uniform_node_list, self.section_class_list,
            self.separate, self.full_section_number,
            self.robot_start_pos, self.robot_goal_pos,
        )
        env_result = self.env.on_init()

        self.uniform_G = env_result[0]
        self.uniform_node_size_list = env_result[1]
        self.uniform_node_color_list = env_result[2]
        self.uniform_pos_dict = env_result[3]
        self.robot_class_list = env_result[4]
        self.goal_class_list = env_result[5]
        self.section_class_list = env_result[6]
        self.robot_start_pos = env_result[7]
        self.robot_goal_pos = env_result[8]
        self.time_wo_set_sp_gp = env_result[9]

    # ---- Allocation pipeline ----

    def initial_allocator(self):
        self.init_allocate = InitialAllocation(
            self.uniform_G, self.uniform_pos_dict,
            self.robot_class_list, self.goal_class_list,
            self.section_class_list, self.robot_num, self.goal_num,
            self.uniform_JC_node_list, self.uniform_additional_JC_node_list,
            self.full_section_number,
        )
        self.diff_seg_section_list = self.init_allocate.on_init()

    def planner(self):
        self.transfer_planner = Arbiter(self.section_class_list, self.diff_seg_section_list)
        result = self.transfer_planner.analyzer()
        self.case1_section_list = result[0]
        self.case2_section_list = result[1]
        self.case3_section_list = result[2]
        self.case4_section_list = result[3]

    def re_allocator(self):
        self.re_assign = FinalAllocator(
            self.uniform_G, self.uniform_pos_dict,
            self.section_class_list, self.robot_class_list, self.goal_class_list,
            self.case1_section_list, self.case2_section_list,
            self.case3_section_list, self.case4_section_list,
            self.full_section_number,
        )
        self.final_allocation_result = self.re_assign.on_init()

    # ---- Cost calculation ----

    def calc_cost(self):
        """Calculate total MRTA-RM cost from final allocation."""
        self.total_cost = 0
        for alloc in self.final_allocation_result:
            robot_idx, goal_idx = alloc[0], alloc[1]
            R2N = self.robot_class_list[robot_idx].dist_to_valid_vertex
            G2N = self.goal_class_list[goal_idx].dist_to_valid_vertex
            wp_list = self.robot_class_list[robot_idx].travel_way_point
            num_wp = len(wp_list)

            if num_wp == 0:
                RN2GN = nx.astar_path_length(
                    self.uniform_G,
                    self.robot_class_list[robot_idx].nearest_valid_vertex,
                    self.goal_class_list[goal_idx].nearest_valid_vertex)
                self.total_cost += R2N + G2N + RN2GN
            elif num_wp == 1:
                RN2J = nx.astar_path_length(
                    self.uniform_G,
                    self.robot_class_list[robot_idx].nearest_valid_vertex, wp_list[0])
                J2GN = nx.astar_path_length(
                    self.uniform_G,
                    wp_list[0], self.goal_class_list[goal_idx].nearest_valid_vertex)
                self.total_cost += R2N + G2N + RN2J + J2GN
            else:
                RN2J = nx.astar_path_length(
                    self.uniform_G,
                    self.robot_class_list[robot_idx].nearest_valid_vertex, wp_list[0])
                J2GN = nx.astar_path_length(
                    self.uniform_G,
                    wp_list[-1], self.goal_class_list[goal_idx].nearest_valid_vertex)
                for i in range(num_wp - 1):
                    J2J = nx.astar_path_length(self.uniform_G, wp_list[i], wp_list[i + 1])
                    self.total_cost += J2J
                self.total_cost += R2N + G2N + RN2J + J2GN

        return self.total_cost

    # ---- Output ----

    def save_result(self, output_dir="output"):
        """Save allocation result as JSON to the output directory."""
        os.makedirs(output_dir, exist_ok=True)

        # Prepare polygon data (y-axis flip for visualization)
        polygon_data = []
        for polygon in self.polygon_vertex_list:
            new_polygon = polygon[:-1] if len(polygon) > 1 else polygon
            new_polygon = [[v[0], self.maph - v[1]] for v in new_polygon]
            polygon_data.append(new_polygon)

        # Build result dict
        result = {
            "map_width": self.mapw,
            "map_height": self.maph,
            "robot_size": self.robot_size,
            "robot_num": self.robot_num,
            "total_cost": self.total_cost,
            "roadmap_time_sec": self.roadmap_time,
            "run_time_sec": self.run_time,
            "allocations": [],
            "polygons": polygon_data,
        }

        for alloc in self.final_allocation_result:
            robot_idx, goal_idx = alloc[0], alloc[1]
            robot = self.robot_class_list[robot_idx]
            goal = self.goal_class_list[goal_idx]
            waypoints = []
            for wp in robot.travel_way_point:
                pos = self.uniform_pos_dict[wp]
                waypoints.append([pos[0], self.maph - pos[1]])

            result["allocations"].append({
                "robot_index": robot_idx,
                "goal_index": goal_idx,
                "robot_start": list(self.robot_start_pos[robot_idx]),
                "goal_pos": list(self.robot_goal_pos[goal_idx]),
                "waypoints": waypoints,
            })

        map_name = MAP_REGISTRY.get(self.test_set, str(self.test_set))
        filename = f"{map_name}_r{self.robot_num}.json"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Result saved to {filepath}")

    # ---- Visualization ----

    def plot_roadmap(self):
        fig = plt.gcf()
        fig.set_size_inches(10, 10)
        ax = fig.gca()

        new_polygon_vertex_list = self._flip_polygons()
        for polygon_vertices in new_polygon_vertex_list:
            polygon = patches.Polygon(polygon_vertices, fill=True, facecolor='grey', edgecolor='grey')
            ax.add_patch(polygon)

        # Remove robot/goal nodes for clean roadmap view
        uniform_G_copy = copy.deepcopy(self.uniform_G)
        color_list = copy.deepcopy(self.uniform_node_color_list)
        size_list = copy.deepcopy(self.uniform_node_size_list)
        for node in list(uniform_G_copy.nodes):
            if isinstance(node, str) and (node[0] == 'R' or node[0] == 'G'):
                idx = list(uniform_G_copy.nodes).index(node)
                uniform_G_copy.remove_node(node)
                del color_list[idx]
                del size_list[idx]

        nx.draw(uniform_G_copy, node_size=7, node_color=color_list,
                pos=self.uniform_pos_dict, width=1, with_labels=False)
        ax.set_xlim([0, self.mapw])
        ax.set_ylim([0, self.maph])
        plt.show()

    def plot_allocation(self):
        edge_color_list = ['black'] * len(self.uniform_G.edges())
        width_list = [1] * len(self.uniform_G.edges())

        fig = plt.gcf()
        fig.set_size_inches(self.mapw / 100, self.maph / 100)
        ax = fig.gca()

        new_polygon_vertex_list = self._flip_polygons()
        for polygon_vertices in new_polygon_vertex_list:
            polygon = patches.Polygon(polygon_vertices, fill=True, facecolor='grey', edgecolor='grey')
            ax.add_patch(polygon)

        for allocation in self.final_allocation_result:
            self.uniform_G.add_edge('R' + str(allocation[0]), 'G' + str(allocation[1]), weight=99999)
            edge_color_list.append('red')
            width_list.append(2)

        nx.draw(self.uniform_G, pos=self.uniform_pos_dict,
                node_size=[i / 8 for i in self.uniform_node_size_list],
                node_color=self.uniform_node_color_list,
                edge_color=edge_color_list, width=width_list, with_labels=False)
        plt.show()

        # Clean up added edges
        for allocation in self.final_allocation_result:
            self.uniform_G.remove_edge('R' + str(allocation[0]), 'G' + str(allocation[1]))

    def _flip_polygons(self):
        """Flip polygon y-coordinates for matplotlib display."""
        flipped = []
        for polygon in self.polygon_vertex_list:
            new_polygon = [(v[0], self.maph - v[1]) for v in polygon]
            flipped.append(new_polygon)
        return flipped

    # ---- Main entry ----

    def run(self, num=1, show_roadmap=False, show_result=False):
        """Run the full MRTA-RM pipeline.

        Args:
            num: Iteration number (1 = build roadmap, >1 = reuse).
            show_roadmap: Display the roadmap visualization.
            show_result: Display the allocation result visualization.

        Returns:
            List of [robot_index, goal_index] pairs.
        """
        print(f"--- Run #{num} | test_set={self.test_set} | robots={self.robot_num} ---")

        if num == 1:
            self.make_roadmap()
        else:
            self.roadmap_reuse()

        self.make_env()

        start_time = time.time()
        self.initial_allocator()
        self.planner()
        self.re_allocator()
        self.run_time = time.time() - start_time + self.time_wo_set_sp_gp

        self.calc_cost()

        print(f"Roadmap time: {self.roadmap_time:.3f} sec")
        print(f"Allocation time (w/o roadmap): {self.run_time:.3f} sec")
        print(f"Total cost: {self.total_cost:.2f}")
        print(f"Allocations: {self.final_allocation_result}")

        if show_roadmap:
            self.plot_roadmap()
        if show_result:
            self.plot_allocation()

        self.save_result()

        return self.final_allocation_result


if __name__ == "__main__":
    # Available test_sets: 1(random), 2(dept_store), 3(warehouse), 4(demo),
    #                      11(random_mini), 22(dept_store_mini), 33(warehouse_mini)
    app = MRTA_RM(test_set=4, robot_num=10)
    app.run(show_roadmap=False, show_result=False)
