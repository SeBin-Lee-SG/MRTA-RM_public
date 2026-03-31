import shapely.geometry as sh
import shapely.ops as sho
from scipy.spatial import Voronoi
import copy
import time
import numpy as np
from func.my_class import section_class, temp_section_class
from func.func import euclidean_distance, dist, a_sub_b
from func.my_map import load_map
from collections import deque

NODE_INTERVAL = 1


class VBRM:
    """Visibility-Based Roadmap generator.

    Builds a Voronoi-based roadmap from polygon obstacles, then creates
    a uniform graph with evenly-spaced nodes partitioned into sections.
    """

    def __init__(self, mapw, maph, test_set, robot_radius):
        self.mapw = mapw
        self.maph = maph
        self.test_set = test_set
        self.polygon_vertex_list = []

        self.Box_line_list = [
            sh.LineString([(0, 0), (0, self.maph)]),
            sh.LineString([(0, self.maph), (self.mapw, self.maph)]),
            sh.LineString([(self.mapw, self.maph), (self.mapw, 0)]),
            sh.LineString([(self.mapw, 0), (0, 0)]),
        ]

        self.robot_radius = robot_radius
        self.goal_radius = robot_radius
        self.sampling_dist = int(self.robot_radius * 2)

    def set_environment(self):
        polygons, _, _, _, _ = load_map(self.test_set)
        self.polygon_vertex_list = polygons

    def set_polygon(self):
        self.polygon_list = []
        multi_polygon = []
        for poly_vertex in self.polygon_vertex_list:
            multi_polygon.append(sh.Polygon(poly_vertex))
        unary_union_result = sho.unary_union(multi_polygon)

        if isinstance(unary_union_result, sh.MultiPolygon):
            individual_polygons = list(unary_union_result.geoms)
        else:
            individual_polygons = [unary_union_result]

        for polygon in individual_polygons:
            self.polygon_vertex_list.append(list(polygon.exterior.coords)[:-1])
            self.polygon_list.append(polygon)

    def _sample_boundary(self, vertex_list, boundary_sample):
        for poly in vertex_list:
            poly.append(poly[0])
        for poly in range(0, len(vertex_list)):
            for edge in range(0, len(vertex_list[poly]) - 1):
                edge_len = euclidean_distance(vertex_list[poly][edge], vertex_list[poly][edge + 1])
                how_many_sample = int(edge_len / self.sampling_dist) + 1
                for i in range(0, how_many_sample):
                    boundary_sample.append((
                        vertex_list[poly][edge][0] + (
                                vertex_list[poly][edge + 1][0] - vertex_list[poly][edge][0]) * i / how_many_sample,
                        vertex_list[poly][edge][1] + (
                                vertex_list[poly][edge + 1][1] - vertex_list[poly][edge][1]) * i / how_many_sample
                    ))

    def sampling_boundary(self):
        self.bounding_box = [(0, 0), (0, self.maph), (self.mapw, self.maph), (self.mapw, 0)]
        self.bounding_box_sample = []
        self.polygon_boundary_sample = []
        self._sample_boundary([self.bounding_box], self.bounding_box_sample)
        self._sample_boundary(self.polygon_vertex_list, self.polygon_boundary_sample)

    def make_voronoi(self):
        temp = self.bounding_box_sample + self.polygon_boundary_sample
        self.vor = Voronoi(temp)

    def find_valid_vertices(self):
        self.vor_valid_vertices = []
        self.vor_valid_vertices_index = []
        Box = sh.Polygon(self.bounding_box)

        for i, vertex in enumerate(self.vor.vertices):
            if Box.contains(sh.Point(vertex)):
                for polygon in self.polygon_list:
                    if polygon.contains(sh.Point(vertex)):
                        break
                else:
                    self.vor_valid_vertices.append(vertex)
                    self.vor_valid_vertices_index.append(i)
        self.vor_valid_vertices = np.array(self.vor_valid_vertices).reshape(-1, 2)

    def find_confine_nodes(self):
        self.confined_vertices_index = []
        for vertex in self.vor_valid_vertices_index:
            width_to_polygon = min(sh.Point(self.vor.vertices[vertex]).distance(poly) for poly in self.polygon_list)
            width_to_boundary = min(sh.Point(self.vor.vertices[vertex]).distance(line) for line in self.Box_line_list)
            width = min(width_to_polygon, width_to_boundary)
            if width < self.robot_radius - 0.5:
                self.confined_vertices_index.append(vertex)

    def find_valid_ridge_vertices_ver2(self):
        self.vor_valid_ridge_vertices = []
        self.unrolled_real_valid_ridge_vertices_index = []
        self.vor_edge_list = []

        for ridge in self.vor.ridge_vertices:
            if ridge[0] != -1 and ridge[1] != -1:
                self.vor_valid_ridge_vertices.append(ridge)

        valid_vertices_set = set(self.vor_valid_vertices_index)
        confined_vertices_set = set(self.confined_vertices_index)

        for ridge in self.vor_valid_ridge_vertices:
            if all(vertex in valid_vertices_set and vertex not in confined_vertices_set for vertex in ridge):
                self.unrolled_real_valid_ridge_vertices_index.extend(ridge)
                self.vor_edge_list.append(tuple(ridge))

        self.unrolled_real_valid_ridge_vertices_index = list(self.unrolled_real_valid_ridge_vertices_index)

    def find_JC_node(self):
        self.JC_node_list = []
        for node in set(self.unrolled_real_valid_ridge_vertices_index):
            if self.unrolled_real_valid_ridge_vertices_index.count(node) > 2:
                self.JC_node_list.append(node)

    def networkX_attribute(self):
        import networkx as nx
        self.pos_dict = {}
        self.color_list = []
        self.size_list = []

        temp_vor_valid_vertices_index = copy.deepcopy(self.vor_valid_vertices_index)
        temp_vor_valid_vertices = copy.deepcopy(self.vor_valid_vertices)

        self.vor_valid_vertices_index = []
        self.vor_valid_vertices = []

        for index in temp_vor_valid_vertices_index:
            if index not in self.confined_vertices_index:
                self.vor_valid_vertices_index.append(index)
                self.vor_valid_vertices.append(temp_vor_valid_vertices[temp_vor_valid_vertices_index.index(index)])

        self.vor_valid_vertices_index = a_sub_b(self.vor_valid_vertices_index, self.confined_vertices_index)
        for i in range(len(self.vor_valid_vertices)):
            self.pos_dict[self.vor_valid_vertices_index[i]] = [self.vor_valid_vertices[i][0],
                                                               self.maph - self.vor_valid_vertices[i][1]]
            if self.vor_valid_vertices_index[i] in self.JC_node_list:
                self.color_list.append('red')
                self.size_list.append(100)
            else:
                self.color_list.append('blue')
                self.size_list.append(20)

    def make_weight_list(self):
        self.weight_list = []
        for edge in self.vor_edge_list:
            self.weight_list.append(dist(edge[0], edge[1], self.pos_dict))

    def make_node(self):
        import networkx as nx
        self.G = nx.Graph()
        self.G.add_nodes_from(self.vor_valid_vertices_index)

    def make_edge(self):
        for i in range(len(self.vor_edge_list)):
            self.G.add_edge(self.vor_edge_list[i][0], self.vor_edge_list[i][1],
                            weight=self.weight_list[i], color='black')

    def insert_end_point_to_JC_node(self):
        self.additional_JC_node_list = [node for node in self.vor_valid_vertices_index if len(self.G.adj[node]) == 1]
        for node in self.additional_JC_node_list:
            node_index = self.vor_valid_vertices_index.index(node)
            self.color_list[node_index] = 'red'
            self.size_list[node_index] = 100

    def calc_path_length(self, path, graph):
        length = 0
        for i in range(len(path) - 1):
            length += graph.get_edge_data(path[i], path[i + 1])['weight']
        return length

    def dist_for_astar(self, a, b):
        return dist(a, b, self.pos_dict)

    def bfs_with_sections(self, graph, start_node):
        visited = {start_node}
        queue = deque([(start_node, [start_node])])
        while queue:
            current_node, path = queue.popleft()
            if current_node in self.JC_node_list and current_node != start_node:
                yield path
                continue
            for node in graph.neighbors(current_node):
                if node not in visited:
                    visited.add(node)
                    queue.append((node, path + [node]))

    def partitioning_temp_section(self):
        self.temp_section_class_list = []
        section_set = set()

        for start in self.JC_node_list:
            for path in self.bfs_with_sections(self.G, start):
                section_tuple = tuple(sorted((path[0], path[-1])))
                if section_tuple not in section_set:
                    section_set.add(section_tuple)
                    path_length = self.calc_path_length(path, self.G)
                    self.temp_section_class_list.append(temp_section_class(
                        start=path[0], end=path[-1],
                        way_point=path, length=path_length
                    ))

        for start in self.additional_JC_node_list:
            for path in self.bfs_with_sections(self.G, start):
                section_tuple = tuple(sorted((path[0], path[-1])))
                if section_tuple not in section_set:
                    section_set.add(section_tuple)
                    path_length = self.calc_path_length(path, self.G)
                    self.temp_section_class_list.append(temp_section_class(
                        start=path[0], end=path[-1],
                        way_point=path, length=path_length
                    ))

    def process_nodes(self, nodes, base_to_uniform_JC_node_dict, offset=0):
        uniform_node_list = []
        uniform_pos_dict = {}
        uniform_color_list = []
        uniform_size_list = []
        for i, node in enumerate(nodes):
            uniform_node = i + offset
            base_to_uniform_JC_node_dict[node] = uniform_node
            uniform_node_list.append(uniform_node)
            uniform_pos_dict[uniform_node] = [self.vor.vertices[node][0], self.maph - self.vor.vertices[node][1]]
            uniform_color_list.append('red')
            uniform_size_list.append(100)
        return uniform_node_list, uniform_pos_dict, uniform_color_list, uniform_size_list, base_to_uniform_JC_node_dict

    def make_uniform_graph_attribute(self):
        self.uniform_JC_node_list = []
        self.uniform_additional_JC_node_list = []
        self.uniform_node_list = []
        self.uniform_pos_dict = {}
        self.uniform_color_list = []
        self.uniform_size_list = []
        self.base_to_uniform_JC_node_dict = {}
        self.uniform_edge_list = []
        self.uniform_section_way_point_list = []

        self.uniform_JC_node_list, self.uniform_pos_dict, self.uniform_color_list, self.uniform_size_list, self.base_to_uniform_JC_node_dict = self.process_nodes(
            self.JC_node_list, self.base_to_uniform_JC_node_dict)

        offset = len(self.uniform_JC_node_list)
        self.uniform_additional_JC_node_list, temp_uniform_pos_dict, temp_uniform_color_list, temp_uniform_size_list, self.base_to_uniform_JC_node_dict = self.process_nodes(
            self.additional_JC_node_list, self.base_to_uniform_JC_node_dict, offset)

        self.uniform_node_list = self.uniform_JC_node_list + self.uniform_additional_JC_node_list
        self.uniform_pos_dict.update(temp_uniform_pos_dict)
        self.uniform_color_list += temp_uniform_color_list
        self.uniform_size_list += temp_uniform_size_list

        for section in self.temp_section_class_list:
            way_point = section.way_point
            section_length = section.length
            if section_length <= (2 * NODE_INTERVAL) * self.robot_radius:
                center_node_index = len(self.uniform_node_list)
                x, y = way_point[0], way_point[-1]
                self.uniform_pos_dict[center_node_index] = self.vor.vertices[x] + (
                        self.vor.vertices[y] - self.vor.vertices[x]) / 2
                self.uniform_pos_dict[center_node_index] = [self.uniform_pos_dict[center_node_index][0],
                                                            self.maph - self.uniform_pos_dict[center_node_index][1]]

                self.uniform_node_list.append(center_node_index)
                self.uniform_color_list.append('blue')
                self.uniform_size_list.append(20)

                self.uniform_edge_list.append([self.base_to_uniform_JC_node_dict[way_point[0]], center_node_index])
                self.uniform_edge_list.append([center_node_index, self.base_to_uniform_JC_node_dict[way_point[-1]]])

                self.uniform_section_way_point_list.append(
                    [self.base_to_uniform_JC_node_dict[way_point[0]], center_node_index,
                     self.base_to_uniform_JC_node_dict[way_point[-1]]])
            else:
                length_list = []
                accumulate_length_list = []
                for edge in range(len(way_point) - 1):
                    length_list.append(self.G.get_edge_data(way_point[edge], way_point[edge + 1])['weight'])
                    accumulate_length_list.append(sum(length_list))
                how_many_node = int(section_length / (2 * NODE_INTERVAL * self.robot_radius)) + 1
                uniform_dist = section_length / how_many_node
                how_many_node -= 1
                ind = []
                for i in range(how_many_node):
                    for j in range(len(accumulate_length_list)):
                        if accumulate_length_list[j] >= uniform_dist * (i + 1):
                            ind.append(j)
                            break

                section_way_point = [self.base_to_uniform_JC_node_dict[way_point[0]]]

                for i in range(len(ind)):
                    new_node_index = len(self.uniform_node_list)
                    section_way_point.append(new_node_index)
                    self.uniform_node_list.append(new_node_index)
                    self.uniform_color_list.append('blue')
                    self.uniform_size_list.append(20)

                    if ind[i] == 0:
                        x, y = way_point[0], way_point[1]
                        scaling_factor = (uniform_dist * (i + 1)) / length_list[0]
                    else:
                        x, y = way_point[ind[i]], way_point[ind[i] + 1]
                        scaling_factor = (uniform_dist * (i + 1) - accumulate_length_list[ind[i] - 1]) / length_list[ind[i]]
                    self.uniform_pos_dict[new_node_index] = self.vor.vertices[x] + (
                                self.vor.vertices[y] - self.vor.vertices[x]) * scaling_factor
                    self.uniform_pos_dict[new_node_index] = [self.uniform_pos_dict[new_node_index][0],
                                                             self.maph - self.uniform_pos_dict[new_node_index][1]]

                section_way_point.append(self.base_to_uniform_JC_node_dict[way_point[-1]])
                self.uniform_section_way_point_list.append(section_way_point)

                self.uniform_edge_list.append(
                    (self.base_to_uniform_JC_node_dict[way_point[0]], len(self.uniform_node_list) - how_many_node))
                self.uniform_edge_list.append(
                    (len(self.uniform_node_list) - 1, self.base_to_uniform_JC_node_dict[way_point[-1]]))
                for i in range(how_many_node - 1):
                    self.uniform_edge_list.append((len(self.uniform_node_list) - how_many_node + i,
                                                   len(self.uniform_node_list) - how_many_node + i + 1))

    def make_uniform_graph(self):
        import networkx as nx
        self.uniform_G = nx.Graph()
        self.uniform_weight_list = []
        self.make_uniform_graph_attribute()

        for edge in self.uniform_edge_list:
            self.uniform_weight_list.append(dist(edge[0], edge[1], self.uniform_pos_dict))

        self.uniform_G.add_nodes_from(self.uniform_node_list)

        for i in range(len(self.uniform_edge_list)):
            self.uniform_G.add_edge(self.uniform_edge_list[i][0], self.uniform_edge_list[i][1],
                                    weight=self.uniform_weight_list[i], color='black')

    def partitioning_section(self):
        self.section_class_list = []
        self.section_class_dict = {}
        for way_point in self.uniform_section_way_point_list:
            self.section_class_list.append(section_class(
                start=way_point[0], end=way_point[-1],
                way_point=way_point,
                length=self.calc_path_length(way_point, self.uniform_G)
            ))
            sorted_start_end = tuple(sorted((way_point[0], way_point[-1])))
            self.section_class_dict[sorted_start_end] = len(self.section_class_list) - 1

        self.full_section_number = len(self.section_class_list) - 1

        for JC_node in self.uniform_JC_node_list:
            self.section_class_list.append(section_class(
                start=JC_node, end=JC_node, way_point=[JC_node], length=0
            ))

        for JC_node in self.uniform_additional_JC_node_list:
            self.section_class_list.append(section_class(
                start=JC_node, end=JC_node, way_point=[JC_node], length=0
            ))

    def on_init(self):
        start_time = time.time()

        self.set_environment()
        self.set_polygon()
        self.sampling_boundary()
        self.make_voronoi()
        self.find_valid_vertices()
        self.find_confine_nodes()
        self.find_valid_ridge_vertices_ver2()
        self.find_JC_node()
        self.networkX_attribute()
        self.make_weight_list()
        self.make_node()
        self.make_edge()
        self.insert_end_point_to_JC_node()
        self.partitioning_temp_section()
        self.make_uniform_graph()
        self.partitioning_section()

        elapsed = time.time() - start_time

        return (self.uniform_G, self.uniform_size_list, self.uniform_color_list, self.uniform_pos_dict,
                self.section_class_list, self.uniform_JC_node_list, self.uniform_additional_JC_node_list,
                self.uniform_edge_list, self.uniform_node_list, self.polygon_list, self.polygon_vertex_list,
                self.section_class_dict, elapsed, self.full_section_number)
