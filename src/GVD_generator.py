# src/GVD_generator.py

from __future__ import annotations

import copy
import json
import os
import time
from collections import deque
from typing import List, Tuple, Dict, Iterable, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import shapely.geometry as sh
import shapely.ops as sho
from scipy.spatial import Voronoi

from func.my_class import section_class, temp_section_class
from func.func import *  # 기존 헬퍼들과 호환 위해 유지 (dist, euclidean_distance, a_sub_b 등)
from func.my_map import return_map as legacy_return_map  # 폴백용


class VBRM:
    """
    Visibility-Based RoadMap (VBRM)
    - 기존 동작은 보존하면서 내부 품질/구조/가독성 개선
    - 맵 로딩: JSON/YAML 우선, 실패시 기존 func.my_map.return_map(test_set) 사용
    """

    def __init__(
        self,
        mapw: int,
        maph: int,
        test_set: int,
        robot_radius: int,
        *,
        maps_dir: Optional[str] = None,
        node_interval: float = 1.0,
    ):
        # 기본 파라미터
        self.mapw = mapw
        self.maph = maph
        self.test_set = test_set
        self.robot_radius = robot_radius
        self.goal_radius = robot_radius

        # 샘플링/균등노드 설정
        self.sampling_dist = int(self.robot_radius * 2)
        self.node_interval = node_interval  # 기존 전역 node_interval 대체

        # 파일 기반 맵 디렉토리
        self.maps_dir = maps_dir

        # 내부 상태 컨테이너
        self.polygon_vertex_list: List[List[Tuple[float, float]]] = []
        self.polygon_list: List[sh.Polygon] = []

        # 경계(바운딩 박스)
        self.Box_line_list = [
            sh.LineString([(0, 0), (0, self.maph)]),
            sh.LineString([(0, self.maph), (self.mapw, self.maph)]),
            sh.LineString([(self.mapw, self.maph), (self.mapw, 0)]),
            sh.LineString([(self.mapw, 0), (0, 0)]),
        ]

    # -------- 맵 로딩 -------- #

    def _load_polygon_from_file(self) -> Optional[List[List[Tuple[float, float]]]]:
        """
        maps_dir 안에서 test_set에 해당하는 JSON/YAML 파일을 찾아 로드.
        스키마:
        - JSON/YAML 모두에서 top-level에 `polygons` 키를 권장.
          예) { "polygons": [ [[x,y],[x,y],...], [[...]], ... ] }
        - 만약 top-level이 곧바로 다각형 리스트여도 허용.

        반환: 로드 성공 시 폴리곤 리스트, 실패 시 None
        """
        if not self.maps_dir:
            return None
        if not os.path.isdir(self.maps_dir):
            return None

        # 후보 파일명 (원하는대로 추가 가능)
        base_names = [
            f"{self.test_set}",
            # 관례적 네이밍 예시 (선택)
            {1: "random_full", 11: "random_mini", 2: "dept_store", 22: "dept_store_mini",
             3: "warehouse", 33: "warehouse_mini", 4: "recording", 44: "recording_mini", 444: "recording_micro"}.get(self.test_set, None)
        ]
        base_names = [b for b in base_names if b]

        candidates = []
        for b in base_names:
            candidates += [os.path.join(self.maps_dir, f"{b}.json"),
                           os.path.join(self.maps_dir, f"{b}.yaml"),
                           os.path.join(self.maps_dir, f"{b}.yml")]

        for path in candidates:
            if not os.path.isfile(path):
                continue
            try:
                if path.endswith(".json"):
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                else:
                    import yaml  # 선택 의존. 없으면 except로 빠짐
                    with open(path, "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f)

                polygons = data.get("polygons", data)
                # (x,y) 튜플로 정규화
                polygons = [
                    [tuple(map(float, pt)) for pt in poly]
                    for poly in polygons
                ]
                return polygons
            except Exception:
                # 파일이 있지만 파싱 실패 → 다음 후보로
                continue
        return None

    def set_environment(self) -> None:
        """
        polygon_vertex_list 준비:
          1) maps_dir에서 JSON/YAML 시도
          2) 실패하면 legacy_return_map(test_set) 폴백
        """
        from_file = self._load_polygon_from_file()
        if from_file:
            self.polygon_vertex_list = from_file
        else:
            # 기존 방식 폴백
            self.polygon_vertex_list = legacy_return_map(self.test_set)
        # print("polygon_vertex_list =", self.polygon_vertex_list)

    # -------- 폴리곤/샘플/보로노이 -------- #

    def set_polygon(self) -> None:
        """polygon_vertex_list -> shapely Polygon/MultiPolygon 병합 & 정리"""
        self.polygon_list = []
        merged = sho.unary_union([sh.Polygon(p) for p in self.polygon_vertex_list])

        individuals = list(merged.geoms) if isinstance(merged, sh.MultiPolygon) else [merged]
        self.polygon_vertex_list = []
        for polygon in individuals:
            self.polygon_vertex_list.append(list(polygon.exterior.coords)[:-1])
            self.polygon_list.append(polygon)

    def _sample_boundary(self, vertex_list: List[List[Tuple[float, float]]], out: List[Tuple[float, float]]) -> None:
        """다각형 경계 샘플링"""
        closed = [poly + [poly[0]] for poly in vertex_list]
        for poly in closed:
            for a, b in zip(poly[:-1], poly[1:]):
                edge_len = euclidean_distance(a, b)
                n = int(edge_len / self.sampling_dist) + 1
                for i in range(n):
                    out.append((
                        a[0] + (b[0] - a[0]) * i / n,
                        a[1] + (b[1] - a[1]) * i / n
                    ))

    def sampling_boundary(self) -> None:
        self.bounding_box = [(0, 0), (0, self.maph), (self.mapw, self.maph), (self.mapw, 0)]
        self.bounding_box_sample: List[Tuple[float, float]] = []
        self.polygon_boundary_sample: List[Tuple[float, float]] = []
        self._sample_boundary([self.bounding_box], self.bounding_box_sample)
        self._sample_boundary(self.polygon_vertex_list, self.polygon_boundary_sample)

    def make_voronoi(self) -> None:
        temp = self.bounding_box_sample + self.polygon_boundary_sample
        self.vor = Voronoi(temp)

    # -------- 보로노이 유효 정점/엣지 필터링 -------- #

    def find_valid_vertices(self) -> None:
        self.vor_valid_vertices: List[List[float]] = []
        self.vor_valid_vertices_index: List[int] = []

        Box = sh.Polygon(self.bounding_box)
        for i, v in enumerate(self.vor.vertices):
            p = sh.Point(v)
            if not Box.contains(p):
                continue
            if any(poly.contains(p) for poly in self.polygon_list):
                continue
            self.vor_valid_vertices.append(v.tolist())
            self.vor_valid_vertices_index.append(i)

        self.vor_valid_vertices = np.array(self.vor_valid_vertices).reshape(-1, 2)

    def find_confine_nodes(self) -> None:
        """폭이 로봇 지름보다 작은 정점 제외"""
        self.confined_vertices_index: List[int] = []
        rr = self.robot_radius - 0.5
        for vidx in self.vor_valid_vertices_index:
            p = sh.Point(self.vor.vertices[vidx])
            width_to_polygon = min(p.distance(poly) for poly in self.polygon_list)
            width_to_boundary = min(p.distance(line) for line in self.Box_line_list)
            if min(width_to_polygon, width_to_boundary) < rr:
                self.confined_vertices_index.append(vidx)

    def find_valid_ridge_vertices(self) -> None:
        self.vor_valid_ridge_vertices = [r for r in self.vor.ridge_vertices if r[0] != -1 and r[1] != -1]
        valid = set(self.vor_valid_vertices_index)
        confined = set(self.confined_vertices_index)

        self.unrolled_real_valid_ridge_vertices_index: List[int] = []
        self.vor_edge_list: List[Tuple[int, int]] = []
        for a, b in self.vor_valid_ridge_vertices:
            if a in valid and b in valid and a not in confined and b not in confined:
                self.unrolled_real_valid_ridge_vertices_index += [a, b]
                self.vor_edge_list.append((a, b))

    def find_JC_node(self) -> None:
        """교차점(차수>2)"""
        counts = {}
        for n in self.unrolled_real_valid_ridge_vertices_index:
            counts[n] = counts.get(n, 0) + 1
        self.JC_node_list = [n for n, c in counts.items() if c > 2]

    # -------- NetworkX 속성 구성 -------- #

    def networkX_attribute(self) -> None:
        self.pos_dict: Dict[int, List[float]] = {}
        self.color_list: List[str] = []
        self.size_list: List[int] = []

        # confined 제거
        keep_idx = [idx for idx in self.vor_valid_vertices_index if idx not in set(self.confined_vertices_index)]
        keep_pts = [self.vor_valid_vertices[self.vor_valid_vertices_index.index(idx)] for idx in keep_idx]

        self.vor_valid_vertices_index = keep_idx
        self.vor_valid_vertices = keep_pts

        # 좌표/색/사이즈
        for i, idx in enumerate(self.vor_valid_vertices_index):
            v = self.vor_valid_vertices[i]
            self.pos_dict[idx] = [v[0], self.maph - v[1]]
            if idx in self.JC_node_list:
                self.color_list.append('red')
                self.size_list.append(100)
            else:
                self.color_list.append('blue')
                self.size_list.append(20)

    def make_weight_list(self) -> None:
        self.weight_list: List[float] = [dist(a, b, self.pos_dict) for a, b in self.vor_edge_list]

    def make_node(self) -> None:
        self.G = nx.Graph()
        self.G.add_nodes_from(self.vor_valid_vertices_index)

    def make_edge(self) -> None:
        for (a, b), w in zip(self.vor_edge_list, self.weight_list):
            self.G.add_edge(a, b, weight=w, color='black')

    def insert_end_point_to_JC_node(self) -> None:
        """끝점(차수=1)도 JC 노드로 취급"""
        self.additional_JC_node_list = [n for n in self.vor_valid_vertices_index if len(self.G.adj[n]) == 1]
        for n in self.additional_JC_node_list:
            i = self.vor_valid_vertices_index.index(n)
            self.color_list[i] = 'red'
            self.size_list[i] = 100

    def calc_path_length(self, path: List[int], graph: nx.Graph) -> float:
        return sum(graph.get_edge_data(path[i], path[i + 1])['weight'] for i in range(len(path) - 1))

    def bfs_with_sections(self, graph: nx.Graph, start_node: int) -> Iterable[List[int]]:
        visited = {start_node}
        q = deque([(start_node, [start_node])])
        while q:
            cur, path = q.popleft()
            if cur in self.JC_node_list and cur != start_node:
                yield path
                continue
            for nxt in graph.neighbors(cur):
                if nxt not in visited:
                    visited.add(nxt)
                    q.append((nxt, path + [nxt]))

    def partitioning_temp_section(self) -> None:
        self.temp_section_class_list: List[temp_section_class] = []
        seen = set()

        def _add(start_list: List[int]) -> None:
            for start in start_list:
                for path in self.bfs_with_sections(self.G, start):
                    tup = tuple(sorted((path[0], path[-1])))
                    if tup in seen:
                        continue
                    seen.add(tup)
                    self.temp_section_class_list.append(temp_section_class(
                        start=path[0],
                        end=path[-1],
                        way_point=path,
                        length=self.calc_path_length(path, self.G),
                    ))

        _add(self.JC_node_list)
        _add(self.additional_JC_node_list)

    # -------- Uniform Graph -------- #

    def _process_nodes(
        self, nodes: List[int], mapping: Dict[int, int], offset: int = 0
    ):
        uniform_nodes = []
        pos = {}
        colors, sizes = [], []
        for i, node in enumerate(nodes):
            un = i + offset
            mapping[node] = un
            uniform_nodes.append(un)
            v = self.vor.vertices[node]
            pos[un] = [v[0], self.maph - v[1]]
            colors.append('red')
            sizes.append(100)
        return uniform_nodes, pos, colors, sizes, mapping

    def make_uniform_graph_attribute(self) -> None:
        self.uniform_JC_node_list: List[int] = []
        self.uniform_additional_JC_node_list: List[int] = []
        self.uniform_node_list: List[int] = []
        self.uniform_pos_dict: Dict[int, List[float]] = {}
        self.uniform_color_list: List[str] = []
        self.uniform_size_list: List[int] = []
        self.base_to_uniform_JC_node_dict: Dict[int, int] = {}
        self.uniform_edge_list: List[Tuple[int, int]] = []
        self.uniform_section_way_point_list: List[List[int]] = []

        # JC / 추가 JC
        (self.uniform_JC_node_list,
         self.uniform_pos_dict,
         self.uniform_color_list,
         self.uniform_size_list,
         self.base_to_uniform_JC_node_dict) = self._process_nodes(
            self.JC_node_list, self.base_to_uniform_JC_node_dict
        )

        offset = len(self.uniform_JC_node_list)
        (self.uniform_additional_JC_node_list,
         temp_pos, temp_color, temp_size,
         self.base_to_uniform_JC_node_dict) = self._process_nodes(
            self.additional_JC_node_list, self.base_to_uniform_JC_node_dict, offset
        )

        self.uniform_node_list = self.uniform_JC_node_list + self.uniform_additional_JC_node_list
        self.uniform_pos_dict.update(temp_pos)
        self.uniform_color_list += temp_color
        self.uniform_size_list += temp_size

        # 구간을 균등 노드로 재샘플
        for section in self.temp_section_class_list:
            way = section.way_point
            length = section.length
            # 짧은 구간: 가운데 노드 하나
            if length <= (2 * self.node_interval) * self.robot_radius:
                center = len(self.uniform_node_list)
                x, y = way[0], way[-1]
                p = self.vor.vertices[x] + (self.vor.vertices[y] - self.vor.vertices[x]) / 2
                self.uniform_pos_dict[center] = [p[0], self.maph - p[1]]

                self.uniform_node_list.append(center)
                self.uniform_color_list.append('blue')
                self.uniform_size_list.append(20)

                a = self.base_to_uniform_JC_node_dict[way[0]]
                b = self.base_to_uniform_JC_node_dict[way[-1]]
                self.uniform_edge_list += [(a, center), (center, b)]
                self.uniform_section_way_point_list.append([a, center, b])
            else:
                # 긴 구간: 균등 거리로 중간 노드들 추가
                edge_len = []
                acc = []
                for e in range(len(way) - 1):
                    w = self.G.get_edge_data(way[e], way[e + 1])['weight']
                    edge_len.append(w)
                    acc.append(sum(edge_len))

                how_many = int(length / (2 * self.node_interval * self.robot_radius)) + 1
                uni = length / how_many
                how_many -= 1

                # 각 균등점이 어느 엣지에 놓이는지 찾기
                idxs = []
                for i in range(how_many):
                    target = uni * (i + 1)
                    for j, s in enumerate(acc):
                        if s >= target:
                            idxs.append(j)
                            break

                section_way = [self.base_to_uniform_JC_node_dict[way[0]]]

                base = len(self.uniform_node_list)
                for i, j in enumerate(idxs):
                    new_n = base + i
                    section_way.append(new_n)
                    self.uniform_node_list.append(new_n)
                    self.uniform_color_list.append('blue')
                    self.uniform_size_list.append(20)

                    if j == 0:
                        x, y = way[0], way[1]
                        t = (uni * (i + 1)) / edge_len[0]
                    else:
                        x, y = way[j], way[j + 1]
                        t = (uni * (i + 1) - acc[j - 1]) / edge_len[j]

                    p = self.vor.vertices[x] + (self.vor.vertices[y] - self.vor.vertices[x]) * t
                    self.uniform_pos_dict[new_n] = [p[0], self.maph - p[1]]

                section_way.append(self.base_to_uniform_JC_node_dict[way[-1]])
                self.uniform_section_way_point_list.append(section_way)

                # 엣지 연결
                a = self.base_to_uniform_JC_node_dict[way[0]]
                b = self.base_to_uniform_JC_node_dict[way[-1]]
                self.uniform_edge_list.append((a, base))
                self.uniform_edge_list.append((base + how_many - 1, b))
                for i in range(how_many - 1):
                    self.uniform_edge_list.append((base + i, base + i + 1))

    def make_uniform_graph(self) -> None:
        self.uniform_G = nx.Graph()
        self.uniform_weight_list = [dist(a, b, self.uniform_pos_dict) for (a, b) in self.uniform_edge_list]

        self.uniform_G.add_nodes_from(self.uniform_node_list)
        for (a, b), w in zip(self.uniform_edge_list, self.uniform_weight_list):
            self.uniform_G.add_edge(a, b, weight=w, color='black')

    def partitioning_section(self) -> None:
        self.section_class_list: List[section_class] = []
        self.section_class_dict: Dict[Tuple[int, int], int] = {}

        for way in self.uniform_section_way_point_list:
            sc = section_class(
                start=way[0],
                end=way[-1],
                way_point=way,
                length=self.calc_path_length(way, self.uniform_G)
            )
            self.section_class_list.append(sc)
            self.section_class_dict[tuple(sorted((way[0], way[-1])))] = len(self.section_class_list) - 1

        # JC/추가JC 단독 섹션도 추가 (기존 로직 유지)
        self.full_section_number = len(self.section_class_list) - 1
        for n in self.uniform_JC_node_list + self.uniform_additional_JC_node_list:
            self.section_class_list.append(section_class(start=n, end=n, way_point=[n], length=0))

    # -------- 엔트리 포인트 -------- #

    def on_init(self):
        t0 = time.time()

        self.set_environment()
        self.set_polygon()
        self.sampling_boundary()
        self.make_voronoi()
        self.find_valid_vertices()
        self.find_confine_nodes()
        self.find_valid_ridge_vertices()
        self.find_JC_node()
        self.networkX_attribute()
        self.make_weight_list()
        self.make_node()
        self.make_edge()
        self.insert_end_point_to_JC_node()
        self.partitioning_temp_section()
        self.make_uniform_graph_attribute()
        self.make_uniform_graph()
        self.partitioning_section()

        elapsed = time.time() - t0
        return (
            self.uniform_G,
            self.uniform_size_list,
            self.uniform_color_list,
            self.uniform_pos_dict,
            self.section_class_list,
            self.uniform_JC_node_list,
            self.uniform_additional_JC_node_list,
            self.uniform_edge_list,
            self.uniform_node_list,
            self.polygon_list,
            self.polygon_vertex_list,
            self.section_class_dict,
            elapsed,
            self.full_section_number,
        )
