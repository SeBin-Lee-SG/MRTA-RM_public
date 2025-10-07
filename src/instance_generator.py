# src/instance_generator.py

from __future__ import annotations

import copy
import time
from typing import List, Tuple, Dict, Iterable, Optional

from tqdm import tqdm
import numpy as np
import shapely.geometry as sh

from func.my_class import obj_node_class
from func.func import euclidean_distance, dist  # 명시적 임포트

class env:
    """
    Environment builder:
    - 로봇/목표 시작점 생성
    - 유니폼 그래프에 R{i}, G{i} 노드 추가 및 최근접 유효 정점 연결
    - 섹션별 로봇/목표 배치 정보 갱신 및 정렬
    """

    def __init__(
        self,
        mapw: int,
        maph: int,
        robot_num: int,
        goal_num: int,
        robot_radius: int,
        test_set: int,
        polygon_list,  # List[sh.Polygon]
        uniform_G,
        uniform_pos_dict: Dict,
        uniform_color_list: List,
        uniform_size_list: List,
        uniform_JC_node_list: List[int],
        uniform_additional_JC_node_list: List[int],
        uniform_node_list: List[int],
        section_class_list: List,
        separate: bool,
        full_section_number: int,
        robot_start_pos: List[Tuple[int, int]] = [],
        robot_goal_pos: List[Tuple[int, int]] = [],
    ):
        self.mapw = mapw
        self.maph = maph
        self.test_set = test_set

        self.robot_num = robot_num
        self.goal_num = goal_num
        self.robot_radius = robot_radius
        self.goal_radius = robot_radius

        # from GVD_generator
        self.polygon_list = polygon_list

        self.uniform_G = uniform_G
        self.uniform_pos_dict = uniform_pos_dict
        self.uniform_color_list = uniform_color_list
        self.uniform_size_list = uniform_size_list

        self.uniform_JC_node_list = uniform_JC_node_list
        self.uniform_additional_JC_node_list = uniform_additional_JC_node_list
        self.uniform_node_list = uniform_node_list

        self.section_class_list = section_class_list
        self.full_section_number = full_section_number

        self.separate = separate

        # may be provided from outside; otherwise generated here
        self.robot_start_pos: List[Tuple[int, int]] = list(robot_start_pos)
        self.robot_goal_pos: List[Tuple[int, int]] = list(robot_goal_pos)

    # -------------------------
    # Helpers
    # -------------------------

    def _in_collision(self, pos: Tuple[int, int], radius: int) -> bool:
        """원형 버퍼와 폴리곤 충돌 여부."""
        pt = sh.Point(pos).buffer(radius)
        for poly in self.polygon_list:
            if poly.intersects(pt):
                return True
        return False

    def _rand_int(self, low: int, high: int) -> int:
        """np.random.randint 대비, 실수 경계 대비 보정."""
        # np.random.randint는 high를 '미포함' 경계로 취급
        high = max(low + 1, int(high))
        return int(np.random.randint(int(low), int(high)))

    def _random_pos_range(self, for_goal: bool) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        시작/목표 포인트에 대한 (x_min, x_max), (y_min, y_max) 범위 반환.
        separate==True인 경우, test_set in {1,2,3}에서 x축 반분할 유지.
        일부 test_set(6,13)은 기존 코드 호환을 위해 특수 범위 유지.
        """
        r = self.robot_radius if not for_goal else self.goal_radius

        # 특수 케이스 (기존 코드 유지)
        if for_goal and self.test_set == 6:
            return (20, 180), (8, 192)
        if for_goal and self.test_set == 13:
            return (50, 950), (50, 950)

        if self.separate and self.test_set in (1, 2, 3):
            if for_goal:
                # 오른쪽 절반
                x_min = self.mapw / 2 + r
                x_max = self.mapw - r
            else:
                # 왼쪽 절반
                x_min = r
                x_max = self.mapw / 2 - r
            y_min, y_max = r, self.maph - r
        else:
            x_min, x_max = r, self.mapw - r
            y_min, y_max = r, self.maph - r

        return (int(x_min), int(x_max)), (int(y_min), int(y_max))

    def _random_free_position(self, for_goal: bool, existing: List[Tuple[int, int]], min_clearance: int) -> Optional[Tuple[int, int]]:
        """
        충돌/간격 조건을 모두 만족하는 랜덤 좌표 1개 샘플링.
        - polygon 충돌 없음
        - 기존 점들과의 간격 >= min_clearance
        - 시도 횟수 상한으로 무한루프 방지
        """
        (x_min, x_max), (y_min, y_max) = self._random_pos_range(for_goal)
        attempts = 0
        max_attempts = 20_000  # 매우 빡센 맵에서도 탈출 가능
        while attempts < max_attempts:
            attempts += 1
            x = self._rand_int(x_min, x_max)
            y = self._rand_int(y_min, y_max)
            pos = (x, y)
            if self._in_collision(pos, self.goal_radius if for_goal else self.robot_radius):
                continue
            # 간격 검사
            ok = True
            for p in existing:
                if euclidean_distance(pos, p) < min_clearance:
                    ok = False
                    break
            if ok:
                return pos
        return None  # 실패 (상위에서 처리)

    def _generate_positions(self, n: int, for_goal: bool) -> List[Tuple[int, int]]:
        """
        목표 개수만큼 유효 좌표 생성. 기존 제공 좌표가 있으면 그대로 사용.
        """
        out = self.robot_goal_pos if for_goal else self.robot_start_pos
        radius = self.goal_radius if for_goal else self.robot_radius
        min_clearance = 2 * radius + 1

        if len(out) >= n:
            return out[:n]

        while len(out) < n:
            pos = self._random_free_position(for_goal, out, min_clearance)
            if pos is None:
                break
            out.append(pos)

        return out

    # -------------------------
    # Public methods
    # -------------------------

    def set_sp_gp(self) -> None:
        """self.robot_start_pos & self.robot_goal_pos 보장."""
        if not (self.robot_num == len(self.robot_start_pos) and self.goal_num == len(self.robot_goal_pos)):
            # 시작/목표를 필요한 만큼 생성
            self._generate_positions(self.robot_num, for_goal=False)
            self._generate_positions(self.goal_num, for_goal=True)

        # 기존 출력 유지(동작 호환성)
        print("self.robot_start_pos =", self.robot_start_pos)
        print("self.robot_goal_pos =", self.robot_goal_pos)

    def make_robot_and_goal_class_list(self) -> None:
        """로봇/목표 obj_node_class 리스트 초기화."""
        self.robot_class_list: List[obj_node_class] = []
        self.goal_class_list: List[obj_node_class] = []

        for i in range(self.robot_num):
            self.robot_class_list.append(
                obj_node_class(
                    pos=self.robot_start_pos[i],
                    nearest_valid_vertex=None,
                    dist_to_valid_vertex=None,
                    nearest_JC_node=None,
                    dist_to_JC_node=None,
                    section=[],
                )
            )
        for i in range(self.goal_num):
            self.goal_class_list.append(
                obj_node_class(
                    pos=self.robot_goal_pos[i],
                    nearest_valid_vertex=None,
                    dist_to_valid_vertex=None,
                    nearest_JC_node=None,
                    dist_to_JC_node=None,
                    section=[],
                )
            )

    def make_uniform_graph(self) -> None:
        """유니폼 그래프에 R{i}, G{i} 노드와 위치/색/사이즈 추가."""
        for i in range(self.robot_num):
            key = f"R{i}"
            self.uniform_G.add_node(key)
            self.uniform_pos_dict[key] = [
                self.robot_start_pos[i][0],
                self.maph - self.robot_start_pos[i][1],
            ]
            self.uniform_color_list.append("green")
            self.uniform_size_list.append(200)

        for i in range(self.goal_num):
            key = f"G{i}"
            self.uniform_G.add_node(key)
            self.uniform_pos_dict[key] = [
                self.robot_goal_pos[i][0],
                self.maph - self.robot_goal_pos[i][1],
            ]
            self.uniform_color_list.append("yellow")
            self.uniform_size_list.append(200)

    def generate_key(self, prefix: str, index: int) -> str:
        return f"{prefix}{index}"

    def check_intersection(self, node_index: int, obj_index: int, robot: bool, goal: bool) -> bool:
        """
        유니폼 노드(node_index)와 객체(R/G obj_index) 사이 선분이 폴리곤과 교차하는지 검사.
        """
        if robot:
            obj_pos = self.uniform_pos_dict[f"R{obj_index}"]
            obj_pos = [obj_pos[0], self.maph - obj_pos[1]]
        elif goal:
            obj_pos = self.uniform_pos_dict[f"G{obj_index}"]
            obj_pos = [obj_pos[0], self.maph - obj_pos[1]]
        else:
            return False

        node_pos = self.uniform_pos_dict[node_index]
        node_pos = [node_pos[0], self.maph - node_pos[1]]

        line = sh.LineString([node_pos, obj_pos])
        for polygon in self.polygon_list:
            if line.intersects(polygon):
                return True
        return False

    def process_class_list(self, class_list: List[obj_node_class], prefix: str, valid_vertices: List[int]) -> None:
        """
        각 객체(로봇/목표)에 대해:
        - 유효 정점들까지의 거리 리스트 계산(dist 함수 사용)
        - 교차 없는 최단 후보를 최근접 유효 정점으로 채택
        """
        is_robot = prefix == "R"
        is_goal = prefix == "G"

        # enumerate로 index 캐시 (list.index 반복 비용 제거)
        for idx, entity_class in enumerate(class_list):
            key = self.generate_key(prefix, idx)

            # (vertex, distance) 리스트 한 방에 생성
            key_distance_list = [
                (vertex, dist(key, vertex, self.uniform_pos_dict))
                for vertex in valid_vertices
            ]
            key_distance_list.sort(key=lambda x: x[1])

            # 교차 없는 첫 후보 채택
            chosen = None
            for vertex, d in key_distance_list:
                if self.check_intersection(vertex, idx, is_robot, is_goal):
                    continue
                chosen = (vertex, d)
                break

            if chosen:
                entity_class.nearest_valid_vertex, entity_class.dist_to_valid_vertex = chosen

    def find_nearest_valid_vertex(self) -> None:
        """
        R/G 노드를 유효 정점에 연결하고, 섹션별 로봇/목표 카운트와 배치정보 갱신.
        """
        JC_nodes = self.uniform_JC_node_list + self.uniform_additional_JC_node_list
        valid_vertices = self.uniform_node_list  # (원래 JC 제외 안함; 기존 동작과 동일)

        # 최근접 유효 정점 선택
        self.process_class_list(self.robot_class_list, "R", valid_vertices)
        self.process_class_list(self.goal_class_list, "G", valid_vertices)

        # 그래프 연결
        for i, robot_class in enumerate(self.robot_class_list):
            self.uniform_G.add_edge(
                f"R{i}",
                robot_class.nearest_valid_vertex,
                weight=robot_class.dist_to_valid_vertex,
                color="black",
            )
        for i, goal_class in enumerate(self.goal_class_list):
            self.uniform_G.add_edge(
                f"G{i}",
                goal_class.nearest_valid_vertex,
                weight=goal_class.dist_to_valid_vertex,
                color="black",
            )

        # 섹션 할당 갱신
        for i, robot_class in enumerate(self.robot_class_list):
            v = robot_class.nearest_valid_vertex
            if v in JC_nodes:
                section = self.section_class_list[self.full_section_number + JC_nodes.index(v) + 1]
            else:
                section = next(s for s in self.section_class_list if v in s.way_point)
            section.robot_num += 1
            section.robot_list.append(i)
            section.robot_vertex_list.append(v)
            robot_class.section.append(self.section_class_list.index(section))

        for i, goal_class in enumerate(self.goal_class_list):
            v = goal_class.nearest_valid_vertex
            if v in JC_nodes:
                section = self.section_class_list[self.full_section_number + JC_nodes.index(v) + 1]
            else:
                section = next(s for s in self.section_class_list if v in s.way_point)
            section.goal_num += 1
            section.goal_list.append(i)
            section.goal_vertex_list.append(v)
            goal_class.section.append(self.section_class_list.index(section))

    def sort_section(self) -> None:
        """
        동일 유효정점을 공유하는 로봇/목표를,
        섹션 시작점과의 거리 기준으로 안정적으로 정렬.
        """
        # (0..full_section_number)까지가 실제 구간, 이후는 JC 단독 섹션
        for section in self.section_class_list[: self.full_section_number + 1]:
            robot_v = section.robot_vertex_list[:]
            goal_v = section.goal_vertex_list[:]
            robot_idx = section.robot_list[:]
            goal_idx = section.goal_list[:]

            start_point = self.uniform_pos_dict[section.start]

            # 로봇 정렬
            sorted_robot = sorted(
                zip(robot_idx, robot_v),
                key=lambda iv: (iv[1], euclidean_distance(start_point, self.uniform_pos_dict[f"R{iv[0]}"])),
            )
            # 목표 정렬
            sorted_goal = sorted(
                zip(goal_idx, goal_v),
                key=lambda iv: (iv[1], euclidean_distance(start_point, self.uniform_pos_dict[f"G{iv[0]}"])),
            )

            section.robot_list = [i for i, _ in sorted_robot]
            section.robot_queue = section.robot_list[:]
            section.robot_vertex_list = [v for _, v in sorted_robot]

            section.goal_list = [i for i, _ in sorted_goal]
            section.goal_vertex_list = [v for _, v in sorted_goal]

        for section in self.section_class_list[self.full_section_number + 1 :]:
            section.robot_queue = copy.deepcopy(section.robot_list)

    def on_init(self):
        # import random
        # import numpy as np
        # a = 15
        # random.seed(a)
        # np.random.seed(a)
        self.set_sp_gp()

        t0 = time.time()
        self.make_robot_and_goal_class_list()
        self.make_uniform_graph()
        self.find_nearest_valid_vertex()
        self.sort_section()
        elapsed = time.time() - t0

        return (
            self.uniform_G,
            self.uniform_size_list,
            self.uniform_color_list,
            self.uniform_pos_dict,
            self.robot_class_list,
            self.goal_class_list,
            self.section_class_list,
            self.robot_start_pos,
            self.robot_goal_pos,
            elapsed,
        )
