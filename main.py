import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import time
import os
from tqdm import tqdm
import networkx as nx

from src.GVD_generator import VBRM
from func.my_map import *
from src.instance_generator import env
from src.initial_allocator import InitialAllocator
from src.transfer_planner import TransferPlanner
from src.final_allocator import FinalAllocator
from func.func import *

sys.path.append('/home/sebin/PycharmProjects/MRTA_RM')

# Global variables
# ----------------
global show_roadmap
global show_result
global debug_mode
global realistic
global vs_CBS
######################
show_roadmap = 0
show_result = 0

debug_mode = 0
realistic = False

vs_CBS = False
######################

class MRTA_RM:
    def __init__(self):
        self.maps_dir = "maps"

        self.mapw = 200
        self.maph = 200
        self.robot_num = 1
        self.goal_num = self.robot_num
        self.robot_size = 8
        self.goal_size = self.robot_size

        ## 내가 만든 맵 ##
        self.test_set = 4  # 1은 랜덤맵, 최대 2000대
                            # 2는 현백, 최대 1343대
                            # 3은 창고, 최대 2288대
                            # 4는 녹화용, 최대 64대
                            # 11은 랜던 미니맵, 최대 826대
                            # 22는 현백 미니맵, 최대 442대
                            # 33는 창고 미니맵, 최대 496대

        # 랜덤
        if self.test_set == 1:
            self.mapw = 1000
            self.maph = 1000
            self.robot_size = 8
            if self.robot_num > 2000:
                print("robot num limit = 2000")
                self.robot_num = 2000

        # 현백
        elif self.test_set == 2:
            self.mapw = 1760
            self.maph = 900
            self.robot_size = 8
            if self.robot_num > 1343:
                print("robot num limit = 1343")
                self.robot_num = 1343

        # 창고
        elif self.test_set == 3:
            self.mapw = 2000
            self.maph = 880
            self.robot_size = 8
            if self.robot_num > 2288:
                print("robot num limit = 2288")
                self.robot_num = 2288

        # 녹화용
        elif self.test_set == 4:
            self.mapw = 200
            self.maph = 200
            self.robot_size = 8
            if self.robot_num > 64:
                print("robot num limit = 64")
                self.robot_num = 64

        # 미니
        elif self.test_set == 11 or self.test_set == 22 or self.test_set == 33:
            self.mapw = 640
            self.maph = 640
            self.robot_size = 8
            if self.test_set == 11:
                if self.robot_num > 826:
                    print("robot num limit = 826")
                    self.robot_num = 826
            elif self.test_set == 22:
                if self.robot_num > 442:
                    print("robot num limit = 442")
                    self.robot_num = 442
            elif self.test_set == 33:
                if self.robot_num > 496:
                    print("robot num limit = 496")
                    self.robot_num = 496

        # 미니 녹화용
        elif self.test_set == 44:
            self.mapw = 100
            self.maph = 100
            self.robot_size = 8

        elif self.test_set == 444:
            self.mapw = 160
            self.maph = 160
            self.robot_size = 8

        self.debug_mode = debug_mode
        self.realistic = realistic

        self.separate = None

    def set_test_set(self, test_set):
        self.test_set = test_set

        # 랜덤
        if self.test_set == 1:
            self.mapw = 1000
            self.maph = 1000
            self.robot_size = 8
            if self.robot_num > 2000:
                print("robot num limit = 2000")
                self.robot_num = 2000

        # 현백
        elif self.test_set == 2:
            self.mapw = 1760
            self.maph = 900
            self.robot_size = 8
            if self.robot_num > 1343:
                print("robot num limit = 1343")
                self.robot_num = 1343

        # 창고
        elif self.test_set == 3:
            self.mapw = 2000
            self.maph = 880
            self.robot_size = 8
            if self.robot_num > 2288:
                print("robot num limit = 2288")
                self.robot_num = 2288

        # 녹화용
        elif self.test_set == 4:
            self.mapw = 200
            self.maph = 200
            self.robot_size = 8
            if self.robot_num > 64:
                print("robot num limit = 64")
                self.robot_num = 64

        # 미니
        elif self.test_set == 11 or self.test_set == 22 or self.test_set == 33:
            self.mapw = 640
            self.maph = 640
            self.robot_size = 8
            if self.test_set == 11:
                if self.robot_num > 826:
                    print("robot num limit = 826")
                    self.robot_num = 826
            elif self.test_set == 22:
                if self.robot_num > 442:
                    print("robot num limit = 442")
                    self.robot_num = 442
            elif self.test_set == 33:
                if self.robot_num > 496:
                    print("robot num limit = 496")
                    self.robot_num = 496

        # 미니 녹화용
        elif self.test_set == 44:
            self.mapw = 100
            self.maph = 100
            self.robot_size = 8

        elif self.test_set == 444:
            self.mapw = 160
            self.maph = 160
            self.robot_size = 6

    # make GVD #
    def make_roadmap(self):
        # make roadmap #
        self.roadmap = VBRM(
            self.mapw,
            self.maph,
            self.test_set,
            self.robot_size,
            maps_dir=self.maps_dir,  # ← 추가
            node_interval=1.0  # ← (선택) 기존 전역 node_interval 대체
        )
        self.reusable_roadmap_result = self.roadmap.on_init()
        self.roadmap_result = copy.deepcopy(self.reusable_roadmap_result)

        self.uniform_G = self.roadmap_result[0]
        self.uniform_node_size_list = self.roadmap_result[1]
        self.uniform_node_color_list = self.roadmap_result[2]
        self.uniform_pos_dict = self.roadmap_result[3]

        self.section_class_list = self.roadmap_result[4]
        self.section_class_dict = self.roadmap_result[11]

        self.uniform_JC_node_list = self.roadmap_result[5]
        self.uniform_additional_JC_node_list = self.roadmap_result[6]
        self.uniform_edge_list = self.roadmap_result[7]
        self.uniform_node_list = self.roadmap_result[8]

        self.polygon_list = self.roadmap_result[9]
        self.polygon_vertex_list = self.roadmap_result[10]

        self.roadmap_time = self.roadmap_result[12]

        self.full_section_number = self.roadmap_result[13]

    def roadmap_reuse(self):
        self.roadmap_result = copy.deepcopy(self.reusable_roadmap_result)

        self.uniform_G = self.roadmap_result[0]
        self.uniform_node_size_list = self.roadmap_result[1]
        self.uniform_node_color_list = self.roadmap_result[2]
        self.uniform_pos_dict = self.roadmap_result[3]

        self.section_class_list = self.roadmap_result[4]
        self.section_class_dict = self.roadmap_result[11]

        self.uniform_JC_node_list = self.roadmap_result[5]
        self.uniform_additional_JC_node_list = self.roadmap_result[6]
        self.uniform_edge_list = self.roadmap_result[7]
        self.uniform_node_list = self.roadmap_result[8]

        self.polygon_list = self.roadmap_result[9]
        self.polygon_vertex_list = self.roadmap_result[10]

        self.roadmap_time = self.roadmap_result[12]

        self.full_section_number = self.roadmap_result[13]

    def make_env(self):
        if self.debug_mode:
            pass
        else:
            self.robot_start_pos = []
            self.robot_goal_pos = []

        if vs_CBS:
            if self.test_set == 1 or self.test_set == 2 or self.test_set == 3 or self.test_set == 4 or self.test_set == 11 or self.test_set == 22 or self.test_set == 33 or self.test_set == 44:
                if self.separate:
                    self.robot_start_pos = return_separate_robot_position(self.test_set, self.robot_num)
                    self.robot_start_grid = self.robot_start_pos[1]
                    self.robot_start_pos = self.robot_start_pos[0]

                    self.robot_goal_pos = return_separate_goal_position(self.test_set, self.goal_num)
                    self.robot_goal_grid = self.robot_goal_pos[1]
                    self.robot_goal_pos = self.robot_goal_pos[0]

                    self.obs, self.dim = return_grid_obs(self.test_set)
                else:
                    self.robot_start_pos = return_position(self.test_set, self.robot_num)
                    self.robot_start_grid = self.robot_start_pos[1]
                    self.robot_start_pos = self.robot_start_pos[0]

                    self.robot_goal_pos = return_position(self.test_set, self.goal_num)
                    self.robot_goal_grid = self.robot_goal_pos[1]
                    self.robot_goal_pos = self.robot_goal_pos[0]

                    self.obs, self.dim = return_grid_obs(self.test_set)
            else:
                pass
        else:
            pass

        self.env = env(self.mapw, self.maph, self.robot_num, self.goal_num, self.robot_size, self.test_set,\
                       self.polygon_list, self.uniform_G, self.uniform_pos_dict, self.uniform_node_color_list,\
                       self.uniform_node_size_list, self.uniform_JC_node_list, self.uniform_additional_JC_node_list,\
                       self.uniform_node_list, self.section_class_list, self.separate, self.full_section_number, self.robot_start_pos, self.robot_goal_pos)
        self.env_result = self.env.on_init()

        self.uniform_G = self.env_result[0]
        self.uniform_node_size_list = self.env_result[1]
        self.uniform_node_color_list = self.env_result[2]
        self.uniform_pos_dict = self.env_result[3]

        self.robot_class_list = self.env_result[4]
        self.goal_class_list = self.env_result[5]
        self.section_class_list = self.env_result[6]

        self.robot_start_pos = self.env_result[7]
        self.robot_goal_pos = self.env_result[8]

        self.time_wo_set_sp_gp = self.env_result[9]

        # print("self.robot_start_pos=", self.robot_start_pos)
        # print("self.robot_goal_pos=", self.robot_goal_pos)



    # local > global Hungarian을 이용한 Initial Allocator #
    '''O(N^3) + 최대 N^2번 A* path search'''
    def initial_allocator(self):
        self.init_allocate = InitialAllocator(self.uniform_G, self.uniform_pos_dict, self.robot_class_list, self.goal_class_list, \
                                           self.section_class_list, self.robot_num, self.goal_num, \
                                           self.uniform_JC_node_list, self.uniform_additional_JC_node_list, self.full_section_number)
        self.diff_seg_section_list = self.init_allocate.on_init()

        # print("\ninit_allocation: ", self.init_allocation)
        # print("init cost: ", sum(self.init_path_dist_list))

    # transfer_analysis 를 통해 얻은 결과를 바탕 #
    # 각 section에 대해 어느 section에서 몇대의 로봇을, 어느 section으로 몇대의 로봇을 #
    # transfer, receive 해야하는지 check #
    '''(O(NS)'''
    def planner(self):
        self.transfer_planner = TransferPlanner(self.section_class_list, self.diff_seg_section_list)
        self.transfer_planner_result = self.transfer_planner.analyze()
        self.case1_section_list = self.transfer_planner_result[0]
        self.case2_section_list = self.transfer_planner_result[1]
        self.case3_section_list = self.transfer_planner_result[2]
        self.case4_section_list = self.transfer_planner_result[3]
        self.section2section_transfer_dict_key_list = self.transfer_planner_result[4]
        self.section2section_transfer_dict = self.transfer_planner_result[5]

    # planner 의 결과를 바탕 #
    # Equalization 및 Allocation #
    '''O(NS^2)'''
    def re_allocator(self):
        self.re_assign = FinalAllocator(self.uniform_G, self.uniform_pos_dict, self.section_class_list, self.robot_class_list, self.goal_class_list,\
                                         self.case1_section_list, self.case2_section_list, self.case3_section_list, self.case4_section_list, self.realistic, self.full_section_number)

        self.final_allocation_result = self.re_assign.on_init()

#############################################################################################


#############################################################################################
                                        # Visualize #
                                        #############
    def plot_roadmap_wo_line(self):
        fig = plt.gcf()
        # fig.set_size_inches(self.mapw/100, self.maph/100)
        fig.set_size_inches(10, 10)
        ax = fig.gca()

        # Draw polygons first
        for polygon_vertices in self.new_polygon_vertex_list:
            polygon = patches.Polygon(polygon_vertices, fill=True, facecolor='grey', edgecolor='grey')
            ax.add_patch(polygon)

        # Then draw the graph
        uniform_G_copy = copy.deepcopy(self.uniform_G)
        color_list = copy.deepcopy(self.uniform_node_color_list)
        size_list = copy.deepcopy(self.uniform_node_size_list)

        # Remove edges connected with robot start/goal
        # robot node는 'R'로 시작, 'R'+str(robot_idx)
        # goal node는 'G'로 시작, 'G'+str(goal_idx)
        for node in list(uniform_G_copy.nodes):
            if type(node) == str:
                if node[0] == 'R' or node[0] == 'G':
                    idx = list(uniform_G_copy.nodes).index(node)
                    uniform_G_copy.remove_node(node)
                    del color_list[idx]
                    del size_list[idx]



        nx.draw(uniform_G_copy, node_size=7, node_color=color_list, pos=self.uniform_pos_dict, width=1, with_labels=False)

        # Set the graph's layout
        ax.set_xlim([0, self.mapw])
        ax.set_ylim([0, self.maph])

        plt.show()

    def plot_roadmap(self):
        fig = plt.gcf()
        # fig.set_size_inches(self.mapw/100, self.maph/100)
        fig.set_size_inches(10, 10)
        ax = fig.gca()

        # Draw polygons first
        for polygon_vertices in self.new_polygon_vertex_list:
            polygon = patches.Polygon(polygon_vertices, fill=True, facecolor='grey')
            ax.add_patch(polygon)

        # Then draw the graph
        nx.draw(self.roadmap_result[0], node_size=self.roadmap_result[1], node_color=self.roadmap_result[2],
                pos=self.roadmap_result[3], width=1, with_labels=True, font_size=5)

        # Set the graph's layout
        ax.set_xlim([0, self.mapw])
        ax.set_ylim([0, self.maph])

        plt.show()

    def plot_allocation(self, allocation_result):
        self.uniform_edge_color_list = ['black' for i in range(len(self.uniform_G.edges()))]
        self.uniform_width_list = [1 for i in range(len(self.uniform_G.edges()))]

        fig = plt.gcf()
        fig.set_size_inches(self.mapw/100, self.maph/100)
        ax = fig.gca()

        # Draw polygons first
        for polygon_vertices in self.new_polygon_vertex_list:
            polygon = patches.Polygon(polygon_vertices, fill=True, facecolor='grey', edgecolor='grey')
            ax.add_patch(polygon)

        # Then draw the graph
        for allocation in allocation_result:
            self.uniform_G.add_edge('R' + str(allocation[0]), 'G' + str(allocation[1]), weight=99999)
            self.uniform_edge_color_list.append('red')
            self.uniform_width_list.append(2)

        # nx.draw(self.uniform_G, pos=self.uniform_pos_dict, node_size=self.uniform_node_size_list,
        #         node_color=self.uniform_node_color_list,
        #         edge_color=self.uniform_edge_color_list, width=self.uniform_width_list, with_labels=False)

        nx.draw(self.uniform_G, pos=self.uniform_pos_dict, node_size=[i/8 for i in self.uniform_node_size_list],
                node_color=self.uniform_node_color_list,
                edge_color=self.uniform_edge_color_list, width=self.uniform_width_list, with_labels=False)

        # Set the graph's layout
        # ax.set_xlim([0, self.mapw])
        # ax.set_ylim([0, self.maph])

        plt.show()

        # remove edge
        for allocation in allocation_result:
            self.uniform_G.remove_edge('R' + str(allocation[0]), 'G' + str(allocation[1]))

    def plot_env(self):
        fig = plt.gcf()
        # fig.set_size_inches(self.mapw / 100, self.maph / 100)
        ax = fig.gca()

        print("self.new_polygon_vertex_list = ", self.new_polygon_vertex_list)

        # Draw polygons first
        for polygon_vertices in self.new_polygon_vertex_list:
            polygon = patches.Polygon(polygon_vertices, fill=True, facecolor='grey', edgecolor='grey')
            ax.add_patch(polygon)

        # save as png
        # plt.savefig('env.png', dpi=300)
        plt.show()

#############################################################################################


    def on_init(self, num=1, test_set = None, robot_num = 0, separate = False, folder_name = None):
        if test_set != None:
            self.test_set = test_set
        if robot_num != 0:
            self.robot_num = robot_num
            self.goal_num = robot_num

        self.separate = separate
        self.folder_name = folder_name
        self.num = num


        self.set_test_set(self.test_set)


        print("for", self.num, "th map")
        if self.num == 1:
            self.make_roadmap()
        else:
            self.roadmap_reuse()



        self.make_env()

        self.start_time = time.time()

        self.initial_allocator()

        self.planner()
        self.re_allocator()

        print("time to make road map = ", self.roadmap_time, "sec")
        self.run_time = time.time() - self.start_time + self.time_wo_set_sp_gp
        print("SB search run time wo roadmap = ", self.run_time, "sec")
        print("Result :")
        print("JC node list = ", self.uniform_JC_node_list)
        print("additional JC node list = ", self.uniform_additional_JC_node_list)
        print(self.final_allocation_result)

        self.polygon_vertex_list_save = copy.deepcopy(self.polygon_vertex_list)
        new_polygon_vertex_list = []
        for polygon in self.polygon_vertex_list_save:
            new_polygon = polygon[:-1]  # exclude the last element
            new_polygon = [list(vertex) for vertex in new_polygon]  # convert all vertices to list
            for vertex in new_polygon:
                vertex[1] = self.maph - vertex[1]
            new_polygon = [tuple(vertex) for vertex in new_polygon]  # convert all vertices back to tuple
            new_polygon_vertex_list.append(new_polygon)
        self.polygon_vertex_list_save = new_polygon_vertex_list  # assign the new list back to the original variable

        if show_roadmap or show_result:
            self.new_polygon_vertex_list = []
            for polygon in self.polygon_vertex_list:
                new_polygon = []
                for vertex in polygon:
                    new_vertex = (vertex[0], self.maph - vertex[1])
                    new_polygon.append(new_vertex)
                self.new_polygon_vertex_list.append(new_polygon)
            if show_roadmap:
                self.plot_roadmap_wo_line()
                self.plot_roadmap()
            if show_result:
                self.plot_allocation(self.final_allocation_result)
        print("\n\n\n\n")
        return self.final_allocation_result

if __name__ == "__main__":
    app = MRTA_RM()
    original_stdout = sys.stdout
    if debug_mode:
        app.on_init(1)
    else:
        # 1은 랜덤맵, 최대 2000대
        # 2는 현백, 최대 1343대
        # 3은 창고, 최대 2288대
        # 4는 녹화용, 최대 64대
        # 11은 랜던 미니맵, 최대 826대
        # 22는 현백 미니맵, 최대 442대
        # 33는 창고 미니맵, 최대 496대

        # env_list = [1, 2, 3]
        # num_list = [50, 100, 200, 400, 800, 1200, 1600, 2000]
        # how_many = 50

        # full map
        env_list = [3]
        non_sep_num_list = [50]
        sep_num_list = [50]
        how_many = 100

        # # mini map
        # env_list = [11, 22, 33]
        # num_list = [5, 10, 15, 20, 25, 30]
        # how_many = 50

        folder_name = "test"
        for env_num in env_list:
            prefix = "HB_" if env_num == 2 else "WH_" if env_num == 3 else "RN_" if env_num == 1 else "mini_RN_" if env_num == 11 else "mini_HB_" if env_num == 22 else "mini_WH_" if env_num == 33 else "Record_" if env_num == 4 else "Unknown_"
            # for separate in [True]:
            for separate in [False]:
            # for separate in [False, True]:
                if separate:
                    num_list = sep_num_list
                else:
                    num_list = non_sep_num_list
                for robot_num in num_list:
                    with tqdm(total=how_many,
                              desc=f'Making environments {prefix}, Separate={separate}, robot_num={robot_num}') as pbar:
                        if not os.path.exists(folder_name):
                            os.makedirs(folder_name)
                        if separate:
                            # sys.stdout = open(f"data/{prefix}Separate_{robot_num}_save_result.txt", "w")
                            sys.stdout = open(f"{folder_name}/{prefix}Separate_{robot_num}_save_result.txt", "w")
                        else:
                            # sys.stdout = open(f"data/{prefix}NonSeparate_{robot_num}_save_result.txt", "w")
                            sys.stdout = open(f"{folder_name}/{prefix}NonSeparate_{robot_num}_save_result.txt", "w")
                        for i in range(1, how_many + 1):
                            app.on_init(i, int(env_num), int(robot_num), separate, folder_name)
                            pbar.update(1)
                        sys.stdout.close()
                        sys.stdout = original_stdout

