import numpy as np
import networkx as nx
from func.numba_utils import euclidean_distance_tuple as euclidean_distance


def dist(a, b, pos_dict):
    return euclidean_distance(
        np.array(pos_dict[a], dtype=np.float64),
        np.array(pos_dict[b], dtype=np.float64),
    )


def a_sub_b(a, b):
    """Return elements in list a that are not in list b."""
    b_set = set(b)
    return [x for x in a if x not in b_set]


def list_summation(l):
    length = len(l[0])
    collected_list = [[] for _ in range(length)]
    for sublist in l:
        for i in range(len(sublist)):
            collected_list[i].append(sublist[i])
    return collected_list


def find_index_with_value(l, value):
    for i, sublist in enumerate(l):
        if sublist[1] == value:
            return i
    return None


def find_index_with_key(l, key):
    for i, sublist in enumerate(l):
        if sublist[0] == key:
            return i
    return None


def calc_path_length(path, graph):
    """Calculate the total length of a path."""
    return sum(graph.get_edge_data(path[i], path[i + 1])['weight'] for i in range(len(path) - 1))


def make_simple_path(path, uniform_JC_node_list, uniform_additional_JC_node_list):
    """Remove nodes from the path that are not junction nodes."""
    return [path[i] for i in range(len(path))
            if i == 0 or i == len(path) - 1
            or path[i] in uniform_JC_node_list
            or path[i] in uniform_additional_JC_node_list]


def extract_path2section(path, JC_nodes, full_section_number):
    section_nums = []
    for i in range(len(path)):
        if isinstance(path[i], str) and path[i].startswith("S") and len(path[i]) > 1:
            section_nums.append(int(path[i][1:]))
        elif path[i] in JC_nodes:
            section_idx = full_section_number + JC_nodes.index(path[i]) + 1
            section_nums.append(section_idx)
    return section_nums


def make_two_list(list1, list2):
    return_list1 = []
    return_list2 = []
    for sub_list in list1:
        return_list1.append(sub_list[0])
        return_list2.append(sub_list[1])
    for sub_list in list2:
        return_list1.append(sub_list[0])
        return_list2.append(sub_list[1])
    return return_list1, return_list2


def find_and_delete_allocate(init_allocate, init_path_list, init_path_dist_list,
                             using_section_list, allocation):
    if allocation not in init_allocate:
        print("error in find_and_delete_allocate")
        return None
    else:
        index = init_allocate.index(allocation)
        del init_allocate[index]
        del init_path_list[index]
        del init_path_dist_list[index]
        del using_section_list[index]


def calc_dist_in_section(graph, section_class_list, section_index, a, b):
    """Calculate the distance between two nodes in a section."""
    section = section_class_list[section_index]
    if a == b:
        return 0
    elif (a == section.start and b == section.end) or (b == section.start and a == section.end):
        return section.length
    elif a == section.start:
        end_ind = section.way_point.index(b)
        return end_ind * section.length / (len(section.way_point) - 1)
    elif b == section.start:
        end_ind = section.way_point.index(a)
        return end_ind * section.length / (len(section.way_point) - 1)
    elif a == section.end:
        start_ind = section.way_point.index(b)
        return (len(section.way_point) - 1 - start_ind) * section.length / (len(section.way_point) - 1)
    elif b == section.end:
        start_ind = section.way_point.index(a)
        return (len(section.way_point) - 1 - start_ind) * section.length / (len(section.way_point) - 1)
    else:
        start = min(a, b)
        end = max(a, b)
        path = list(range(start, end + 1))
        return calc_path_length(path, graph)
