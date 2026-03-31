

class obj_node_class:
    """Represents a robot or goal node."""
    def __init__(self, pos=None, nearest_valid_vertex=None, dist_to_valid_vertex=None,
                 nearest_JC_node=None, dist_to_JC_node=None, section=None):
        self.pos = pos
        self.nearest_valid_vertex = nearest_valid_vertex
        self.dist_to_valid_vertex = dist_to_valid_vertex
        self.nearest_JC_node = nearest_JC_node
        self.dist_to_JC_node = dist_to_JC_node
        self.section = section if section is not None else []
        self.traveled_distance = 0
        self.travel_way_point = []


class temp_section_class:
    """Temporary section used during roadmap construction."""
    def __init__(self, start=None, end=None, way_point=None, length=None):
        self.start = start
        self.end = end
        self.way_point = way_point
        self.length = length


class section_class:
    """A section of the roadmap between two junction nodes."""
    def __init__(self, start=None, end=None, way_point=None, length=None,
                 width=0, robot_size=None):
        self.start = start
        self.end = end
        self.way_point = way_point
        self.length = length
        self.robot_num = 0
        self.robot_list = []
        self.robot_vertex_list = []
        self.goal_num = 0
        self.goal_list = []
        self.goal_vertex_list = []

        self.remain_robot_num = 0
        self.remain_goal_num = 0

        # key: target section idx, value: ["D" or "R", robot_count]
        # "D"/"R" indicates direction from this section's perspective
        self.transfer_section_index_dict = {}
        self.receive_section_index_dict = {}

        # "D": toward end, "R": toward start, "E": either endpoint
        self.num_of_transfer = {"D": 0, "R": 0, "E": 0}
        self.num_of_receive = {"D": 0, "R": 0, "E": 0}

        # Sorted so index 0 is the robot closest to the relevant endpoint
        self.transfer_robot_dict = {"D": [], "R": [], "E": []}
        self.receive_robot_dict = {"D": [], "R": [], "E": []}

        self.robot_queue = []
        self.goal_dict_for_received_robot = {"D": [], "R": [], "E": []}


class allocation_set_class:
    """A single robot-goal allocation result."""
    def __init__(self, robot_index, goal_index):
        self.robot_index = robot_index
        self.goal_index = goal_index
        self.travel_dist = 0
        self.travel_path = []
        self.travel_section = []


class set_class:
    """Container for a complete set of allocation results."""
    def __init__(self, allocation_result=None, cost_sum=None,
                 using_section=None, case1_section=None,
                 case2_section=None, case3_section=None):
        self.allocation_result = allocation_result
        self.init_allocation = None
        self.init_path_list = None
        self.init_path_dist_list = None
        self.same_seg_allocation = None
        self.same_seg_path_list = None
        self.same_seg_path_dist_list = None
        self.diff_seg_allocation = None
        self.diff_seg_path_list = None
        self.diff_seg_path_dist_list = None
        self.cost_sum = cost_sum
        self.using_section = using_section
        self.case1_section = case1_section
        self.case2_section = case2_section
        self.case3_section = case3_section
