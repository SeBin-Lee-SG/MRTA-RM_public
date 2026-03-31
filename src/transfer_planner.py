from collections import defaultdict


class Arbiter:
    """Analyzes section-to-section robot transfers and classifies sections
    into four cases based on transfer/receive patterns."""

    def __init__(self, section_class_list, diff_seg_section_list):
        self.section_class_list = section_class_list
        self.diff_seg_section_list = diff_seg_section_list

    def analyzer(self):
        self.section2section_transfer_dict = defaultdict(int)
        self.section2section_transfer_dict_key_list = []

        for lst in self.diff_seg_section_list:
            for i in range(len(lst) - 1):
                key = tuple(lst[i:i + 2])
                if key not in self.section2section_transfer_dict_key_list:
                    self.section2section_transfer_dict_key_list.append(key)
                self.section2section_transfer_dict[key] += 1

                source_section_idx = lst[i]
                destination_section_idx = lst[i + 1]

                source_section = self.section_class_list[source_section_idx]
                destination_section = self.section_class_list[destination_section_idx]

                source_section_start = source_section.start
                source_section_end = source_section.end
                destination_section_start = destination_section.start
                destination_section_end = destination_section.end

                source_section_is_JC_node = (source_section_start == source_section_end)
                destination_section_is_JC_node = (destination_section_start == destination_section_end)

                # Update source transfer dict
                if destination_section_idx in source_section.transfer_section_index_dict:
                    source_section.transfer_section_index_dict[destination_section_idx][1] += 1
                else:
                    if source_section_is_JC_node:
                        source_section.transfer_section_index_dict[destination_section_idx] = ["E", 1]
                    elif source_section_end == destination_section_start or \
                            source_section_end == destination_section_end:
                        source_section.transfer_section_index_dict[destination_section_idx] = ["D", 1]
                    elif source_section_start == destination_section_start or \
                            source_section_start == destination_section_end:
                        source_section.transfer_section_index_dict[destination_section_idx] = ["R", 1]

                # Update destination receive dict
                if source_section_idx in destination_section.receive_section_index_dict:
                    destination_section.receive_section_index_dict[source_section_idx][1] += 1
                else:
                    if destination_section_is_JC_node:
                        destination_section.receive_section_index_dict[source_section_idx] = ["E", 1]
                    elif source_section_end == destination_section_start or \
                            source_section_start == destination_section_start:
                        destination_section.receive_section_index_dict[source_section_idx] = ["D", 1]
                    elif source_section_end == destination_section_end or \
                            source_section_start == destination_section_end:
                        destination_section.receive_section_index_dict[source_section_idx] = ["R", 1]

                # Update transfer/receive counts
                if source_section_end == destination_section_start:
                    if source_section_is_JC_node:
                        source_section.num_of_transfer["E"] += 1
                    else:
                        source_section.num_of_transfer["D"] += 1
                    if destination_section_is_JC_node:
                        destination_section.num_of_receive["E"] += 1
                    else:
                        destination_section.num_of_receive["D"] += 1
                elif source_section_end == destination_section_end:
                    if source_section_is_JC_node:
                        source_section.num_of_transfer["E"] += 1
                    else:
                        source_section.num_of_transfer["D"] += 1
                    if destination_section_is_JC_node:
                        destination_section.num_of_receive["E"] += 1
                    else:
                        destination_section.num_of_receive["R"] += 1
                elif source_section_start == destination_section_end:
                    if source_section_is_JC_node:
                        source_section.num_of_transfer["E"] += 1
                    else:
                        source_section.num_of_transfer["R"] += 1
                    if destination_section_is_JC_node:
                        destination_section.num_of_receive["E"] += 1
                    else:
                        destination_section.num_of_receive["R"] += 1
                elif source_section_start == destination_section_start:
                    if source_section_is_JC_node:
                        source_section.num_of_transfer["E"] += 1
                    else:
                        source_section.num_of_transfer["R"] += 1
                    if destination_section_is_JC_node:
                        destination_section.num_of_receive["E"] += 1
                    else:
                        destination_section.num_of_receive["D"] += 1

        # Classify sections into 4 cases
        # Case 1: no transfer, no receive
        # Case 2: transfer only (no receive)
        # Case 3: receive only (no transfer)
        # Case 4: both transfer and receive
        self.case1_section_list = []
        self.case2_section_list = []
        self.case3_section_list = []
        self.case4_section_list = []

        for section_idx in range(len(self.section_class_list)):
            section = self.section_class_list[section_idx]
            transfer_num = section.num_of_transfer["D"] + section.num_of_transfer["R"] + section.num_of_transfer["E"]
            receive_num = section.num_of_receive["D"] + section.num_of_receive["R"] + section.num_of_receive["E"]

            if transfer_num == 0 and receive_num == 0:
                self.case1_section_list.append(section_idx)
            elif transfer_num != 0 and receive_num == 0:
                self.case2_section_list.append(section_idx)
            elif transfer_num == 0 and receive_num != 0:
                self.case3_section_list.append(section_idx)
            else:
                self.case4_section_list.append(section_idx)

        # Sort case4 sections by order of appearance in diff_seg paths
        section_appear_dict = {}
        for section_idx in self.case4_section_list:
            for path in self.diff_seg_section_list:
                if section_idx in path:
                    if section_idx not in section_appear_dict:
                        section_appear_dict[section_idx] = path.index(section_idx)
                    else:
                        if path.index(section_idx) > section_appear_dict[section_idx]:
                            section_appear_dict[section_idx] = path.index(section_idx)

        self.case4_section_list = sorted(self.case4_section_list, key=lambda x: section_appear_dict[x])

        return (self.case1_section_list, self.case2_section_list, self.case3_section_list,
                self.case4_section_list, self.section2section_transfer_dict_key_list,
                self.section2section_transfer_dict)
