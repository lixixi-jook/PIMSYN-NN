import math
from copy import copy

'''
Allocate macros for each layer
Each layer's outputs will be gathered into its last macro's local memory
And its consumer layer will access inputs from it
'''


class LayerMacroMapper():

    def __init__(self, name):
        self.name = name
        self.macro_axis = []
        self.merge_axis = []
        self.transfer_links = []
        self.transfer_hops = 0
        self.merge_links = []
        self.merge_hops = 0


class NeuralNetworkMacroMapper():

    def __init__(self, layer_dict, macro_alloc, macro_sharing):
        self.layer_dict = layer_dict
        self.macro_alloc = macro_alloc
        self.macro_sharing = macro_sharing
        self.layout = {key: LayerMacroMapper(key) for key in layer_dict.keys()}
        self.macro_num = sum(list(macro_alloc.values()))
        self.row = math.ceil(math.sqrt(self.macro_num))
        self.col = math.ceil(self.macro_num/self.row)

    def config_layers_macro_cnt(self):
        for key, value in self.macro_alloc.items():
            nn_layer = self.layer_dict[key]
            to_share = self.macro_sharing[key]
            if to_share:
                nn_layer.macro_cnt = self.macro_alloc[to_share] + value
                self.layer_dict[to_share].macro_cnt += value
            else:
                nn_layer.macro_cnt = value

    def calculate_axis(self, current_axis, offset, macro_cnt, macro_axis):
        for _ in range(macro_cnt):
            current_axis[1] = current_axis[1] + offset
            if current_axis[1] == self.col:
                current_axis[1] -= 1
                current_axis[0] += 1
                offset = -1
            if current_axis[1] == -1:
                current_axis[1] += 1
                current_axis[0] += 1
                offset = 1
            macro_axis.append(copy(current_axis))
        return offset

    def determine_layout(self):
        self.specify_macro_axis()
        self.specify_occupied_links()
        noc_conflict = self.add_noc_conflict()
        return noc_conflict

    def specify_macro_axis(self):
        self.config_layers_macro_cnt()
        current_axis = [0, -1]
        offset = 1
        for key, nn_layer in self.layer_dict.items():
            layer_mapper = self.layout[key]
            if nn_layer.op in ['OP_CONV', 'OP_FC']:
                to_share = self.macro_sharing[key]
                if to_share:
                    layer_mapper.macro_axis = self.layout[to_share].macro_axis
                    try:
                        layer_mapper.merge_axis = layer_mapper.macro_axis[0]
                    except IndexError:
                        print('INDEXERROR', layer_mapper.macro_axis)
                        assert False
                else:
                    offset = self.calculate_axis(current_axis, offset,
                                                 nn_layer.macro_cnt,
                                                 layer_mapper.macro_axis)
                    layer_mapper.merge_axis = layer_mapper.macro_axis[-1]
                # print(f'\n{key} \nMACRO_AXIS{layer_mapper.macro_axis}\nMERGE_AXIS{layer_mapper.merge_axis}')
            else:
                name = nn_layer.provider[0]
                layer_mapper.macro_axis.append(self.layout[name].merge_axis)
                layer_mapper.merge_axis = layer_mapper.macro_axis[-1]
                # print(f'\n{key} \nMACRO_AXIS{layer_mapper.macro_axis}\nMERGE_AXIS{layer_mapper.merge_axis}')

    def find_paths(self, src_points, dst_points, links, max_hops):
        for src in src_points:
            for dst in dst_points:
                y_offset = 1 if dst[1] > src[1] else -1
                x_offset = 1 if dst[0] > src[0] else -1
                hops = abs(src[0]-dst[0]) + abs(src[1]-dst[1])
                for y in range(src[1], dst[1], y_offset):
                    name = f'{src[0]}_{y}-{src[0]}_{y+y_offset}'
                    links.append(name)
                for x in range(src[0], dst[0], x_offset):
                    name = f'{x}_{dst[1]}-{x+x_offset}_{dst[1]}'
                    links.append(name)
                max_hops = max(max_hops, hops)
        return max_hops

    def specify_occupied_links(self):
        for key, nn_layer in self.layer_dict.items():
            layer_mapper = self.layout[key]
            max_hops = 0
            for name in nn_layer.provider:
                src = [self.layout[name].merge_axis]
                dst = layer_mapper.macro_axis
                max_hops = self.find_paths(src, dst,
                                           layer_mapper.transfer_links,
                                           max_hops)
            layer_mapper.transfer_hops = max_hops
            src = layer_mapper.macro_axis
            dst = [layer_mapper.merge_axis]
            layer_mapper.merge_hops = self.find_paths(src, dst,
                                                      layer_mapper.merge_links,
                                                      max_hops=0)
            layer_mapper.transfer_links = set(layer_mapper.transfer_links)
            layer_mapper.merge_links = set(layer_mapper.merge_links)
            # print(f'\n{key} \nTRANS_LINKS{layer_mapper.transfer_links}\nMERGE_LINKS{layer_mapper.merge_links}')
            # print(f'\nTRANS_HOPS {layer_mapper.transfer_hops}\nMERGE_HOPS {layer_mapper.merge_hops}')

    def add_noc_conflict(self):
        keys = list(self.layer_dict.keys())
        noc_conflict = []
        for i, key in enumerate(keys):
            layer_mapper = self.layout[key]
            if layer_mapper.transfer_links & layer_mapper.merge_links:
                noc_conflict.append([f'{key}-OP_TRAN', f'{key}-OP_MRG'])
            for key_to_cmp in keys[i+1:]:
                to_cmp = self.layout[key_to_cmp]
                if layer_mapper.transfer_links & to_cmp.transfer_links:
                    noc_conflict.append([f'{key}-OP_TRAN', f'{key_to_cmp}-OP_TRAN'])
                if layer_mapper.transfer_links & to_cmp.merge_links:
                    noc_conflict.append([f'{key}-OP_TRAN', f'{key_to_cmp}-OP_MRG'])
                if layer_mapper.merge_links & to_cmp.transfer_links:
                    noc_conflict.append([f'{key}-OP_MRG', f'{key_to_cmp}-OP_TRAN'])
                if layer_mapper.merge_links & to_cmp.merge_links:
                    noc_conflict.append([f'{key}-OP_MRG', f'{key_to_cmp}-OP_MRG'])
        return noc_conflict
