import math


BIT_LEVEL_OPERATION = ['OP_DAC', 'OP_ADC', 'OP_SHIFTADD']


class IRNode:
    def __init__(self,
                 name,
                 op_name,
                 op_idx,
                 cb_idx,
                 bit_idx,
                 layer
                 ):
        self.name = name
        if layer == 'BUBBLE':
            self.layer = layer
            self.cpi = 1
            self.rep = 1
        else:
            self.layer = layer.name
            self.cpi = layer.op_delay[op_idx]
        self.child = []
        self.parent = []
        self.indegree = 0
        self.outdegree = 0
        self.op_name = op_name
        self.op_idx = op_idx    # index of op in layer's op_order
        self.cb_idx = cb_idx    # index of computation block
        self.bit_idx = bit_idx  # index of bit iteration
        self.te = 0             # the most early execution time of op
        self.delay = {}         # delay between ops


def compute_depend_cb_idx(cb_idx, provider, nn_layer):
    pixel_id = min(nn_layer.W*nn_layer.W, (cb_idx+1)*nn_layer.dup)
    if nn_layer.op in ['OP_CONV', 'OP_POOL']:
        x = math.ceil(pixel_id/nn_layer.W)
        y = pixel_id % nn_layer.W
        y = nn_layer.W if y == 0 else y
        x = min(provider.W, (x-1)*nn_layer.stride_h + nn_layer.kernel_h - nn_layer.pad_h0)
        y = min(provider.W, (y-1)*nn_layer.stride_w + nn_layer.kernel_w - nn_layer.pad_w0)
        result = math.ceil(((x-1)*provider.W + y)/provider.dup)
    elif nn_layer.op == 'OP_ELTWISE':
        result = math.ceil(pixel_id/provider.dup)
    else:
        result = provider.op_cnt
    return result


def compute_depend_pixel_range(pixel_id, provider, nn_layer):
    if nn_layer.op in ['OP_CONV', 'OP_POOL']:
        x = math.ceil(pixel_id/nn_layer.W) if pixel_id > 0 else 1
        y = pixel_id % nn_layer.W
        y = nn_layer.W if y == 0 else y
        start_x = (x-1)*nn_layer.stride_h + 1 - nn_layer.pad_h0
        start_y = max(0, (y-1)*nn_layer.stride_w + 1 - nn_layer.pad_w0)
        start_id = (start_x-1)*provider.W + start_y if start_x > 0 else 0
        end_x = min(provider.W, (x-1)*nn_layer.stride_h + nn_layer.kernel_h - nn_layer.pad_h0)
        end_y = min(provider.W, (y-1)*nn_layer.stride_w + nn_layer.kernel_w - nn_layer.pad_w0)
        end_id = (end_x-1)*provider.W + end_y
    elif nn_layer.op == 'OP_ELTWISE':
        start_id = pixel_id
        end_id = pixel_id
    else:
        start_id = 0
        end_id = provider.W * provider.H
    return start_id, end_id


def calculate_input_buffer_capacity(cb_idx, provider, nn_layer):
    input_div = math.ceil(nn_layer.macro_cnt/nn_layer.dup)
    dup_div = math.ceil(nn_layer.macro_cnt/input_div)
    start_id = nn_layer.dup * cb_idx
    end_id = start_id + math.ceil(nn_layer.dup/dup_div)
    start_id = compute_depend_pixel_range(start_id, provider, nn_layer)[0]
    end_id = compute_depend_pixel_range(end_id, provider, nn_layer)[1]
    input_cap = (end_id-start_id) * provider.C / input_div
    return input_cap


class DataflowCompiler():

    def __init__(self,
                 layer_dict,
                 bit_loop_cnt
                 ):
        self.layer_dict = layer_dict
        self.bit_loop_cnt = bit_loop_cnt

    def init_dataflow_graph(self):
        self.step = 0
        self.node_dict = {}
        self.DAG = []
        self.node_cnt = 0
        self.input_buffer = {key: 0 for key in self.layer_dict.keys()}
        self.output_buffer = {key: 0 for key in self.layer_dict.keys()}
        self.buffer = {key: 0 for key in self.layer_dict.keys()}
        self.output_layout = {key: [0, 0] for key in self.layer_dict.keys()}
        src_list = self.generate_ir_nodes()
        return src_list

    def add_inter_op_dependency(self, node, layer):
        if node.op_idx > 0:
            prev_op = layer.op_order[node.op_idx-1]
            node.parent.append(
                f'{node.layer}-cb{node.cb_idx}-{prev_op}-iter{0}'
            )
            node.indegree += 1
        if node.op_idx < len(layer.op_order) - 1:
            next_op = layer.op_order[node.op_idx+1]
            name = f'{node.layer}-cb{node.cb_idx}-{next_op}-iter{0}'
            node.child.append(name)
            node.delay[name] = node.cpi * node.rep if \
                node.op_name in BIT_LEVEL_OPERATION \
                and next_op not in BIT_LEVEL_OPERATION else node.cpi
            node.outdegree += 1

    def add_inter_cb_dependency(self, node, layer):
        if node.cb_idx > 0:
            prev_cb = node.cb_idx - 1
            node.parent.append(
                f'{node.layer}-cb{prev_cb}-{node.op_name}-iter{0}'
            )
            node.indegree += 1
        if node.cb_idx < layer.op_cnt - 1:
            next_cb = node.cb_idx + 1
            name = f'{node.layer}-cb{next_cb}-{node.op_name}-iter{0}'
            node.child.append(name)
            if node.op_idx < len(layer.op_order) - 1:
                next_op = layer.op_order[node.op_idx+1]
                cpi = layer.op_delay[node.op_idx+1]
                rep = self.bit_loop_cnt if \
                    layer.op_order[node.op_idx+1] in BIT_LEVEL_OPERATION else 1
                inter_op_delay = node.cpi * node.rep if \
                    node.op_name in BIT_LEVEL_OPERATION \
                    and next_op not in BIT_LEVEL_OPERATION else node.cpi
            else:
                cpi, rep, inter_op_delay = 0, 0, 0
            node.delay[name] = max(node.cpi*node.rep, inter_op_delay+cpi*rep-1)
            node.outdegree += 1

    def add_inter_layer_dependency(self, node, layer):
        for key in layer.provider:
            parent_op = self.layer_dict[key].op_order[-1]
            parent_cb = compute_depend_cb_idx(node.cb_idx, self.layer_dict[key], layer) - 1
            parent_name = f'{key}-cb{parent_cb}-{parent_op}-iter{0}'
            node.parent.append(parent_name)
            node.indegree += 1
            parent_node = self.node_dict[parent_name]
            parent_node.child.append(node.name)
            parent_node.delay[node.name] = self.layer_dict[key].op_delay[-1]
            parent_node.outdegree += 1

    def add_inter_bit_dependency(self, node):
        if node.bit_idx > 0:
            prev_bit = node.bit_idx - 1
            node.parent.append(
                f'{node.layer}-cb{node.cb_idx}-{node.op_name}-iter{prev_bit}'
            )
            node.indegree += 1
        if node.bit_idx < self.bit_loop_cnt - 1:
            next_bit = node.bit_idx + 1
            name = f'{node.layer}-cb{node.cb_idx}-{node.op_name}-iter{next_bit}'
            node.child.append(name)
            node.delay[name] = node.cpi
            node.outdegree += 1

    def add_dependency(self, node, layer):
        if node.bit_idx == 0:
            self.add_inter_op_dependency(node, layer)
            self.add_inter_cb_dependency(node, layer)
        if node.op_idx == 0:
            self.add_inter_layer_dependency(node, layer)
        if node.op_name in BIT_LEVEL_OPERATION:
            self.add_inter_bit_dependency(node)

    def generate_ir_nodes(self):
        src = []
        for layer in self.layer_dict.values():
            for cb_idx in range(0, layer.op_cnt):
                for op_idx, op_name in enumerate(layer.op_order):
                    bit_iter = self.bit_loop_cnt if \
                        op_name in BIT_LEVEL_OPERATION else 1
                    for bit_idx in range(0, bit_iter):
                        name = f'{layer.name}-cb{cb_idx}-{op_name}-iter{bit_idx}'
                        node = IRNode(name, op_name, op_idx, cb_idx, bit_idx, layer)
                        node.rep = self.bit_loop_cnt if \
                            op_name in BIT_LEVEL_OPERATION else 1
                        self.add_dependency(node, layer)
                        self.node_cnt += 1
                        self.node_dict[name] = node
                        if node.parent == []:
                            src.append(name)
        return src

    def topology_traverse(self, src_list, active_node, semi_active):
        for src_name in src_list:
            src = self.node_dict[src_name]
            for child_name in src.child:
                child = self.node_dict[child_name]
                child.indegree = child.indegree - 1
                child.te = max(child.te, src.te + src.delay[child_name])
                if child.indegree == 0:
                    if self.step == child.te:
                        active_node.append(child_name)
                    else:
                        semi_active.append(child_name)

    def delete_finished_nodes(self, processing_nodes):
        for i in range(len(processing_nodes) - 1, -1, -1):
            item = processing_nodes[i]
            item[1] = item[1] + 1
            node = self.node_dict[item[0]]
            if item[1] == node.cpi:
                processing_nodes.pop(i)

    def add_processing_nodes(self, active_node, processing_nodes):
        processing_nodes.extend([[node, 0] for node in active_node])

    def check_adc_conflict(self, processing_nodes, active_node, conflict):
        adc_bubble = []
        processing_adc_nodes = [item for item in processing_nodes if
                                'ADC' in item[0] and 'bubble' not in item[0]]
        for i, item in enumerate(processing_adc_nodes):
            node = self.node_dict[item[0]]
            bit_idx = node.bit_idx
            idx = self.layer_dict[node.layer].index
            for item_to_cmp in processing_adc_nodes[i+1:]:
                node_to_cmp = self.node_dict[item_to_cmp[0]]
                bit_to_cmp = node_to_cmp.bit_idx
                idx_to_cmp = self.layer_dict[node_to_cmp.layer].index
                if [node.layer, node_to_cmp.layer] in conflict or \
                        [node_to_cmp.layer, node.layer] in conflict:
                    target = item if bit_idx < bit_to_cmp or \
                        (bit_idx == bit_to_cmp and idx > idx_to_cmp) else item_to_cmp
                    assert target[0] in active_node, print(f'active node {active_node} target {target[0]}')
                    active_node.remove(target[0])
                    processing_nodes.remove(target)
                    adc_bubble.append(target[0])
                    break
        return adc_bubble

    def check_noc_conflict(self, processing_nodes, active_node, conflict):
        noc_bubble = []
        processing_noc_nodes = [item for item in processing_nodes if
                                ('TRAN' in item[0] or 'MRG' in item[0]) and
                                'bubble' not in item[0]]
        for i, item in enumerate(processing_noc_nodes):
            if item[0] == 'has_deleted':
                continue
            node = self.node_dict[item[0]]
            op = node.op_name
            para = f'{node.layer}-{op}'
            delay = node.cpi - item[1]
            for item_to_cmp in processing_noc_nodes[i+1:]:
                if item_to_cmp[0] == 'has_deleted':
                    continue
                node_to_cmp = self.node_dict[item_to_cmp[0]]
                op_to_cmp = node_to_cmp.op_name
                para_to_cmp = f'{node_to_cmp.layer}-{op_to_cmp}'
                delay_to_cmp = node_to_cmp.cpi - item_to_cmp[1]
                if [para, para_to_cmp] in conflict or [para_to_cmp, para] in conflict:
                    target = item if item[1] < item_to_cmp[1] or \
                        (item[1] == item_to_cmp[1] and delay > delay_to_cmp) else item_to_cmp
                    active_node.remove(target[0])
                    processing_nodes.remove(target)
                    noc_bubble.append(target[0])
                    if target == item:
                        break
                    item_to_cmp[0] = 'has_deleted'
        return noc_bubble

    def add_bubble(self, bubble_list, active_node, processing_nodes):
        for name in bubble_list:
            node = self.node_dict[name]
            node.indegree = 1
            bubble_name = f'{name}-bubble'
            bubble = IRNode(bubble_name, 'OP_BUBBLE', 0, 0, 0, 'BUBBLE')
            self.node_dict[bubble_name] = bubble
            bubble.child.append(name)
            bubble.delay[name] = 1
            bubble.te = node.te
            active_node.append(bubble_name)
            processing_nodes.append([bubble_name, 0])

    def check_semi_active(self, semi_active, active_node):
        for name in semi_active[:]:
            node = self.node_dict[name]
            if node.te != self.step:
                continue
            semi_active.remove(name)
            active_node.append(name)
            assert node.te >= self.step, 'The te has become smaller than step'

    def update_local_memory_capacity(self, active_node, data_res):
        for key in active_node:
            input_buff_cap = 0
            node = self.node_dict[key]
            nn_layer = self.layer_dict[node.layer]
            if node.op_name in ['OP_LD', 'OP_ST']:
                # print(f'=========Node {key}===========')
                for name in nn_layer.provider:
                    provider = self.layer_dict[name]
                    input_buff_cap += calculate_input_buffer_capacity(node.cb_idx, provider, nn_layer)
                    start_id = nn_layer.dup * node.cb_idx
                    start_id = compute_depend_pixel_range(start_id, provider, nn_layer)[0]
                    self.output_layout[name][0] = start_id
                    output_buff_cap = (self.output_layout[name][1] - self.output_layout[name][0]) * provider.C
                    self.output_buffer[name] = max(self.output_buffer[name], output_buff_cap)
                    self.buffer[name] = max(self.buffer[name], self.input_buffer[name]+output_buff_cap)
                    # print(f'{node.layer}|{name} \
                    #       \noutput_layout {self.output_layout[name]} \
                    #       \noutput_buffer {self.output_buffer[name]*(data_res/8) / 1024}')
                self.input_buffer[node.layer] = max(self.input_buffer[node.layer], input_buff_cap)
                self.buffer[node.layer] = max(self.buffer[node.layer], self.output_buffer[node.layer]+input_buff_cap)
                # print(f'{node.layer} \
                #         \ninput_buff {self.input_buffer[node.layer]*(data_res/8) / 1024} \
                #         \noutput_buffer {self.output_buffer[node.layer]*(data_res/8) / 1024}')
            if node.op_name == 'OP_ST':
                self.output_layout[node.layer][1] = nn_layer.dup * (node.cb_idx+1)
                output_buff_cap = (self.output_layout[node.layer][1] -
                                   self.output_layout[node.layer][0]) * nn_layer.C
                self.output_buffer[node.layer] = max(self.output_buffer[node.layer], output_buff_cap)
                self.buffer[node.layer] = max(self.buffer[node.layer], self.input_buffer[node.layer]+output_buff_cap)
                # print(f'{node.layer} \
                #       \noutput_layout {self.output_layout[node.layer]} \
                #       \noutput_buffer {self.output_buffer[node.layer]*(data_res/8) / 1024}')

    def generate_dataflow_graph(self, adc_conflict, noc_conflict, data_res):
        src_list = self.init_dataflow_graph()
        semi_active = []
        self.DAG.append(src_list)
        processing_nodes = []
        self.add_processing_nodes(src_list, processing_nodes)
        while src_list or semi_active:
            active_node = []
            self.step = self.step + 1
            self.check_semi_active(semi_active, active_node)
            self.topology_traverse(src_list, active_node, semi_active)
            self.delete_finished_nodes(processing_nodes)
            self.add_processing_nodes(active_node, processing_nodes)
            self.update_local_memory_capacity(active_node, data_res)
            adc_bubble = self.check_adc_conflict(processing_nodes, active_node, adc_conflict)
            noc_bubble = self.check_noc_conflict(processing_nodes, active_node, noc_conflict)
            self.add_bubble(adc_bubble+noc_bubble, active_node, processing_nodes)
            self.DAG.append(active_node)
            src_list = active_node
        buff_cap = [self.input_buffer[key]+self.output_buffer[key] for key in self.layer_dict.keys()]
        memory_capacity = math.ceil((max(buff_cap) * (data_res/8)))
        memory_capacity = int(pow(2, math.ceil(math.log2(memory_capacity*0.9))) / 1024)
        return memory_capacity

    def print_DAG(self, processing_nodes):
        print(f'\nSTEP {self.step}')
        print(f'======Processing Node======\n{", ".join(map(lambda x: " | ".join(map(str, x)), processing_nodes))}')
