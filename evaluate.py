import math


def evaluate_memory_related_metrics(layer_dict,
                                    data_width,
                                    bandwidth,
                                    macro_num,
                                    memory_paras,
                                    xbar_size,
                                    rrams_for_weight
                                    ):
    leak_power = memory_paras['leak_power']
    read_latency = memory_paras['read_latency']
    write_latency = memory_paras['write_latency']
    read_energy = memory_paras['read_energy']
    write_energy = memory_paras['write_energy']
    static_power = macro_num * leak_power
    peak_power = leak_power * macro_num
    ld_latency, st_latency = {}, {}
    dynamic_energy = 0
    for key, nn_layer in layer_dict.items():
        input_div = math.ceil(nn_layer.macro_cnt/nn_layer.dup)
        dup_div = math.ceil(nn_layer.macro_cnt/input_div)
        xbars_share_input = math.ceil(nn_layer.C/xbar_size) * rrams_for_weight
        if nn_layer.op == 'OP_CONV':
            input_size = nn_layer.input_channel * nn_layer.kernel_h * nn_layer.kernel_w
            ld_data = (input_size/input_div) * (nn_layer.dup/dup_div) * xbars_share_input
            ld_volumn = nn_layer.dup * input_size * xbars_share_input
        elif nn_layer.op == 'OP_POOL':
            input_size = nn_layer.C * nn_layer.kernel_h * nn_layer.kernel_w
            ld_data = input_size * nn_layer.dup
            ld_volumn = ld_data
        elif nn_layer.op == 'OP_ELTWISE':
            ld_data = nn_layer.dup * nn_layer.C
            ld_volumn = ld_data
        else:
            input_size = nn_layer.input_channel
            ld_data = (input_size/input_div) * (nn_layer.dup/dup_div) * xbars_share_input
            ld_volumn = nn_layer.dup * input_size * xbars_share_input
        st_data = nn_layer.dup * nn_layer.C / dup_div
        st_volumn = nn_layer.dup * nn_layer.C
        ld_latency[key] = math.ceil(ld_data*data_width/bandwidth) * read_latency
        st_latency[key] = math.ceil(st_data*data_width/bandwidth) * write_latency
        ld_energy = math.ceil(ld_volumn*data_width/bandwidth) * read_energy
        st_energy = math.ceil(st_volumn*data_width/bandwidth) * write_energy
        dynamic_energy += (ld_energy + st_energy) * nn_layer.op_cnt
    return ld_latency, st_latency, dynamic_energy, static_power, peak_power


def evaluate_components_energy(hardware_config,
                               workload,
                               comp_alloc,
                               layer_dict,
                               bit_loop_cnt,
                               dac_res,
                               adc_res,
                               total_latency
                               ):
    components_energy = 0
    static_energy = 0
    dynamic_energy = 0
    for key, value in workload.items():
        nn_layer = layer_dict[key]
        for comp, wl in value.items():
            if comp == 'ADC':
                paras = hardware_config[comp][str(adc_res)]
            elif comp == 'DAC':
                paras = hardware_config[comp][str(dac_res)]
            else:
                paras = hardware_config[comp]
            static_power = paras['static_power']
            dynamic_power = paras['dynamic_power']
            latency = paras['latency']
            loop_cnt = bit_loop_cnt if comp in ['ADC', 'DAC', 'SHIFTADD'] else 1
            static_energy += total_latency * static_power * comp_alloc[key][comp]
            dynamic_energy += latency * dynamic_power * wl * nn_layer.op_cnt * loop_cnt
    components_energy = dynamic_energy + static_energy
    return components_energy


def evaluate_rram_energy(layer_dict,
                         paras,
                         xbar_size,
                         bit_loop_cnt,
                         rrams_for_weight,
                         total_latency):
    rram_energy = 0
    for nn_layer in layer_dict.values():
        if nn_layer.op in ['OP_POOL', 'OP_ELTWISE']:
            continue
        if nn_layer.op == 'OP_CONV':
            input_size = nn_layer.input_channel * nn_layer.kernel_h * nn_layer.kernel_w
        if nn_layer.op == 'OP_FC':
            input_size = nn_layer.input_channel
        xbars_share_input = math.ceil(nn_layer.C/xbar_size) * rrams_for_weight
        xbar_num = nn_layer.dup * xbars_share_input * math.ceil(input_size/xbar_size)
        static_power = paras['static_power']
        dynamic_power = paras['dynamic_power']
        latency = paras['latency']
        static_energy = total_latency * static_power * xbar_num
        dynamic_energy = latency * dynamic_power * xbar_num * nn_layer.op_cnt * bit_loop_cnt
        rram_energy += dynamic_energy + static_energy
    return rram_energy


def evaluate_inter_macro_communication(layer_dict, layout, noc_bw, data_width):
    transfer_delay, merge_delay = {}, {}
    for key, nn_layer in layer_dict.items():
        layer_mapper = layout[key]
        input_div = math.ceil(nn_layer.macro_cnt/nn_layer.dup)
        dup_div = math.ceil(nn_layer.macro_cnt/input_div)
        input_per_macro = math.ceil(nn_layer.input_channel/input_div) \
            if hasattr(nn_layer, 'input_channel') else 1
        dup_per_macro = math.ceil(nn_layer.dup/dup_div)
        merge_flit = math.ceil(dup_per_macro * nn_layer.C * data_width / noc_bw)
        if nn_layer.op == 'OP_CONV':
            transfer_flit = math.ceil((dup_per_macro*nn_layer.stride_w + nn_layer.kernel_w)
                                      * input_per_macro * data_width / noc_bw)
        elif nn_layer.op == 'OP_POOL':
            transfer_flit = math.ceil((nn_layer.dup*nn_layer.stride_w + nn_layer.kernel_w)
                                      * nn_layer.C * data_width / noc_bw)
        elif nn_layer.op == 'OP_ELTWISE':
            transfer_flit = math.ceil(nn_layer.dup * nn_layer.C * data_width / noc_bw)
        else:
            transfer_flit = math.ceil(input_per_macro * data_width / noc_bw)
        if layer_mapper.merge_hops != 0:
            nn_layer.op_order.append('OP_MRG')
            nn_layer.op_delay.append(1)
            merge_delay[key] = math.ceil(layer_mapper.merge_hops + merge_flit)
        if layer_mapper.transfer_hops != 0:
            nn_layer.op_order.insert(0, 'OP_TRAN')
            nn_layer.op_delay.insert(0, 1)
            transfer_delay[key] = math.ceil(layer_mapper.transfer_hops + transfer_flit)
    return transfer_delay, merge_delay
