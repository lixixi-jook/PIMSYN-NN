
import math
import copy


def calculate_workload(layer_dict,
                       xbar_size,
                       rrams_for_weight,
                       vfu_width=32
                       ):
    workload = {}
    for key, nn_layer in layer_dict.items():
        workload[key] = {}
        xbars_share_input = math.ceil(nn_layer.C/xbar_size) * rrams_for_weight
        if nn_layer.op == 'OP_CONV':
            input_size = nn_layer.input_channel * nn_layer.kernel_h * nn_layer.kernel_w
        if nn_layer.op == 'OP_FC':
            input_size = nn_layer.input_channel
        for op in nn_layer.op_order:
            comp = op.split('_')[-1]
            if op in ['OP_LD', 'OP_ST', 'OP_TRAN', 'OP_MRG']:
                continue
            elif op == 'OP_DAC':
                workload[key][comp] = nn_layer.dup * input_size * xbars_share_input
            elif op == 'OP_ADC':
                workload[key][comp] = nn_layer.dup * math.ceil(nn_layer.C/xbar_size) * xbar_size * \
                    math.ceil(input_size/xbar_size) * rrams_for_weight
            elif op == 'OP_SHIFTADD':
                workload[key][comp] = nn_layer.dup * math.ceil(nn_layer.C/xbar_size) * xbar_size * \
                    math.ceil(input_size/xbar_size) * rrams_for_weight / vfu_width
            elif op == 'OP_ADDITION':
                workload[key][comp] = nn_layer.dup * math.ceil(input_size/xbar_size) * \
                    rrams_for_weight / vfu_width
            else:
                workload[key][comp] = nn_layer.dup * nn_layer.C / vfu_width
            if nn_layer.op == 'OP_FC':
                workload[key][comp] = min(math.ceil(workload[key][comp] / 100), 100)
    return workload


def allocate_components(workload,
                        hardware_config,
                        max_power,
                        macro_sharing,
                        adc_res,
                        dac_res,
                        macro_setting,
                        layer_dict
                        ):

    wl = copy.deepcopy(workload)
    comp_alloc = {key: {comp: 0 for comp in value.keys()}
                  for key, value in wl.items()}
    const_value, clk = 0, 0
    for key, value in macro_sharing.items():
        if not value:
            continue
        if wl[key]['ADC'] < wl[value]['ADC']:
            wl[key]['ADC'] = 0
        else:
            wl[value]['ADC'] = 0
    for key, value in wl.items():
        for comp in value.keys():
            paras = hardware_config[comp]
            if comp == 'DAC':
                comp_alloc[key][comp] = wl[key][comp]
                max_power -= wl[key][comp] * paras[str(dac_res)]['peak_power']
            elif comp == 'ADC':
                const_value += wl[key][comp] * paras[str(adc_res)]['peak_power'] \
                    / paras[str(adc_res)]['frequency']
            else:
                const_value += wl[key][comp] * paras['peak_power'] \
                    / paras['frequency']
    const_value = const_value / max_power
    max_comp = {}
    for key, value in wl.items():
        nn_layer = layer_dict[key]
        for comp in value.keys():
            if comp in ['DAC']:
                continue
            paras = hardware_config[comp][str(adc_res)] if comp == 'ADC' \
                else hardware_config[comp]
            comp_alloc[key][comp] = math.ceil(wl[key][comp]/paras['frequency']/const_value) \
                if wl[key][comp] > 0 else 0
            if nn_layer.op == 'OP_FC':
                continue
            latency = wl[key][comp]/(paras['frequency']*comp_alloc[key][comp]) \
                if comp_alloc[key][comp] > 0 else 0
            clk = max(clk, latency)
            if comp not in max_comp.keys():
                max_comp[comp] = 0
            max_comp[comp] = max(max_comp[comp],
                                 math.ceil(comp_alloc[key][comp]/nn_layer.macro_cnt))

    if macro_setting == 'specified':
        return clk, comp_alloc

    for key, value in comp_alloc.items():
        nn_layer = layer_dict[key]
        for comp in value.keys():
            paras = hardware_config[comp][str(adc_res)] if comp == 'ADC' \
                else hardware_config[comp]
            if comp not in max_comp.keys():
                continue
            comp_alloc[key][comp] = max_comp[comp] * nn_layer.macro_cnt \
                if comp_alloc[key][comp] > 0 else 0
            if nn_layer.op == 'OP_FC':
                continue
            latency = wl[key][comp]/(paras['frequency']*comp_alloc[key][comp]) \
                if comp_alloc[key][comp] > 0 else 0
            clk = max(clk, latency)
    return clk, comp_alloc


def calculate_power_requirements(workload,
                                 hardware_config,
                                 dac_res,
                                 adc_res,
                                 layer_dict):
    mini_power = 0
    for key, value in workload.items():
        nn_layer = layer_dict[key]
        for comp in value.keys():
            paras = hardware_config[comp]
            if comp == 'DAC':
                mini_power += workload[key]['DAC'] * paras[str(dac_res)]['peak_power']
            elif comp == 'ADC':
                mini_power += nn_layer.macro_cnt * paras[str(adc_res)]['peak_power']
            else:
                mini_power += nn_layer.macro_cnt * paras['peak_power']
    return mini_power
