import math
import time
from tqdm import tqdm
from neural_network_parse import NeuralNetworkParser
from evolution_algorithm import EvolutionAlgorithm
from simulated_annealing import SimulatedAnnealingAlgorithm


DEFAULT_DAC_RESOLUTION = [1, 2, 4]
DEFAULT_ADC_RESOLUTION = [4, 6, 7, 8, 9, 10, 11, 12, 13, 14]


def set_weight_duplication(layer_dict, dup):
    i = 0
    for key, nn_layer in layer_dict.items():
        if nn_layer.op == 'OP_CONV':
            nn_layer.dup = dup[i]
            nn_layer.op_cnt = math.ceil(nn_layer.W*nn_layer.H/nn_layer.dup)
            i = i + 1
        elif nn_layer.op == 'OP_FC':
            nn_layer.dup = 1
            nn_layer.op_cnt = math.ceil(nn_layer.W*nn_layer.H/nn_layer.dup)
        else:
            for name in nn_layer.provider:
                provider = layer_dict[name]
                nn_layer.op_cnt = max(nn_layer.op_cnt, provider.op_cnt)
            nn_layer.op_cnt = min(nn_layer.op_cnt, nn_layer.W*nn_layer.H)
            nn_layer.dup = math.ceil(nn_layer.W*nn_layer.H/nn_layer.op_cnt)


def design_space_exploration(rram_ratio, rram_res, xbar_size, pimsyn_cfg):

    best_ever = -1
    best_arch = {
        'rram_ratio': rram_ratio,
        'rram_res': rram_res,
        'xbar_size': xbar_size,
        'power_efficiency': -1,
        'dup': None,
        'gene': None,
        'xbar_alloc': None
    }

    nn_parser = NeuralNetworkParser(pimsyn_cfg['network'])
    nn_parser.parse_neural_network()
    layer_paras = nn_parser.layer_paras
    layer_dict = nn_parser.layer_dict
    layer_list = nn_parser.conv_list + nn_parser.fc_list

    rrams_for_weight = math.ceil(pimsyn_cfg["weight_res"]/rram_res)
    conv_weight_volumn = [rrams_for_weight * math.ceil(x/xbar_size) * math.ceil(y/xbar_size) for x, y
                          in zip(layer_paras['conv_input_lenth'], layer_paras['conv_output_channel'])]
    fc_weight_volumn = [rrams_for_weight * math.ceil(x/xbar_size) * math.ceil(y/xbar_size)
                        for x, y in zip(layer_paras['fc_input_channel'],
                                        layer_paras['fc_output_channel'])]
    total_power = pimsyn_cfg["total_power"]
    rram_power = total_power * rram_ratio
    xbar_paras = pimsyn_cfg['RRAM'][f'{xbar_size}_{rram_res}']
    total_xbar_num = int(rram_power / xbar_paras['peak_power'])
    conv_xbar_num = total_xbar_num - sum(fc_weight_volumn)

    if conv_xbar_num <= sum(conv_weight_volumn):
        print(f"Processing ({rram_ratio}, {rram_res}, {xbar_size}) RRAM CAPACITY IS NOT ENOUGH")
        return best_arch

    ea_engine = EvolutionAlgorithm(layer_dict=layer_dict,
                                   layer_list=layer_list,
                                   layer_paras=layer_paras,
                                   config=pimsyn_cfg,
                                   max_power=total_power-rram_power,
                                   rram_res=rram_res,
                                   xbar_size=xbar_size
                                   )
    sa_engine = SimulatedAnnealingAlgorithm(layer_parameters=layer_paras,
                                            weight_volumn=conv_weight_volumn,
                                            xbar_size=xbar_size,
                                            rrams_for_weight=rrams_for_weight,
                                            sa_config=pimsyn_cfg["SA"]
                                            )

    start = time.time()
    sa_engine.run(conv_xbar_num)

    for dup in tqdm(sa_engine.candidates,
                    desc=f'Iterate dup candidates rram_ratio={rram_ratio} rram_res={rram_res} xbar_size={xbar_size}'):
        set_weight_duplication(layer_dict, dup)
        ea_engine.build_mutate_space()
        # print('dup is', dup)
        for dac_res in DEFAULT_DAC_RESOLUTION:
            # print('dac_res is', dac_res)
            adc_res = int(math.log(xbar_size, 2) + dac_res + rram_res - 1) \
                if dac_res > 1 and rram_res > 1 else \
                int(math.log(xbar_size, 2) + dac_res + rram_res - 2)
            if adc_res not in DEFAULT_ADC_RESOLUTION:
                continue
            ea_engine.run(dac_res, adc_res)

            if ea_engine.best_perf > best_ever:
                best_ever = ea_engine.best_perf
                xbar_alloc = [x * y for x, y in zip(dup, conv_weight_volumn)] + fc_weight_volumn
                best_arch['gene'] = ea_engine.best_gene
                best_arch['dac_res'] = dac_res
                best_arch['adc_res'] = adc_res
                best_arch['dup'] = dup
                best_arch['xbar_alloc'] = xbar_alloc
                best_arch['power_efficiency'] = ea_engine.best_perf/1e9

    set_weight_duplication(layer_dict, dup)
    ea_engine.evaluate_fitness(best_arch['gene'],
                               best_arch['dac_res'],
                               best_arch['adc_res'],
                               loginfo=best_arch
                               )
    end = time.time()
    duration = (end-start)/60
    print(f"Processing ({rram_ratio}, {rram_res}, {xbar_size}) cost {duration}min \
          with efficienct power efficiency = {best_ever/1e+9}GOPS/W")

    return best_arch


if __name__ == '__main__':

    import argparse
    import json
    parser = argparse.ArgumentParser(description="Please specify the dse arguments")
    parser.add_argument('--network', '-n',
                        type=str,
                        required=True,
                        help='Specify the output path of onnx2json frontend')
    parser.add_argument('--total_power', '-p',
                        type=float,
                        required=True,
                        help='Specify the peak power limitation')
    parser.add_argument('--macro_setting', '-m',
                        type=str,
                        default='unified',
                        choices=['unified', 'specified'],
                        help='Possible values: unified, specified')
    parser.add_argument('--cfg', '-c',
                        type=str,
                        required=True,
                        help='Specify configuration file path'
                        )
    parser.add_argument('--macro_reuse', '-r',
                        type=bool,
                        default=False,
                        help='Whether enable inter-layer macro sharing')
    parser.add_argument('-o',
                        type=str)
    parser.add_argument('-ratio',
                        type=float)
    parser.add_argument('-res',
                        type=int)
    parser.add_argument('-size',
                        type=int)
    args = parser.parse_args()

    config_file = open(args.cfg, 'r')
    pimsyn_cfg = json.load(config_file)
    pimsyn_cfg.update(vars(args))
    best_arch = design_space_exploration(rram_ratio=args.ratio,
                                         rram_res=args.res,
                                         xbar_size=args.size,
                                         pimsyn_cfg=pimsyn_cfg)
