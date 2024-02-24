import math
import random
from copy import copy

from macro_allocate import NeuralNetworkMacroMapper
from evaluate import evaluate_memory_related_metrics, \
                     evaluate_components_energy, \
                     evaluate_rram_energy, \
                     evaluate_inter_macro_communication
from dataflow_compile import DataflowCompiler
from component_allocate import calculate_workload, \
                               allocate_components, \
                               calculate_power_requirements


class EvolutionAlgorithm():
    def __init__(self,
                 layer_dict,
                 layer_list,
                 layer_paras,
                 config,
                 max_power,
                 rram_res,
                 xbar_size,
                 ):

        self.best_perf = 0
        self.best_gene = []
        self.max_power = max_power
        self.layer_paras = layer_paras
        self.cfg = config
        self.rram_res = rram_res
        self.xbar_size = xbar_size
        self.gene_length = len(layer_list)
        self.macro_setting = config["macro_setting"]
        self.macro_reuse = config["macro_reuse"]
        self.layer_dict = layer_dict
        self.layer_list = layer_list                # conv + fc
        self.macro_num_limits = []
        self.macro_num_mutate_space = []
        self.macro_size_mutate_space = []
        self.mini_macro_size = []
        for key, value in config['EA'].items():
            setattr(self, key, value)

    def reset_ea_engine(self):
        self.population, self.fitness = [], []
        self.parents, self.parents_fitness = [], []
        self.childs, self.child_fitness = [], []

    def init_population(self):
        for _ in range(self.gene_num):
            if self.macro_setting == 'unified':
                self.macro_size = random.choice(self.macro_size_mutate_space)
                macro_num = [math.ceil(x / math.ceil(self.macro_size/y)) for x, y in
                             zip(self.macro_num_limits, self.mini_macro_size)]
                gene = [(i+1)*1000 + macro_num[i] for i in range(self.gene_length)]
            else:
                gene = [(i+1)*1000 + random.choice(self.macro_num_mutate_space[i])
                        for i in range(self.gene_length)]
            self.population.append(gene)

    def select_parents(self):
        self.parents = []
        pops_num = len(self.population)
        elite_num = math.ceil(self.elite_ratio*self.gene_num)
        random_num = self.gene_num - elite_num
        idx = sorted(list(range(pops_num)), key=lambda i: self.fitness[i], reverse=True)
        idx = idx[:elite_num] + random.sample(idx[elite_num:], random_num)
        self.parents = [self.population[id] for id in idx]
        self.parents_fitness = [self.fitness[id] for id in idx]

    def mutate_specified_macro_num(self, gene, idx):
        layer_idx = [int(x / 1000) for x in gene]
        shared_idx = [x for x, y in enumerate(layer_idx) if y == layer_idx[idx] and x != idx]
        if shared_idx:
            num_limits = min(self.macro_num_limits[idx], self.macro_num_limits[shared_idx[0]])
            candidates = [x for x in self.macro_num_mutate_space[idx]
                          if x <= num_limits-int(gene[shared_idx[0]] % 1000)]
            macro_num = random.choice(candidates)
        else:
            macro_num = random.choice(self.macro_num_mutate_space[idx])
        gene[idx] = layer_idx[idx] * 1000 + macro_num

    def mutate_unified_macro_num(self, gene):
        layer_idx = [int(x / 1000) for x in gene]
        macro_num = [int(x % 1000) for x in gene]
        macro_size = self.default_min_macro_size
        for idx in range(self.gene_length):
            shared_idx = [x for x, y in enumerate(layer_idx[idx+1:]) if y == layer_idx[idx]]
            if shared_idx:
                shared_idx = [x for x, y in enumerate(layer_idx) if y == layer_idx[idx] and x != idx]
                macro_num_sum = macro_num[idx] + macro_num[shared_idx[0]]
                macro_size = max(macro_size, int(self.mini_macro_size[idx]*(macro_num_sum/macro_num[idx])))
                macro_size = max(macro_size,
                                 int(self.mini_macro_size[shared_idx[0]]*(macro_num_sum/macro_num[shared_idx[0]])))
        start_id = math.ceil((macro_size-self.default_min_macro_size)/self.default_macro_size_stride)
        self.macro_size = random.choice(self.macro_size_mutate_space[start_id:])

        for idx, key in enumerate(self.layer_list):
            nn_layer = self.layer_dict[key]
            input_length = math.ceil(nn_layer.kernel_h*nn_layer.kernel_w*nn_layer.input_channel/self.xbar_size) \
                if nn_layer.op == 'OP_CONV' else math.ceil(nn_layer.input_channel/self.xbar_size)
            output_length = math.ceil(nn_layer.C/self.xbar_size)
            xbar_num = nn_layer.dup * input_length * output_length * math.ceil(self.cfg['weight_res']/self.rram_res)
            num = math.ceil(xbar_num/self.macro_size)
            gene[idx] = layer_idx[idx]*1000 + num

    def mutate_macro_sharing(self, gene, idx):
        layer_idx = [int(x / 1000) for x in gene]
        macro_num = [int(x % 1000) for x in gene]
        shared_idx = [x for x, y in enumerate(layer_idx) if y == layer_idx[idx] and x != idx]
        if shared_idx:
            gene[shared_idx[0]] = (shared_idx[0]+1) * 1000 + macro_num[shared_idx[0]]
            gene[idx] = (idx+1) * 1000 + macro_num[idx]
        else:
            candidates = [x + 1 for x, y in enumerate(layer_idx[:idx])
                          if layer_idx.count(y) == 1
                          and macro_num[x] + macro_num[idx] <=
                          min(self.macro_num_limits[idx], self.macro_num_limits[x])]
            if candidates:
                gene[idx] = random.choice(candidates) * 1000 + macro_num[idx]

    def validate_childs(self, gene):
        layer_idx = [int(x / 1000) for x in gene]
        macro_num = [int(x % 1000) for x in gene]
        assert all(layer_idx.count(x) <= 2 for x in layer_idx), \
            print('MUTATE MACRO SHARING DID NOT PASS')
        for i in range(self.gene_length):
            shared_idx = [x for x, y in enumerate(layer_idx) if y == layer_idx[i] and x != i]
            if shared_idx:
                num_limits = min(self.macro_num_limits[i], self.macro_num_limits[shared_idx[0]])
            else:
                num_limits = self.macro_num_limits[i]
            assert macro_num[i] <= num_limits, print('MUTATE MACRO NUM DID NOT PASS')

    def mutate(self):
        for parent in self.parents:
            child = copy(parent)
            if self.macro_setting == 'unified' \
                    and random.random() < self.macro_num_mutate_prob:
                self.mutate_unified_macro_num(child)
            for i in range(self.gene_length):
                point = random.random()
                if point < self.macro_num_mutate_prob + self.macro_sharing_mutate_prob \
                        and point >= self.macro_sharing_mutate_prob \
                        and self.macro_setting == 'specified':
                    self.mutate_specified_macro_num(child, i)
                if point < self.macro_sharing_mutate_prob \
                        and self.macro_reuse:
                    self.mutate_macro_sharing(child, i)
            self.validate_childs(child)

    def build_mutate_space(self):
        for key in self.layer_list:
            nn_layer = self.layer_dict[key]
            input_length = math.ceil(nn_layer.kernel_h*nn_layer.kernel_w*nn_layer.input_channel/self.xbar_size) \
                if nn_layer.op == 'OP_CONV' else math.ceil(nn_layer.input_channel/self.xbar_size)
            max_macro_num = nn_layer.dup * input_length
            macro_size = math.ceil(nn_layer.C/self.xbar_size) * math.ceil(self.cfg['weight_res']/self.rram_res)
            self.macro_num_limits.append(max_macro_num)
            self.mini_macro_size.append(macro_size)
            if self.macro_setting == 'specified':
                input_div_space = [i for i in range(1, input_length+1) if input_length % i == 0]
                dup_div_space = [i for i in range(1, nn_layer.dup+1) if nn_layer.dup % i == 0]
                candidates = set([x * y for x in input_div_space for y in dup_div_space if x * y < 1000])
                self.macro_num_mutate_space.append(sorted(list(candidates)))
            else:
                macro_size = math.ceil(macro_size/self.default_macro_size_stride) * self.default_macro_size_stride
                self.default_min_macro_size = max(self.default_min_macro_size, macro_size)
        if self.macro_setting == 'unified':
            self.macro_size_mutate_space = list(range(self.default_min_macro_size, self.default_max_macro_size+1,
                                                      self.default_macro_size_stride))

    def run(self, dac_res, adc_res):
        self.reset_ea_engine()
        self.init_population()
        self.fitness = [self.evaluate_fitness(gene, dac_res, adc_res)
                        for gene in self.population]
        for _ in range(self.max_iter):
            self.select_parents()
            self.mutate()
            self.child_fitness = [self.evaluate_fitness(gene, dac_res, adc_res)
                                  for gene in self.childs]
            self.population = self.parents + self.childs
            self.fitness = self.parents_fitness + self.child_fitness
        self.best_perf = max(self.fitness)
        max_id = self.fitness.index(self.best_perf)
        self.best_gene = self.population[max_id]

    def init_op_delay(self):
        for nn_layer in self.layer_dict.values():
            if 'OP_MRG' in nn_layer.op_order:
                mrg_id = nn_layer.op_order.index('OP_MRG')
                nn_layer.op_order.pop(mrg_id)
                nn_layer.op_delay.pop(mrg_id)
            if 'OP_TRAN' in nn_layer.op_order:
                tran_id = nn_layer.op_order.index('OP_TRAN')
                nn_layer.op_order.pop(tran_id)
                nn_layer.op_delay.pop(tran_id)
            nn_layer.op_delay = [1 for _ in nn_layer.op_order]

    def set_macro_alloc(self, gene):
        macro_alloc, macro_sharing = {}, {}
        adc_conflict = []
        for i, key in enumerate(self.layer_list):
            nn_layer = self.layer_dict[key]
            layer_idx = int(gene[i] / 1000)
            macro_cnt = int(gene[i] % 1000)
            nn_layer.macro_cnt = macro_cnt
            macro_alloc[key] = macro_cnt
            if layer_idx == i + 1:
                macro_sharing[key] = None
            else:
                macro_sharing[key] = self.layer_list[layer_idx]
                adc_conflict.append([key, self.layer_list[layer_idx]])
        return macro_alloc, macro_sharing, adc_conflict

    def set_inter_macro_communication_delay(self, transfer_delay, merge_delay, clk):
        for key, nn_layer in self.layer_dict.items():
            if 'OP_MRG' in nn_layer.op_order:
                mrg_id = nn_layer.op_order.index('OP_MRG')
                nn_layer.op_delay[mrg_id] = math.ceil(merge_delay[key]/clk)
            if 'OP_TRAN' in nn_layer.op_order:
                tran_id = nn_layer.op_order.index('OP_TRAN')
                nn_layer.op_delay[tran_id] = math.ceil(transfer_delay[key]/clk)

    def set_intra_macro_communication_delay(self, ld_delay, st_delay, clk):
        for key, nn_layer in self.layer_dict.items():
            ld_id = nn_layer.op_order.index('OP_LD')
            st_id = nn_layer.op_order.index('OP_ST')
            nn_layer.op_delay[ld_id] = math.ceil(ld_delay[key]/clk)
            nn_layer.op_delay[st_id] = math.ceil(st_delay[key]/clk)

    def evaluate_fitness(self, gene, dac_res, adc_res, rram_read_lat=100, loginfo=None):

        feasible = True
        clk = rram_read_lat
        self.init_op_delay()
        bit_loop_cnt = math.ceil(self.cfg['data_res']/dac_res)
        rrams_for_weight = math.ceil(self.cfg['weight_res']/self.rram_res)
        macro_alloc, macro_sharing, adc_conflict = self.set_macro_alloc(gene)

        nn_macro_mapper = NeuralNetworkMacroMapper(self.layer_dict,
                                                   macro_alloc,
                                                   macro_sharing
                                                   )
        noc_conflict = nn_macro_mapper.determine_layout()
        transfer_delay, merge_delay = evaluate_inter_macro_communication(self.layer_dict,
                                                                         nn_macro_mapper.layout,
                                                                         self.cfg['noc_bw'],
                                                                         self.cfg['data_res']
                                                                         )
        total_macro_num = nn_macro_mapper.macro_num

        self.set_inter_macro_communication_delay(transfer_delay, merge_delay, clk)
        compiler = DataflowCompiler(self.layer_dict, bit_loop_cnt)

        # Here the clk is not accurate so that the op delay of each IR is not accurate
        # which makes the memory capacity an estimate value
        approximate_memory_capacity = compiler.generate_dataflow_graph(adc_conflict,
                                                                       noc_conflict,
                                                                       self.cfg['data_res'])
        if approximate_memory_capacity > 4096:
            return 0
        memory_capacity = 0
        iteration = 0

        # In normal circumstances, the storage capacity will converge to a fixed value after multiple iterations
        while (iteration < 10):
            iteration = iteration + 1
            ld_latency, st_latency, memory_dynamic_energy, memory_static_power, memory_peak_power = \
                evaluate_memory_related_metrics(self.layer_dict,
                                                self.cfg['data_res'],
                                                self.cfg['memory_bw'],
                                                total_macro_num,
                                                self.cfg['memory'][str(approximate_memory_capacity)],
                                                self.xbar_size,
                                                rrams_for_weight
                                                )
            # print(f'memory power is {memory_peak_power/self.max_power}\n \
            #       capacity is {approximate_memory_capacity}\n \
            #       total_macro_num is {total_macro_num}')

            component_power = self.max_power - memory_peak_power - self.cfg['noc_power']
            workload = calculate_workload(self.layer_dict,
                                          self.xbar_size,
                                          rrams_for_weight,
                                          self.cfg['vfu_width']
                                          )
            mini_power = calculate_power_requirements(workload,
                                                      self.cfg,
                                                      dac_res,
                                                      adc_res,
                                                      self.layer_dict
                                                      )
            if mini_power >= component_power:
                feasible = False
                break
            max_ir_latency, comp_alloc = allocate_components(workload, self.cfg,
                                                             component_power,
                                                             macro_sharing,
                                                             adc_res, dac_res,
                                                             self.cfg['macro_setting'],
                                                             self.layer_dict
                                                             )
            if not comp_alloc:
                break
            clk = max(rram_read_lat, max_ir_latency)
            self.set_intra_macro_communication_delay(ld_latency, st_latency, clk)
            self.set_inter_macro_communication_delay(transfer_delay, merge_delay, clk)

            memory_capacity = compiler.generate_dataflow_graph(adc_conflict,
                                                               noc_conflict,
                                                               self.cfg['data_res'])
            if memory_capacity == approximate_memory_capacity or memory_capacity > 4096:
                break
            approximate_memory_capacity = memory_capacity

        # local memory capacity is too large to achieve high power efficiency
        if not feasible or memory_capacity > 4096:
            return 0

        cycle = compiler.step
        total_time = cycle * clk
        component_energy = evaluate_components_energy(self.cfg, workload, comp_alloc,
                                                      self.layer_dict, bit_loop_cnt,
                                                      dac_res, adc_res, total_time
                                                      )
        rram_energy = evaluate_rram_energy(self.layer_dict,
                                           self.cfg['RRAM'][f'{self.xbar_size}_{self.rram_res}'],
                                           self.xbar_size, bit_loop_cnt,
                                           rrams_for_weight, total_time
                                           )
        memory_energy = memory_dynamic_energy + memory_static_power * total_time
        noc_energy = self.cfg['noc_power'] * total_time
        energy = component_energy + rram_energy + memory_energy + noc_energy
        ops = sum([2 * x * y * z for x, y, z in zip(self.layer_paras['conv_input_lenth'],
                                                    self.layer_paras['conv_output_size'],
                                                    self.layer_paras['conv_output_channel'])])
        ops += sum([2 * x * y for x, y in zip(self.layer_paras['fc_input_channel'],
                                              self.layer_paras['fc_output_channel'])])
        efficient_power_efficiency = ops / (energy*1e-9)

        # print(f'efficienct power efficiency is {efficient_power_efficiency/1e9}\n \
        #         clk is {clk} {max_ir_latency/clk}\n \
        #         energy is {energy}\n \
        #         time is {total_time}\n \
        #         memory power is {memory_peak_power/self.max_power}\n \
        #         component power is {component_power/self.max_power}\n \
        #         memory energy is {memory_energy/energy}\n \
        #         memory static energy is {memory_static_power * total_time/energy}\n \
        #         memory dynamic energy is {memory_dynamic_energy/energy}\n \
        #         component energy is {component_energy/energy}\n \
        #         rram energy is {rram_energy/energy}')

        if loginfo:
            loginfo["macro_h"] = nn_macro_mapper.row
            loginfo["macro_w"] = nn_macro_mapper.col
            loginfo["spm_capacity"] = memory_capacity
            idx = 0
            max_macro_size = 0
            for key, layer in self.layer_dict.items():
                layer_info = {}
                layer_info['duplication'] = layer.dup
                if layer.op in ['OP_CONV', 'OP_FC']:
                    layer_info['macro_num'] = macro_alloc[key]
                    layer_info['macro_sharing'] = str(macro_sharing[key])
                    layer_info['macro_size'] = math.ceil(loginfo['xbar_alloc'][idx]/macro_alloc[key])
                    max_macro_size = max(layer_info['macro_size'], max_macro_size)
                    idx = idx + 1
                for comp in comp_alloc[key].keys():
                    if comp == 'DAC':
                        continue
                    layer_info[comp] = math.ceil(comp_alloc[key][comp]/macro_alloc[key]) \
                        if layer.op in ['OP_CONV', 'OP_FC'] else comp_alloc[key][comp]
                loginfo[key] = layer_info
            if self.cfg['macro_setting'] == 'unified':
                for key in self.layer_list:
                    loginfo[key]['macro_size'] = math.ceil((max_macro_size-self.default_min_macro_size)
                                                           / self.default_macro_size_stride) * \
                                                            self.default_macro_size_stride + self.default_min_macro_size
            return

        return efficient_power_efficiency
