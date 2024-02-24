import math
from copy import copy
from statistics import mean, stdev
from random import randint, random, choice


def random_increase_duplication(candidate, weight_volumn, max_dup, index, delta):
    while True:
        to_inc = choice(index)
        dup_max = max_dup[to_inc] - candidate[to_inc]
        dup_min = math.ceil(delta/weight_volumn[to_inc])
        if dup_max > dup_min:
            break
    dup_inc = randint(dup_min, math.ceil(dup_min+(dup_max-dup_min)/5))
    index.remove(to_inc)
    candidate[to_inc] += dup_inc


def random_decrease_duplication(candidate, weight_volumn, index, delta):
    if index:
        to_dec = choice(index)
        dup_range = min(candidate[to_dec]-1, math.floor(delta/weight_volumn[to_dec]))
        if dup_range > 0:
            dup_dec = randint(1, dup_range)
            candidate[to_dec] -= dup_dec
        else:
            index.remove(to_dec)
    else:
        diff = [abs(weight_volumn[i]-delta) if candidate[i] > 1
                else math.inf for i in range(len(candidate))]
        to_dec = diff.index(min(diff))
        candidate[to_dec] -= 1


class SimulatedAnnealingAlgorithm:

    def __init__(self,
                 layer_parameters,
                 weight_volumn,
                 xbar_size,
                 rrams_for_weight,
                 sa_config
                 ):

        self.candidates = []
        self.value = []
        self.xbar_size = xbar_size
        self.rrams_for_weight = rrams_for_weight
        self.layer_paras = layer_parameters
        self.weight_volumn = weight_volumn
        self.layer_num = len(weight_volumn)
        for key, value in sa_config.items():
            setattr(self, key, value)

    def _evaluate(self, dup):
        is_sorted = all([(dup[i] >= 0.6 * dup[i+1]) for i in range(self.layer_num-1)])
        if not is_sorted:
            return math.inf
        xbars_share_input = [math.ceil(x/self.xbar_size) * self.rrams_for_weight
                             for x in self.layer_paras['conv_output_channel']]
        xbars_share_output = [math.ceil(x/self.xbar_size) * self.rrams_for_weight
                              for x in self.layer_paras['conv_input_lenth']]
        ld_volumn = [dup[i] * self.layer_paras['conv_input_lenth'][i] * xbars_share_input[i]
                     for i in range(self.layer_num)]
        st_volumn = [dup[i] * self.layer_paras['conv_output_channel'][i] * xbars_share_output[i]
                     for i in range(self.layer_num)]
        compute_volumn = [math.ceil(self.layer_paras['conv_output_size'][i]/dup[i])
                          for i in range(self.layer_num)]
        access_volumn = [ld_volumn[i] + st_volumn[i] for i in range(self.layer_num)]
        if self.func_name == 'computation':
            value = max(compute_volumn)
        else:
            value = stdev(compute_volumn) + self.alpha * stdev(access_volumn)
        return value

    def init_candidates(self, xbar_num):
        ratio = [x*y for x, y in zip(self.layer_paras['conv_output_size'], self.weight_volumn)]
        seed = [math.ceil((xbar_num*x/sum(ratio))/y) for x, y in zip(ratio, self.weight_volumn)]
        delta = sum([x*y for x, y in zip(seed, self.weight_volumn)]) - xbar_num
        idx = 0
        while (delta > mean(self.weight_volumn)):
            if seed[idx] <= 1:
                idx = idx + 1 if idx < len(seed)-1 else 0
                continue
            seed[idx] = seed[idx] - 1
            delta = delta - self.weight_volumn[idx]
            idx = idx + 1 if idx < len(seed)-1 else 0
        value = min(self._evaluate(seed), 1e+4)
        self.T0 = pow(10, math.floor(math.log10(value)))
        self.Tf = (self.T0/1000)
        self.T = self.T0
        self.candidates = [copy(seed) for _ in range(self.length)]
        self.value = [value for _ in range(self.length)]

    def metrospolis(self, value, value_new):
        if value_new <= value:
            return True
        else:
            point = math.exp((value - value_new) / self.T)
            if random() < point:
                return True
            else:
                return False

    def generate_new_candidates(self, seed, xbar_num, max_dup, max_iter):
        iter = 0
        candidate = copy(seed)
        index = list(range(self.layer_num))
        used_xbar_num = sum([x*y for x, y in zip(self.weight_volumn, candidate)])
        delta = max(xbar_num-used_xbar_num, 0)
        random_increase_duplication(candidate, self.weight_volumn, max_dup, index, delta)
        used_xbar_num = sum([x*y for x, y in zip(self.weight_volumn, candidate)])
        delta = abs(used_xbar_num-xbar_num)
        while delta > mean(self.weight_volumn) and iter <= max_iter:
            random_decrease_duplication(candidate, self.weight_volumn, index, delta)
            used_xbar_num = sum([x*y for x, y in zip(self.weight_volumn, candidate)])
            delta = abs(used_xbar_num-xbar_num)
            iter = iter + 1
        if iter > max_iter:
            print('Hard to generate new candidates')
            return seed
        return candidate

    def run(self, xbar_num, max_iter=100):
        dup_range = [math.floor((xbar_num-sum(self.weight_volumn))/x) + 1 for x in self.weight_volumn]
        self.init_candidates(xbar_num)
        while self.T > self.Tf:
            for _ in range(self.iter):
                for idx in range(self.length):
                    candidate = self.generate_new_candidates(self.candidates[idx], xbar_num, dup_range, max_iter)
                    value = self._evaluate(candidate)
                    if self.metrospolis(self.value[idx], value):
                        self.candidates[idx] = copy(candidate)
                        self.value[idx] = value
                    used_xbar_num = sum([x*y for x, y in zip(candidate, self.weight_volumn)])
                    assert abs(used_xbar_num-xbar_num) <= mean(self.weight_volumn), \
                        "The candidate has not fully used all xbars, left"
            self.T = self.T * self.sigma
