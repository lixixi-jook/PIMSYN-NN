import json
from copy import copy

SUPPORTED_LINEAR_OPERATION = ['OP_CONV', 'OP_POOL', 'OP_ELTWISE', 'OP_FC']
SUPPORTED_NONLINEAR_OPERATION = ['OP_SIGMOID', 'OP_RELU']
# OP_DAC includes MVM and DAC
CONV_OPERATION = ['OP_DAC', 'OP_ADC', 'OP_SHIFTADD', 'OP_ADDITION']
POOL_OPERATION = ['OP_POOL']
ELTWISE_OPERATION = ['OP_ELTWISE']
FC_OPERATION = ['OP_DAC', 'OP_ADC', 'OP_SHIFTADD', 'OP_ADDITION']


class NeuralNetworkLayer():

    def __init__(self,
                 name,
                 operation,
                 output_dim,
                 kwargs: dict
                 ):
        self.name = name
        self.op = operation
        self.N, self.C, self.W, self.H = output_dim if len(output_dim) == 4 else output_dim + [1, 1]
        self.op_cnt = -1
        self.dup = -1
        self.macro_cnt = 1
        self.index = -1
        self.op_order = []
        self.op_delay = []
        self.provider = []
        self.consumer = []
        for key, value in kwargs.items():
            setattr(self, key, value)


def modify_layer_attr(layer, **kwargs):
    for key, value in kwargs.items():
        setattr(layer, key, value)


class NeuralNetworkParser():

    def __init__(self,
                 file_path
                 ):

        file = open(file_path, 'r')
        self.network = json.load(file)
        self.layer_dict = {}
        self.lookup = {}
        self.conv_list = []
        self.conv_num = 0
        self.fc_list = []
        for layer in self.network:
            self.lookup[layer['name']] = layer
        self.layer_paras = {key: [] for key in ['conv_output_width',
                                                'conv_input_lenth',
                                                'conv_output_size',
                                                'conv_output_channel',
                                                'dup_range',
                                                'conv_weight_cap',
                                                'fc_input_channel',
                                                'fc_output_channel',
                                                'fc_weight_cap']}

    def iterative_search_consumer(self, target, consumer):
        for key in consumer:
            layer = self.lookup[key]
            op = layer['operation']
            if op in SUPPORTED_LINEAR_OPERATION:
                target.consumer.append(key)
            else:
                if op in SUPPORTED_NONLINEAR_OPERATION:
                    target.op_order.append(op)
                self.iterative_search_consumer(target, layer['consumer'])

    def extract_layer_infos(self):
        for index, layer in enumerate(self.network):
            if layer['operation'] not in SUPPORTED_LINEAR_OPERATION:
                continue
            name = layer['name']
            if 'pad_h0' not in layer['param']:
                layer['param']['pad_h0'] = 0
            if 'pad_w0' not in layer['param']:
                layer['param']['pad_w0'] = 0
            nn_layer = NeuralNetworkLayer(name=name,
                                          operation=layer['operation'],
                                          output_dim=layer['output_dim'],
                                          kwargs=layer['param'])
            if layer['operation'] == 'OP_CONV':
                self.conv_num += 1
                self.conv_list.append(name)
                self.layer_paras['conv_output_channel'].append(nn_layer.C)
                self.layer_paras['conv_output_width'].append(nn_layer.W)
                self.layer_paras['conv_input_lenth'].append(
                    nn_layer.kernel_h*nn_layer.kernel_w*nn_layer.input_channel)
                self.layer_paras['conv_output_size'].append(
                    nn_layer.W * nn_layer.H)
                nn_layer.op_order = copy(CONV_OPERATION)
            elif layer['operation'] == 'OP_POOL':
                nn_layer.op_order = copy(POOL_OPERATION)
            elif layer['operation'] == 'OP_ELTWISE':
                nn_layer.op_order = copy(ELTWISE_OPERATION)
            else:
                self.fc_list.append(name)
                self.layer_paras['fc_output_channel'].append(nn_layer.C)
                self.layer_paras['fc_input_channel'].append(nn_layer.input_channel)
                nn_layer.op_order = copy(FC_OPERATION)
            self.layer_dict[name] = nn_layer
            nn_layer.index = index
            if 'consumer' in layer.keys():
                self.iterative_search_consumer(nn_layer, layer['consumer'])
            else:
                nn_layer.consumer = []
            nn_layer.op_order.insert(0, 'OP_LD')
            nn_layer.op_order.append('OP_ST')

    def specify_layer_provider(self):
        for key, layer in self.layer_dict.items():
            for consumer in layer.consumer:
                self.layer_dict[consumer].provider.append(key)

    def parse_neural_network(self):
        self.extract_layer_infos()
        self.specify_layer_provider()

    def print_network_topo(self):
        for name, nn_layer in self.layer_dict.items():
            print(f'\n{name}:')
            for key, value in nn_layer.__dict__.items():
                if isinstance(value, list):
                    print(f'{key}: {", ".join(map(str, value))}')
                else:
                    print(f'{key}: {value}')
