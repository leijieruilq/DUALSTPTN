import torch.nn as nn
from .model import Model

class cat_ptnet_api(nn.Module):
    def __init__(self, configs, graph_generator, fixed_adjs):
        super(cat_ptnet_api, self).__init__()
        model_configs = self.load_configs(configs)
        self.adjs = fixed_adjs
        if graph_generator is not None:
            self.with_GraphGen = True
            self.graph_generator = graph_generator
        else:
            self.with_GraphGen = False
        # spatial temporal predictor
        self.model = Model(model_configs)
 
    def load_configs(self, configs):
        model_configs = configs['model']
        model_configs['c_date'] = configs['dataset']['c_date']
        model_configs['n_nodes'] = configs['dataset']['n_nodes']
        model_configs['c_in'] = configs['envs']['c_in']
        model_configs['c_out'] = configs['envs']['c_out']
        model_configs['device'] = configs['envs']['device']
        model_configs['inp_len'] = configs['envs']['inp_len']
        model_configs['pred_len'] = configs['envs']['pred_len']
        model_configs['dropout'] = configs['envs']['dropout']
        return model_configs
    
    def forward(self, seq_x, seq_x_mark, seq_y_mark, **args):
        assert seq_x.size(2) == 1
        predicts = self.model(seq_x, seq_x_mark, seq_y_mark, adjs=None)# scaler=args['scaler'])
        return predicts, 0.0