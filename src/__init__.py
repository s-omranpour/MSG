import os
import pickle

class Experiment:
    def __init__(self, root_dir, name=None, data_conf=None, model_conf=None):
        os.makedirs(root_dir, exist_ok=True)
        
        if name is None:
            exps = sorted(filter(lambda x: x.startswith('exp_') ,os.listdir(root_dir)))
            name = f'exp_{len(exps)}'

        self.save_path = root_dir + name + '/'
        self.weights_path = self.save_path + 'weights/'
        self.logs_path = self.save_path + 'logs/'
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.weights_path, exist_ok=True)
        os.makedirs(self.logs_path, exist_ok=True)
        
        if self.load_configs():
            assert data_conf is not None and model_conf is not None
            self.data_conf = data_conf
            self.model_conf = model_conf
            self.save_configs()
        
        
    def save_configs(self):
        pickle.dump(self.data_conf, open(self.save_path + 'data_conf.pkl', 'wb'))
        pickle.dump(self.model_conf, open(self.save_path + 'model_conf.pkl', 'wb'))
        
    def load_configs(self):
        if not os.path.exists(self.save_path + 'data_conf.pkl'):
            self.data_conf = None
            self.model_conf = None
            return True
        
        self.data_conf = pickle.load(open(self.save_path + 'data_conf.pkl', 'rb'))
        self.model_conf = pickle.load(open(self.save_path + 'model_conf.pkl', 'rb'))
        return False