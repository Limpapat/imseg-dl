from imsegdl.train import train
from imsegdl.eval import eval
import json

class ImsegDL:
    def __init__(self, params_path:str):
        self.params_path = params_path
        with open(params_path, 'r') as f:
            self.params = json.loads(f.read())
    
    def train_model(self):
        print("-"*40)
        print("--- Start taining: params path is {}".format(self.params_path))
        trained_model_path = train(self.params)
        message = "trained model is saved to {}".format(trained_model_path) if trained_model_path is not None else "there is no better model saved"
        print("--- Stop taining: {}".format(message))
        print("-"*40)
    
    def eval_model(self):
        print("-"*40)
        print("--- Start evaluation: trained model path is {}".format(self.params["EVALUATION"]["MODEL_PATH"]))
        saving_eval_path = eval(self.params)
        print("--- Stop evaluation: results are saved to {}".format(saving_eval_path))
        print("-"*40)