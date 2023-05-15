from imsegdl.train import train
from imsegdl.eval import eval
from imsegdl.plots import plot_test_gt, plot_cc
from imsegdl.dataset.dataset import COCODataset
import json

class ImsegDL:
    def __init__(self, params_path:str, **kwarg):
        self.params_path = params_path
        with open(params_path, 'r') as f:
            self.params = json.loads(f.read())
        self.params = {**self.params, **kwarg}
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.ground_truth_dataset = None
        self.__gen_dataset()

    def __gen_dataset(self):
        cs = self.params['cs'] if 'cs' in self.params.keys() else {}
        self.ground_truth_dataset = COCODataset(self.params["DATASET"]["TEST_DIR"], self.params["DATASET"]["GROUND_TRUTH_ANN_FILE"], cs=cs)
        self.train_dataset = COCODataset(self.params["DATASET"]["TRAIN_DIR"], self.params["DATASET"]["TRAIN_ANN_FILE"], cs=self.ground_truth_dataset.coco.cs)
        self.val_dataset = COCODataset(self.params["DATASET"]["VAL_DIR"], self.params["DATASET"]["VAL_ANN_FILE"], cs=self.ground_truth_dataset.coco.cs)
        self.test_dataset = COCODataset(self.params["DATASET"]["TEST_DIR"], self.params["DATASET"]["TEST_ANN_FILE"], cs=self.ground_truth_dataset.coco.cs)
    
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
    
    @property
    def show_plot(self):
        _mapping = plot_test_gt(self.test_dataset, self.ground_truth_dataset)
        return _mapping
    
    @property
    def show_cc(self):
        plot_cc(self.test_dataset)