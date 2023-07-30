from imsegdl.plots import plot_test_gt, plot_cc
from imsegdl.dataset import COCODataset
from imsegdl.train import train
from imsegdl.eval import eval
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
        self.cache = None

    def __gen_dataset(self):
        cs = self.params['cs'] if 'cs' in self.params.keys() else {}
        pad = self.params['pad'] if 'pad' in self.params.keys() else 0
        self.ground_truth_dataset = COCODataset(self.params["DATASET"]["TEST_DIR"], self.params["DATASET"]["GROUND_TRUTH_ANN_FILE"], cs=cs, pad=pad)
        self.train_dataset = COCODataset(self.params["DATASET"]["TRAIN_DIR"], self.params["DATASET"]["TRAIN_ANN_FILE"], cs=self.ground_truth_dataset.coco.cs, pad=pad)
        self.val_dataset = COCODataset(self.params["DATASET"]["VAL_DIR"], self.params["DATASET"]["VAL_ANN_FILE"], cs=self.ground_truth_dataset.coco.cs, pad=pad)
        self.test_dataset = COCODataset(self.params["DATASET"]["TEST_DIR"], self.params["DATASET"]["TEST_ANN_FILE"], cs=self.ground_truth_dataset.coco.cs, pad=pad)
    
    def train_model(self):
        print("-"*40)
        print("--- Start taining: params path is {}".format(self.params_path))
        trained_model_path = train(self.params)
        message = "trained model is saved to {}".format(trained_model_path) if trained_model_path is not None else "there is no better model saved"
        print("--- Stop taining: {}".format(message))
        print("-"*40)
        return trained_model_path
    
    def eval_model(self):
        print("-"*40)
        print("--- Start evaluation: trained model path is {}".format(self.params["EVALUATION"]["MODEL_PATH"]))
        saving_eval_path, cache = eval(self.params)
        print("--- Stop evaluation: results are saved to {}".format(saving_eval_path))
        print("-"*40)
        self.cache = cache
        return saving_eval_path
    
    @property
    def train_eval_model(self):
        print("-"*40)
        print("--- Start train & eval")
        print("-"*40)
        trained_model_path = self.train_model()
        root_trained_model_path = "/".join(trained_model_path.split("/")[0:-1])
        print("-"*40)
        print("--- Train DONE")
        print("-"*40)
        self.params["EVALUATION"]["MODEL_PATH"] = trained_model_path
        self.params["EVALUATION"]["SAVE_PATH"] = root_trained_model_path
        self.params["plots_test_save"] = f"{root_trained_model_path}/test"
        self.params["DATASET"]["TEST_ANN_FILE"] = f"{root_trained_model_path}/_annotations.coco.json"
        saving_eval_path = self.eval_model()
        print("-"*40)
        print("--- Eval DONE")
        print("-"*40)
        self.show_plot
        print("-"*40)
        print("--- Stop train & eval")
        print("-"*40)
    
    @property
    def show_plot(self):
        if self.test_dataset is None or self.ground_truth_dataset is None:
            self.__gen_dataset()
        plots_test_save = self.params["plots_test_save"] if "plots_test_save" in self.params.keys() else None
        _mapping = plot_test_gt(self.test_dataset, self.ground_truth_dataset, plots_test_save=plots_test_save)
        return _mapping
    
    @property
    def show_cc(self):
        if self.test_dataset is None:
            self.__gen_dataset()
        plot_cc(self.test_dataset)