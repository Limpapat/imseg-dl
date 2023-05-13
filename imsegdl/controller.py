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
    
    # TODO
    def get_dataset(self):
        pass
    
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

class ComparePlot:
    def __init__(self, root_dir, ann_dir, ground_truth_ann_dir):
        self.root_dir = root_dir
        self.ann_dir = ann_dir
        self.ground_truth_ann_dir = ground_truth_ann_dir
        self._gt_dataset = COCODataset(root_dir, ground_truth_ann_dir)
        self._dataset = COCODataset(root_dir, ann_dir, cs=self._gt_dataset.coco.cs)
        self.mapping = None
    
    @property
    def show_plot(self):
        self.mapping = plot_test_gt(self._dataset, self._gt_dataset)
    
    @property
    def show_cc(self):
        plot_cc(self._dataset)