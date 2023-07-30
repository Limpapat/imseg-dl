# filename : eval.py
# updated : 12-03-2023
# version : v1.0

from imsegdl.utils import gen_empty_annf, mask2ann, iou_score
from imsegdl.dataset import COCODataset
from torch.utils.data import DataLoader
from imsegdl.model import UNet
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
import argparse
import torch
import json
import os

def eval(params:dict):
    print("-"*40)
    # cuda device setting
    if torch.cuda.is_available():
        DEVICE = 'cuda:0'
        print('Running on the GPU')
    else:
        DEVICE = "cpu"
        print('Running on the CPU')
    print("-"*40)

    # params setting
    TEST_DIR = params["DATASET"]["TEST_DIR"]
    TEST_ANN_FILE = params["DATASET"]["TEST_ANN_FILE"]
    GROUND_TRUTH_ANN_FILE = params["DATASET"]["GROUND_TRUTH_ANN_FILE"] if "GROUND_TRUTH_ANN_FILE" in params["DATASET"].keys() else TEST_ANN_FILE
    CATEGORIES = params["EVALUATION"]["CATEGORIES"]
    RESULT_PATH = params["EVALUATION"]["SAVE_PATH"]
    model_path = params["model_path"] if "model_path" in params.keys() else params["EVALUATION"]["MODEL_PATH"]
    checkpoint = torch.load(model_path, map_location=torch.device(DEVICE))
    N_CLASSES = checkpoint['n_classes']
    LEARNING_RATE = checkpoint['learning_rate']
    VERSION = checkpoint['version']
    BATCH_SIZE = params["batch_size"] if "batch_size" in params.keys() else 1
    DISP_PLOT = params["disp_plot"] if "disp_plot" in params.keys() else False
    RES_PLOT = params['res_plot'] if "res_plot" in params.keys() else True
    PTYPE = params["ptype"] if "ptype" in params.keys() else "segmentation"
    IMFORMAT = params["imformat"] if "imformat" in params.keys() else "png"
    IMAGE_SIZE = params["image_size"] if "image_size" in params.keys() else 512
    CS = params['cs'] if 'cs' in params.keys() else {}
    PAD = params['pad'] if 'pad' in params.keys() else 0
    N_CHANNELS = params['n_channels'] if 'n_channels' in params.keys() else 3

    # create empty _annotation.coco.json
    with open(CATEGORIES, 'r') as f:
        cats = json.loads(f.read())
    now = datetime.now()
    gen_empty_annf(root_dir=TEST_DIR,
                   ann_dir=TEST_ANN_FILE,
                   version=VERSION, 
                   stamp=now.strftime("%Y-%m-%dT%H:%M:%S+00:00"), 
                   cats=cats,
                   imformat=IMFORMAT,
                   image_size=IMAGE_SIZE)

    # load test dataset
    transform = params["transform"] if "transform" in params.keys() else None
    test_dataset = COCODataset(TEST_DIR, GROUND_TRUTH_ANN_FILE, categories_path=CATEGORIES, transforms=transform, ptype=PTYPE, cs=CS, pad=PAD)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # load trained model
    model = UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES).to(DEVICE).train()
    model.load_state_dict(checkpoint['model_state_dict'])

    # define saving evaluation results path
    saving_path = os.path.join(RESULT_PATH, "evaluation_{}".format(datetime.now().strftime("%Y%m%d%H%M%S")))
    os.mkdir(saving_path)
    print("-"*40)

    # evaluation
    model.eval()
    anns={"last_ann_id":-1, "annotation":[]}
    with torch.no_grad():
        iou_scores = []
        for idx, batch in enumerate(tqdm(test_loader)):
            X, y = batch
            X, y = X.to(DEVICE), y.to(DEVICE)
            logits = model(X)
            pred = nn.functional.softmax(logits.detach(), dim=1)
            pred_argmax = torch.argmax(pred, dim=1, keepdims=True)
            pred = torch.zeros_like(pred).scatter_(1, pred_argmax, 1)
            # gen annotation
            pred_mask = pred.detach().cpu().squeeze()
            sc = iou_score(pred_mask, y.float().cpu())
            print("{}={}".format(test_loader.dataset.samples(idx), sc))
            iou_scores.append(sc)
            anns = mask2ann(pred_mask.numpy(), image_id=idx, annotation=anns, cats_idx=test_dataset.cats_idx_for_target)
            if RES_PLOT:
                # plot prediction
                fig = plt.gcf()
                fig.set_size_inches(28, 18)
                sp = plt.subplot(1, 2, 1)
                sp.axis('Off')
                y_detach = y.detach().cpu()
                y_plot = torch.zeros([1, 1, y_detach.shape[-2], y_detach.shape[-1]])
                for i in range(N_CLASSES):
                    y_plot += i*y_detach[:,i,:,:]
                plt.imshow(y_plot.squeeze().numpy())
                plt.title("ground truth")
                sp = plt.subplot(1, 2, 2)
                sp.axis('Off')
                pred_detach = pred.detach().cpu()
                pred_plot = torch.zeros([BATCH_SIZE, 1, pred_detach.shape[-2], pred_detach.shape[-1]])
                for i in range(N_CLASSES):
                    pred_plot += i*pred_detach[:,i,:,:]
                plt.imshow(pred_plot.squeeze().numpy())
                plt.title("pred")
                sample_fname = test_loader.dataset.samples(idx)
                plt.savefig(f'{saving_path}/eval_{idx}_{sample_fname}.png')
                if DISP_PLOT:
                    print(sample_fname)
                    plt.show()
    print("----- TEST IoU : {}".format(sum(iou_scores)/len(iou_scores)))
    print("-"*20)
    # update & save _annotation.coco.json
    with open(TEST_ANN_FILE, 'r') as f:
        annf = json.loads(f.read())
    # annf_annotation = annf['annotations']
    # annf_annotation.extend(anns['annotation'])
    # annf['annotations'] = annf_annotation
    annf['annotations'] = anns['annotation']
    with open(TEST_ANN_FILE, 'w') as f:
        f.write(json.dumps(annf, indent=4))
    return saving_path, pred
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p', '--params',
        type=str,
        default="params.json",
        help="A path to parameters setting JSON file"
    )
    args = parser.parse_args()
    with open(args.params, 'r') as f:
        params_json = json.loads(f.read())
    print("-"*40)
    print("--- Start evaluation: trained model path is {}".format(params_json["EVALUATION"]["MODEL_PATH"]))
    saving_eval_path = eval(params_json)
    print("--- Stop evaluation: results are saved to {}".format(saving_eval_path))
    print("-"*40)