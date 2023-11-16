# filename : train.py
# updated : 04-03-2023
# version : v1.0

from imsegdl.dataset import COCODataset
from imsegdl.utils import iou_score
from imsegdl.model import UNet, models
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
from math import sqrt
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
import torch.nn as nn
import argparse
import torch
import json
import os

def train(params:dict):
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
    TRAIN_DIR = params["DATASET"]["TRAIN_DIR"]
    TRAIN_ANN_FILE = params["DATASET"]["TRAIN_ANN_FILE"]
    VAL_DIR = params["DATASET"]["VAL_DIR"]
    VAL_ANN_FILE = params["DATASET"]["VAL_ANN_FILE"]
    CATEGORIES = params["TRAIN"]["CATEGORIES"]
    EPOCHS = params["TRAIN"]["EPOCHS"]
    EARLY_STOPPING_TOLERANCE = params["TRAIN"]["EARLY_STOPPING_TOLERANCE"]
    EARLY_STOPPING_THRESHOLD = params["TRAIN"]["EARLY_STOPPING_THRESHOLD"]
    RESULT_PATH = params["TRAIN"]["RESULT_PATH"]
    SAVE_ALL_MODELS = params["save_all_models"] if "save_all_models" in params.keys() else False
    TRANSFORM = params["transform"] if "transform" in params.keys() else None
    BATCH_SIZE = params["batch_size"] if "batch_size" in params.keys() else 1
    SHUFFLE = params["shuffle"] if "shuffle" in params.keys() else True
    NUM_WORKERS = params["num_workers"] if "num_workers" in params.keys() else (1 if torch.cuda.is_available() else 0)
    LEARNING_RATE = params["learning_rate"] if "learning_rate" in params.keys() else 0.005
    DISP_PLOT = params["disp_plot"] if "disp_plot" in params.keys() else False
    RES_PLOT = params['res_plot'] if "res_plot" in params.keys() else True
    PTYPE = params["ptype"] if "ptype" in params.keys() else "segmentation"
    PRETRAINED_MODEL = params["pretrained_model"] if "pretrained_model" in params.keys() else None
    checkpoint = torch.load(PRETRAINED_MODEL, map_location=torch.device(DEVICE)) if PRETRAINED_MODEL else {}
    INIT_EPOCHS = checkpoint['epoch'] + 1 if PRETRAINED_MODEL else 1
    CS = params['cs'] if 'cs' in params.keys() else {}
    OPTIM_TYPE = params['optimizer'] if 'optimizer' in params.keys() else 'adam'
    CLASS_WEIGHT = params['class_weight'] if 'class_weight' in params.keys() else []
    INIT_WEIGHTS = params['init_weights'] if 'init_weights' in params.keys() else False
    PAD = params['pad'] if 'pad' in params.keys() else 0
    N_CHANNELS = params['n_channels'] if 'n_channels' in params.keys() else 3
    MODEL_TYPE = params['model_type'] if 'model_type' in params.keys() else "unet"

    # load dataset
    train_dataset = COCODataset(TRAIN_DIR, TRAIN_ANN_FILE, categories_path=CATEGORIES, transforms=TRANSFORM, ptype=PTYPE, cs=CS, pad=PAD)
    val_dataset = COCODataset(VAL_DIR, VAL_ANN_FILE, categories_path=CATEGORIES, transforms=TRANSFORM, ptype=PTYPE, cs=CS, pad=PAD)
    N_CLASSES = checkpoint['n_classes'] if PRETRAINED_MODEL else train_dataset.n_classes
    VERSION = train_dataset.version
    print("-"*40)
    print("DATASET VERSION : {}".format(VERSION))
    print("N_TRAIN : {}, N_VAL : {}, N_CLASSES : {}".format(len(train_dataset), len(val_dataset), N_CLASSES))
    print("-"*40)

    # define data loader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS)
    N_TRAIN, N_VAL = len(train_loader), len(val_loader)
    
    # create an instance of the U-Net model
    if MODEL_TYPE == 'unet':
        model = UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES, init_weights=INIT_WEIGHTS).to(DEVICE).train()
    elif MODEL_TYPE == 'vit':
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
        model.heads.head = nn.Linear(in_features=model.heads.head.in_features, out_features=len(N_CLASSES), bias=True)
        k1, k2 = model.heads.head.weight.data.shape
        model.heads.head.weight.data.normal_(0., sqrt(2/(k1*k2)))
        model = model.to(DEVICE).train()
    else:
        raise KeyError(f"unsupport model : {MODEL_TYPE}")

    # define the loss function and optimizer
    weights = torch.tensor(CLASS_WEIGHT).cuda(DEVICE) if len(CLASS_WEIGHT) > 0 else None # adding class weight for imbanlance training dataset
    criterion = nn.CrossEntropyLoss(weight=weights)
    # criterion = nn.BCEWithLogitsLoss()
    if OPTIM_TYPE.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    elif OPTIM_TYPE.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.99)
    elif OPTIM_TYPE.lower() == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-8, momentum=0.9)
    else:
        raise ValueError(f"Incorrect optimization : {OPTIM_TYPE} found")
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if model.n_classes > 1 else 'max', patience=2)
    print("-"*40)
    print("Construct model : U-net")
    print("Initial weights : {}".format(INIT_WEIGHTS))
    print("Construct optimizer : {} - learning_rate : {}".format(optimizer.__class__.__name__, LEARNING_RATE))
    print("Define criterion : BCEWithLogitsLoss")

    # load pre-trained model
    if PRETRAINED_MODEL is not None:
        print("-"*40)
        print(f"Pre-trained model is detected : path : {PRETRAINED_MODEL}")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Loading pre-trained model success")

    # define result path
    saving_path = os.path.join(RESULT_PATH, "train_{}".format(datetime.now().strftime("%Y%m%d%H%M%S")))
    os.mkdir(saving_path)
    os.mkdir(os.path.join(saving_path, "train"))
    os.mkdir(os.path.join(saving_path, "val"))
    os.mkdir(os.path.join(saving_path, "test"))

    # train
    print("-"*40)
    LOSS_TRAIN_VALS = checkpoint["train_loss"] if PRETRAINED_MODEL else []
    LOSS_VALIDATION_VALS = checkpoint["val_loss"] if PRETRAINED_MODEL else []
    IOU_TRAIN_SCORES = checkpoint["train_iou_score"] if PRETRAINED_MODEL else []
    IOU_VALIDATION_SCORES = checkpoint["val_iou_score"] if PRETRAINED_MODEL else []
    early_stopping_counter = 0
    model_saving_path = None
    for e in range(INIT_EPOCHS, EPOCHS + 1):
        print("EPOCH : {}".format(e))
        model.train()
        train_loss_vals, train_iou_scores = [], []
        train_loader = tqdm(train_loader)
        if RES_PLOT:
            fig = plt.gcf()
            fig.set_size_inches((36//9 + 1) * 9, 12) # TODO : N_TRAIN
        for idx, batch in enumerate(train_loader):
            X, y = batch
            X, y = X.to(DEVICE), y.to(DEVICE)
            logits = model(X)
            loss = criterion(logits, y.float())
            train_loss_vals.append(loss.item())
            train_iou_scores.append(iou_score(logits.detach().cpu(), y.float().cpu()))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), 0.1)
            optimizer.step()
            if RES_PLOT:
                # plot train prediction
                pred = nn.functional.softmax(logits.detach(), dim=1)
                pred_argmax = torch.argmax(pred, dim=1, keepdims=True)
                pred_detach = torch.zeros_like(pred).scatter_(1, pred_argmax, 1)
                # pred_detach = pred_detach.sigmoid()
                pred_detach = pred_detach.cpu()
                pred_plot = torch.zeros([BATCH_SIZE, 1, pred_detach.shape[-2], pred_detach.shape[-1]])
                for i in range(N_CLASSES):
                    pred_plot += i*pred_detach[:,i,:,:]
                if idx < 36:
                    sp = plt.subplot(36//9, 9, idx+1) # TODO : N_TRAIN
                    sp.axis('Off')
                    plt.imshow(pred_plot.squeeze().numpy())
        if RES_PLOT:
            plt.savefig(f'{saving_path}/train/train_{e}.png')
            if DISP_PLOT:
                plt.show()
        LOSS_TRAIN_VALS.append(sum(train_loss_vals)/len(train_loss_vals))
        IOU_TRAIN_SCORES.append(sum(train_iou_scores)/len(train_iou_scores))
        print("----- TRAIN Loss : {}".format(LOSS_TRAIN_VALS[-1]))
        print("----- TRAIN IoU : {}".format(IOU_TRAIN_SCORES[-1]))
        print("-"*20)

        # validation
        model.eval()
        val_loss_vals, val_iou_scores = [], []
        val_loader = tqdm(val_loader)
        if RES_PLOT:
            fig = plt.gcf()
            fig.set_size_inches((18//9 + 1) * 9, 12) # TODO : N_VAL
        with torch.no_grad():
            for idx, batch in enumerate(val_loader):
                X, y = batch
                X, y = X.to(DEVICE), y.to(DEVICE)
                logits = model(X)
                val_loss = criterion(logits, y.float())
                val_loss_vals.append(val_loss.item())
                val_iou_scores.append(iou_score(logits.detach().cpu(), y.float().cpu()))
                if RES_PLOT:
                    # plot val prediction
                    pred = nn.functional.softmax(logits.detach(), dim=1)
                    pred_argmax = torch.argmax(pred, dim=1, keepdims=True)
                    pred_detach = torch.zeros_like(pred).scatter_(1, pred_argmax, 1)
                    # pred_detach = pred_detach.sigmoid()
                    pred_detach = pred_detach.cpu()
                    pred_plot = torch.zeros([BATCH_SIZE, 1, pred_detach.shape[-2], pred_detach.shape[-1]])
                    for i in range(N_CLASSES):
                        pred_plot += i*pred_detach[:,i,:,:]
                    if idx < 18:
                        sp = plt.subplot(18//9, 9, idx+1) # TODO : N_VAL
                        sp.axis('Off')
                        plt.imshow(pred_plot.squeeze().numpy())
            if RES_PLOT:
                plt.savefig(f'{saving_path}/val/val_{e}.png')
                if DISP_PLOT:
                    plt.show()
            cum_loss = sum(val_loss_vals)/len(val_loss_vals)
            before_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(cum_loss)
            after_lr = optimizer.param_groups[0]["lr"]
            LOSS_VALIDATION_VALS.append(cum_loss)
            IOU_VALIDATION_SCORES.append(sum(val_iou_scores)/len(val_iou_scores))
            print("----- VALIDATION Loss : {}".format(LOSS_VALIDATION_VALS[-1]))
            print("----- VALIDATION IoU : {}".format(IOU_VALIDATION_SCORES[-1]))
            print("-"*20)
            best_loss = min(LOSS_VALIDATION_VALS)
            print("----- Best VALIDATION Loss : {}".format(best_loss))

            # save best model
            if SAVE_ALL_MODELS:
                best_model_wts = model.state_dict()
                checkpoint_dict = {
                    'epoch' : e,
                    'model_state_dict' : best_model_wts,
                    'optimizer_state_dict' : optimizer.state_dict(),
                    'learning_rate' : LEARNING_RATE,
                    'n_classes' : N_CLASSES,
                    'version' : VERSION,
                    'train_loss' : LOSS_TRAIN_VALS,
                    'val_loss' : LOSS_VALIDATION_VALS, 
                    'train_iou_score' : IOU_TRAIN_SCORES,
                    'val_iou_score' : IOU_VALIDATION_SCORES
                    }
                # save best model
                filename = "model_e{}.pth".format(str(e))
                model_saving_path = os.path.join(saving_path, filename)
                torch.save(checkpoint_dict, model_saving_path)

            # save best model
            if cum_loss <= best_loss:
                best_model_wts = model.state_dict()
                checkpoint_dict = {
                    'epoch' : e,
                    'model_state_dict' : best_model_wts,
                    'optimizer_state_dict' : optimizer.state_dict(),
                    'learning_rate' : LEARNING_RATE,
                    'n_classes' : N_CLASSES,
                    'version' : VERSION,
                    'train_loss' : LOSS_TRAIN_VALS,
                    'val_loss' : LOSS_VALIDATION_VALS,
                    'train_iou_score' : IOU_TRAIN_SCORES,
                    'val_iou_score' : IOU_VALIDATION_SCORES
                    }
                # save best model
                model_saving_path = os.path.join(saving_path, "model.pth")
                torch.save(checkpoint_dict, model_saving_path)
            
            # early stopping
            if cum_loss > best_loss:
                early_stopping_counter += 1
            else:
                early_stopping_counter = 0
            print("----- Early Stopping : counter: {} - tolerance: {}".format(early_stopping_counter, EARLY_STOPPING_TOLERANCE))
            if before_lr != after_lr:
                print(f"----- Optimizer lr was changed - {before_lr} -> {after_lr}")
            
            if (early_stopping_counter == EARLY_STOPPING_TOLERANCE) or (best_loss <= EARLY_STOPPING_THRESHOLD):
                print("----- Terminating: early stopping: Best loss: {}, Theshold: {}".format(best_loss, EARLY_STOPPING_THRESHOLD))
                break # terminate training
            print("-"*20)

    # plot loss at the last training epoch
    plt.figure()
    plt.plot(list(range(1, len(LOSS_TRAIN_VALS)+1)), LOSS_TRAIN_VALS, label='train')
    plt.plot(list(range(1, len(LOSS_VALIDATION_VALS)+1)), LOSS_VALIDATION_VALS, label='validation')
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Average loss")
    plt.title("Loss")
    plt.savefig(f'{saving_path}/loss.png')
    if DISP_PLOT:
        plt.show()
    return model_saving_path if model_saving_path else PRETRAINED_MODEL

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
    print("--- Start taining: params path is {}".format(args.params))
    trained_model_path = train(params_json)
    message = "trained model is saved to {}".format(trained_model_path) if trained_model_path is not None else "there is no better model saved"
    print("--- Stop taining: {}".format(message))
    print("-"*40)