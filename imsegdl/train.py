# filename : train.py
# updated : 04-03-2023
# version : v1.0

from imsegdl.dataset.dataset import COCODataset
from imsegdl.model.model import UNet
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
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
    GEN_SEG = params["gen_segmentation"] if "gen_segmentation" in params.keys() else False
    PRETRAINED_MODEL = params["pretrained_model"] if "pretrained_model" in params.keys() else None
    checkpoint = torch.load(PRETRAINED_MODEL, map_location=torch.device(DEVICE)) if PRETRAINED_MODEL else {}
    INIT_EPOCHS = checkpoint['epoch'] + 1 if PRETRAINED_MODEL else 1

    # load dataset
    train_dataset = COCODataset(TRAIN_DIR, TRAIN_ANN_FILE, categories_path=CATEGORIES, transforms=TRANSFORM, gen_segmentation=GEN_SEG)
    val_dataset = COCODataset(VAL_DIR, VAL_ANN_FILE, categories_path=CATEGORIES, transforms=TRANSFORM, gen_segmentation=GEN_SEG)
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
    model = UNet(n_channels=3, n_classes=N_CLASSES).to(DEVICE).train()

    # define the loss function and optimizer
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print("-"*40)
    print("Construct model : U-net")
    print("Construct optimizer : Adam - learning_rate : {}".format(LEARNING_RATE))
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

    # train
    print("-"*40)
    LOSS_TRAIN_VALS = checkpoint["train_loss"] if PRETRAINED_MODEL else []
    LOSS_VALIDATION_VALS = checkpoint["val_loss"] if PRETRAINED_MODEL else []
    early_stopping_counter = 0
    model_saving_path = None
    for e in range(INIT_EPOCHS, EPOCHS + 1):
        print("EPOCH : {}".format(e))
        model.train()
        train_loss_vals = []
        train_loader = tqdm(train_loader)
        if RES_PLOT:
            fig = plt.gcf()
            fig.set_size_inches((N_TRAIN//9 + 1) * 9, 12)
        for idx, batch in enumerate(train_loader):
            X, y = batch
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            loss = criterion(pred, y.float())
            train_loss_vals.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if RES_PLOT:
                # plot train prediction
                pred_detach = pred.detach()
                pred_detach = pred_detach.sigmoid()
                pred_detach = pred_detach.cpu()
                pred_plot = torch.zeros([BATCH_SIZE, 1, pred_detach.shape[-2], pred_detach.shape[-1]])
                for i in range(N_CLASSES):
                    pred_plot += pred_detach[:,i,:,:]
                sp = plt.subplot(N_TRAIN//9, 9, idx+1)
                sp.axis('Off')
                plt.imshow(pred_plot.squeeze().numpy())
        if RES_PLOT:
            plt.savefig(f'{saving_path}/train/train_{e}.png')
            if DISP_PLOT:
                plt.show()
        LOSS_TRAIN_VALS.append(sum(train_loss_vals)/len(train_loss_vals))
        print("----- TRAIN Loss : {}".format(LOSS_TRAIN_VALS[-1]))
        print("-"*20)

        # validation
        model.eval()
        val_loss_vals = []
        val_loader = tqdm(val_loader)
        if RES_PLOT:
            fig = plt.gcf()
            fig.set_size_inches((N_VAL//9 + 1) * 9, 12)
        with torch.no_grad():
            for idx, batch in enumerate(val_loader):
                X, y = batch
                X, y = X.to(DEVICE), y.to(DEVICE)
                pred = model(X)
                val_loss = criterion(pred, y.float())
                val_loss_vals.append(val_loss.item())
                if RES_PLOT:
                    # plot val prediction
                    pred_detach = pred.detach()
                    pred_detach = pred_detach.sigmoid()
                    pred_detach = pred_detach.cpu()
                    pred_plot = torch.zeros([BATCH_SIZE, 1, pred_detach.shape[-2], pred_detach.shape[-1]])
                    for i in range(N_CLASSES):
                        pred_plot += pred_detach[:,i,:,:]
                    sp = plt.subplot(N_VAL//9, 9, idx+1)
                    sp.axis('Off')
                    plt.imshow(pred_plot.squeeze().numpy())
            if RES_PLOT:
                plt.savefig(f'{saving_path}/val/val_{e}.png')
                if DISP_PLOT:
                    plt.show()
            cum_loss = sum(val_loss_vals)/len(val_loss_vals)
            LOSS_VALIDATION_VALS.append(cum_loss)
            print("----- VALIDATION Loss : {}".format(LOSS_VALIDATION_VALS[-1]))
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
                    'val_loss' : LOSS_VALIDATION_VALS
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
                    'val_loss' : LOSS_VALIDATION_VALS
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
    return model_saving_path

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