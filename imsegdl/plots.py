from torchvision.transforms.functional import to_tensor
from imsegdl.dataset import COCODataset
from imsegdl.utils import iou_score
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch

def idxmapping(imgs1:dict, imgs2:dict)->dict:
    imgs1 = {k:v['file_name'] for k,v in imgs1.items()}
    imgs2 = {k:v['file_name'] for k,v in imgs2.items()}
    klist, vlist = list(imgs2.keys()), list(imgs2.values())
    return {k:klist[vlist.index(v)] for k,v in imgs1.items()}

def plot_test_gt(ds:COCODataset, gt_ds:COCODataset, plots_test_save:str=None)->dict:
    _mapping = idxmapping(ds.coco.imgs, gt_ds.coco.imgs)
    for idx in range(len(ds)):
        fig = plt.gcf()
        fig.set_size_inches(38, 18)
        # ds
        sp1 = plt.subplot(1, 3, 1)
        sp1.axis('Off')
        img = ds.coco.loadImgs(ds.ids[idx])[0]
        ann_ids = ds.coco.getAnnIds(imgIds=img['id'])
        anns = ds.coco.loadAnns(ann_ids)
        image = Image.open(f'{ds.root_dir}/{img["file_name"]}').convert('RGB')
        pred = np.zeros((ds.n_classes, to_tensor(np.array(image)).shape[-2], to_tensor(np.array(image)).shape[-1]), dtype=np.float32)
        for ann in anns:
            mask = ds.coco.annToMask(ann).astype(np.float32)
            pred[ds.cats_idx_for_target[ann['category_id']]] += mask
            pred[0] += mask
        pred[pred > 1] = 1
        pred[0] = -1 * (pred[0] - 1)
        pred = torch.as_tensor(pred, dtype=torch.long)
        sp1.imshow(image)
        ds.coco.showAnns(anns, draw_bbox=False)
        plt.title(f"======= {ds.samples(idx)} =======")

        # gt_ds
        sp2 = plt.subplot(1, 3, 2)
        sp2.axis('Off')
        img = gt_ds.coco.loadImgs(gt_ds.ids[_mapping[idx]])[0]
        ann_ids = gt_ds.coco.getAnnIds(imgIds=img['id'])
        anns = gt_ds.coco.loadAnns(ann_ids)
        image = Image.open(f'{gt_ds.root_dir}/{img["file_name"]}').convert('RGB')
        tar = np.zeros((gt_ds.n_classes, to_tensor(np.array(image)).shape[-2], to_tensor(np.array(image)).shape[-1]), dtype=np.float32)
        for ann in anns:
            mask = gt_ds.coco.annToMask(ann).astype(np.float32)
            tar[gt_ds.cats_idx_for_target[ann['category_id']]] += mask
            tar[0] += mask
        tar[tar > 1] = 1
        tar[0] = -1 * (tar[0] - 1)
        tar = torch.as_tensor(tar, dtype=torch.long)
        sp2.imshow(image)
        gt_ds.coco.showAnns(anns, draw_bbox=False)
        plt.title(f"======= {gt_ds.samples(_mapping[idx])} =======")

        # iou scores for each class
        sp3 = plt.subplot(1, 3, 3)
        sp3.axis('Off')
        scores_plotting = {i:iou_score(pred[i],tar[i]) for i in range(ds.n_classes)}
        score = iou_score(pred, tar)
        sp3.plot(list(scores_plotting.keys()), list(scores_plotting.values()))
        plt.xlabel("Class")
        plt.ylabel("IoU_score")
        plt.title(f"======= IoU = {score} =======")
        if plots_test_save is not None:
            plt.savefig(f'{plots_test_save}/cp_{ds.samples(idx)}.png')
        plt.show()
    return _mapping

def plot_cc(ds):
    # Create a figure and axis
    fig, ax = plt.subplots()
    l = len(list(ds.coco.cs.values()))
    bbs = [[i,0,1,1] for i in range(l)]

    for bb, c in zip(bbs, list(ds.coco.cs.values())):
        # Create a rectangle patch
        x, y, h, w = bb
        rect = patches.Rectangle((x, y), h, w, linewidth=1, edgecolor=c, facecolor=c, alpha=0.4)

        # Add the patch to the axis
        ax.add_patch(rect)

    # Set limits for the x and y axis
    ax.set_xlim([0, l])
    ax.set_ylim([0, 1])

    # Display the plot
    plt.show()