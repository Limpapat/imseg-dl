from PIL import Image
from imsegdl.dataset.dataset import COCODataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def idxmapping(imgs1:dict, imgs2:dict)->dict:
    imgs1 = {k:v['file_name'] for k,v in imgs1.items()}
    imgs2 = {k:v['file_name'] for k,v in imgs2.items()}
    klist, vlist = list(imgs2.keys()), list(imgs2.values())
    return {k:klist[vlist.index(v)] for k,v in imgs1.items()}

def plot_test_gt(ds:COCODataset, gt_ds:COCODataset)->dict:
    _mapping = idxmapping(ds.coco.imgs, gt_ds.coco.imgs)
    fig = plt.gcf()
    fig.set_size_inches(28, 18)
    for idx in range(len(ds)):
        # ds
        sp1 = plt.subplot(1, 2, 1)
        sp1.axis('Off')
        img = ds.coco.loadImgs(ds.ids[idx])[0]
        ann_ids = ds.coco.getAnnIds(imgIds=img['id'])
        anns = ds.coco.loadAnns(ann_ids)
        fig1, ax1 = plt.subplots(figsize=(10,10))
        ax1.imshow(Image.open(f'{ds.root_dir}/{img["file_name"]}').convert('RGB'))
        ds.coco.showAnns(anns, draw_bbox=False)
        plt.title(f"======= {ds.samples(idx)} =======")

        # gt_ds
        sp2 = plt.subplot(1, 2, 2)
        sp2.axis('Off')
        img = gt_ds.coco.loadImgs(gt_ds.ids[_mapping[idx]])[0]
        ann_ids = gt_ds.coco.getAnnIds(imgIds=img['id'])
        anns = gt_ds.coco.loadAnns(ann_ids)
        fig2, ax2 = plt.subplots(figsize=(10,10))
        ax2.imshow(Image.open(f'{gt_ds.root_dir}/{img["file_name"]}').convert('RGB'))
        gt_ds.coco.showAnns(anns, draw_bbox=True)
        plt.title(f"======= {gt_ds.samples(_mapping[idx])} =======")
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