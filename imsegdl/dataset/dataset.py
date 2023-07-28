# filename : dataset.py
# updated : 10-03-2023
# version : v1.0

from torchvision.transforms.functional import to_tensor
from imsegdl.utils import load_categories_json
from imsegdl.dataset.imsegcoco import ImsegCOCO
from torchvision.utils import save_image
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

class COCODataset(Dataset):
  def __init__(self, root_dir, ann_file, categories_path=None, transforms=None, dbtype="train", ptype:str="segmentation", cs:dict={}, pad:int=0):
    self.root_dir = root_dir
    self.coco = ImsegCOCO(annotation_file=ann_file, cs=cs)
    self.ids = list(sorted(self.coco.imgs.keys()))
    self.transforms = transforms
    self.categories = load_categories_json(categories_path) if categories_path else self.coco.categories
    self.n_classes = len(self.categories)
    self.version = self.coco.dataset['info']['version']
    self.cats_idx_for_target = {j['id']:i for i, j in enumerate(self.categories)}
    if dbtype not in ["train", "test"]:
      raise ValueError("Invalid dbtype: {}".format(dbtype))
    self.dbtype = dbtype
    if ptype not in ["segmentation", "object_detection"]:
      raise ValueError(f"Invalid ptype: ptype should be one of \'segmentation\', \'object_detection\' but found {ptype}")
    self.ptype = ptype
    self.pad = pad

  def __len__(self):
    return len(self.ids)

  def __getitem__(self, idx):
    img_id = self.ids[idx]
    ann_ids = self.coco.getAnnIds(imgIds=img_id)
    anns = self.coco.loadAnns(ann_ids)
    image_path = os.path.join(self.root_dir, self.coco.loadImgs(img_id)[0]['file_name'])
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    image = to_tensor(image)
    target = np.zeros((self.n_classes, image.shape[-2], image.shape[-1]), dtype=np.float32)

    if self.ptype == "object_detection":
      for ann in anns:
        if ann['category_id'] in self.cats_idx_for_target.keys():
          x, y, w, h = ann['bbox']
          bb_array = np.array([x,y,x+w,y+h]).astype(np.int32)
          mask = np.zeros((image.shape[-2], image.shape[-1]), dtype=np.float32)
          mask[bb_array[0]:bb_array[2], bb_array[1]:bb_array[3]] = 1.
          target[self.cats_idx_for_target[ann['category_id']]] += mask
    elif self.ptype == "segmentation":
      for ann in anns:
        if ann['category_id'] in self.cats_idx_for_target.keys():
          mask = self.coco.annToMask(ann).astype(np.float32)
          target[self.cats_idx_for_target[ann['category_id']]] += mask
          target[0] += mask
    else:
      pass

    target[target > 1] = 1
    target[0] = -1 * (target[0] - 1)
    target = torch.as_tensor(target, dtype=torch.long)
    if self.pad > 0:
      target = self.clean_overlapping_pixel(target.detach())
    if self.transforms:
      image = self.transforms(image)
      # target = self.transforms(target) # TODO
    if image.dim() < 3:
      image = torch.unsqueeze(image, 0)
    if self.dbtype == "test":
      save_image(image, image_path)

    return image, target
  
  def samples(self, idx):
    img_id = self.ids[idx]
    img = self.coco.loadImgs(img_id)[0]
    return img["file_name"]
  
  def draw_bbox(self, idx):
    print(f"======= {self.samples(idx)} =======")
    img_id = self.ids[idx]
    img = self.coco.loadImgs(img_id)[0]
    img_path = f'{self.root_dir}/{img["file_name"]}'
    image = Image.open(img_path).convert('RGB')
    ann_ids = self.coco.getAnnIds(imgIds=img['id'])
    anns = self.coco.loadAnns(ann_ids)
    fig, ax = plt.subplots()
    # Draw boxes and add label to each box
    for ann in anns:
        color = self.coco.cs[ann['category_id']]
        box = ann['bbox']
        bb = patches.Rectangle((box[0],box[1]), box[2],box[3], linewidth=2, edgecolor=color, facecolor="none")
        ax.add_patch(bb)
    ax.imshow(image)
    ax.axis('off')
    plt.show()
  
  def disp(self, idx, draw_bbox=False):
    """
      This function isn't accepted for evaluation !
    """
    print(f"======= {self.samples(idx)} =======")
    img_id = self.ids[idx]
    img = self.coco.loadImgs(img_id)[0]
    img_path = f'{self.root_dir}/{img["file_name"]}'
    image = Image.open(img_path).convert('RGB')
    ann_ids = self.coco.getAnnIds(imgIds=img['id'])
    anns = self.coco.loadAnns(ann_ids)
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(image)
    self.coco.showAnns(anns, draw_bbox=draw_bbox)
    plt.axis('off')
    plt.show()

  def clean_overlapping_pixel(self, tar:torch.Tensor)->torch.Tensor:
    tar_sum = torch.sum(tar, (0))
    px = (tar_sum > 1).nonzero().tolist()
    for opx in px:
        pxx, pxy = opx
        overlapping_pixel = tar[:,pxx, pxy]
        padding_layers = tar[:,pxx-self.pad:pxx+(self.pad+1),pxy-self.pad:pxy+(self.pad+1)]
        pad_sum = torch.sum(padding_layers, (1,2))
        pad_sum = overlapping_pixel * pad_sum
        max_value = torch.max(pad_sum)
        pad_sum[pad_sum < max_value] = 0
        pad_sum[pad_sum >= max_value] = 1
        tar[:,pxx,pxy] = pad_sum
    return tar.detach()
