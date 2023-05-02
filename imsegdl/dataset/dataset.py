# filename : dataset.py
# updated : 10-03-2023
# version : v1.0

from PIL import Image
from torch.utils.data import Dataset
from imsegdl.dataset.imsegcoco import ImsegCOCO
from imsegdl.utils.utils import load_categories_json
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

class COCODataset(Dataset):
  def __init__(self, root_dir, ann_file, categories_path=None, transforms=None, dbtype="train", gen_segmentation=False, cs:dict={}):
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
    self.gen_segmentation = gen_segmentation

  def __len__(self):
    return len(self.ids)

  def __getitem__(self, idx):
    img_id = self.ids[idx]
    ann_ids = self.coco.getAnnIds(imgIds=img_id)
    anns = self.coco.loadAnns(ann_ids)
    target = np.zeros((self.n_classes, self.coco.imgs[img_id]['height'], self.coco.imgs[img_id]['width']), dtype=np.float32)

    nanns = []
    if self.gen_segmentation:
      for ann in anns:
        x1, y1, x2, y2 = ann['bbox']
        ann["segmentation"] = [[x1,y1,x1,(y1 + y2), (x1 + x2), (y1 + y2), (x1 + x2), y1]]
        nanns.append(ann)
      anns = nanns

    for ann in anns:
      if ann['category_id'] in self.cats_idx_for_target.keys():
        mask = self.coco.annToMask(ann).astype(np.float32)
        target[self.cats_idx_for_target[ann['category_id']]] += mask

    target[target > 1] = 1
    image_path = os.path.join(self.root_dir, self.coco.loadImgs(img_id)[0]['file_name'])
    image = Image.open(image_path).convert('RGB')

    target = torch.as_tensor(target, dtype=torch.long)
    if self.transforms:
      image = self.transforms(image)
      target = self.transforms(target)
    
    image = np.array(image)
    image = to_tensor(image)
    if self.dbtype == "test":
      save_image(image, image_path)

    return image, target
  
  def samples(self, idx):
    img_id = self.ids[idx]
    img = self.coco.loadImgs(img_id)[0]
    return img["file_name"]
  
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