# filename : dataset.py
# updated : 10-03-2023
# version : v1.0

from PIL import Image
from torch.utils.data import Dataset
from imsegdl.dataset.imsegcoco import ImsegCOCO
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

class COCODataset(Dataset):
  def __init__(self, root_dir, ann_file, transforms=None, dbtype="train"):
    self.root_dir = root_dir
    self.coco = ImsegCOCO(ann_file)
    self.ids = list(sorted(self.coco.imgs.keys()))
    self.transforms = transforms
    self.n_classes = len(self.coco.categories) - 1
    self.version = self.coco.dataset['info']['version']
    if dbtype not in ["train", "test"]:
      raise ValueError("Invalid dbtype: {}".format(dbtype))
    self.dbtype = dbtype

  def __len__(self):
    return len(self.ids)

  def __getitem__(self, idx):
    img_id = self.ids[idx]
    ann_ids = self.coco.getAnnIds(imgIds=img_id)
    anns = self.coco.loadAnns(ann_ids)
    target = np.zeros((self.n_classes, self.coco.imgs[img_id]['height'], self.coco.imgs[img_id]['width']), dtype=np.float32)

    for i, ann in enumerate(anns):
      mask = self.coco.annToMask(ann).astype(np.float32)
      target[i] = mask

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
  
  def disp(self, img_id, draw_bbox=False):
    """
      This function isn't accepted for evaluation !
    """
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