# filename : utils.py
# updated : 10-03-2023
# version : v1.0

from pycocotools import mask
from skimage import measure
import numpy as np
import glob
import json
import os

def gen_images_form(image_id, file_name, date_captured)->dict:
  return {
      'id': image_id,
      'license': 1,
      'file_name': file_name,
      'height': 512,          # TODO update later
      'width': 512,           # TODO update later
      'date_captured': date_captured
  }

def gen_annotations_form(ann_id, image_id, ann)->dict:
  return {
      'id': ann_id,
      'image_id': image_id,
      'category_id': ann["category_id"],
      'bbox': ann["bbox"],
      'area': ann["area"],
      'segmentation': ann["segmentation"],
      'iscrowd': 0
  }

def mask2ann(ground_truth_mask:np.array, image_id, annotation:dict={"last_ann_id":-1, "annotation":[]})->dict:
  """
  ground_truth_mask.shape = (n_classes, h, w)
  """
  ann_id = annotation["last_ann_id"]
  for i in range(ground_truth_mask.shape[0]):
    ann_id += 1
    ground_truth_binary_mask = ground_truth_mask[i,:,:]
    ground_truth_binary_mask = ground_truth_binary_mask.astype(np.uint8)
    fortran_ground_truth_binary_mask = np.asfortranarray(ground_truth_binary_mask)
    encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
    ground_truth_area = mask.area(encoded_ground_truth)
    ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
    contours = measure.find_contours(ground_truth_binary_mask, 0.5)
    segmentation = []
    for contour in contours:
        contour = np.flip(contour, axis=1)
        segmentation.append(contour.ravel().tolist())
    ann = {
          "segmentation": segmentation,
          "area": ground_truth_area.tolist(),
          "bbox": ground_truth_bounding_box.tolist(),
          "category_id": i + 1} # adding 1 because category_id:0 is background
    annotation["annotation"].append(gen_annotations_form(ann_id,image_id,ann))
  annotation["last_ann_id"] = ann_id
  return annotation

def gen_empty_annf(root_dir:str, ann_dir:str, version:str, stamp:str, cats:list):
    # TODO : update "categories" for other dataset!
    annf = {
       "info" : {
            "year" : stamp.split("-")[0],
            "version" : version,
            "description" : "Generated from imseg-backend",
            "contributor" : "",
            "url" : "",
            "date_created" : stamp
        },
        "licenses" : {
            "id" : 1,
            "url" : "https://creativecommons.org/licenses/by/4.0/",
            "name": "CC BY 4.0"
        },
        "categories" : cats,
        "images" : [],
        "annotations" : [],
    }
    for i, n in enumerate(glob.glob(os.path.join(root_dir,"*.png"))):
        annf["images"].append(gen_images_form(i, n.split("/")[-1], stamp))
    with open(ann_dir, 'w') as f:
        f.write(json.dumps(annf, indent=4))