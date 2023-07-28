import torch

def iou_score(pred_mask:torch.Tensor, true_mask:torch.Tensor)->float:
    # Convert the masks to binary tensors (0 or 1) for calculation
    pred_mask = (pred_mask > 0.5).int()
    true_mask = (true_mask > 0.5).int()
    # Compute the intersection and union of the masks
    intersection = torch.sum(pred_mask & true_mask).float()
    union = torch.sum(pred_mask | true_mask).float()
    iou = torch.tensor(0.0) if union==0 else intersection / union
    return iou.item()

def dice_coefficient(pred_mask, true_mask): # TODO draft
    # Convert the masks to binary tensors (0 or 1) for calculation
    pred_mask = (pred_mask > 0.5).int()
    true_mask = (true_mask > 0.5).int()

    # Compute the intersection and the sum of masks
    intersection = torch.sum(pred_mask & true_mask).float()
    sum_masks = torch.sum(pred_mask) + torch.sum(true_mask).float()

    # Handle special cases where the sum of masks is 0 (to avoid division by zero)
    if sum_masks == 0:
        dice = torch.tensor(0.0)
    else:
        dice = (2.0 * intersection) / sum_masks

    return dice.item()

def mean_iou(pred_masks, true_masks): # TODO draft
    # Convert the masks to binary tensors (0 or 1) for calculation
    pred_masks = (pred_masks > 0.5).int()
    true_masks = (true_masks > 0.5).int()

    num_classes = pred_masks.size(1)  # Assuming the channel dimension represents the number of classes

    # Initialize the mIoU for each class
    class_iou = torch.zeros(num_classes)

    for class_idx in range(num_classes):
        pred_class = pred_masks[:, class_idx, :, :]
        true_class = true_masks[:, class_idx, :, :]

        # Compute the intersection and union of the masks for the current class
        intersection = torch.sum(pred_class & true_class).float()
        union = torch.sum(pred_class | true_class).float()

        # Handle special cases where the union is 0 (to avoid division by zero)
        if union == 0:
            class_iou[class_idx] = torch.tensor(0.0)
        else:
            class_iou[class_idx] = intersection / union

    # Compute the average mIoU across all classes
    miou = torch.mean(class_iou)

    return miou.item()