import torch
from iou import iou

def nms(preds, iou_threshold, prob_threshold, box_format="midpoint"):
    """
    preds will be in the format [[class, probability, x1, y1, x2, y2]]
    """
    
    bboxes = [box for box in preds if box[1] > prob_threshold]
    # Sort the bounding box from higher probability to lowest probability
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []
    
    while bboxes:
        chosen_box = bboxes.pop(0)
        
        bboxes = [
            box for box in bboxes
            if box[0] != chosen_box[0]
            or iou(torch.tensor(chosen_box[2:]), torch.tensor(box[2:]), box_format=box_format) < iou_threshold
        ]
        
        bboxes_after_nms.append(chosen_box)
        
    return bboxes_after_nms
