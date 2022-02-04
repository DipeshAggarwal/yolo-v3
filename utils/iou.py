import torch

def iou(boxes_pred, boxes_label, box_format="midpoint"):
    
    if box_format == "corners":
        box1_x1 = boxes_pred[..., 0:1]
        box1_y1 = boxes_pred[..., 1:2]
        box1_x2 = boxes_pred[..., 2:3]
        box1_y2 = boxes_pred[..., 3:4]
        box2_x1 = boxes_label[..., 0:1]
        box2_y1 = boxes_label[..., 1:2]
        box2_x2 = boxes_label[..., 2:3]
        box2_y2 = boxes_label[..., 3:4]
        
    elif box_format == "midpoint":
        box1_x1 = boxes_pred[..., 0:1] - boxes_pred[..., 2:3] / 2
        box1_y1 = boxes_pred[..., 1:2] - boxes_pred[..., 3:4] / 2
        box1_x2 = boxes_pred[..., 2:3] + boxes_pred[..., 0:1] / 2
        box1_y2 = boxes_pred[..., 3:4] + boxes_pred[..., 1:2] / 2
        box2_x1 = boxes_label[..., 0:1] - boxes_label[..., 2:3] / 2
        box2_y1 = boxes_label[..., 1:2] - boxes_label[..., 3:4] / 2
        box2_x2 = boxes_label[..., 2:3] + boxes_label[..., 0:1] / 2
        box2_y2 = boxes_label[..., 3:4] + boxes_label[..., 1:2] / 2
        
    
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.max(box1_x2, box2_x2)
    y2 = torch.max(box1_y2, box2_y2)

    # clamp(0) is for the edgecase where boxes do not overlap
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    
    box1_area  = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area  = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
    
    # Add a small epsilon value to prevent Divide by Zero error
    return intersection / (box1_area + box2_area - intersection + 1e-5)
