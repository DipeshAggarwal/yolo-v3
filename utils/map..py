import torch
from collections import Counter
from iou import iou

def m_ap(pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20):
    """
    pred_boxes is [[train_idx, class, prob_score, x1, y1, x2, y2], ...]
    """
    average_precision = []
    epsilon = 1e-5
    
    for c in range(num_classes):
        detections = []
        gts = []
        
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)
                
        for true_box in true_boxes:
            if true_box[1] == c:
                gts.append(true_box)
                
        # Find the number of boxes for each training example
        amount_boxes = Counter([gt[0] for gt in gts])
        
        for key, val in amount_boxes:
            amount_boxes[key] = torch.zeros(val)
            
        detections.sort(key=lambda x:x[2], reverse=True)
        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))
        total_true_boxes = len(gts)
        
        # If no true boxes for this class exist, move to the next class
        if total_true_boxes == 0:
            continue
        
        for detection_idx, detection in enumerate(detections):
            gt_img = [bbox for bbox in gts if bbox[0] == detection[0]]
            best_iou = 0
            
            for c_iou, gt in enumerate(gt_img):
                c_iou = iou(torch.tensor(detection[3:]), torch.tensor(gt[3:]), box_format=box_format)
                
                if c_iou > best_iou:
                    best_iou = c_iou
                    best_gt_idx = detection_idx
            
            if best_iou > iou_threshold:
                if amount_boxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_boxes[detection[0]][best_gt_idx]
                else:
                    FP[detection_idx] = 1
            
            else:
                FP[detection_idx] = 1
                
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        
        recall = TP_cumsum / (total_true_boxes + epsilon)
        recall = torch.cat((torch.tensor([0]), recall))
        precision = FP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precision = torch.cat((torch.tensor([1]), precision))
        
        # To find the total area under graph
        average_precision.append(torch.trapz(precision, recall))
    
    return sum(average_precision) / len(average_precision)
