import torch
import torch.nn as nn
from utils.iou import iou

class YOLOLoss(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()
        
        self.lambda_class = 1
        self.lambda_no_obj = 10
        self.lambda_obj = 1
        self.lambda_box = 10
        
    def forward(self, predictions, target, anchors):
        obj = target[..., 0] == 1
        no_obj = target[..., 0] == 0
        
        no_obj_loss = self.bce((predictions[..., 0:1][no_obj], (target[..., 0:1][no_obj])))
        
        # Object Loss
        anchors = anchors.reshape(1, 3, 1, 1, 2)
        box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5] * anchors)], dim=-1)
        ious = iou(box_preds[obj], target[..., 1:5][obj]).detach()
        obj_loss = self.bce((predictions[..., 0:1][obj]), (ious * target[..., 0:1]))
        
        # Box Co-ordinate Loss
        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])
        targets[..., 3:5] = torch.log((1e-16 + target[..., 3:5] / anchors))
        box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])

        # Class loss
        class_loss = self.entropy((predictions[..., 5:][obj]), target[..., 5][obj].long())
