import torch
import torch.nn as nn

def box_iou_giou(boxes1, boxes2):
    """
    Computes pairwise IoU and GIoU between two sets of boxes.
    boxes1: [N, 4] (cx, cy, w, h)
    boxes2: [M, 4] (cx, cy, w, h)
    Returns: iou [N, M], giou [N, M]
    """
    b1_x1 = boxes1[:, 0] - boxes1[:, 2] / 2
    b1_y1 = boxes1[:, 1] - boxes1[:, 3] / 2
    b1_x2 = boxes1[:, 0] + boxes1[:, 2] / 2
    b1_y2 = boxes1[:, 1] + boxes1[:, 3] / 2
    
    b2_x1 = boxes2[:, 0] - boxes2[:, 2] / 2
    b2_y1 = boxes2[:, 1] - boxes2[:, 3] / 2
    b2_x2 = boxes2[:, 0] + boxes2[:, 2] / 2
    b2_y2 = boxes2[:, 1] + boxes2[:, 3] / 2
    
    inter_x1 = torch.max(b1_x1[:, None], b2_x1[None, :])
    inter_y1 = torch.max(b1_y1[:, None], b2_y1[None, :])
    inter_x2 = torch.min(b1_x2[:, None], b2_x2[None, :])
    inter_y2 = torch.min(b1_y2[:, None], b2_y2[None, :])
    
    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    inter_area = inter_w * inter_h
    
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area[:, None] + b2_area[None, :] - inter_area
    
    iou = inter_area / torch.clamp(union_area, min=1e-6)
    
    hull_x1 = torch.min(b1_x1[:, None], b2_x1[None, :])
    hull_y1 = torch.min(b1_y1[:, None], b2_y1[None, :])
    hull_x2 = torch.max(b1_x2[:, None], b2_x2[None, :])
    hull_y2 = torch.max(b1_y2[:, None], b2_y2[None, :])
    
    hull_area = torch.clamp((hull_x2 - hull_x1) * (hull_y2 - hull_y1), min=1e-6)
    
    giou = iou - (hull_area - union_area) / hull_area
    return iou, giou

def bbox_iou_giou_elementwise(boxes1, boxes2):
    """
    Computes element-wise IoU and GIoU between two sets of boxes of the same shape.
    boxes1: [K, 4]
    boxes2: [K, 4]
    """
    b1_x1 = boxes1[:, 0] - boxes1[:, 2] / 2
    b1_y1 = boxes1[:, 1] - boxes1[:, 3] / 2
    b1_x2 = boxes1[:, 0] + boxes1[:, 2] / 2
    b1_y2 = boxes1[:, 1] + boxes1[:, 3] / 2
    
    b2_x1 = boxes2[:, 0] - boxes2[:, 2] / 2
    b2_y1 = boxes2[:, 1] - boxes2[:, 3] / 2
    b2_x2 = boxes2[:, 0] + boxes2[:, 2] / 2
    b2_y2 = boxes2[:, 1] + boxes2[:, 3] / 2
    
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)
    
    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    inter_area = inter_w * inter_h
    
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area - inter_area
    
    iou = inter_area / torch.clamp(union_area, min=1e-6)
    
    hull_x1 = torch.min(b1_x1, b2_x1)
    hull_y1 = torch.min(b1_y1, b2_y1)
    hull_x2 = torch.max(b1_x2, b2_x2)
    hull_y2 = torch.max(b1_y2, b2_y2)
    
    hull_area = torch.clamp((hull_x2 - hull_x1) * (hull_y2 - hull_y1), min=1e-6)
    
    giou = iou - (hull_area - union_area) / hull_area
    return iou, giou

@torch.no_grad()
def assign_targets(p_boxes, p_cls, gt_boxes, gt_classes, k=10):
    """
    Matches predictions to ground truth.
    p_boxes: [P, 4]
    p_cls: [P, nc]
    gt_boxes: [N, 4]
    gt_classes: [N]
    """
    num_anchors = p_boxes.shape[0]
    num_gt = gt_boxes.shape[0]
    
    if num_gt == 0:
        return torch.zeros(num_anchors, dtype=torch.bool, device=p_boxes.device), \
               torch.zeros(num_anchors, dtype=torch.long, device=p_boxes.device), \
               torch.zeros((num_anchors, 4), dtype=p_boxes.dtype, device=p_boxes.device)
        
    gt_classes_idx = gt_classes.long()
    p_cls_for_gt = p_cls[:, gt_classes_idx] # [P, N]
    cls_cost = -torch.log(p_cls_for_gt + 1e-7)
    
    _, giou = box_iou_giou(p_boxes, gt_boxes) # [P, N]
    box_cost = 1.0 - giou
    
    # Simple spatial prior: center of predicted box should be somewhat close to GT
    # To avoid matching anchors on the other side of the image
    dist_x = p_boxes[:, 0:1] - gt_boxes[None, :, 0]
    dist_y = p_boxes[:, 1:2] - gt_boxes[None, :, 1]
    dist_sq = (dist_x**2 + dist_y**2)
    # Normalize by GT size
    scale_sq = (gt_boxes[None, :, 2]**2 + gt_boxes[None, :, 3]**2).clamp(min=1e-6)
    spatial_cost = dist_sq / scale_sq
    
    # Add spatial cost if center is out of box, but don't make it impossible to match
    out_of_box = (torch.abs(dist_x) > gt_boxes[None, :, 2] / 2) | (torch.abs(dist_y) > gt_boxes[None, :, 3] / 2)
    spatial_cost[out_of_box] += 10.0
    
    cost = cls_cost + 3.0 * box_cost + spatial_cost
    
    # Select top K for each GT
    k_actual = min(k, num_anchors)
    _, topk_idx = torch.topk(cost, k=k_actual, dim=0, largest=False) # [K, N]
    
    target_cls = torch.full((num_anchors,), -1, dtype=torch.long, device=p_boxes.device)
    target_boxes = torch.zeros_like(p_boxes)
    mask = torch.zeros(num_anchors, dtype=torch.bool, device=p_boxes.device)
    
    topk_idx_flat = topk_idx.flatten()
    gt_idx_flat = torch.arange(num_gt, device=p_boxes.device).repeat(k_actual)
    cost_flat = cost[topk_idx_flat, gt_idx_flat]
    
    sort_idx = torch.argsort(cost_flat)
    
    for idx in sort_idx:
        anchor_idx = topk_idx_flat[idx]
        if not mask[anchor_idx]:
            mask[anchor_idx] = True
            gt_idx = gt_idx_flat[idx]
            target_cls[anchor_idx] = gt_classes[gt_idx]
            target_boxes[anchor_idx] = gt_boxes[gt_idx]
            
    return mask, target_cls, target_boxes

class YOLOLoss(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        self.bce = nn.BCELoss(reduction='none')
        
    def forward(self, preds, targets):
        """
        preds: [B, 4 + nc, P]
        targets: [N, 6] (batch_idx, cls_id, cx, cy, w, h)
        """
        device = preds.device
        B = preds.shape[0]
        
        loss_cls = torch.zeros(1, device=device)
        loss_box = torch.zeros(1, device=device)
        
        for i in range(B):
            # p_boxes format: [P, 4]
            p_boxes = preds[i, :4, :].transpose(1, 0)
            p_cls = preds[i, 4:, :].transpose(1, 0)
            
            gt_mask = targets[:, 0] == i
            gt_boxes = targets[gt_mask, 2:6]
            gt_classes = targets[gt_mask, 1]
            
            mask, target_cls, target_boxes = assign_targets(p_boxes, p_cls, gt_boxes, gt_classes)
            
            target_cls_onehot = torch.zeros_like(p_cls)
            if mask.any():
                target_cls_onehot[mask, target_cls[mask]] = 1.0
                
            p_cls_clamped = torch.clamp(p_cls, 1e-7, 1.0 - 1e-7)
            l_cls = self.bce(p_cls_clamped, target_cls_onehot)
            # Normalize by number of matched anchors to keep gradients stable
            num_matched = max(mask.sum().item(), 1.0)
            loss_cls += l_cls.sum() / num_matched
            
            if mask.any():
                _, giou = bbox_iou_giou_elementwise(p_boxes[mask], target_boxes[mask])
                l_box = 1.0 - giou
                loss_box += l_box.sum() / num_matched
                
        loss_cls = loss_cls / B
        loss_box = loss_box / B
        
        total_loss = loss_cls + 2.0 * loss_box
        return total_loss, loss_cls, loss_box
