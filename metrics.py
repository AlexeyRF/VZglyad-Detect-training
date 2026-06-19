import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys

# Ensure inference tools are accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'inference')))
try:
    from vzglyad_vision.utils import non_max_suppression
except ImportError:
    pass

def compute_ap(recall, precision):
    """Compute average precision."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def compute_metrics(preds, targets, iou_thresh=0.5, conf_thresh=0.001):
    """
    preds: list of tensors [N_preds, 6] (x1, y1, x2, y2, conf, cls) from NMS
    targets: list of tensors [N_targets, 5] (cls, cx, cy, w, h)
    Returns: precision, recall, mAP@0.5
    """
    stats = []
    
    for pred, tgt in zip(preds, targets):
        if len(pred) == 0:
            if len(tgt) > 0:
                stats.append((np.zeros(0), np.zeros(0), np.zeros(0), tgt[:, 0].cpu().numpy()))
            continue
            
        if len(tgt) == 0:
            stats.append((np.zeros(len(pred)), pred[:, 4].cpu().numpy(), pred[:, 5].cpu().numpy(), np.zeros(0)))
            continue
            
        pred_boxes = pred[:, :4]
        
        # Convert target cx,cy,w,h to x1,y1,x2,y2
        tgt_boxes = torch.zeros_like(tgt[:, 1:5])
        tgt_boxes[:, 0] = tgt[:, 1] - tgt[:, 3] / 2
        tgt_boxes[:, 1] = tgt[:, 2] - tgt[:, 4] / 2
        tgt_boxes[:, 2] = tgt[:, 1] + tgt[:, 3] / 2
        tgt_boxes[:, 3] = tgt[:, 2] + tgt[:, 4] / 2
        
        inter_x1 = torch.max(pred_boxes[:, 0:1], tgt_boxes[:, 0:1].T)
        inter_y1 = torch.max(pred_boxes[:, 1:2], tgt_boxes[:, 1:2].T)
        inter_x2 = torch.min(pred_boxes[:, 2:3], tgt_boxes[:, 2:3].T)
        inter_y2 = torch.min(pred_boxes[:, 3:4], tgt_boxes[:, 3:4].T)
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        tgt_area = (tgt_boxes[:, 2] - tgt_boxes[:, 0]) * (tgt_boxes[:, 3] - tgt_boxes[:, 1])
        
        union_area = pred_area[:, None] + tgt_area[None, :] - inter_area
        iou = inter_area / torch.clamp(union_area, min=1e-6)
        
        correct = np.zeros(len(pred))
        detected = []
        
        for i, p in enumerate(pred):
            if p[4] < conf_thresh:
                continue
            
            p_cls = int(p[5])
            matches = torch.where(tgt[:, 0] == p_cls)[0]
            if len(matches) > 0:
                ious = iou[i, matches]
                max_iou, max_idx = ious.max(0)
                if max_iou > iou_thresh and matches[max_idx].item() not in detected:
                    correct[i] = 1
                    detected.append(matches[max_idx].item())
                    
        stats.append((correct, pred[:, 4].cpu().numpy(), pred[:, 5].cpu().numpy(), tgt[:, 0].cpu().numpy()))
        
    if not stats:
        return 0.0, 0.0, 0.0
        
    all_correct = np.concatenate([s[0] for s in stats])
    all_conf = np.concatenate([s[1] for s in stats])
    all_tgt_cls = np.concatenate([s[3] for s in stats])
    
    if len(all_conf) == 0 or len(all_tgt_cls) == 0:
        return 0.0, 0.0, 0.0
        
    sort_i = np.argsort(-all_conf)
    all_correct = all_correct[sort_i]
    all_conf = all_conf[sort_i]
    
    tp = np.cumsum(all_correct)
    fp = np.cumsum(1 - all_correct)
    
    num_gt = len(all_tgt_cls)
    
    recalls = tp / num_gt
    precisions = tp / (tp + fp + 1e-16)
    
    ap = compute_ap(recalls, precisions)
    
    # Calculate PR at threshold 0.25 for logging display
    f1 = 2 * precisions * recalls / (precisions + recalls + 1e-16)
    idx = np.argmax(f1)
    
    return precisions[idx], recalls[idx], ap

def plot_dataset_stats(dataset, save_dir):
    """Generate basic dataset statistics and plot them."""
    labels_all = []
    for idx in range(min(1000, len(dataset))): # Limit to 1000 for speed if huge
        _, labels = dataset[idx]
        if len(labels) > 0:
            labels_all.append(labels.numpy())
            
    if not labels_all:
        print("No labels found for stats.")
        return
        
    labels_all = np.concatenate(labels_all, axis=0)
    classes = labels_all[:, 0]
    widths = labels_all[:, 3]
    heights = labels_all[:, 4]
    
    plt.figure(figsize=(15, 5))
    
    # Class Distribution
    plt.subplot(1, 3, 1)
    plt.hist(classes, bins=len(np.unique(classes)), align='left', rwidth=0.8, color='steelblue')
    plt.title('Class Distribution')
    plt.xlabel('Class ID')
    plt.ylabel('Count')
    
    # Width Distribution
    plt.subplot(1, 3, 2)
    plt.hist(widths, bins=50, color='coral')
    plt.title('Bounding Box Width')
    plt.xlabel('Width (pixels)')
    
    # Height Distribution
    plt.subplot(1, 3, 3)
    plt.hist(heights, bins=50, color='mediumseagreen')
    plt.title('Bounding Box Height')
    plt.xlabel('Height (pixels)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'dataset_stats.png'))
    plt.close()

def plot_training_results(history, save_dir):
    """Plot training and validation metrics history."""
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 10))
    
    # Losses
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Total Loss', color='darkblue')
    if 'val_loss' in history and len(history['val_loss']) > 0:
        plt.plot(epochs, history['val_loss'], label='Val Total Loss', color='darkorange')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Precision & Recall
    if 'precision' in history and 'recall' in history:
        plt.subplot(2, 2, 2)
        plt.plot(epochs, history['precision'], label='Precision', color='purple')
        plt.plot(epochs, history['recall'], label='Recall', color='green')
        plt.title('Validation Precision & Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Metric')
        plt.legend()
        
    # mAP
    if 'map50' in history:
        plt.subplot(2, 2, 3)
        plt.plot(epochs, history['map50'], label='mAP@0.5', color='red')
        plt.title('Validation mAP@0.5')
        plt.xlabel('Epoch')
        plt.ylabel('mAP')
        plt.legend()
        
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_results.png'))
    plt.close()
