import os
import sys
import argparse
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add the parent directory to sys.path to import vzglyad_vision
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'inference')))

from vzglyad_vision.model import CustomVisionModel
from dataset import YOLODataset, collate_fn
from loss import YOLOLoss
from metrics import compute_metrics, plot_dataset_stats, plot_training_results

# Fallback for NMS if it isn't available
try:
    from vzglyad_vision.utils import non_max_suppression
except ImportError:
    non_max_suppression = None

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)
    
    metadata = {}
    yaml_config = {}

    if args.cfg:
        print(f"Loading architecture from {args.cfg}")
        with open(args.cfg, 'r') as f:
            yaml_config = yaml.safe_load(f)
        if args.scale:
            yaml_config['scale'] = args.scale
            
        nc = yaml_config.get('nc', 80)
        model = CustomVisionModel(yaml_config).to(device)
    elif args.model_path:
        print(f"Loading base model from {args.model_path}")
        data = torch.load(args.model_path, map_location='cpu')
        if 'yaml' not in data:
            raise ValueError("Model does not contain 'yaml' config. Please provide a converted standalone .pth model.")
            
        yaml_config = data['yaml']
        metadata = data.get('metadata', {})
        nc = yaml_config.get('nc', 80)
        
        model = CustomVisionModel(yaml_config).to(device)
        
        if not args.scratch:
            print("Loading pretrained weights...")
            model.load_state_dict(data['state_dict'], strict=False)
            
        if 'stride' in metadata:
            model.model[-1].stride = torch.tensor(metadata['stride']).to(device)
    else:
        raise ValueError("You must provide either --cfg or --model-path.")

    print(f"Number of classes: {nc}")

    # Datasets and Dataloaders
    import random
    import glob
    print(f"Loading datasets from {args.data_dir}")
    
    has_splits = os.path.exists(os.path.join(args.data_dir, 'images', 'train')) and \
                 os.path.exists(os.path.join(args.data_dir, 'images', 'val'))
                 
    if has_splits:
        print("Found explicit train/val splits.")
        train_dataset = YOLODataset(args.data_dir, split='train', img_size=args.img_size, augment=not args.no_augment)
        val_dataset = YOLODataset(args.data_dir, split='val', img_size=args.img_size, augment=False)
    else:
        print("No explicit train/val splits found. Auto-splitting 80/20...")
        all_images = sorted(glob.glob(os.path.join(args.data_dir, 'images', '*.*')))
        if not all_images:
            raise ValueError(f"No images found in {os.path.join(args.data_dir, 'images')}")
            
        random.seed(42)
        random.shuffle(all_images)
        
        split_idx = int(len(all_images) * 0.8)
        train_paths = all_images[:split_idx]
        val_paths = all_images[split_idx:]
        
        train_dataset = YOLODataset(img_paths=train_paths, img_size=args.img_size, augment=not args.no_augment)
        val_dataset = YOLODataset(img_paths=val_paths, img_size=args.img_size, augment=False)
        print(f"Auto-split complete: {len(train_dataset)} train, {len(val_dataset)} val.")
    
    print("Generating dataset statistics...")
    plot_dataset_stats(train_dataset, args.save_dir)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn, 
        num_workers=args.workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.workers,
        pin_memory=True
    )

    # Loss and Optimizer
    criterion = YOLOLoss(num_classes=nc).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {
        'train_loss': [],
        'val_loss': [],
        'precision': [],
        'recall': [],
        'map50': []
    }

    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        
        epoch_loss = 0
        
        for batch_idx, (imgs, targets) in enumerate(pbar):
            imgs = imgs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            preds = model(imgs)
            loss, loss_cls, loss_box = criterion(preds, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'cls': f"{loss_cls.item():.4f}", 'box': f"{loss_box.item():.4f}"})
            
        scheduler.step()
        avg_train_loss = epoch_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Validation Loop
        model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
        with torch.no_grad():
            for imgs, targets in val_pbar:
                imgs = imgs.to(device)
                targets = targets.to(device)
                
                # Forward pass
                preds_raw = model(imgs)
                
                # Loss
                loss, _, _ = criterion(preds_raw, targets)
                val_loss += loss.item()
                
                # Inference metrics (if NMS is available)
                if non_max_suppression is not None:
                    # Model outputs [B, 4+nc, anchors]
                    # non_max_suppression expects [B, anchors, 4+nc]
                    preds_for_nms = preds_raw.transpose(1, 2)
                    preds_nms = non_max_suppression(preds_for_nms, conf_thres=0.01, iou_thres=0.5)
                    
                    # Split targets by batch_idx
                    for i in range(len(imgs)):
                        tgt_mask = targets[:, 0] == i
                        all_targets.append(targets[tgt_mask])
                        all_preds.append(preds_nms[i])
                        
        avg_val_loss = val_loss / max(len(val_loader), 1)
        history['val_loss'].append(avg_val_loss)
        
        precision, recall, map50 = 0.0, 0.0, 0.0
        if non_max_suppression is not None and len(all_preds) > 0:
            precision, recall, map50 = compute_metrics(all_preds, all_targets)
            
        history['precision'].append(precision)
        history['recall'].append(recall)
        history['map50'].append(map50)
        
        print(f"Epoch {epoch+1} Results: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | P: {precision:.4f} | R: {recall:.4f} | mAP@50: {map50:.4f}")
        
        # Plot metrics
        plot_training_results(history, args.save_dir)
        
        # Update metadata stride dynamically if needed
        dummy_input = torch.zeros(1, 3, args.img_size, args.img_size, device=device)
        with torch.no_grad():
            _ = model(dummy_input)
            current_stride = model.model[-1].stride.cpu().numpy().tolist() if hasattr(model.model[-1], 'stride') else [8, 16, 32]
        metadata['stride'] = current_stride
        
        # Save checkpoint
        checkpoint = {
            'yaml': yaml_config,
            'metadata': metadata,
            'state_dict': model.state_dict(),
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
            'history': history
        }
        
        torch.save(checkpoint, os.path.join(args.save_dir, f"epoch_{epoch+1}.pth"))
        torch.save(checkpoint, os.path.join(args.save_dir, "last.pth"))
        
        # Save best model
        if map50 > 0 and map50 >= max(history['map50']):
            torch.save(checkpoint, os.path.join(args.save_dir, "best.pth"))
            
    print("Training finished!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Standalone YOLO-compatible Training")
    parser.add_argument('--data-dir', type=str, required=True, help="Path to YOLO format dataset")
    parser.add_argument('--cfg', type=str, default=None, help="Path to YAML configuration")
    parser.add_argument('--scale', type=str, default=None, help="Model scale (n, s, m, l, x)")
    parser.add_argument('--model-path', type=str, default=None, help="Path to base .pth model")
    parser.add_argument('--scratch', action='store_true', help="Train from scratch")
    parser.add_argument('--no-augment', action='store_true', help="Disable augmentation")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--save-dir', type=str, default='runs/train', help="Save directory")
    
    args = parser.parse_args()
    train(args)
