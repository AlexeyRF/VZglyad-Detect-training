import os
import cv2
import numpy as np
import torch
import random
import glob
from torch.utils.data import Dataset

def add_digital_noise(image, noise_type='gaussian'):
    if noise_type == 'gaussian':
        mean = 0
        std = 25
        noise = np.random.normal(mean, std, image.shape).astype(np.float32)
        noisy_image = cv2.add(image.astype(np.float32), noise)
        return np.clip(noisy_image, 0, 255).astype(np.uint8)
    elif noise_type == 'salt_pepper':
        s_vs_p = 0.5
        amount = 0.04
        noisy_image = np.copy(image)
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        noisy_image[tuple(coords)] = 255
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        noisy_image[tuple(coords)] = 0
        return noisy_image

def add_analog_noise(image):
    mean = 0
    std = 20
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy_image = cv2.add(image.astype(np.float32), noise)
    
    kernel_size = 5
    kernel_h = np.zeros((kernel_size, kernel_size))
    kernel_h[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
    kernel_h /= kernel_size
    noisy_image = cv2.filter2D(noisy_image, -1, kernel_h)
    
    num_lines = np.random.randint(1, 4)
    for _ in range(num_lines):
        y = np.random.randint(0, image.shape[0])
        cv2.line(noisy_image, (0, y), (image.shape[1], y), (255, 255, 255), 1)
        
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def apply_random_transform(image, labels):
    new_labels = []
    new_image = image.copy()
    
    transforms = ['hflip', 'vflip', 'rot90', 'rot180', 'rot270', 'analog', 'digital_gaussian', 'digital_salt_pepper', 'none']
    # Choose a random transform (or none)
    transform_type = random.choice(transforms)
    
    if transform_type == 'none':
        return image, labels
        
    if transform_type == 'hflip':
        new_image = cv2.flip(image, 1)
        for cls_id, x, y, w, h in labels:
            new_labels.append([cls_id, 1.0 - x, y, w, h])
            
    elif transform_type == 'vflip':
        new_image = cv2.flip(image, 0)
        for cls_id, x, y, w, h in labels:
            new_labels.append([cls_id, x, 1.0 - y, w, h])
            
    elif transform_type == 'rot90':
        new_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        for cls_id, x, y, w, h in labels:
            new_labels.append([cls_id, 1.0 - y, x, h, w])
            
    elif transform_type == 'rot180':
        new_image = cv2.rotate(image, cv2.ROTATE_180)
        for cls_id, x, y, w, h in labels:
            new_labels.append([cls_id, 1.0 - x, 1.0 - y, w, h])
            
    elif transform_type == 'rot270':
        new_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        for cls_id, x, y, w, h in labels:
            new_labels.append([cls_id, y, 1.0 - x, h, w])
            
    elif transform_type == 'analog':
        new_image = add_analog_noise(image)
        new_labels = list(labels)
        
    elif transform_type == 'digital_gaussian':
        new_image = add_digital_noise(image, 'gaussian')
        new_labels = list(labels)
        
    elif transform_type == 'digital_salt_pepper':
        new_image = add_digital_noise(image, 'salt_pepper')
        new_labels = list(labels)
        
    new_labels = np.array(new_labels, dtype=np.float32) if len(new_labels) > 0 else np.zeros((0, 5), dtype=np.float32)
    return new_image, new_labels

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)

class YOLODataset(Dataset):
    def __init__(self, data_dir=None, split='train', img_size=640, augment=True, img_paths=None):
        self.img_size = img_size
        self.augment = augment
        
        if img_paths is not None:
            self.img_paths = img_paths
        else:
            img_dir = os.path.join(data_dir, 'images', split)
            if not os.path.exists(img_dir):
                img_dir = os.path.join(data_dir, 'images')
            self.img_paths = sorted(glob.glob(os.path.join(img_dir, '*.*')))
            
        self.label_paths = [
            p.replace('images', 'labels').replace('images\\', 'labels\\').rsplit('.', 1)[0] + '.txt' for p in self.img_paths
        ]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        label_path = self.label_paths[index]

        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Image not found or corrupted: {img_path}")

        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(float(parts[0]))
                        cx = float(parts[1])
                        cy = float(parts[2])
                        bw = float(parts[3])
                        bh = float(parts[4])
                        labels.append([cls_id, cx, cy, bw, bh])
                        
        labels = np.array(labels, dtype=np.float32) if len(labels) else np.zeros((0, 5), dtype=np.float32)

        if self.augment:
            img, labels = apply_random_transform(img, labels)

        h0, w0 = img.shape[:2]
        
        # Convert relative labels to absolute before letterbox
        if len(labels) > 0:
            labels[:, 1] *= w0
            labels[:, 2] *= h0
            labels[:, 3] *= w0
            labels[:, 4] *= h0

        # Resize and pad
        img, ratio, pad = letterbox(img, (self.img_size, self.img_size), auto=False)
        
        if len(labels) > 0:
            labels[:, 1] = labels[:, 1] * ratio[0] + pad[0]
            labels[:, 2] = labels[:, 2] * ratio[1] + pad[1]
            labels[:, 3] = labels[:, 3] * ratio[0]
            labels[:, 4] = labels[:, 4] * ratio[1]

        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to CHW
        img = np.ascontiguousarray(img)
        img_tensor = torch.from_numpy(img).float() / 255.0
        
        return img_tensor, torch.from_numpy(labels)

def collate_fn(batch):
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs, 0)
    
    out_labels = []
    for i, l in enumerate(labels):
        if len(l) > 0:
            batch_idx = torch.full((len(l), 1), i, dtype=torch.float32)
            out_labels.append(torch.cat((batch_idx, l), dim=1))
            
    if len(out_labels) > 0:
        out_labels = torch.cat(out_labels, 0)
    else:
        out_labels = torch.zeros((0, 6), dtype=torch.float32)
        
    return imgs, out_labels
