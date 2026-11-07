train_dir = "/content/data/Beach Bird Detection Dataset v1.0/images/train"
train_anns = "/content/data/Beach Bird Detection Dataset v1.0/labels/train"
val_dir = "/content/data/Beach Bird Detection Dataset v1.0/images/val"
val_anns = "/content/data/Beach Bird Detection Dataset v1.0/labels/val"

import os
import time
import math
import torch
import torchvision
from torch.cuda.amp import autocast, GradScaler
from dataset import get_dataloaders
from model import get_faster_rcnn_model

def adjust_learning_rate(optimizer, epoch, iteration, num_iterations_per_epoch, total_epochs, peak_lr, warm_up_epochs=1):
    """
    Implements linear warm-up from 10% of peak to peak, followed by cosine decay to 1% of peak.
    Advanced by optimizer updates.
    """
    total_steps = total_epochs * num_iterations_per_epoch
    current_step = epoch * num_iterations_per_epoch + iteration
    warm_up_steps = warm_up_epochs * num_iterations_per_epoch

    min_lr = peak_lr * 0.01
    warm_up_start_lr = peak_lr * 0.1

    if current_step < warm_up_steps:
        # Linear warm-up
        lr = warm_up_start_lr + (peak_lr - warm_up_start_lr) * (current_step / warm_up_steps)
    else:
        # Cosine decay
        progress = (current_step - warm_up_steps) / (total_steps - warm_up_steps)
        lr = min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * progress))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def train_model():
    # Configuration
    train_dir = "./data/train"  # Update paths as needed
    val_dir = "./data/val"
    train_anns = []  # Load or pass your annotation list here
    val_anns = []    # Load or pass your annotation list here
    
    weights_path = "./fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth"
    output_dir = "./checkpoints_m3"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data loaders (batch_size=8 as per M3 spec)
    train_loader, val_loader = get_dataloaders(train_dir, train_anns, val_dir, val_anns, batch_size=8)

    # Initialize model
    model = get_faster_rcnn_model(weights_path=weights_path, num_classes=2)
    model.to(device)

    # Optimizer: SGD (momentum 0.9, weight decay 0.0001, Nesterov off)
    # Filter parameters requiring gradients
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.0025, momentum=0.9, nesterov=False, weight_decay=0.0001)

    # AMP Scaler
    scaler = GradScaler()

    num_epochs = 100
    accumulation_steps = 2
    peak_lr = 0.0025
    best_ap50 = 0.0.0
    best_ap50_95 = 0.0

    print("Starting M3 Training...")

    for epoch in range(num_epochs):
        model.train()
        epoch_start_time = time.time()
        running_loss = 0.0
        optimizer.zero_grad()

        num_batches = len(train_loader)

        for i, (images, targets) in enumerate(train_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass with AMP FP16
            with autocast():
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                # Normalize loss by accumulation steps
                losses = losses / accumulation_steps

            # Backward pass
            scaler.scale(losses).backward()

            running_loss += losses.item() * accumulation_steps

            # Gradient accumulation step
            if (i + 1) % accumulation_steps == 0 or (i + 1) == num_batches:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # Adjust learning rate per optimizer update
                current_lr = adjust_learning_rate(optimizer, epoch, i, num_batches, num_epochs, peak_lr, warm_up_epochs=1)

        epoch_duration = time.time() - epoch_start_time
        avg_train_loss = running_loss / num_batches
        peak_gpu_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

        print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {avg_train_loss:.4f} | Time: {epoch_duration:.2f}s | GPU Mem: {peak_gpu_mem:.2f}MB")

        # Validation step every epoch
        model.eval()
        # TODO: Run evaluation on val_loader using COCO evaluator to compute AP50, AP50:95, AP75, recall, etc.
        # placeholder evaluation results for checkpoint saving logic:
        val_ap50 = 0.0  # Replace with actual evaluator output
        val_ap50_95 = 0.0

        # Checkpoint saving rules:
        # - Save last after every epoch
        # - Archive every 10 epochs
        # - Save best using validation AP50 (tie-break AP50:95, then earlier epoch)
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
        }

        last_path = os.path.join(output_dir, "checkpoint_last.pth")
        torch.save(checkpoint, last_path)

        if (epoch + 1) % 10 == 0:
        archived_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save(checkpoint, archived_path)

        # Update best model based on AP50 and tie-breakers
        if val_ap50 > best_ap50 or (val_ap50 == best_ap50 and val_ap50_95 > best_ap50_95):
            best_ap50 = val_ap50
            best_ap50_95 = val_ap50_95
            best_path = os.path.join(output_dir, "checkpoint_best.pth")
            torch.save(checkpoint, best_path)
            print(f"--> Saved new best checkpoint at epoch {epoch+1} with AP50: {val_ap50:.4f}")

if __name__ == "__main__":
    train_model()