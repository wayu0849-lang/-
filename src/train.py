import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

# Set multi-threading for PyTorch CPU optimization
cpu_threads = os.cpu_count() or 4
import torch
torch.set_num_threads(cpu_threads)

# Reconfigure stdout/stderr for Unicode safety on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.dataset import get_dataloaders, compute_class_weights
from src.model import build_model


def calculate_topk_accuracy(outputs: torch.Tensor, targets: torch.Tensor, topk=(1, 5)) -> List[float]:
    """
    Computes the accuracy over the k top predictions for the specified values of k.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)

        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


def train_one_epoch(model: nn.Module,
                    dataloader: torch.utils.data.DataLoader,
                    criterion: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device) -> Dict[str, float]:
    """
    Run one full training epoch.
    """
    model.train()
    running_loss = 0.0
    top1_correct = 0
    total_samples = 0

    pbar = tqdm(dataloader, desc='Training', leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Gradient clipping for training stability
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        _, preds = outputs.max(1)
        top1_correct += preds.eq(targets).sum().item()
        total_samples += batch_size

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{(top1_correct / total_samples) * 100:.2f}%'})

    epoch_loss = running_loss / total_samples
    epoch_acc = (top1_correct / total_samples) * 100.0
    return {'loss': epoch_loss, 'top1_acc': epoch_acc}


@torch.no_grad()
def validate(model: nn.Module,
             dataloader: torch.utils.data.DataLoader,
             criterion: nn.Module,
             device: torch.device) -> Dict[str, float]:
    """
    Run validation loop and calculate loss, Top-1, and Top-5 accuracy.
    """
    model.eval()
    running_loss = 0.0
    top1_total = 0.0
    top5_total = 0.0
    total_samples = 0

    pbar = tqdm(dataloader, desc='Validation', leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

        top1, top5 = calculate_topk_accuracy(outputs, targets, topk=(1, 5))
        top1_total += top1 * batch_size
        top5_total += top5 * batch_size

    val_loss = running_loss / total_samples
    val_top1 = top1_total / total_samples
    val_top5 = top5_total / total_samples

    return {'loss': val_loss, 'top1_acc': val_top1, 'top5_acc': val_top5}


def plot_and_save_curves(history: Dict[str, List[float]], save_path: Path):
    """
    Plot and save training and validation metrics curves.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss Curve
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r--s', label='Val Loss', linewidth=2)
    ax1.set_title('Cross-Entropy Loss vs. Epochs', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(fontsize=10)

    # Accuracy Curve
    ax2.plot(epochs, history['train_acc'], 'b-o', label='Train Top-1 Acc (%)', linewidth=2)
    ax2.plot(epochs, history['val_top1'], 'g--^', label='Val Top-1 Acc (%)', linewidth=2)
    ax2.plot(epochs, history['val_top5'], 'm-.d', label='Val Top-5 Acc (%)', linewidth=2)
    ax2.set_title('Accuracy vs. Epochs', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def generate_training_report(args: argparse.Namespace,
                             history: Dict[str, List[float]],
                             best_epoch: int,
                             best_val_acc: float,
                             total_time: float,
                             save_path: Path):
    """
    Generate markdown report summarizing training configuration and final results.
    """
    report_content = f"""# รายงานสรุปผลการฝึกสอนโมเดล (Model Training Report)

**วันที่และเวลา:** {time.strftime('%Y-%m-%d %H:%M:%S')}  
**โมเดล Architecture:** `{args.model_name}`  
**จำนวนคลาสทั้งหมด:** 104 Classes  
**ระยะเวลาการเทรนทั้งหมด:** {total_time:.2f} วินาที ({total_time / 60:.2f} นาที)  

---

## 1. การกำหนดค่าไฮเปอร์พารามิเตอร์ (Hyperparameters)

| Parameter | Value |
| :--- | :--- |
| **Model Backbone** | `{args.model_name}` (Pretrained on ImageNet) |
| **Batch Size** | `{args.batch_size}` |
| **Initial Learning Rate** | `{args.lr}` |
| **Optimizer** | `AdamW` (weight_decay={args.weight_decay}) |
| **LR Scheduler** | `CosineAnnealingLR` (T_max={args.epochs}) |
| **Label Smoothing** | `{args.label_smoothing}` |
| **Early Stopping Patience** | `{args.patience}` epochs |
| **Device Used** | `{args.device}` |

---

## 2. ผลลัพธ์ที่ดีที่สุด (Best Validation Results)

- **Best Epoch:** Epoch `{best_epoch}`
- **Best Validation Top-1 Accuracy:** **`{best_val_acc:.2f}%`**
- **Best Validation Top-5 Accuracy:** **`{history['val_top5'][best_epoch-1]:.2f}%`**
- **Validation Loss:** **`{history['val_loss'][best_epoch-1]:.4f}`**

---

## 3. ประวัติการเทรนในแต่ละ Epoch (Epoch History)

| Epoch | Train Loss | Train Acc (%) | Val Loss | Val Top-1 Acc (%) | Val Top-5 Acc (%) |
| :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for ep in range(len(history['train_loss'])):
        report_content += f"| {ep+1} | {history['train_loss'][ep]:.4f} | {history['train_acc'][ep]:.2f}% | {history['val_loss'][ep]:.4f} | {history['val_top1'][ep]:.2f}% | {history['val_top5'][ep]:.2f}% |\n"

    report_content += """
---

## 4. กราฟแสดงประสิทธิภาพ (Training Curves)

![Training and Validation Curves](training_curves.png)

---

## 5. ขั้นตอนถัดไป (Next Steps)
- ทำการประเมินผลบนชุดทดสอบ (Test Set) ด้วยคำสั่ง `python src/evaluate.py`
- ทดสอบการทำนายภาพเดี่ยว (Inference) ด้วยคำสั่ง `python src/predict.py --image <path_to_image>`
"""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report_content)


def main():
    parser = argparse.ArgumentParser(description='Train Dog & Cat Breed Classifier')
    parser.add_argument('--data_dir', type=str, default='data', help='Path to data directory')
    parser.add_argument('--model_name', type=str, default='mobilenetv3_large_100', help='timm model name')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--label_smoothing', type=float, default=0.08, help='Label smoothing value')
    parser.add_argument('--use_class_weights', action='store_true', help='Use class weights in loss')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume/fine-tune from')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader workers')
    parser.add_argument('--save_dir', type=str, default='models', help='Directory to save model weights')
    parser.add_argument('--reports_dir', type=str, default='reports/training', help='Directory for training reports')
    parser.add_argument('--device', type=str, default='auto', help='Device (cuda/cpu/auto)')

    args = parser.parse_args()

    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f'Using device: {device} (CPU Threads: {torch.get_num_threads()})')

    # Output directories
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. DataLoaders
    print('\n[1/5] Loading datasets and building class mappings...')
    mapping_path = save_dir / 'class_mapping.json'
    train_loader, val_loader, test_loader, class_to_idx = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mapping_save_path=mapping_path
    )
    num_classes = len(class_to_idx)
    print(f'[OK] Found {num_classes} classes.')
    print(f'[OK] Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}, Test samples: {len(test_loader.dataset)}')

    # 2. Build Model
    print(f'\n[2/5] Initializing model ({args.model_name})...')
    model = build_model(
        model_name=args.model_name,
        num_classes=num_classes,
        pretrained=True
    ).to(device)

    best_val_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_top1': [],
        'val_top5': []
    }

    # Resume from checkpoint if specified
    if args.resume and Path(args.resume).exists():
        print(f'[*] Resuming and fine-tuning from checkpoint: {args.resume}')
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        prev_history = checkpoint.get('history')
        if prev_history:
            history = prev_history
        print(f'[OK] Loaded weights successfully! Current Best Val Acc: {best_val_acc:.2f}%')

    # 3. Loss, Optimizer, Scheduler
    print('\n[3/5] Setting up Loss, Optimizer, and LR Scheduler...')
    if args.use_class_weights:
        weights = compute_class_weights(train_loader.dataset, num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
        print('[OK] Using Class-Weighted Cross-Entropy Loss.')
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        print('[OK] Using Standard Cross-Entropy Loss with Label Smoothing.')

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 4. Training Loop
    print(f'\n[4/5] Starting Training for {args.epochs} epochs...')
    best_epoch = len(history['train_loss'])
    patience_counter = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        ep_start = time.time()
        current_total_epoch = len(history['train_loss']) + 1
        print(f'\n--- Epoch {current_total_epoch} (Session Epoch {epoch}/{args.epochs} | LR: {scheduler.get_last_lr()[0]:.6f}) ---')

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        ep_duration = time.time() - ep_start

        # Record metrics
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['top1_acc'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_top1'].append(val_metrics['top1_acc'])
        history['val_top5'].append(val_metrics['top5_acc'])

        print(f'Train Loss: {train_metrics["loss"]:.4f} | Train Acc: {train_metrics["top1_acc"]:.2f}% | '
              f'Val Loss: {val_metrics["loss"]:.4f} | Val Top-1 Acc: {val_metrics["top1_acc"]:.2f}% | '
              f'Val Top-5 Acc: {val_metrics["top5_acc"]:.2f}% | Time: {ep_duration:.1f}s')

        # Checkpoint Best Model
        if val_metrics['top1_acc'] > best_val_acc:
            best_val_acc = val_metrics['top1_acc']
            best_epoch = current_total_epoch
            patience_counter = 0

            checkpoint = {
                'epoch': current_total_epoch,
                'model_name': args.model_name,
                'num_classes': num_classes,
                'state_dict': model.state_dict(),
                'best_val_acc': best_val_acc,
                'history': history,
                'class_to_idx': class_to_idx
            }
            torch.save(checkpoint, save_dir / 'best_model.pth')
            print(f'[*] Saved new best model checkpoint with Val Acc: {best_val_acc:.2f}% to {save_dir / "best_model.pth"}')
        else:
            patience_counter += 1
            print(f'Early Stopping counter: {patience_counter}/{args.patience}')
            if patience_counter >= args.patience:
                print(f'Early Stopping triggered.')
                break

    total_training_time = time.time() - start_time
    print(f'\n[5/5] Training completed in {total_training_time:.1f}s ({total_training_time/60:.2f} mins).')
    print(f'Best Validation Top-1 Accuracy: {best_val_acc:.2f}%')

    # Save Latest Checkpoint
    latest_checkpoint = {
        'epoch': len(history['train_loss']),
        'model_name': args.model_name,
        'num_classes': num_classes,
        'state_dict': model.state_dict(),
        'history': history,
        'class_to_idx': class_to_idx
    }
    torch.save(latest_checkpoint, save_dir / 'latest_checkpoint.pth')

    # Generate Learning Curves and Report
    curves_path = reports_dir / 'training_curves.png'
    plot_and_save_curves(history, curves_path)
    print(f'[OK] Saved training curves to: {curves_path}')

    report_path = reports_dir / 'training_report.md'
    generate_training_report(args, history, best_epoch, best_val_acc, total_training_time, report_path)
    print(f'[OK] Saved training report to: {report_path}')


if __name__ == '__main__':
    main()
