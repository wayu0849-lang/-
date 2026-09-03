import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any

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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from PIL import Image

from src.dataset import DogsCatsDataset, get_transforms
from src.model import build_model


@torch.no_grad()
def evaluate_model(model: nn.Module,
                   dataloader: DataLoader,
                   device: torch.device,
                   class_names: List[str],
                   use_tta: bool = False,
                   test_dir: str = 'data/test',
                   class_to_idx: dict = None) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """
    Run evaluation on the test set with optional Test-Time Augmentation (TTA).
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    if use_tta:
        flip_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        flip_dataset = DogsCatsDataset(test_dir, class_to_idx=class_to_idx, transform=flip_transform)
        flip_loader = DataLoader(flip_dataset, batch_size=dataloader.batch_size, shuffle=False, num_workers=0)
        
        loader_iter = zip(dataloader, flip_loader)
        desc = 'Evaluating on Test Set (with TTA)'
    else:
        loader_iter = ((batch, None) for batch in dataloader)
        desc = 'Evaluating on Test Set'

    for (images, targets), flip_batch in tqdm(loader_iter, desc=desc, total=len(dataloader)):
        images = images.to(device)
        outputs1 = model(images)
        probs1 = torch.softmax(outputs1, dim=1)

        if use_tta and flip_batch is not None:
            flip_images, _ = flip_batch
            flip_images = flip_images.to(device)
            outputs2 = model(flip_images)
            probs2 = torch.softmax(outputs2, dim=1)
            probs = ((probs1 + probs2) / 2.0).cpu().numpy()
        else:
            probs = probs1.cpu().numpy()

        preds = np.argmax(probs, axis=1)

        all_probs.append(probs)
        all_preds.extend(preds)
        all_targets.extend(targets.numpy())

    all_probs = np.vstack(all_probs)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Top-1 and Top-5 accuracy calculation
    top1_correct = (all_preds == all_targets).sum()
    top1_acc = (top1_correct / len(all_targets)) * 100.0

    # Top-5
    top5_preds = np.argsort(all_probs, axis=1)[:, -5:]
    top5_correct = sum([target in top5_preds[i] for i, target in enumerate(all_targets)])
    top5_acc = (top5_correct / len(all_targets)) * 100.0

    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0) * 100.0
    weighted_f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0) * 100.0
    macro_precision = precision_score(all_targets, all_preds, average='macro', zero_division=0) * 100.0
    macro_recall = recall_score(all_targets, all_preds, average='macro', zero_division=0) * 100.0

    metrics = {
        'total_samples': len(all_targets),
        'top1_acc': top1_acc,
        'top5_acc': top5_acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall
    }

    return metrics, all_targets, all_preds, all_probs


def plot_confusion_matrix(targets: np.ndarray,
                          preds: np.ndarray,
                          class_names: List[str],
                          save_path: Path):
    """
    Generate and save Confusion Matrix heatmap.
    """
    cm = confusion_matrix(targets, preds)
    plt.figure(figsize=(24, 20))
    sns.heatmap(cm, cmap='Blues', xticklabels=False, yticklabels=False)
    plt.title(f'Confusion Matrix on Test Set ({len(class_names)} Classes)', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Predicted Class Index', fontsize=12)
    plt.ylabel('True Class Index', fontsize=12)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_error_analysis(dataset: DogsCatsDataset,
                        targets: np.ndarray,
                        preds: np.ndarray,
                        probs: np.ndarray,
                        idx_to_class: Dict[int, str],
                        save_path: Path,
                        max_samples: int = 12):
    """
    Plot misclassified image samples with their predicted probabilities.
    """
    error_indices = np.where(targets != preds)[0]
    if len(error_indices) == 0:
        return

    selected_indices = error_indices[:max_samples]
    num_plots = len(selected_indices)
    rows = (num_plots + 3) // 4
    fig, axes = plt.subplots(rows, 4, figsize=(18, 4 * rows))
    axes = axes.flatten()

    for i, idx in enumerate(selected_indices):
        img_path, true_label = dataset.samples[idx]
        pred_label = preds[idx]
        pred_prob = probs[idx][pred_label] * 100.0

        try:
            img = Image.open(img_path)
            axes[i].imshow(img)
        except Exception:
            axes[i].text(0.5, 0.5, 'Image Load Error', ha='center', va='center')

        true_name = idx_to_class[true_label]
        pred_name = idx_to_class[pred_label]

        axes[i].set_title(f'True: {true_name[:18]}\nPred: {pred_name[:18]} ({pred_prob:.1f}%)',
                          color='crimson', fontsize=9, fontweight='bold')
        axes[i].axis('off')

    for j in range(num_plots, len(axes)):
        axes[j].axis('off')

    plt.suptitle('Misclassified Examples from Test Set (Error Analysis)', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()


def generate_evaluation_report(metrics: Dict[str, Any],
                               clf_report_df: pd.DataFrame,
                               model_name: str,
                               save_path: Path):
    """
    Save markdown test evaluation report.
    """
    report_content = f"""# รายงานการประเมินผลบนชุดทดสอบ (Test Set Evaluation Report)

**ชุดข้อมูลทดสอบ:** `data/test` (จำนวน {metrics['total_samples']} ภาพ, 104 คลาส)  
**สถาปัตยกรรมโมเดล:** `{model_name}`  

---

## 1. ผลลัพธ์ประสิทธิภาพหลัก (Overall Performance Metrics)

| Metric | Score | คำอธิบาย |
| :--- | :---: | :--- |
| **Top-1 Accuracy** | **`{metrics['top1_acc']:.2f}%`** | สัดส่วนการทายสายพันธุ์ถูกต้องอันดับ 1 |
| **Top-5 Accuracy** | **`{metrics['top5_acc']:.2f}%`** | สัดส่วนที่คำตอบที่ถูกต้องติด 1 ใน 5 อันดับแรก |
| **Macro F1-Score** | **`{metrics['macro_f1']:.2f}%`** | ค่า F1 เฉลี่ยทุกคลาส (ทนต่อ Class Imbalance) |
| **Weighted F1-Score** | **`{metrics['weighted_f1']:.2f}%`** | ค่า F1 ถ่วงน้ำหนักตามจำนวนภาพจริง |
| **Macro Precision** | **`{metrics['macro_precision']:.2f}%`** | ความแม่นยำเฉลี่ยของการทำนาย |
| **Macro Recall** | **`{metrics['macro_recall']:.2f}%`** | ความครอบคลุมเฉลี่ยในการตรวจจับ |

---

## 2. Confusion Matrix & การวิเคราะห์ข้อผิดพลาด (Visualizations)

| ชาร์ตแสดงผล | ลิงก์ไฟล์ | สาระสำคัญ |
| :--- | :--- | :--- |
| **Confusion Matrix Heatmap** | `reports/evaluation/confusion_matrix.png` | แสดงการกระจายตัวของการจำแนกระหว่าง 104 คลาส |
| **Error Analysis Samples** | `reports/evaluation/error_analysis.png` | ตัวอย่างภาพที่โมเดลทำนายผิดพลาดพร้อมค่าความมั่นใจ |

---

## 3. สรุปผลการประเมินรายคลาส (Top & Bottom Performing Classes)

### คลาสที่มีประสิทธิภาพสูงสุด 5 อันดับแรก (Top-5 F1-Score):
"""
    top_5_classes = clf_report_df.sort_values(by='f1-score', ascending=False).head(5)
    report_content += top_5_classes[['precision', 'recall', 'f1-score', 'support']].to_markdown()

    report_content += """

### คลาสที่โมเดลยังจำแนกได้ยาก 5 อันดับ (Bottom-5 F1-Score):
"""
    bottom_5_classes = clf_report_df[clf_report_df['support'] > 0].sort_values(by='f1-score', ascending=True).head(5)
    report_content += bottom_5_classes[['precision', 'recall', 'f1-score', 'support']].to_markdown()

    report_content += """

---

## 4. ข้อสังเกตและข้อเสนอแนะในการปรับปรุง (Insights & Recommendations)
1. **Top-5 Accuracy vs Top-1 Accuracy:** เนื่องจากสายพันธุ์สุนัขและแมวบางชนิดมีความคล้ายคลึงกันทางกายภาพสูงมาก (Fine-grained classification) ค่า Top-5 Accuracy จะช่วยสะท้อนว่าโมเดลมองเห็นลักษณะเด่นของกลุ่มสายพันธุ์ใกล้เคียงได้เป็นอย่างดี
2. **การต่อยอดสู่การใช้งานจริง:** สามารถนำโมเดลไปใช้งานผ่านสคริปต์ `src/predict.py` เพื่อแสดงผล Top-5 Predictions พร้อม Confidence Score
"""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report_content)


def main():
    parser = argparse.ArgumentParser(description='Evaluate Breed Classifier on Test Set')
    parser.add_argument('--test_dir', type=str, default='data/test', help='Path to test set')
    parser.add_argument('--checkpoint', type=str, default='models/best_model.pth', help='Path to model checkpoint')
    parser.add_argument('--mapping_path', type=str, default='models/class_mapping.json', help='Path to class mapping json')
    parser.add_argument('--reports_dir', type=str, default='reports/evaluation', help='Path for evaluation reports')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--use_tta', action='store_true', help='Use Test-Time Augmentation')
    parser.add_argument('--device', type=str, default='auto', help='Device (cuda/cpu/auto)')

    args = parser.parse_args()

    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f'Using device: {device}')

    # Output directory
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Checkpoint & Class Mapping
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f'Error: Checkpoint file {checkpoint_path} not found. Please train model first.')
        sys.exit(1)

    print(f'Loading checkpoint from: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    class_to_idx = checkpoint.get('class_to_idx')
    if not class_to_idx and Path(args.mapping_path).exists():
        with open(args.mapping_path, 'r', encoding='utf-8') as f:
            class_to_idx = json.load(f)

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    num_classes = len(class_names)
    model_name = checkpoint.get('model_name', 'mobilenetv3_large_100')

    # 2. Build Model and load weights
    model = build_model(model_name=model_name, num_classes=num_classes, pretrained=False).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    print(f'[OK] Loaded model {model_name} with {num_classes} classes.')

    # 3. Load Test Dataset
    _, eval_transform = get_transforms()
    test_dataset = DogsCatsDataset(args.test_dir, class_to_idx=class_to_idx, transform=eval_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f'[OK] Test dataset loaded: {len(test_dataset)} samples.')

    # 4. Evaluate
    metrics, targets, preds, probs = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=device,
        class_names=class_names,
        use_tta=args.use_tta,
        test_dir=args.test_dir,
        class_to_idx=class_to_idx
    )

    print('\n' + '='*50)
    print(f'   TEST EVALUATION RESULTS {"(WITH TTA)" if args.use_tta else ""}')
    print('='*50)
    print(f'Total Test Samples : {metrics["total_samples"]}')
    print(f'Top-1 Accuracy     : {metrics["top1_acc"]:.2f}%')
    print(f'Top-5 Accuracy     : {metrics["top5_acc"]:.2f}%')
    print(f'Macro F1-Score     : {metrics["macro_f1"]:.2f}%')
    print(f'Weighted F1-Score  : {metrics["weighted_f1"]:.2f}%')
    print(f'Macro Precision    : {metrics["macro_precision"]:.2f}%')
    print(f'Macro Recall       : {metrics["macro_recall"]:.2f}%')
    print('='*50)

    # Classification Report
    clf_dict = classification_report(targets, preds, target_names=class_names, output_dict=True, zero_division=0)
    clf_df = pd.DataFrame(clf_dict).transpose()
    clf_df.to_csv(reports_dir / 'classification_report.csv')

    # Visualizations
    cm_path = reports_dir / 'confusion_matrix.png'
    plot_confusion_matrix(targets, preds, class_names, cm_path)
    print(f'[OK] Saved Confusion Matrix to: {cm_path}')

    error_path = reports_dir / 'error_analysis.png'
    plot_error_analysis(test_dataset, targets, preds, probs, idx_to_class, error_path)
    print(f'[OK] Saved Error Analysis to: {error_path}')

    report_path = reports_dir / 'test_evaluation_report.md'
    generate_evaluation_report(metrics, clf_df.iloc[:-3], model_name, report_path)
    print(f'[OK] Saved Test Evaluation Report to: {report_path}')


if __name__ == '__main__':
    main()
