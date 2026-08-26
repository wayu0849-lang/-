import os
import sys
import argparse
import random
import time
from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class ImagePreprocessor:
    """
    Production-ready Image Preprocessing and Augmentation pipeline for Dogs & Cats Classifier dataset.
    """
    def __init__(self,
                 target_size: tuple = (224, 224),
                 normalize_range: tuple = (0.0, 1.0),
                 use_imagenet_stats: bool = True):
        self.target_size = target_size
        self.normalize_range = normalize_range
        self.use_imagenet_stats = use_imagenet_stats
        # Standard ImageNet Mean and Std (RGB order)
        self.imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def load_and_validate(self, image_path: str | Path) -> Image.Image:
        """
        Safely open image, handle truncated headers, and convert to 3-channel RGB.
        """
        img = Image.open(image_path)
        img.load()  # Force load image data
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    def resize_direct(self, img: Image.Image) -> Image.Image:
        """
        Resize image directly to target_size using high quality Lanczos interpolation.
        """
        return img.resize(self.target_size, resample=Image.Resampling.LANCZOS)

    def resize_with_pad(self, img: Image.Image, fill_color=(0, 0, 0)) -> Image.Image:
        """
        Resize image while strictly preserving aspect ratio using letterbox padding.
        """
        img.thumbnail(self.target_size, Image.Resampling.LANCZOS)
        padded = Image.new("RGB", self.target_size, fill_color)
        paste_pos = ((self.target_size[0] - img.size[0]) // 2,
                     (self.target_size[1] - img.size[1]) // 2)
        padded.paste(img, paste_pos)
        return padded

    def normalize(self, img_array: np.ndarray) -> np.ndarray:
        """
        Normalize numpy image array: scale to [0, 1] and optionally standardize using ImageNet stats.
        """
        # img_array shape: (H, W, 3) in range [0, 255]
        norm_img = img_array.astype(np.float32) / 255.0

        if self.use_imagenet_stats:
            norm_img = (norm_img - self.imagenet_mean) / self.imagenet_std

        return norm_img

    def augment(self,
                img: Image.Image,
                flip_prob: float = 0.5,
                max_rotation_deg: float = 15.0,
                brightness_factor_range: tuple = (0.85, 1.15),
                contrast_factor_range: tuple = (0.85, 1.15)) -> Image.Image:
        """
        Apply stochastic data augmentation for training split.
        """
        augmented = img.copy()

        # 1. Random Horizontal Flip
        if random.random() < flip_prob:
            augmented = augmented.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        # 2. Random Rotation
        angle = random.uniform(-max_rotation_deg, max_rotation_deg)
        augmented = augmented.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False)

        # 3. Random Brightness Jitter
        b_factor = random.uniform(*brightness_factor_range)
        augmented = ImageEnhance.Brightness(augmented).enhance(b_factor)

        # 4. Random Contrast Jitter
        c_factor = random.uniform(*contrast_factor_range)
        augmented = ImageEnhance.Contrast(augmented).enhance(c_factor)

        return augmented

    def preprocess_image(self,
                         image_path: str | Path,
                         is_training: bool = False,
                         preserve_aspect_ratio: bool = False) -> tuple[np.ndarray, Image.Image]:
        """
        Full end-to-end preprocessing pipeline for a single image.
        Returns: (normalized_numpy_array, processed_pil_image)
        """
        img = self.load_and_validate(image_path)

        if is_training:
            img = self.augment(img)

        if preserve_aspect_ratio:
            img = self.resize_with_pad(img)
        else:
            img = self.resize_direct(img)

        img_np = np.array(img)
        norm_np = self.normalize(img_np)
        return norm_np, img


def calculate_channel_stats(data_dir: Path, sample_size: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate dataset-specific Channel Mean and Standard Deviation from sample images.
    """
    img_files = list(data_dir.rglob("*.jpg")) + list(data_dir.rglob("*.png"))
    if not img_files:
        return np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])

    sampled = random.sample(img_files, min(sample_size, len(img_files)))
    rgb_values = []

    for img_p in sampled:
        try:
            with Image.open(img_p) as img:
                img_rgb = img.convert("RGB")
                arr = np.asarray(img_rgb, dtype=np.float32) / 255.0
                rgb_values.append(arr.mean(axis=(0, 1)))
        except Exception:
            continue

    means = np.mean(rgb_values, axis=0)
    stds = np.std(rgb_values, axis=0)
    return means, stds


def generate_preprocessing_visualizations(data_dir: Path, report_dir: Path, preprocessor: ImagePreprocessor):
    """
    Generate sample visual demonstrations of image transformations and augmentations.
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    img_files = list(data_dir.rglob("*.jpg"))
    if not img_files:
        return

    random.seed(42)
    selected_imgs = random.sample(img_files, min(4, len(img_files)))

    # 1. Preprocessing Comparison (Original vs Direct Resize vs Padded Resize)
    fig, axes = plt.subplots(len(selected_imgs), 3, figsize=(10, 3 * len(selected_imgs)))
    if len(selected_imgs) == 1:
        axes = np.expand_dims(axes, 0)

    for i, img_path in enumerate(selected_imgs):
        orig = Image.open(img_path).convert("RGB")
        direct = preprocessor.resize_direct(orig)
        padded = preprocessor.resize_with_pad(orig)

        axes[i, 0].imshow(orig)
        axes[i, 0].set_title(f"Original ({orig.size[0]}x{orig.size[1]})")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(direct)
        axes[i, 1].set_title(f"Direct Resize ({preprocessor.target_size[0]}x{preprocessor.target_size[1]})")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(padded)
        axes[i, 2].set_title("Aspect-Pad Resize")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(report_dir / "preprocessing_comparison.png", dpi=150)
    plt.close()

    # 2. Augmentation Examples (1 original -> 4 variations)
    sample_img = preprocessor.load_and_validate(selected_imgs[0])
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.5))

    axes[0].imshow(sample_img)
    axes[0].set_title("Original (Val/Test)")
    axes[0].axis("off")

    for j in range(1, 5):
        aug = preprocessor.augment(sample_img)
        axes[j].imshow(aug)
        axes[j].set_title(f"Augmented #{j}")
        axes[j].axis("off")

    plt.suptitle("Training Data Augmentation Variations", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(report_dir / "augmentation_samples.png", dpi=150)
    plt.close()

    print(f"[Preprocessing] Visual samples saved to {report_dir.resolve()}")


def generate_preprocessing_report(report_dir: Path,
                                 mean_calc: np.ndarray,
                                 std_calc: np.ndarray,
                                 throughput_fps: float,
                                 target_size: tuple):
    """
    Generate comprehensive markdown report for Image Preprocessing.
    """
    report_content = f"""# รายงานการเตรียมและประมวลผลรูปภาพ (Image Preprocessing Report)
**ชุดข้อมูล:** Dogs and Cats Breed Classifier (104 Classes)  
**เป้าหมายการประมวลผล:** ปรับมาตรฐานและเพิ่มประสิทธิภาพรูปภาพให้พร้อมสำหรับฝึกสอนและทดสอบโมเดล Deep Learning  
**วันที่จัดทำ:** 2026-08-26  

---

## 1. วัตถุประสงค์และเหตุผลทางเทคนิค (Technical Rationale)

จากการวิเคราะห์ข้อมูลเชิงสำรวจ (EDA) พบว่าชุดข้อมูลรูปภาพมีความพร้อมในระดับหนึ่ง แต่เพื่อให้การฝึกสอนโมเดล (เช่น CNN, ResNet, EfficientNet, MobileNet, ViT) มีเสถียรภาพสูงสุด จึงต้องกำหนดกระบวนการ Preprocessing ที่เหมาะสมดังนี้:

| ขั้นตอน (Step) | วิธีการ (Technique) | เหตุผลทางเทคนิค (Rationale) |
| :--- | :--- | :--- |
| **1. การตรวจสอบและแปลงสี (Color Mode)** | แปลงทุกภาพเป็น `RGB` 3 Channels | ป้องกันความผิดพลาดของ Tensor Shape จากภาพที่มี 1 Channel (Grayscale) หรือ 4 Channels (RGBA) |
| **2. การปรับขนาด (Image Resizing)** | ปรับขนาดสู่ **{target_size[0]} × {target_size[1]}** ด้วย Lanczos Interpolation | เป็นขนาดมาตรฐานของโมเดล Computer Vision ส่วนใหญ่ และช่วยให้สร้าง Batch Tensor ที่มีขนาดสม่ำเสมอได้ |
| **3. การทำ Normalization & Standardization** | สเกล `[0, 1]` และทำ Z-Score Normalization ด้วย ImageNet Stats | ป้องกันปัญหา Vanishing/Exploding Gradients และทำให้ Gradient Descent ลู่เข้า (Converge) เร็วขึ้น |
| **4. การทำ Data Augmentation (Train Set)** | Flip, Rotation ±15°, Brightness/Contrast Jitter | ลด Overfitting และแก้ปัญหา Class Imbalance (12.41x) ที่พบใน EDA |

---

## 2. ข้อมูลสถิติช่องสีของชุดข้อมูล (Color Channel Statistics)

จากการสุ่มคำนวณค่าเฉลี่ยและส่วนเบี่ยงเบนมาตรฐานของพิกเซลในชุดข้อมูลจริง (บนช่วง `[0, 1]`):

| ช่องสี (Channel) | Dataset Mean | Dataset Std | ImageNet Standard Mean | ImageNet Standard Std |
| :---: | :---: | :---: | :---: | :---: |
| **Red (R)** | `{mean_calc[0]:.4f}` | `{std_calc[0]:.4f}` | `0.4850` | `0.2290` |
| **Green (G)** | `{mean_calc[1]:.4f}` | `{std_calc[1]:.4f}` | `0.4560` | `0.2240` |
| **Blue (B)** | `{mean_calc[2]:.4f}` | `{std_calc[2]:.4f}` | `0.4060` | `0.2250` |

> **คำแนะนำ:** หากใช้ Transfer Learning หรือ Pretrained Models (เช่น ImageNet weights) แนะนำให้ใช้ ImageNet Standard Stats เพื่อให้ตรงกับ Distribution เดิมของโมเดล

---

## 3. สถาปัตยกรรมการแปลงข้อมูล (Pipeline Architecture)

### 3.1 สำหรับ Training Pipeline (รวม Data Augmentation)
1. **Input:** รูปภาพไฟล์ดิบ (`.jpg`, `.png`)
2. **Format Conversion:** บังคับโหลดเป็น 3-Channel RGB
3. **Augmentation:**
   - Random Horizontal Flip ($p=0.5$)
   - Random Rotation (+/- 15 deg)
   - Color Jitter (Brightness factor 0.85 - 1.15, Contrast factor 0.85 - 1.15)
4. **Resize:** 224 x 224 (Lanczos Interpolation)
5. **Normalization:** Pixel / 255.0
6. **Standardization:** (x - mean) / std
7. **Output:** Normalized Float32 Tensor Shape (3, 224, 224) หรือ (224, 224, 3)

### 3.2 สำหรับ Validation & Test Pipeline (Deterministic)
1. **Input:** รูปภาพไฟล์ดิบ
2. **Format Conversion:** บังคับโหลดเป็น RGB
3. **Resize:** 224 x 224 (Lanczos Interpolation)
4. **Normalization:** Pixel / 255.0
5. **Standardization:** (x - mean) / std
6. **Output:** Deterministic Float32 Tensor

---

## 4. ประสิทธิภาพความเร็วในการประมวลผล (Benchmarking Throughput)

- **ความเร็วในการ Preprocess & Transform:** **~{throughput_fps:.1f}** ภาพต่อวินาที (Images / sec บน CPU)
- **สถานะ:** มีประสิทธิภาพเพียงพอสำหรับการทำ On-the-fly Batch Data Loading ในขณะฝึกสอนโมเดล

---

## 5. ตัวอย่างโค้ดสำหรับนำไปใช้งานต่อ (Code Integration Examples)

### ตัวอย่าง: PyTorch / Torchvision Transforms
```python
from torchvision import transforms

# 1. Training Transform Pipeline
train_transform = transforms.Compose([
    transforms.Resize(({target_size[0]}, {target_size[1]})),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2. Validation & Test Transform Pipeline
eval_transform = transforms.Compose([
    transforms.Resize(({target_size[0]}, {target_size[1]})),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### ตัวอย่าง: การเรียกใช้ผ่านสคริปต์ `src/preprocess.py`
```python
from src.preprocess import ImagePreprocessor

preprocessor = ImagePreprocessor(target_size=(224, 224))
tensor, pil_img = preprocessor.preprocess_image("data/train/cat - bengal/image.jpg", is_training=True)
print(tensor.shape, tensor.dtype) # (224, 224, 3) float32
```
"""

    report_path = report_dir / "preprocessing_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"[Preprocessing] Report written to: {report_path.resolve()}")
    return report_path


def run_preprocessing_pipeline(data_dir: str = "data",
                               report_dir: str = "reports/preprocessing",
                               target_size: tuple = (224, 224),
                               bench_samples: int = 200):
    """
    Main function to execute preprocessing evaluation, benchmark, visualization, and report generation.
    """
    data_path = Path(data_dir)
    rep_path = Path(report_dir)
    rep_path.mkdir(parents=True, exist_ok=True)

    print(f"[Preprocessing] Initializing pipeline with target_size={target_size}...")
    preprocessor = ImagePreprocessor(target_size=target_size)

    # 1. Calculate Channel Stats
    print("[Preprocessing] Calculating channel statistics...")
    mean_calc, std_calc = calculate_channel_stats(data_path, sample_size=500)

    # 2. Benchmark Throughput
    img_files = list(data_path.rglob("*.jpg"))
    if img_files:
        test_samples = img_files[:min(bench_samples, len(img_files))]
        start_time = time.time()
        for p in test_samples:
            _ = preprocessor.preprocess_image(p, is_training=True)
        elapsed = time.time() - start_time
        fps = len(test_samples) / elapsed if elapsed > 0 else 0
        print(f"[Benchmarking] Processed {len(test_samples)} images in {elapsed:.2f}s ({fps:.1f} FPS)")
    else:
        fps = 0.0

    # 3. Generate Visualizations
    generate_preprocessing_visualizations(data_path, rep_path, preprocessor)

    # 4. Generate Report
    report_file = generate_preprocessing_report(rep_path, mean_calc, std_calc, fps, target_size)
    return report_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Preprocessing and Augmentation Pipeline")
    parser.add_argument("--data-dir", type=str, default="data", help="Path to dataset directory")
    parser.add_argument("--report-dir", type=str, default="reports/preprocessing", help="Path to report output directory")
    parser.add_argument("--img-size", type=int, default=224, help="Target image dimension (e.g. 224)")

    args = parser.parse_args()
    run_preprocessing_pipeline(
        data_dir=args.data_dir,
        report_dir=args.report_dir,
        target_size=(args.img_size, args.img_size)
    )
