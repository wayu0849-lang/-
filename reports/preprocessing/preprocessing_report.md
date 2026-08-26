# รายงานการเตรียมและประมวลผลรูปภาพ (Image Preprocessing Report)
**ชุดข้อมูล:** Dogs and Cats Breed Classifier (104 Classes)  
**เป้าหมายการประมวลผล:** ปรับมาตรฐานและเพิ่มประสิทธิภาพรูปภาพให้พร้อมสำหรับฝึกสอนและทดสอบโมเดล Deep Learning  
**วันที่จัดทำ:** 2026-08-26  

---

## 1. วัตถุประสงค์และเหตุผลทางเทคนิค (Technical Rationale)

จากการวิเคราะห์ข้อมูลเชิงสำรวจ (EDA) พบว่าชุดข้อมูลรูปภาพมีความพร้อมในระดับหนึ่ง แต่เพื่อให้การฝึกสอนโมเดล (เช่น CNN, ResNet, EfficientNet, MobileNet, ViT) มีเสถียรภาพสูงสุด จึงต้องกำหนดกระบวนการ Preprocessing ที่เหมาะสมดังนี้:

| ขั้นตอน (Step) | วิธีการ (Technique) | เหตุผลทางเทคนิค (Rationale) |
| :--- | :--- | :--- |
| **1. การตรวจสอบและแปลงสี (Color Mode)** | แปลงทุกภาพเป็น `RGB` 3 Channels | ป้องกันความผิดพลาดของ Tensor Shape จากภาพที่มี 1 Channel (Grayscale) หรือ 4 Channels (RGBA) |
| **2. การปรับขนาด (Image Resizing)** | ปรับขนาดสู่ **224 × 224** ด้วย Lanczos Interpolation | เป็นขนาดมาตรฐานของโมเดล Computer Vision ส่วนใหญ่ และช่วยให้สร้าง Batch Tensor ที่มีขนาดสม่ำเสมอได้ |
| **3. การทำ Normalization & Standardization** | สเกล `[0, 1]` และทำ Z-Score Normalization ด้วย ImageNet Stats | ป้องกันปัญหา Vanishing/Exploding Gradients และทำให้ Gradient Descent ลู่เข้า (Converge) เร็วขึ้น |
| **4. การทำ Data Augmentation (Train Set)** | Flip, Rotation ±15°, Brightness/Contrast Jitter | ลด Overfitting และแก้ปัญหา Class Imbalance (12.41x) ที่พบใน EDA |

---

## 2. ข้อมูลสถิติช่องสีของชุดข้อมูล (Color Channel Statistics)

จากการสุ่มคำนวณค่าเฉลี่ยและส่วนเบี่ยงเบนมาตรฐานของพิกเซลในชุดข้อมูลจริง (บนช่วง `[0, 1]`):

| ช่องสี (Channel) | Dataset Mean | Dataset Std | ImageNet Standard Mean | ImageNet Standard Std |
| :---: | :---: | :---: | :---: | :---: |
| **Red (R)** | `0.5367` | `0.1589` | `0.4850` | `0.2290` |
| **Green (G)** | `0.5004` | `0.1543` | `0.4560` | `0.2240` |
| **Blue (B)** | `0.4461` | `0.1648` | `0.4060` | `0.2250` |

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

- **ความเร็วในการ Preprocess & Transform:** **~357.6** ภาพต่อวินาที (Images / sec บน CPU)
- **สถานะ:** มีประสิทธิภาพเพียงพอสำหรับการทำ On-the-fly Batch Data Loading ในขณะฝึกสอนโมเดล

---

## 5. ตัวอย่างโค้ดสำหรับนำไปใช้งานต่อ (Code Integration Examples)

### ตัวอย่าง: PyTorch / Torchvision Transforms
```python
from torchvision import transforms

# 1. Training Transform Pipeline
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2. Validation & Test Transform Pipeline
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
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
