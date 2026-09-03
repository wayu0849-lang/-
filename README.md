# Dogs and Cats Breed Classifier - End-to-End Data Pipeline

[![GitHub repo](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/wayu0849-lang/-)
[![Dataset](https://img.shields.io/badge/Kaggle-Dogs%20%26%20Cats%20Classifier-20BEFF?logo=kaggle)](https://www.kaggle.com/datasets/rajarshi2712/dogs-and-cats-classifier)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green?logo=python)](https://www.python.org/)

โครงการพัฒนา Data Pipeline แบบอัตโนมัติตามหลักวิศวกรรม Machine Learning สำหรับชุดข้อมูลภาพถ่ายจำแนกสายพันธุ์สุนัขและแมว (104 คลาส, 28,983 รูปภาพ) ครอบคลุมตั้งแต่การดึงข้อมูล, การทำ EDA, การสร้าง Image Preprocessing Pipeline พร้อม Data Augmentation, และการตรวจสอบ Data Splitting

---

## สารบัญ (Table of Contents)

1. [โครงสร้างโปรเจกต์ (Project Structure)](#โครงสร้างโปรเจกต์-project-structure)
2. [การรัน Pipeline แบบอัตโนมัติ (Automated Execution)](#การรัน-pipeline-แบบอัตโนมัติ-automated-execution)
3. [การรันทีละขั้นตอน (Step-by-Step Execution)](#การรันทีละขั้นตอน-step-by-step-execution)
4. [รายงานสรุปผลในโปรเจกต์ (Reports & Documentation)](#รายงานสรุปผลในโปรเจกต์-reports--documentation)
5. [ภาพรวมของ Git Branches](#ภาพรวมของ-git-branches)

---

## โครงสร้างโปรเจกต์ (Project Structure)

```text
.
├── src/
│   ├── __init__.py
│   ├── download_data.py         # ดาวน์โหลดข้อมูลจาก KaggleHub API
│   ├── eda.py                   # การวิเคราะห์ข้อมูลเชิงสำรวจ (EDA)
│   ├── preprocess.py            # Image Preprocessing & Data Augmentation Pipeline
│   ├── split_data.py             # ตรวจสอบและจัดการ Train/Valid/Test Splitting
│   ├── dataset.py               # PyTorch Dataset, DataLoaders & Class Mapping
│   ├── model.py                 # สถาปัตยกรรมโมเดล Deep Learning (Transfer Learning)
│   ├── train.py                 # สคริปต์ฝึกสอนโมเดล (Training & Validation Loop)
│   ├── evaluate.py              # สคริปต์ประเมินผลบนชุดทดสอบ (Test Evaluation)
│   └── predict.py               # สคริปต์ทำนายผลภาพเดี่ยว (Inference)
├── models/                      # โฟลเดอร์จัดเก็บโมเดล Checkpoints
│   ├── best_model.pth           # โมเดล Weights ที่ได้ความแม่นยำสูงสุด
│   ├── class_mapping.json       # พจนานุกรมจับคู่ 104 คลาส
│   └── latest_checkpoint.pth    # Checkpoint ล่าสุด
├── reports/
│   ├── project_summary_report.md# รายงานสรุปภาพรวมโครงการ (Master Project Report)
│   ├── eda/                     # รายงานและกราฟการสำรวจข้อมูล (EDA)
│   ├── preprocessing/           # รายงานและกราฟการเตรียมข้อมูลภาพ
│   ├── data_splitting/          # รายงานและชาร์ตการแบ่งชุดข้อมูล
│   ├── training/                # รายงานผลการเทรนและ Learning Curves
│   │   ├── training_report.md
│   │   └── training_curves.png
│   └── evaluation/              # รายงานผลการประเมินบน Test Set
│       ├── test_evaluation_report.md
│       ├── confusion_matrix.png
│       ├── error_analysis.png
│       └── classification_report.csv
├── data/                        # โฟลเดอร์เก็บข้อมูลภาพดิบ
│   ├── train/ (26,752 images)
│   ├── valid/ (1,339 images)
│   └── test/  (892 images)
├── run_pipeline.bat             # รัน Data Pipeline ครบวงจร
├── run_train.bat                # รัน Train -> Evaluate -> Demo Inference ในคลิกเดียว
├── run_app.bat                  # รัน Web Application ในคลิกเดียว
├── app.py                       # แอปพลิเคชัน Web UI (Gradio) สำหรับนำเสนอ
├── requirements.txt             # รายการไลบรารีที่จำเป็น
└── README.md
```

---

## การรัน Pipeline แบบอัตโนมัติ (Automated Execution)

### 1. รัน Web Application สำหรับใช้งานและนำเสนอ (Interactive Web App):
```cmd
run_app.bat
```
> หรือรันคำสั่ง `python app.py` แล้วเปิดเบราว์เซอร์ไปที่ `http://localhost:7860`

### 2. รัน Data Pipeline (ดาวน์โหลด, EDA, Preprocessing, Splitting):
```cmd
run_pipeline.bat
```

### 3. รัน Model Training & Evaluation (เทรนโมเดล, ประเมินผล Test Set, ทดสอบ Predict):
```cmd
run_train.bat
```

---

## การรันทีละขั้นตอน (Step-by-Step Execution)

### 1. ติดตั้ง Dependencies
```bash
pip install -r requirements.txt
```

### 2. ดาวน์โหลดชุดข้อมูลจาก Kaggle
```bash
python src/download_data.py
```

### 3. รัน Exploratory Data Analysis (EDA)
```bash
python src/eda.py
```

### 4. รัน Image Preprocessing & Augmentation Benchmark
```bash
python src/preprocess.py
```

### 5. ตรวจสอบการแบ่งชุดข้อมูล (Data Splitting)
```bash
python src/split_data.py
```

### 6. ฝึกสอนโมเดล Deep Learning (Model Training)
```bash
python src/train.py --model_name mobilenetv3_large_100 --epochs 5 --batch_size 32
```
> สามารถระบุ Model Architecture ได้ เช่น `mobilenetv3_large_100`, `efficientnet_b0`, `resnet50`

### 7. ประเมินผลบนชุดทดสอบ (Model Evaluation)
```bash
python src/evaluate.py --checkpoint models/best_model.pth
```
> รายงานผลและชาร์ต Confusion Matrix / Error Analysis จะถูกบันทึกที่ `reports/evaluation/`

### 8. ทำนายผลภาพเดี่ยว (Inference / Prediction)
```bash
python src/predict.py --image path/to/image.jpg --top_k 5
```

---

## รายงานสรุปผลในโปรเจกต์ (Reports & Documentation)

| หัวข้อรายงาน | ลิงก์ไฟล์รายงาน | ผลลัพธ์ / สาระสำคัญ |
| :--- | :--- | :--- |
| **สไลด์นำเสนอโครงการ (PPTX)** | [`presentation_dogs_cats_pipeline.pptx`](presentation_dogs_cats_pipeline.pptx) | สไลด์นำเสนอ 8 หน้าครอบคลุม Pipeline, ปัญหา/อุปสรรค, และแนวทางต่อยอด |
| **สคริปต์พูดนำเสนอ (Speaker Notes)** | [`reports/presentation_slides.md`](reports/presentation_slides.md) | บทพูดนำเสนอและสาระสำคัญของแต่ละสไลด์แบบละเอียด |
| **รายงานภาพรวมโครงการ** | [`reports/project_summary_report.md`](reports/project_summary_report.md) | สรุปภาพรวมของทุกขั้นตอน และสถาปัตยกรรมระบบ |
| **รายงาน EDA** | [`reports/eda/eda_report.md`](reports/eda/eda_report.md) | สถิติ 104 คลาส, ขนาดภาพ, และ Imbalance (12.41x) |
| **รายงาน Preprocessing** | [`reports/preprocessing/preprocessing_report.md`](reports/preprocessing/preprocessing_report.md) | การ Resize $224\times224$, ImageNet Normalization, Augmentation |
| **รายงาน Data Splitting** | [`reports/data_splitting/data_splitting_report.md`](reports/data_splitting/data_splitting_report.md) | การแบ่ง Train (92.3%) / Valid (4.6%) / Test (3.1%) |
| **รายงานผลการเทรน** | [`reports/training/training_report.md`](reports/training/training_report.md) | Val Top-1 Accuracy: **86.18%**, Learning Curves |
| **รายงานผลบน Test Set** | [`reports/evaluation/test_evaluation_report.md`](reports/evaluation/test_evaluation_report.md) | **Test Top-1: 86.88%**, **Test Top-5: 99.22%**, **F1: 85.01%** |

---

## สรุปประสิทธิภาพโมเดล (Model Performance Benchmark)

- **Backbone Model:** `MobileNetV3-Large` (Transfer Learning from ImageNet)
- **จำนวนคลาส:** 104 Classes (สายพันธุ์สุนัขและแมว)
- **Test Top-1 Accuracy:** **`86.88%`**
- **Test Top-5 Accuracy:** **`99.22%`**
- **Macro F1-Score:** **`85.01%`**
- **Weighted F1-Score:** **`86.37%`**
- **Inference Speed:** ~15-25 ms ต่อภาพบน GPU / ~40-60 ms บน CPU

