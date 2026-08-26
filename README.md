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
│   └── split_data.py             # ตรวจสอบและจัดการ Train/Valid/Test Splitting
├── reports/
│   ├── project_summary_report.md# รายงานสรุปภาพรวมโครงการ (Master Project Report)
│   ├── eda/
│   │   ├── eda_report.md        # รายงาน EDA ฉบับละเอียด
│   │   ├── dataset_metadata.csv # เมทาดาทาภาพทั้งหมด 28,983 ภาพ
│   │   ├── split_distribution.png
│   │   ├── species_breakdown.png
│   │   ├── dimension_distribution.png
│   │   └── top_classes.png
│   ├── preprocessing/
│   │   ├── preprocessing_report.md   # รายงานการเตรียมข้อมูลภาพและ Benchmark
│   │   ├── preprocessing_comparison.png
│   │   └── augmentation_samples.png
│   └── data_splitting/
│       ├── data_splitting_report.md  # รายงานสรุปการตรวจสอบการแบ่งชุดข้อมูล
│       ├── class_split_breakdown.csv # สรุปการกระจายตัวคลาสต่อ Split
│       └── data_split_distribution.png
├── data/                        # โฟลเดอร์เก็บข้อมูลดิบ (ถูก ignore ใน .gitignore)
│   ├── train/ (26,752 images)
│   ├── valid/ (1,339 images)
│   └── test/  (892 images)
├── run_pipeline.bat             # สคริปต์ Batch File รัน Pipeline ทั้งหมดในคลิกเดียว
├── .gitignore                   # กรองไฟล์ภาพดิบและ Secret Token ไม่ให้ขึ้น Git
├── requirements.txt             # รายการไลบรารีที่จำเป็น
└── README.md
```

---

## การรัน Pipeline แบบอัตโนมัติ (Automated Execution)

คุณสามารถสั่งรันกระบวนการทั้งหมดตั้งแต่ติดตั้ง dependencies, ดาวน์โหลดข้อมูล, รัน EDA, ทำ Preprocessing และตรวจสอบ Data Splitting ได้ทันทีในคลิกเดียว:

### บน Windows:
ดับเบิลคลิกไฟล์ **`run_pipeline.bat`** หรือรันคำสั่งใน Command Prompt / PowerShell:
```cmd
run_pipeline.bat
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
> รายงานและชาร์ตสถิติจะถูกบันทึกที่ `reports/eda/`

### 4. รัน Image Preprocessing & Augmentation Benchmark
```bash
python src/preprocess.py
```
> รายงานและภาพตัวอย่างการทำ Transform จะถูกบันทึกที่ `reports/preprocessing/`

### 5. ตรวจสอบการแบ่งชุดข้อมูล (Data Splitting)
```bash
python src/split_data.py
```
> รายงานและชาร์ตสัดส่วน Split จะถูกบันทึกที่ `reports/data_splitting/`

---

## รายงานสรุปผลในโปรเจกต์ (Reports & Documentation)

| หัวข้อรายงาน | ลิงก์ไฟล์รายงาน | สาระสำคัญ |
| :--- | :--- | :--- |
| **รายงานภาพรวมโครงการ** | [`reports/project_summary_report.md`](reports/project_summary_report.md) | สรุปภาพรวมของทุกขั้นตอน, สถาปัตยกรรมระบบ, และคำแนะนำการเทรนโมเดล |
| **รายงาน EDA** | [`reports/eda/eda_report.md`](reports/eda/eda_report.md) | สถิติขนาดภาพ, ระบบสี, จำนวนคลาส (104 คลาส), และ Class Imbalance (12.41x) |
| **รายงาน Preprocessing** | [`reports/preprocessing/preprocessing_report.md`](reports/preprocessing/preprocessing_report.md) | การแปลงสี RGB, การ Resize 224x224, ImageNet Normalization, Augmentation, และ Benchmark (357 FPS) |
| **รายงาน Data Splitting** | [`reports/data_splitting/data_splitting_report.md`](reports/data_splitting/data_splitting_report.md) | การประเมิน Train (92.3%) / Valid (4.6%) / Test (3.1%), Class Coverage (100%), และ Data Leakage Check |

---

## ภาพรวมของ Git Branches

- **`main`**: รวมโค้ดและรายงานที่ผ่านการทดสอบสมบูรณ์แล้ว พร้อม `run_pipeline.bat`
- **`feat/datacollection`**: สคริปต์ดาวน์โหลดข้อมูลผ่าน Kaggle API + `.gitignore`
- **`feat/exploratorydataanlyisi`**: สคริปต์ EDA และรายงาน `reports/eda/`
- **`feat/imagepreprocessing`**: สคริปต์ Preprocessing, Augmentation และรายงาน `reports/preprocessing/`
- **`feat/datasplitting`**: สคริปต์ตรวจสอบการแบ่งชุดข้อมูลและรายงาน `reports/data_splitting/`
