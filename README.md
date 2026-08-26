# Dogs and Cats Breed Classifier

โปรเจกต์ดาวน์โหลดชุดข้อมูลรูปภาพ Dogs and Cats Breed Classifier (104 classes, 28,983 images) จาก Kaggle พร้อมสถาปัตยกรรมโค้ดและการวิเคราะห์ข้อมูลตามหลักวิศวกรรม Machine Learning

## โครงสร้างโปรเจกต์ (Project Structure)

```text
├── src/
│   ├── __init__.py
│   ├── download_data.py         # ดาวน์โหลดข้อมูลจาก KaggleHub
│   ├── eda.py                   # วิเคราะห์ข้อมูลเชิงสำรวจ (EDA)
│   ├── preprocess.py            # Image Preprocessing & Data Augmentation Pipeline
│   └── split_data.py             # ตรวจสอบและวิเคราะห์การแบ่งชุดข้อมูล (Data Splitting)
├── reports/
│   ├── eda/
│   │   ├── eda_report.md        # รายงานสรุป EDA ละเอียด
│   │   ├── dataset_metadata.csv # เมทาดาทาภาพทั้งหมด 28,983 ภาพ
│   │   ├── split_distribution.png
│   │   ├── species_breakdown.png
│   │   ├── dimension_distribution.png
│   │   └── top_classes.png
│   ├── preprocessing/
│   │   ├── preprocessing_report.md   # รายงานการเตรียมข้อมูลภาพ
│   │   ├── preprocessing_comparison.png
│   │   └── augmentation_samples.png
│   └── data_splitting/
│       ├── data_splitting_report.md  # รายงานการตรวจสอบการแบ่งข้อมูล
│       ├── class_split_breakdown.csv # สรุปการกระจายตัวคลาสต่อ Split
│       └── data_split_distribution.png
├── data/                        # โฟลเดอร์เก็บข้อมูลดิบ (ถูก ignore ใน .gitignore)
│   ├── train/
│   ├── valid/
│   └── test/
├── .gitignore
├── requirements.txt
└── README.md
```

## วิธีติดตั้งและรัน Pipeline ทั้งหมด

### 1. ติดตั้ง Dependencies

```bash
pip install -r requirements.txt
```

### 2. ดาวน์โหลด Dataset

```bash
python src/download_data.py
```

### 3. รัน Exploratory Data Analysis (EDA)

```bash
python src/eda.py
```
> รายงานฉบับสมบูรณ์จะถูกสร้างไว้ที่ `reports/eda/eda_report.md`

### 4. รัน Image Preprocessing & Augmentation Benchmark

```bash
python src/preprocess.py
```
> รายงานฉบับสมบูรณ์จะถูกสร้างไว้ที่ `reports/preprocessing/preprocessing_report.md`

### 5. ตรวจสอบการแบ่งชุดข้อมูล (Data Splitting Verification)

```bash
python src/split_data.py
```
> รายงานฉบับสมบูรณ์จะถูกสร้างไว้ที่ `reports/data_splitting/data_splitting_report.md`

---

## สรุป Git Branches ในโปรเจกต์

- `main`: โครงสร้างเริ่มต้น
- `feat/datacollection`: สคริปต์ดาวน์โหลดข้อมูลผ่าน Kaggle API + `.gitignore`
- `feat/exploratorydataanlyisi`: โครงสร้าง `src/`, สคริปต์ EDA และรายงาน `reports/eda/`
- `feat/imagepreprocessing`: สคริปต์ Image Preprocessing, Data Augmentation และรายงาน `reports/preprocessing/`
- `feat/datasplitting`: การตรวจสอบการแบ่งชุดข้อมูล Train / Valid / Test และรายงาน `reports/data_splitting/`
