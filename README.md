# Dogs and Cats Breed Classifier

โปรเจกต์ดาวน์โหลดชุดข้อมูลรูปภาพ Dogs and Cats Breed Classifier (104 classes, 28,983 images) และจัดเตรียม Data Pipeline สำหรับการเทรนโมเดล

## โครงสร้างโปรเจกต์ (Project Structure)

```text
├── src/
│   ├── __init__.py
│   ├── download_data.py   # สคริปต์ดาวน์โหลดข้อมูลจาก Kaggle
│   ├── eda.py             # สคริปต์วิเคราะห์ข้อมูลเชิงสำรวจ (EDA)
│   ├── preprocess.py      # สคริปต์ Image Preprocessing & Data Augmentation
│   └── split_data.py       # สคริปต์ตรวจสอบและจัดการ Data Splitting
├── reports/
│   ├── eda/
│   │   ├── eda_report.md  # รายงานผลการวิเคราะห์ EDA แบบละเอียด
│   │   ├── dataset_metadata.csv
│   │   ├── split_distribution.png
│   │   ├── species_breakdown.png
│   │   ├── dimension_distribution.png
│   │   └── top_classes.png
│   ├── preprocessing/
│   └── data_splitting/
├── data/                  # ชุดข้อมูลรูปภาพ (ถูก ignore ไว้ ไม่ push ขึ้น git)
├── .gitignore
├── requirements.txt
└── README.md
```

## วิธีติดตั้งและใช้งาน

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
รายงานผลฉบับสมบูรณ์จะอยู่ที่ `reports/eda/eda_report.md`
