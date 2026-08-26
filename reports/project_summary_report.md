# รายงานสรุปผลโครงการภาพรวม (Master Project Summary Report)
**ชื่อโครงการ:** Dogs and Cats Breed Classifier Pipeline  
**ชุดข้อมูล:** `rajarshi2712/dogs-and-cats-classifier` (104 Classes, 28,983 Images)  
**ผู้จัดทำ:** `wayu0849-lang` (wayu0849@gmail.com)  
**วันที่จัดทำ:** 2026-08-26  

---

## 1. บทสรุปผู้บริหาร (Executive Summary)

โครงการนี้จัดทำขึ้นเพื่อสร้างและวางระบบ Data Pipeline แบบอัตโนมัติตามหลักวิศวกรรม Machine Learning ครอบคลุมตั้งแต่การดาวน์โหลดข้อมูล (Data Collection), การสำรวจและวิเคราะห์ข้อมูลเชิงลึก (Exploratory Data Analysis - EDA), การจัดเตรียมและเพิ่มประสิทธิภาพรูปภาพ (Image Preprocessing & Augmentation), จนถึงการตรวจสอบและแบ่งชุดข้อมูล (Data Splitting) เพื่อเตรียมความพร้อมขั้นสูงสุดสำหรับการฝึกสอนโมเดล Deep Learning

### สรุปผลลัพธ์สำคัญ (Key Highlights):
- **จำนวนรูปภาพทั้งหมด:** **28,983 ภาพ** แบ่งเป็น 104 สายพันธุ์สุนัขและแมว
- **คุณภาพของไฟล์ภาพ:** **100% Valid** ไม่พบไฟล์รูปภาพชำรุดเสียหาย (0 Corrupted Images)
- **ระบบอัตโนมัติ (Automation):** มีไฟล์ `run_pipeline.bat` สำหรับสั่งประมวลผลกระบวนการทั้งหมดในคลิกเดียว
- **การจัดการความปลอดภัยและ Git:** กรองไฟล์รูปภาพและ Token ไม่ให้ push ขึ้น GitHub ผ่าน `.gitignore` อย่างรัดกุม

---

## 2. สถาปัตยกรรมและโครงสร้างโปรเจกต์ (Project Architecture)

โครงสร้างโฟลเดอร์ถูกจัดวางให้โค้ดโปรแกรมอยู่ใน `src/`, เอกสารรายงานสรุปผลและชาร์ตอยู่ใน `reports/`, และชุดข้อมูลอยู่ใน `data/`:

```text
.
├── src/
│   ├── __init__.py
│   ├── download_data.py         # โมดูลดาวน์โหลดข้อมูลผ่าน KaggleHub API
│   ├── eda.py                   # โมดูลวิเคราะห์ข้อมูลเชิงสำรวจและสร้างชาร์ตสถิติ
│   ├── preprocess.py            # โมดูล Preprocessing, Normalization และ Augmentation
│   └── split_data.py             # โมดูลตรวจสอบและจัดการ Train/Valid/Test Split
├── reports/
│   ├── project_summary_report.md# รายงานสรุปภาพรวมโครงการ (Master Report)
│   ├── eda/
│   │   ├── eda_report.md        # รายงาน EDA ฉบับละเอียด
│   │   ├── dataset_metadata.csv # เมทาดาทาภาพทั้งหมด 28,983 ภาพ
│   │   ├── split_distribution.png
│   │   ├── species_breakdown.png
│   │   ├── dimension_distribution.png
│   │   └── top_classes.png
│   ├── preprocessing/
│   │   ├── preprocessing_report.md   # รายงานการเตรียมข้อมูลรูปภาพ
│   │   ├── preprocessing_comparison.png
│   │   └── augmentation_samples.png
│   └── data_splitting/
│       ├── data_splitting_report.md  # รายงานสรุปการแบ่งชุดข้อมูล
│       ├── class_split_breakdown.csv # การกระจายตัวคลาสต่อ Split
│       └── data_split_distribution.png
├── data/                        # โฟลเดอร์เก็บข้อมูลภาพ (ถูก ignore โดย .gitignore)
│   ├── train/ (26,752 images)
│   ├── valid/ (1,339 images)
│   └── test/  (892 images)
├── run_pipeline.bat             # สคริปต์รัน Pipeline ทั้งหมดในคลิกเดียว
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 3. สรุปรายละเอียดแต่ละขั้นตอนของ Pipeline (Pipeline Stages)

### 3.1 ขั้นตอนที่ 1: การดาวน์โหลดชุดข้อมูล (Data Collection)
- **แหล่งที่มา:** Kaggle Dataset `rajarshi2712/dogs-and-cats-classifier`
- **การทำงาน:** สคริปต์ `src/download_data.py` เชื่อมต่อผ่าน Kaggle API Token และไลบรารี `kagglehub`
- **การจัดเก็บ:** บันทึกลงในเครื่องที่โฟลเดอร์ `data/` และทำการซิงค์โครงสร้าง `train/`, `valid/`, `test/`
- **การควบคุม Git:** กำหนด `.gitignore` กรองโฟลเดอร์ `data/`, `dataset/` และนามสกุลไฟล์ภาพดิบทั้งหมด ป้องกันไม่ให้ไฟล์ขนาดใหญ่ถูก push ขึ้น Git

### 3.2 ขั้นตอนที่ 2: การวิเคราะห์ข้อมูลเชิงสำรวจ (Exploratory Data Analysis - EDA)
- **สคริปต์:** `src/eda.py`
- **รายงาน:** `reports/eda/eda_report.md`
- **ข้อค้นพบหลัก (Key Findings):**
  - **จำนวนคลาส:** 104 คลาส (สายพันธุ์สุนัขและแมว)
  - **การแจกแจงหมวดหมู่:** Dogs = 19,658 ภาพ (67.8%), Cats = 9,325 ภาพ (32.2%)
  - **Class Imbalance:** คลาสที่มากที่สุดคือ `cat - bengal` (1,080 ภาพ) และน้อยที่สุดคือ `dog - bernard-dog - saint` (87 ภาพ) คิดเป็น Imbalance Ratio = **12.41 เท่า**
  - **คุณสมบัติภาพ:** ความละเอียด $224 \times 224$ พิกเซล, รูปแบบสี RGB 100%, นามสกุล JPEG 100%
  - **Integrity Check:** ไม่พบภาพเสียหรืออ่านไม่ได้ (0 Corrupted Images)

### 3.3 ขั้นตอนที่ 3: การเตรียมและประมวลผลรูปภาพ (Image Preprocessing & Augmentation)
- **สคริปต์:** `src/preprocess.py`
- **รายงาน:** `reports/preprocessing/preprocessing_report.md`
- **กระบวนการที่ออกแบบ:**
  1. **Image Validation:** โหลดและบังคับแปลงเป็น 3-Channel RGB
  2. **Resizing:** รองรับทั้ง Direct Resize สู่ $224 \times 224$ (Lanczos) และ Aspect-Ratio Preserving (Letterbox Padding)
  3. **Normalization & Standardization:** สเกลค่าพิกเซลสู่ $[0, 1]$ และ Standardization ด้วย ImageNet Mean/Std
  4. **Data Augmentation (สำหรับ Train Set):**
     - Random Horizontal Flip ($p=0.5$)
     - Random Rotation ($\pm 15^\circ$)
     - Brightness & Contrast Jitter ($0.85 - 1.15$)
  5. **Benchmark Throughput:** ประมวลผลได้เร็วถึง **~357 ภาพต่อวินาที** บน CPU

### 3.4 ขั้นตอนที่ 4: การตรวจสอบการแบ่งชุดข้อมูล (Data Splitting Verification)
- **สคริปต์:** `src/split_data.py`
- **รายงาน:** `reports/data_splitting/data_splitting_report.md`
- **ผลการประเมิน:**
  - **การตัดสินใจ:** เนื่องจากชุดข้อมูลถูกแบ่งโครงสร้างมาอย่างสมบูรณ์แล้ว (`train: 92.30%`, `valid: 4.62%`, `test: 3.08%`) จึง **ไม่ต้องทำการแบ่งไฟล์ซ้ำซ้อน**
  - **Class Stratification:** ทุกคลาส (104 คลาส) มีตัวแทนอยู่ในทุกชุดย่อยครบ 100%
  - **Data Leakage Check:** สุ่มตรวจสอบ MD5 Fingerprint Hash พบว่าข้อมูลแต่ละชุดย่อยมีความเป็นอิสระต่อกัน (Zero/Near-zero Data Leakage) ทำให้การวัดผลมีความเที่ยงตรง

---

## 4. วิธีการรันระบบ (How to Run)

### วิธีที่ 1: รัน Pipeline อัตโนมัติทั้งหมดในคลิกเดียว (Recommended)
ดับเบิลคลิกไฟล์ **`run_pipeline.bat`** หรือพิมพ์ใน Terminal:

```cmd
run_pipeline.bat
```

### วิธีที่ 2: รันทีละขั้นตอนผ่าน Python
```bash
# 1. ติดตั้ง Dependencies
pip install -r requirements.txt

# 2. ดาวน์โหลดข้อมูล
python src/download_data.py

# 3. รัน EDA
python src/eda.py

# 4. รัน Preprocessing
python src/preprocess.py

# 5. รัน Data Splitting
python src/split_data.py
```

---

## 5. ข้อเสนอแนะสำหรับขั้นตอนการเทรนโมเดล (Next Steps & Model Training)

1. **เลือกใช้ Transfer Learning Architecture:**
   - แนะนำโมเดลเช่น **EfficientNet-B0 / B2**, **MobileNetV3**, หรือ **ResNet-50** ซึ่งมี Pretrained Weights บน ImageNet และทำงานได้มีประสิทธิภาพสูงกับ Input $224 \times 224$
2. **การจัดการ Class Imbalance (12.41x):**
   - ควรใช้ Loss Function เช่น **Focal Loss** หรือกำหนด **Class Weights** ใน Cross-Entropy Loss
   - นำไปป์ไลน์ Data Augmentation ที่เตรียมไว้ใน `src/preprocess.py` ไปประยุกต์ใช้ใน `DataLoader`
3. **การวัดผลและประเมินประสิทธิภาพ (Evaluation Metrics):**
   - แนะนำให้ใช้ **Macro F1-Score**, **Top-1 Accuracy**, และ **Top-5 Accuracy** เนื่องจากมีคลาสจำนวนมาก (104 คลาส)
