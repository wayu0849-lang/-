# รายงานการวิเคราะห์ข้อมูลเชิงสำรวจ (Exploratory Data Analysis - EDA Report)
**ชุดข้อมูล:** Dogs and Cats Breed Classifier  
**แหล่งที่มา:** Kaggle (`rajarshi2712/dogs-and-cats-classifier` / Roboflow Universe)  
**วันที่จัดทำ:** 2026-08-26  

---

## 1. บทสรุปภาพรวมของชุดข้อมูล (Executive Summary)

ชุดข้อมูลนี้เป็นชุดข้อมูลภาพถ่ายสำหรับการจำแนกสายพันธุ์สุนัขและแมว (Multi-class / Multi-breed Image Classification) โดยมีคุณลักษณะพื้นฐานดังนี้:

| รายการ | ค่าสถิติ |
| :--- | :--- |
| **จำนวนรูปภาพทั้งหมดที่ถูกต้อง (Valid Images)** | **28,983** ภาพ |
| **จำนวนภาพที่เสียหาย (Corrupted/Unreadable)** | **0** ภาพ |
| **จำนวนคลาสสายพันธุ์ทั้งหมด (Number of Classes)** | **104** คลาส |
| **หมวดหมู่หลัก (Species Categories)** | Dogs: 19,658 ภาพ, Cats: 9,325 ภาพ |
| **สัดส่วนข้อมูลที่ถูกแบ่งมาแล้ว (Pre-split Sets)** | Train: 26,752 | Valid: 1,339 | Test: 892 |
| **ความละเอียดเฉลี่ยของภาพ (Mean Dimensions)** | 224.0 × 224.0 พิกเซล |
| **ขนาดไฟล์เฉลี่ย (Mean File Size)** | 9.48 KB |

---

## 2. การแจกแจงตามชุดข้อมูล (Dataset Split Distribution)

ชุดข้อมูลมีการแบ่งโครงสร้างโฟลเดอร์เริ่มต้นออกเป็น 3 ชุดย่อย:

| Split Name | จำนวนภาพ | สัดส่วน (%) | จำนวนคลาสที่มี |
| :--- | :--- | :--- | :--- |
| `train` | 26,752 | 92.30% | 104 คลาส |
| `valid` | 1,339 | 4.62% | 104 คลาส |
| `test` | 892 | 3.08% | 104 คลาส |

> **ข้อสังเกต:**
> - ชุดข้อมูลถูกแบ่งเบื้องต้นไว้เป็น 92.3% Train, 4.6% Validation, และ 3.1% Test
> - ทุกชุดย่อย (train, valid, test) มีคลาสครบทั้ง 104 คลาส

---

## 3. การวิเคราะห์คลาสและความสมดุลของข้อมูล (Class Distribution & Balance Analysis)

- **อัตราส่วนความไม่สมดุล (Imbalance Ratio - Max/Min):** `12.41` เท่า
- **จำนวนภาพเฉลี่ยต่อคลาส (Mean images per class):** `278.7` ภาพ (Min: 87, Max: 1080, Median: 242)

### 3.1 10 อันดับคลาสที่มีจำนวนภาพมากที่สุด (Top 10 Classes)
| อันดับ | ชื่อคลาส | หมวดหมู่ | จำนวนภาพ | สัดส่วน (%) |
| :---: | :--- | :---: | :---: | :---: |
| 1 | `cat - bengal` | Cat | 1,080 | 3.73% |
| 2 | `cat - ragdoll` | Cat | 1,006 | 3.47% |
| 3 | `cat - persian` | Cat | 534 | 1.84% |
| 4 | `cat - sphinx` | Cat | 533 | 1.84% |
| 5 | `cat - abyssinian` | Cat | 532 | 1.84% |
| 6 | `cat - scottishfold` | Cat | 527 | 1.82% |
| 7 | `cat - birman` | Cat | 526 | 1.81% |
| 8 | `cat - mainecoon` | Cat | 522 | 1.80% |
| 9 | `cat - siamese` | Cat | 520 | 1.79% |
| 10 | `cat - british shorthair` | Cat | 518 | 1.79% |

### 3.2 10 อันดับคลาสที่มีจำนวนภาพน้อยที่สุด (Bottom 10 Classes)
| อันดับ | ชื่อคลาส | หมวดหมู่ | จำนวนภาพ | สัดส่วน (%) |
| :---: | :--- | :---: | :---: | :---: |
| 1 | `dog - bernard-dog - saint` | Dog | 87 | 0.30% |
| 2 | `dog - doberman-dog - pinscher` | Dog | 91 | 0.31% |
| 3 | `dog - miniature-dog - schnauzer` | Dog | 91 | 0.31% |
| 4 | `dog - labrador-dog - retriever` | Dog | 92 | 0.32% |
| 5 | `dog - dachshund` | Dog | 93 | 0.32% |
| 6 | `dog - aspin` | Dog | 95 | 0.33% |
| 7 | `dog - shih-dog - tzu` | Dog | 101 | 0.35% |
| 8 | `dog - golden-dog - retriever` | Dog | 102 | 0.35% |
| 9 | `dog - german-dog - sheperd` | Dog | 106 | 0.37% |
| 10 | `dog - sheepdog-dog - shetland` | Dog | 109 | 0.38% |

---

## 4. คุณสมบัติทางกายภาพของรูปภาพ (Image Properties Analysis)

### 4.1 ความละเอียดและมิติของภาพ (Dimensions)
| สถิติ | ความกว้าง (Width) | ความสูง (Height) | สัดส่วนภาพ (Aspect Ratio W/H) | ขนาดไฟล์ (KB) |
| :--- | :---: | :---: | :---: | :---: |
| **ต่ำสุด (Min)** | 224 px | 224 px | 1.00 | 2.93 KB |
| **เปอร์เซ็นไทล์ที่ 25 (Q1)** | 224 px | 224 px | 1.00 | 7.69 KB |
| **มัธยฐาน (Median)** | 224 px | 224 px | 1.00 | 9.19 KB |
| **เฉลี่ย (Mean)** | 224.0 px | 224.0 px | 1.00 | 9.48 KB |
| **เปอร์เซ็นไทล์ที่ 75 (Q3)** | 224 px | 224 px | 1.00 | 10.97 KB |
| **สูงสุด (Max)** | 224 px | 224 px | 1.00 | 22.65 KB |

### 4.2 ระบบสี (Color Modes) และนามสกุลไฟล์ (File Formats)

| ระบบสี (Color Mode) | จำนวนภาพ | สัดส่วน (%) |
| :--- | :---: | :---: |
| `RGB` | 28,983 | 100.00% |

| ชนิดไฟล์ (Format) | จำนวนภาพ | สัดส่วน (%) |
| :--- | :---: | :---: |
| `JPEG` | 28,983 | 100.00% |

---

## 5. การตรวจสอบความสมบูรณ์ของไฟล์ (Data Integrity & Quality Check)

- **จำนวนไฟล์รูปภาพที่ชำรุด (Corrupted / Broken Files):** `0` ไฟล์
- **สถานะ:** ข้อมูลรูปภาพทั้งหมดสามารถโหลดและอ่าน header/data ได้อย่างถูกต้อง 100%

---

## 6. ข้อสรุปและข้อเสนอแนะสำหรับการทำ Image Preprocessing & Modeling (Recommendations)

1. **การปรับขนาดรูปภาพ (Resize & Aspect Ratio Handling):**
   - ภาพมีความละเอียดแตกต่างกันอย่างมาก (ตั้งแต่ 224px จนถึง 224px)
   - แนะนำให้ปรับขนาดภาพให้เป็นมาตรฐาน **224 × 224** หรือ **256 × 256** พิกเซล ซึ่งเป็นขนาดมาตรฐานสำหรับโมเดล Convolutional Neural Networks (CNNs เช่น ResNet, EfficientNet, MobileNet) และ Vision Transformers (ViT)
   
2. **การจัดการระบบสี (Color Channel Standardization):**
   - ควรแปลงรูปภาพทุกภาพให้เป็น **RGB 3 ช่องสี** เพื่อป้องกันข้อผิดพลาดกรณีมีภาพ Grayscale (L) หรือภาพที่มี Alpha Channel (RGBA)

3. **การทำ Data Augmentation เพื่อลดผลกระทบจาก Class Imbalance:**
   - เนื่องจากคลาสมี Imbalance Ratio อยู่ที่ประมาณ `12.41` เท่า ควรประยุกต์ใช้ Data Augmentation (เช่น Random Horizontal Flip, Rotation ±15°, Color Jitter, Random Affine Scaling) ในขั้นตอนฝึกสอน เพื่อเพิ่มความหลากหลายและลด Overfitting

4. **การแปลงข้อมูลตัวเลข (Normalization / Standardization):**
   - ทำ Normalization ปรับช่วงค่าพิกเซลจาก `[0, 255]` ให้อยู่ในช่วง `[0, 1]` หรือทำการ Standardize ด้วย ImageNet Mean (`[0.485, 0.456, 0.406]`) และ Std (`[0.229, 0.224, 0.225]`)
