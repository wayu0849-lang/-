# รายงานการประเมินผลบนชุดทดสอบ (Test Set Evaluation Report)

**ชุดข้อมูลทดสอบ:** `data/test` (จำนวน 892 ภาพ, 104 คลาส)  
**สถาปัตยกรรมโมเดล:** `mobilenetv3_large_100`  

---

## 1. ผลลัพธ์ประสิทธิภาพหลัก (Overall Performance Metrics)

| Metric | Score | คำอธิบาย |
| :--- | :---: | :--- |
| **Top-1 Accuracy** | **`87.56%`** | สัดส่วนการทายสายพันธุ์ถูกต้องอันดับ 1 |
| **Top-5 Accuracy** | **`98.77%`** | สัดส่วนที่คำตอบที่ถูกต้องติด 1 ใน 5 อันดับแรก |
| **Macro F1-Score** | **`87.50%`** | ค่า F1 เฉลี่ยทุกคลาส (ทนต่อ Class Imbalance) |
| **Weighted F1-Score** | **`87.40%`** | ค่า F1 ถ่วงน้ำหนักตามจำนวนภาพจริง |
| **Macro Precision** | **`88.82%`** | ความแม่นยำเฉลี่ยของการทำนาย |
| **Macro Recall** | **`88.00%`** | ความครอบคลุมเฉลี่ยในการตรวจจับ |

---

## 2. Confusion Matrix & การวิเคราะห์ข้อผิดพลาด (Visualizations)

| ชาร์ตแสดงผล | ลิงก์ไฟล์ | สาระสำคัญ |
| :--- | :--- | :--- |
| **Confusion Matrix Heatmap** | `reports/evaluation/confusion_matrix.png` | แสดงการกระจายตัวของการจำแนกระหว่าง 104 คลาส |
| **Error Analysis Samples** | `reports/evaluation/error_analysis.png` | ตัวอย่างภาพที่โมเดลทำนายผิดพลาดพร้อมค่าความมั่นใจ |

---

## 3. สรุปผลการประเมินรายคลาส (Top & Bottom Performing Classes)

### คลาสที่มีประสิทธิภาพสูงสุด 5 อันดับแรก (Top-5 F1-Score):
|                           |   precision |   recall |   f1-score |   support |
|:--------------------------|------------:|---------:|-----------:|----------:|
| cat - bombay              |           1 |        1 |          1 |        13 |
| dog - bermaise            |           1 |        1 |          1 |         5 |
| dog - bernese mountain    |           1 |        1 |          1 |         2 |
| dog - bernard-dog - saint |           1 |        1 |          1 |         1 |
| dog - bichon frise        |           1 |        1 |          1 |         2 |

### คลาสที่โมเดลยังจำแนกได้ยาก 5 อันดับ (Bottom-5 F1-Score):
|                                |   precision |   recall |   f1-score |   support |
|:-------------------------------|------------:|---------:|-----------:|----------:|
| dog - labrador-dog - retriever |    0        | 0        |   0        |         1 |
| dog - pekinese                 |    0.5      | 0.5      |   0.5      |         2 |
| dog - german sheperd           |    1        | 0.375    |   0.545455 |         8 |
| dog - lhasa                    |    0.714286 | 0.555556 |   0.625    |         9 |
| dog - boston terrier           |    0.5      | 1        |   0.666667 |         5 |

---

## 4. ข้อสังเกตและข้อเสนอแนะในการปรับปรุง (Insights & Recommendations)
1. **Top-5 Accuracy vs Top-1 Accuracy:** เนื่องจากสายพันธุ์สุนัขและแมวบางชนิดมีความคล้ายคลึงกันทางกายภาพสูงมาก (Fine-grained classification) ค่า Top-5 Accuracy จะช่วยสะท้อนว่าโมเดลมองเห็นลักษณะเด่นของกลุ่มสายพันธุ์ใกล้เคียงได้เป็นอย่างดี
2. **การต่อยอดสู่การใช้งานจริง:** สามารถนำโมเดลไปใช้งานผ่านสคริปต์ `src/predict.py` เพื่อแสดงผล Top-5 Predictions พร้อม Confidence Score
