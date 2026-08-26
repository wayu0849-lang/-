# Dogs and Cats Classifier - Data Collection

โปรเจกต์ดาวน์โหลดชุดข้อมูลรูปภาพ Dogs and Cats Classifier จาก Kaggle เพื่อนำมาเตรียมสำหรับการเทรนโมเดล

## โครงสร้างโปรเจกต์

- `download_data.py`: สคริปต์ Python สำหรับดาวน์โหลดและจัดการชุดข้อมูลจาก Kaggle ผ่าน `kagglehub`
- `.gitignore`: ละเว้นโฟลเดอร์รูปภาพ/ชุดข้อมูล (`data/`, `*.jpg`, `*.png` ฯลฯ) และ token ไม่ให้ push ขึ้น Git
- `requirements.txt`: รายการไลบรารีที่จำเป็น

## วิธีติดตั้งและใช้งาน

### 1. ติดตั้ง Dependencies

```bash
pip install -r requirements.txt
```

### 2. ดาวน์โหลด Dataset

รันสคริปต์เพื่อดึงข้อมูลจาก Kaggle มาไว้ในโฟลเดอร์ `data/`:

```bash
python download_data.py
```

หรือระบุโฟลเดอร์ปลายทาง:

```bash
python download_data.py --output-dir data
```
