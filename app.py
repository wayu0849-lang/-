import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple

# Reconfigure stdout/stderr for Unicode safety on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Add project root to sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
from PIL import Image
import gradio as gr

from src.dataset import get_transforms
from src.model import build_model


# 1. Global Model Loader
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_PATH = Path('models/best_model.pth')
MAPPING_PATH = Path('models/class_mapping.json')

print(f'[INFO] Loading model weights from {CHECKPOINT_PATH} on {DEVICE}...')

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(f"Model checkpoint '{CHECKPOINT_PATH}' not found. Please train model first.")

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)

# Load class mapping
class_to_idx = checkpoint.get('class_to_idx')
if not class_to_idx and MAPPING_PATH.exists():
    with open(MAPPING_PATH, 'r', encoding='utf-8') as f:
        class_to_idx = json.load(f)

idx_to_class = {v: k for k, v in class_to_idx.items()}
num_classes = len(class_to_idx)
model_name = checkpoint.get('model_name', 'mobilenetv3_large_100')

# Initialize model
model = build_model(model_name=model_name, num_classes=num_classes, pretrained=False).to(DEVICE)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

_, eval_transform = get_transforms()
print(f'[OK] Model {model_name} loaded successfully with {num_classes} classes.')


def format_class_name(raw_name: str) -> str:
    """Format raw class name into readable display name with emoji."""
    parts = raw_name.split(' - ')
    if len(parts) >= 2:
        species, breed = parts[0], parts[1]
        emoji = "🐱 แมว" if species.lower() == 'cat' else "🐶 สุนัข"
        breed_clean = breed.replace('-', ' ').title()
        return f"{emoji}: {breed_clean} ({raw_name})"
    return raw_name.title()


def predict_breed(image: Image.Image) -> Tuple[Dict[str, float], str]:
    """
    Inference function for Gradio: returns top class probabilities and summary card.
    """
    if image is None:
        return {}, """
<div style="background-color: #F8FAFC; border: 2px dashed #CBD5E1; border-radius: 12px; padding: 24px; text-align: center;">
    <h3 style="color: #64748B; margin: 0;">⏳ รอการอัปโหลดรูปภาพเพื่อวิเคราะห์...</h3>
    <p style="color: #94A3B8; margin-top: 8px;">ลากรูปภาพมาวาง, วางรูปจากเน็ต (Ctrl+V), หรือคลิกภาพตัวอย่างด้านล่าง</p>
</div>
"""

    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Image tensor transform
    input_tensor = eval_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        
        # Get top-5 predictions
        top_probs, top_indices = torch.topk(probabilities, k=5)

    top_probs = top_probs.cpu().numpy()
    top_indices = top_indices.cpu().numpy()

    # Build dictionary for gr.Label
    confidences = {}
    for idx, prob in zip(top_indices, top_probs):
        raw_name = idx_to_class[idx]
        display_name = format_class_name(raw_name)
        confidences[display_name] = float(prob)

    # Top-1 Winner
    top1_idx = top_indices[0]
    top1_prob = top_probs[0] * 100.0
    top1_raw = idx_to_class[top1_idx]
    top1_species = "แมว (Cat)" if "cat" in top1_raw.lower() else "สุนัข (Dog)"
    top1_breed = top1_raw.split(' - ')[-1].replace('-', ' ').title()
    top1_emoji = "🐱" if "cat" in top1_raw.lower() else "🐶"
    top1_badge_bg = "#ECFDF5" if top1_prob >= 70 else "#FFFBEB"
    top1_badge_color = "#059669" if top1_prob >= 70 else "#D97706"

    summary_html = f"""
<div style="background: linear-gradient(135deg, #FFFFFF, #F8FAFC); border: 2px solid #E2E8F0; border-radius: 12px; padding: 20px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);">
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
        <span style="font-size: 14px; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.5px;">ผลการวิเคราะห์อันดับ 1</span>
        <span style="background: {top1_badge_bg}; color: {top1_badge_color}; padding: 4px 12px; border-radius: 20px; font-size: 14px; font-weight: 800;">
            ความมั่นใจ {top1_prob:.2f}%
        </span>
    </div>
    <div style="font-size: 26px; font-weight: 800; color: #1E293B; margin-bottom: 8px;">
        {top1_emoji} {top1_breed}
    </div>
    <div style="font-size: 15px; color: #475569; line-height: 1.6;">
        <b>ประเภท:</b> <span style="color: #2563EB;">{top1_species}</span> &nbsp;|&nbsp; 
        <b>รหัสคลาส:</b> <code style="background: #F1F5F9; padding: 2px 6px; border-radius: 4px; color: #334155;">{top1_raw}</code>
    </div>
</div>
"""
    return confidences, summary_html


# 2. Example Images List
examples_list = []
test_dir = Path('data/test')
popular_breeds = [
    'dog - golden retriever', 'cat - siamese', 'dog - siberian husky', 
    'cat - persian', 'dog - beagle', 'cat - bengal', 'dog - pug', 'cat - british shorthair'
]
for pop in popular_breeds:
    folder = test_dir / pop
    if folder.exists():
        for f in folder.glob('*.jpg'):
            examples_list.append(str(f))
            break


# 3. Custom High-Contrast CSS
custom_css = """
.gradio-container {
    max-width: 1050px !important;
    margin: auto !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
}
.header-card {
    background: linear-gradient(135deg, #1E3A8A 0%, #0F172A 100%) !important;
    padding: 26px 20px !important;
    border-radius: 16px !important;
    text-align: center !important;
    margin-bottom: 24px !important;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.15) !important;
}
.header-title {
    color: #FFFFFF !important;
    font-size: 30px !important;
    font-weight: 800 !important;
    margin: 0 0 10px 0 !important;
    letter-spacing: -0.5px !important;
}
.header-subtitle {
    color: #38BDF8 !important;
    font-size: 17px !important;
    font-weight: 600 !important;
    margin: 0 0 8px 0 !important;
}
.header-desc {
    color: #E2E8F0 !important;
    font-size: 14px !important;
    margin: 0 !important;
    opacity: 0.9 !important;
}
"""

with gr.Blocks(title="Dogs & Cats Breed Classifier (104 Classes)", css=custom_css, theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate")) as demo:
    with gr.Column(elem_classes=["header-card"]):
        gr.HTML(
            """
            <div style="text-align: center;">
                <h1 class="header-title">🐾 AI จำแนกสายพันธุ์สุนัขและแมว (104 สายพันธุ์)</h1>
                <div class="header-subtitle">Deep Learning Transfer Learning (MobileNetV3) | ความแม่นยำ Top-2: 95.74% | Top-5: 99.33%</div>
                <p class="header-desc">อัปโหลดรูปภาพ, วางรูปจากอินเทอร์เน็ต (Ctrl + V), ถ่ายภาพผ่านกล้อง, หรือคลิกภาพตัวอย่างเพื่อวิเคราะห์</p>
            </div>
            """
        )

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.TabItem("📁 อัปโหลดภาพ / วางรูป (Upload & Paste)"):
                    input_upload = gr.Image(
                        type="pil",
                        label="ลากรูปมาวาง หรือ กด Ctrl+V วางรูปจากเน็ตได้ทันที",
                        sources=["upload", "clipboard"],
                        height=330
                    )
                with gr.TabItem("📸 ถ่ายภาพสดผ่านกล้อง (Webcam)"):
                    input_webcam = gr.Image(
                        type="pil",
                        label="กล้องเว็บแคม (กดอนุญาตสิทธิ์กล้องในเบราว์เซอร์)",
                        sources=["webcam"],
                        height=330
                    )
                    gr.Markdown("💡 *หมายเหตุ: หากกล้องไม่ขึ้น ให้กดไอคอนแม่กุญแจ 🔒 ที่แถบ URL ของเบราว์เซอร์ แล้วเลือก 'อนุญาตกล้อง (Allow Camera)'*")

            with gr.Row():
                btn_clear = gr.Button("🗑️ ล้างภาพ", variant="secondary")
                btn_predict = gr.Button("🔍 วิเคราะห์สายพันธุ์ (Predict)", variant="primary", scale=2)

        with gr.Column(scale=1):
            output_summary = gr.HTML("""
            <div style="background-color: #F8FAFC; border: 2px dashed #CBD5E1; border-radius: 12px; padding: 24px; text-align: center;">
                <h3 style="color: #64748B; margin: 0;">⏳ รอการอัปโหลดรูปภาพเพื่อวิเคราะห์...</h3>
                <p style="color: #94A3B8; margin-top: 8px;">ลากรูปภาพมาวาง, วางรูปจากเน็ต (Ctrl+V), หรือคลิกภาพตัวอย่างด้านล่าง</p>
            </div>
            """)
            output_label = gr.Label(
                num_top_classes=5,
                label="📊 5 อันดับสายพันธุ์ที่มั่นใจที่สุด (Top-5 Predictions)"
            )

    # Event Handlers for Upload & Clipboard
    input_upload.change(
        fn=predict_breed,
        inputs=[input_upload],
        outputs=[output_label, output_summary]
    )
    
    # Event Handlers for Webcam
    input_webcam.change(
        fn=predict_breed,
        inputs=[input_webcam],
        outputs=[output_label, output_summary]
    )

    # Predict Button
    btn_predict.click(
        fn=lambda img_up, img_cam: predict_breed(img_up if img_up is not None else img_cam),
        inputs=[input_upload, input_webcam],
        outputs=[output_label, output_summary]
    )

    # Clear Button
    def clear_inputs():
        return None, None, {}, """
        <div style="background-color: #F8FAFC; border: 2px dashed #CBD5E1; border-radius: 12px; padding: 24px; text-align: center;">
            <h3 style="color: #64748B; margin: 0;">⏳ รอการอัปโหลดรูปภาพเพื่อวิเคราะห์...</h3>
            <p style="color: #94A3B8; margin-top: 8px;">ลากรูปภาพมาวาง, วางรูปจากเน็ต (Ctrl+V), หรือคลิกภาพตัวอย่างด้านล่าง</p>
        </div>
        """

    btn_clear.click(
        fn=clear_inputs,
        outputs=[input_upload, input_webcam, output_label, output_summary]
    )

    # Example Gallery
    if examples_list:
        gr.Markdown("### 🖼️ ภาพตัวอย่างสายพันธุ์ยอดนิยม (คลิกเพื่อทดสอบได้ทันที):")
        gr.Examples(
            examples=examples_list,
            inputs=input_upload,
            outputs=[output_label, output_summary],
            fn=predict_breed,
            cache_examples=False
        )

    with gr.Accordion("ℹ️ ข้อมูลทางเทคนิคและสถิติโมเดล (Model Specifications)", open=False):
        gr.Markdown(
            """
            - **สถาปัตยกรรมโมเดล:** `MobileNetV3-Large` (Transfer Learning Pre-trained on ImageNet-1k)
            - **จำนวนคลาสทั้งหมด:** 104 สายพันธุ์ (สุนัข 68 สายพันธุ์, แมว 36 สายพันธุ์)
            - **ชุดข้อมูลฝึกสอน (Dataset):** 28,983 ภาพ ($224 \\times 224$ RGB)
            - **ผลการวัดผลบน Test Set จริง (892 ภาพ):**
              - **Top-1 Accuracy:** `87.78%`
              - **Top-2 Accuracy:** `95.74%`
              - **Top-3 Accuracy:** `97.76%`
              - **Top-5 Accuracy:** `99.33%`
            """
        )


def main():
    parser = argparse.ArgumentParser(description="Launch Dogs & Cats Breed Classifier Web App")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the app on")
    parser.add_argument("--share", action="store_true", help="Generate a public Gradio URL")
    args = parser.parse_args()

    print(f"\n[INFO] Starting Web Server on http://localhost:{args.port} ...")
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
