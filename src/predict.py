import os
import sys
import json
import argparse
from pathlib import Path

# Reconfigure stdout/stderr for Unicode safety on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
from PIL import Image

from src.dataset import get_transforms
from src.model import build_model


def predict(image_path: str | Path,
            checkpoint_path: str | Path = 'models/best_model.pth',
            top_k: int = 5,
            device: str = 'auto'):
    """
    Predict the top_k classes for a given image.
    """
    img_path = Path(image_path)
    if not img_path.exists():
        print(f'Error: Image file {img_path} not found.')
        sys.exit(1)

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        print(f'Error: Checkpoint file {ckpt_path} not found.')
        sys.exit(1)

    if device == 'auto':
        torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        torch_device = torch.device(device)

    # 1. Load Checkpoint
    checkpoint = torch.load(ckpt_path, map_location=torch_device, weights_only=False)
    class_to_idx = checkpoint.get('class_to_idx')
    if not class_to_idx:
        mapping_file = ckpt_path.parent / 'class_mapping.json'
        if mapping_file.exists():
            with open(mapping_file, 'r', encoding='utf-8') as f:
                class_to_idx = json.load(f)
        else:
            print('Error: Could not find class mapping.')
            sys.exit(1)

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)
    model_name = checkpoint.get('model_name', 'mobilenetv3_large_100')

    # 2. Build Model & Load Weights
    model = build_model(model_name=model_name, num_classes=num_classes, pretrained=False).to(torch_device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # 3. Load and Transform Image
    _, eval_transform = get_transforms()
    image = Image.open(img_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    input_tensor = eval_transform(image).unsqueeze(0).to(torch_device)

    # 4. Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        top_probs, top_indices = torch.topk(probabilities, k=min(top_k, num_classes))

    top_probs = top_probs.cpu().numpy() * 100.0
    top_indices = top_indices.cpu().numpy()

    print('\n' + '='*60)
    print(f'  PREDICTION RESULTS FOR: {img_path.name}')
    print('='*60)
    for rank, (idx, prob) in enumerate(zip(top_indices, top_probs), start=1):
        class_name = idx_to_class[idx]
        bar = '#' * int(prob / 4)
        print(f' {rank}. {class_name:<30} {prob:6.2f}%  {bar}')
    print('='*60 + '\n')

    return [{'rank': rank, 'class': idx_to_class[idx], 'probability': float(prob)}
            for rank, (idx, prob) in enumerate(zip(top_indices, top_probs), start=1)]


def main():
    parser = argparse.ArgumentParser(description='Predict Dog & Cat Breed from Image')
    parser.add_argument('--image', type=str, required=True, help='Path to image file')
    parser.add_argument('--checkpoint', type=str, default='models/best_model.pth', help='Path to model checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top classes to display')
    parser.add_argument('--device', type=str, default='auto', help='Device (cuda/cpu/auto)')

    args = parser.parse_args()
    predict(args.image, args.checkpoint, args.top_k, args.device)


if __name__ == '__main__':
    main()
