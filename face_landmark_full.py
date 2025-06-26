import os
import sys
import time
import json
import cv2
import torch
import random
import numpy as np
from ultralytics import YOLO
from face2face import Face2Face  # IMPORT THÊM

sys.path.append(os.getcwd())
from lib.config import config, update_config
from lib.models import hrnet
from lib.utils.utils import get_final_preds
from lib.utils.transforms import get_affine_transform

# ==================== CẤU HÌNH ====================
CFG = 'experiments/300w/face_alignment_300w_hrnet_w18.yaml'
HRNET_WEIGHTS = 'hrnetv2_pretrained/HR18-300W.pth'
YOLO_WEIGHTS = 'yolov8-face/weights/yolov11l-face.pt'
SIZE = (256, 256)
CONF_THRESH = 0.7
IMAGE_EXT = ('.jpg', '.jpeg', '.png')

# TỰ ĐIỀU KHIỂN Ở ĐÂY
INPUT_DIR = r'D:\DeepLearning\Learning\Do research\replace_face\data_test\241119_IN'
SOURCE_FOLDER = r'D:\DeepLearning\Learning\Do research\replace_face\face2face_project\replace_face\image_source'
OUT_DIR = 'out_landmarks'
FLAT_MODE = True  # True = ảnh nằm trong thư mục chính, False = có thư mục con

# ==================== MODEL ====================
def load_models(device):
    yolo = YOLO(YOLO_WEIGHTS)
    model = hrnet.get_face_alignment_net(config)
    state = torch.load(HRNET_WEIGHTS, map_location='cpu')
    state = {k.replace('module.', ''): v for k, v in state.get('state_dict', state).items()}
    model.load_state_dict(state, strict=False)
    return yolo, model.to(device).eval()

def preprocess(img, box):
    x1, y1, x2, y2 = map(int, box)
    center = np.array([(x1 + x2) / 2, (y1 + y2) / 2], np.float32)
    scale_val = max(x2 - x1, y2 - y1) / 200 * 1.25
    scale = np.array([scale_val, scale_val], dtype=np.float32)
    trans = get_affine_transform(center, scale, 0, SIZE)
    crop = cv2.warpAffine(img, trans, SIZE, flags=cv2.INTER_LINEAR)
    crop = (crop[..., ::-1] / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    tensor = torch.from_numpy(crop.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor, center, scale

def process_image(path, yolo, model, device, out_path, f2f, source_paths, collect_results=None):
    # BƯỚC 1: Hoán đổi mặt
    random_source = random.choice(source_paths)
    swapped_img = f2f.swap_img_to_img(random_source, path)

    if swapped_img is None:
        print(f"Không thể hoán đổi: {os.path.basename(path)}")
        return

    # BƯỚC 2: Dự đoán landmark
    img = swapped_img
    res = yolo(img)[0]
    boxes = res.boxes.xyxy.cpu().numpy() if res.boxes else np.array([])
    scores = res.boxes.conf.cpu().numpy() if res.boxes else np.array([])

    if boxes is None or boxes.size == 0:
        print(f"Không phát hiện khuôn mặt: {os.path.basename(path)}")
        return

    total = 0
    t0 = time.time()
    landmark_data = []

    for box, conf in zip(boxes, scores):
        if conf < CONF_THRESH:
            continue
        tensor, center, scale = preprocess(img, box)
        with torch.no_grad():
            out = model(tensor.to(device))
        preds, _ = get_final_preds(out.cpu(), torch.tensor([center]), torch.tensor([scale]))
        landmarks = preds[0].tolist()
        for x, y in landmarks:
            cv2.circle(img, (int(x), int(y)), 3, (0, 0, 0), -1)
            cv2.circle(img, (int(x), int(y)), 2, (0, 255, 255), -1)
            total += 1
        landmark_data.append({
            "box": [int(i) for i in box],
            "score": float(conf),
            "landmarks": landmarks
        })

    elapsed = time.time() - t0
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.putText(img, f"{total} pts | {elapsed:.2f}s", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imwrite(out_path, img)

    if collect_results is not None:
        collect_results.append({
            "image": os.path.relpath(out_path, OUT_DIR),
            "faces": landmark_data
        })

    print(f"{os.path.basename(path)} ({total} điểm, {elapsed:.2f}s)")


def process_images_flat(input_dir, out_dir, yolo, model, device, f2f, source_paths):
    results = []
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(input_dir):
        if f.lower().endswith(IMAGE_EXT):
            in_path = os.path.join(input_dir, f)
            out_path = os.path.join(out_dir, f)
            process_image(in_path, yolo, model, device, out_path, f2f, source_paths, results)
    with open(os.path.join(out_dir, 'all_results.json'), 'w') as jf:
        json.dump(results, jf, indent=2)

def process_images_nested(input_dir, out_dir, yolo, model, device, f2f, source_paths):
    results = []
    for root, _, files in os.walk(input_dir):
        rel_dir = os.path.relpath(root, input_dir)
        out_subdir = os.path.join(out_dir, rel_dir)
        os.makedirs(out_subdir, exist_ok=True)
        for f in files:
            if f.lower().endswith(IMAGE_EXT):
                in_path = os.path.join(root, f)
                out_path = os.path.join(out_subdir, f)
                process_image(in_path, yolo, model, device, out_path, f2f, source_paths, results)
    with open(os.path.join(out_dir, 'all_results.json'), 'w') as jf:
        json.dump(results, jf, indent=2)

def main():
    update_config(config, type('Namespace', (), {'cfg': CFG})())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    yolo, model = load_models(device)

    print("Khởi tạo Face2Face...")
    try:
        f2f = Face2Face(device_id=0 if torch.cuda.is_available() else None)
    except:
        f2f = Face2Face()

    source_paths = [os.path.join(SOURCE_FOLDER, f) for f in os.listdir(SOURCE_FOLDER)
                    if f.lower().endswith(IMAGE_EXT)]

    if not source_paths:
        print("Không tìm thấy ảnh nguồn trong SOURCE_FOLDER.")
        return

    if FLAT_MODE:
        process_images_flat(INPUT_DIR, OUT_DIR, yolo, model, device, f2f, source_paths)
    else:
        process_images_nested(INPUT_DIR, OUT_DIR, yolo, model, device, f2f, source_paths)


if __name__ == '__main__':
    main()
