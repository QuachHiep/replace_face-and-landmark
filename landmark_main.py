import os
import sys
import time
import json
import cv2
import torch
import argparse
import numpy as np
from ultralytics import YOLO

sys.path.append(os.getcwd())
from lib.config import config, update_config
from lib.models import hrnet
from lib.utils.utils import get_final_preds
from lib.utils.transforms import get_affine_transform

cuda_available = torch.cuda.is_available()
# Print the result
print(f"CUDA Available: {cuda_available}")
CFG = 'experiments/300w/face_alignment_300w_hrnet_w18.yaml'
#HRNET_WEIGHTS = 'hrnetv2_pretrained/model_best.pth'
HRNET_WEIGHTS = 'hrnetv2_pretrained/model_ver2.pth'
YOLO_WEIGHTS = 'yolov8-face/weights/yolov11l-face.pt'
INPUT_DIR = r'D:\DeepLearning\Learning\Do research\replace_face\face2face_project\replace_face\results\images'
OUT_DIR = 'out_landmarks'
SIZE = (256, 256)
CONF_THRESH = 0.7

os.makedirs(OUT_DIR, exist_ok=True)

def load_models(device):
    """Load mô hình YOLO và HRNet với trọng số, chuyển sang eval mode."""
    yolo = YOLO(YOLO_WEIGHTS)
    model = hrnet.get_face_alignment_net(config)
    # state = torch.load(HRNET_WEIGHTS, map_location='cpu')
    # #state = torch.load(HRNET_WEIGHTS, map_location=torch.device('cpu'))
    # state = {k.replace('module.', ''): v for k, v in state.get('state_dict', state).items()}
    # model.load_state_dict(state, strict=False)
    # Load model weights
    state_dict = torch.load(HRNET_WEIGHTS, map_location='cpu')
    
    # If 'state_dict' is in the loaded dictionary, use that
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict)
    else:
        # If the model weights are not wrapped in 'state_dict', load them directly
        model.load_state_dict(state_dict)
    
    # Transfer the model to the selected device (GPU or CPU)
    model = model.to(device)
    
    # Switch model to evaluation mode
    model.eval()
    # return yolo, model.to(device).eval()
    return yolo, model

def preprocess(img, box):
    """Cắt và chuẩn hóa vùng khuôn mặt từ bounding box."""
    x1, y1, x2, y2 = map(int, box)
    center = np.array([(x1 + x2) / 2, (y1 + y2) / 2], np.float32)
    scale_val = max(x2 - x1, y2 - y1) / 200 * 1.25
    scale = np.array([scale_val, scale_val], dtype=np.float32)
    trans = get_affine_transform(center, scale, 0, SIZE)
    crop = cv2.warpAffine(img, trans, SIZE, flags=cv2.INTER_LINEAR)
    if crop is None:
        return None, None, None
    crop = (crop[..., ::-1] / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    tensor = torch.from_numpy(crop.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor, center, scale

def process_image(path, yolo, model, device, output_subdir, collect_results=None):
    """Xử lý ảnh: phát hiện mặt, dự đoán landmark, lưu kết quả hình và JSON."""
    img = cv2.imread(path)
    if img is None:
        print(f"Lỗi ảnh: {path}")
        return

    res = yolo(img)[0]
    boxes = res.boxes.xyxy.cpu().numpy() if res.boxes else []
    scores = res.boxes.conf.cpu().numpy() if res.boxes else []

    if len(boxes) == 0:
        print(f"Không mặt: {os.path.basename(path)}")
        return

    total = 0
    t0 = time.time()
    landmark_data = []

    for box, conf in zip(boxes, scores):
        if conf < CONF_THRESH:
            continue

        tensor, center, scale = preprocess(img, box)
        if tensor is None:
            continue

        with torch.no_grad():
            out = model(tensor.to(device))

        preds, _ = get_final_preds(out.cpu(), torch.tensor([center]), torch.tensor([scale]))
        landmarks = preds[0].tolist()

        for x, y in landmarks:
            cv2.circle(img, (int(x), int(y)), 6, (0, 0, 0), -1)
            cv2.circle(img, (int(x), int(y)), 4, (0, 255, 255), -1)
            total += 1

        x1, y1, x2, y2 = map(int, box)
        # BỎ bounding box đỏ (không còn dòng cv2.rectangle)

        landmark_data.append({
            "box": [x1, y1, x2, y2],
            "score": float(conf),
            "landmarks": landmarks
        })

    elapsed = time.time() - t0
    filename = os.path.basename(path)
    out_path = os.path.join(output_subdir, filename)
    cv2.putText(img, f"{total} pts | {elapsed:.2f}s", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imwrite(out_path, img)

    if collect_results is not None:
        rel_dir = os.path.relpath(output_subdir, OUT_DIR)
        collect_results.append({
            "image": os.path.join(rel_dir, filename),
            "faces": landmark_data
        })
    print(f"{filename} ({total} điểm, {elapsed:.2f}s)")

def main():
    """Hàm chính xử lý toàn bộ ảnh và xuất JSON."""
    update_config(config, argparse.Namespace(cfg=CFG))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    yolo, model = load_models(device)

    all_results = []
    for root, _, files in os.walk(INPUT_DIR):
        rel_dir = os.path.relpath(root, INPUT_DIR)
        out_subdir = os.path.join(OUT_DIR, rel_dir)
        os.makedirs(out_subdir, exist_ok=True)

        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(root, f)
                process_image(img_path, yolo, model, device, out_subdir, collect_results=all_results)

    json_path = os.path.join(OUT_DIR, 'all_results.json')
    with open(json_path, 'w') as jf:
        json.dump(all_results, jf, indent=2)

if __name__ == '__main__':
    main()
