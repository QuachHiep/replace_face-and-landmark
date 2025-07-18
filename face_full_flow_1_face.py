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
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import pandas as pd
import warnings
warnings.filterwarnings("ignore")  # Tắt tất cả các cảnh báo

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
IMAGE_EXT = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')

# TỰ ĐIỀU KHIỂN Ở ĐÂY
INPUT_DIR = r'D:\DeepLearning\Learning\Do research\replace_face\data_test\origin_image_data_26062025\New folder'
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

def classify_face(model_fair_7, face_image, device):
    """
    Hàm phân loại giới tính, độ tuổi và chủng tộc
    """

    # Tiền xử lý ảnh trước khi phân loại
    trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    face_tensor = trans(face_image).unsqueeze(0).to(device)

    with torch.no_grad():
        # Dự đoán từ mô hình
        outputs = model_fair_7(face_tensor)
        outputs = outputs.cpu().detach().numpy()
        outputs = np.squeeze(outputs)

        # Tách ra các lớp race, gender và age
        race_outputs = outputs[:7]  # 7 lớp chủng tộc
        gender_outputs = outputs[7:9]  # 2 lớp giới tính
        age_outputs = outputs[9:18]  # 9 lớp độ tuổi

        # Tính toán xác suất cho mỗi lớp
        race_score = np.exp(race_outputs) / np.sum(np.exp(race_outputs))
        gender_score = np.exp(gender_outputs) / np.sum(np.exp(gender_outputs))
        age_score = np.exp(age_outputs) / np.sum(np.exp(age_outputs))

        # Dự đoán chủng tộc, giới tính và độ tuổi
        race_pred = np.argmax(race_score)
        gender_pred = np.argmax(gender_score)
        age_pred = np.argmax(age_score)

        # Chuyển kết quả dự đoán thành nhãn
        race_labels = ['White', 'Black', 'Latino_Hispanic', 'East Asian', 'Southeast Asian', 'Indian', 'Middle Eastern']
        gender_labels = ['Male', 'Female']
        age_labels = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']

        race = race_labels[race_pred]
        gender = gender_labels[gender_pred]
        age = age_labels[age_pred]

        return race, gender, age

def get_random_image_from_folder(folder_path):
    # Lấy danh sách các tệp ảnh trong thư mục
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp'))]
    
    # Nếu không có tệp ảnh nào trong thư mục, trả về None
    if not image_files:
        return None

    # Chọn ngẫu nhiên một tệp ảnh từ danh sách
    selected_image = random.choice(image_files)
    
    # Trả về đường dẫn đầy đủ của ảnh
    return os.path.join(folder_path, selected_image)

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

def process_image(img_path, yolo, model, device, out_path, f2f, source_paths, model_fair_7, collect_results=None):
    # BƯỚC 1: Đọc ảnh khuôn mặt
    img = cv2.imread(img_path)

    # BƯỚC 2: Phân loại khuôn mặt để quyết định nguồn ảnh
    race_pred, gender_pred, age_pred = classify_face(model_fair_7, img, device)

    # Lựa chọn ảnh nguồn dựa trên kết quả phân loại
    selected_source = select_source_based_on_classification(gender_pred, age_pred, race_pred, source_paths)

    if selected_source is None:
        print(f"Không tìm thấy ảnh nguồn phù hợp cho {age_pred}_{gender_pred}_{race_pred}")
        #print(f"Không tìm thấy ảnh nguồn phù hợp cho {os.path.basename(path)}")
        return

    # Chọn ngẫu nhiên một ảnh trong thư mục nguồn đã chọn
    selected_image = get_random_image_from_folder(selected_source)

    # BƯỚC 3: Hoán đổi mặt với ảnh nguồn đã chọn
    swapped_img = f2f.swap_img_to_img(selected_image, img_path)

    if swapped_img is None:
        print(f"Không thể hoán đổi: {os.path.basename(img_path)}")
        return

    # BƯỚC 4: Dự đoán landmark và lưu kết quả
    res = yolo(swapped_img)[0]
    boxes = res.boxes.xyxy.cpu().numpy() if res.boxes else np.array([])
    scores = res.boxes.conf.cpu().numpy() if res.boxes else np.array([])

    if boxes is None or boxes.size == 0:
        print(f"Không phát hiện khuôn mặt: {os.path.basename(img_path)}")
        return

    total = 0
    t0 = time.time()
    landmark_data = []

    for box, conf in zip(boxes, scores):
        if conf < CONF_THRESH:
            continue
        tensor, center, scale = preprocess(swapped_img, box)
        with torch.no_grad():
            out = model(tensor.to(device))
        preds, _ = get_final_preds(out.cpu(), torch.tensor([center]), torch.tensor([scale]))
        landmarks = preds[0].tolist()
        total += len(landmarks)
        landmark_data.append({
            "box": [int(i) for i in box],
            "score": float(conf),
            "landmarks": landmarks
        })

    elapsed = time.time() - t0
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, swapped_img)

    if collect_results is not None:
        collect_results.append({
            "image": os.path.relpath(out_path, OUT_DIR),
            "faces": landmark_data
        })

    print(f"{os.path.basename(img_path)} ({total} điểm, {elapsed:.2f}s)")

def parse_metadata_from_folder(folder_name):
    # Giả sử tên thư mục có định dạng "age_pred_gender_pred_race_pred"
    parts = folder_name.split('_')
    if len(parts) == 3:
        age_pred = parts[0]  # Ví dụ: "0-2"
        gender_pred = parts[1]  # Ví dụ: "Female"
        race_pred = parts[2]  # Ví dụ: "Indian"
        return age_pred, gender_pred, race_pred
    return None, None, None  # Nếu tên thư mục không hợp lệ

def is_age_in_range(age_pred, source_age):
    """
    Kiểm tra xem age_pred có nằm trong khoảng của source_age hay không.
    Ví dụ: age_pred = '3-9', source_age = '0-10' thì phù hợp.
    """
    # Chuyển đổi độ tuổi thành các giá trị đầu và cuối
    age_pred_start, age_pred_end = map(int, age_pred.split('-'))  # Chia và chuyển đổi thành số
    source_age_start, source_age_end = map(int, source_age.split('-'))  # Làm tương tự cho source_age
    
    # Kiểm tra xem age_pred có nằm trong khoảng source_age không
    return source_age_start <= age_pred_start and source_age_end >= age_pred_end

def select_source_based_on_classification(gender_pred, age_pred, race_pred, source_paths):
    for source_path in source_paths:
        folder_name = os.path.basename(source_path)  # Lấy tên thư mục từ đường dẫn

        # Phân tích thông tin từ tên thư mục
        source_age, source_gender, source_race = parse_metadata_from_folder(folder_name)

        # So sánh các giá trị với các dự đoán
        if source_gender == gender_pred and source_race == race_pred and is_age_in_range(age_pred, source_age):
            return source_path  # Trả về đường dẫn của thư mục nguồn
    return None  # Không tìm thấy phù hợp

def process_images_flat(input_dir, out_dir, yolo, model, device, f2f, source_paths, model_fair_7):
    results = []
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(input_dir):
        if f.lower().endswith(IMAGE_EXT):
            in_path = os.path.join(input_dir, f)
            out_path = os.path.join(out_dir, f)
            process_image(in_path, yolo, model, device, out_path, f2f, source_paths, model_fair_7, results)
    with open(os.path.join(out_dir, 'all_results.json'), 'w') as jf:
        json.dump(results, jf, indent=2)

def main():
    update_config(config, type('Namespace', (), {'cfg': CFG})())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    yolo, model_landmark = load_models(device)

    print("Khởi tạo Face2Face...")
    try:
        #f2f = Face2Face(device_id=0 if torch.cuda.is_available() else None)
        f2f = Face2Face(device_id=None)
    except:
        f2f = Face2Face()

    # Lấy các thư mục con từ SOURCE_FOLDER
    source_paths = [os.path.join(SOURCE_FOLDER, folder) for folder in os.listdir(SOURCE_FOLDER)
                    if os.path.isdir(os.path.join(SOURCE_FOLDER, folder))]  # Kiểm tra xem có phải là thư mục

    if not source_paths:
        print("Không tìm thấy thư mục nguồn trong SOURCE_FOLDER.")
        return

    model_fair_7 = torchvision.models.resnet34(pretrained=True)
    model_fair_7.fc = nn.Linear(model_fair_7.fc.in_features, 18)
    model_fair_7.load_state_dict(torch.load('fair_face_models/res34_fair_align_multi_7_20190809.pt', map_location=torch.device('cpu')))
    model_fair_7 = model_fair_7.to(device)
    model_fair_7.eval()

    if FLAT_MODE:
        process_images_flat(INPUT_DIR, OUT_DIR, yolo, model_landmark, device, f2f, source_paths, model_fair_7)

if __name__ == '__main__':
    main()
