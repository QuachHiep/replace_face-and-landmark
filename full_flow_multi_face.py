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
INPUT_DIR = r'E:\hiepqn\repalce_face\classify\data\Input_image\Young adults 16~35(female)'
SOURCE_FOLDER = r'E:\hiepqn\repalce_face\classify\data\New folder'
OUT_DIR = r'E:\hiepqn\repalce_face\classify\data\Output_image'
FLAT_MODE = True  # True = ảnh nằm trong thư mục chính, False = có thư mục con

# ==================== MODEL ====================
def load_models(device):
    yolo = YOLO(YOLO_WEIGHTS)
    model = hrnet.get_face_alignment_net(config)
    state = torch.load(HRNET_WEIGHTS, map_location='cpu')
    state = {k.replace('module.', ''): v for k, v in state.get('state_dict', state).items()}
    model.load_state_dict(state, strict=False)
    return yolo, model.to(device).eval()

def classify_face(model_fair_7, face_tensor, device):
    """
    Hàm phân loại giới tính, độ tuổi và chủng tộc từ một TENSOR đã được tiền xử lý.
    """
    
    # BỎ TOÀN BỘ PHẦN TIỀN XỬ LÝ Ở ĐÂY.
    # face_tensor đã được tạo và chuẩn hóa sẵn từ hàm preprocess_for_classification.
    
    with torch.no_grad():
        # Đưa tensor lên device (nếu nó chưa ở đó) và thực hiện dự đoán
        outputs = model_fair_7(face_tensor.to(device))
        outputs = outputs.cpu().detach().numpy()
        outputs = np.squeeze(outputs)

        # Tách ra các lớp race, gender và age
        race_outputs = outputs[:7]      # 7 lớp chủng tộc
        gender_outputs = outputs[7:9]   # 2 lớp giới tính
        age_outputs = outputs[9:18]     # 9 lớp độ tuổi

        # Tính toán xác suất cho mỗi lớp (softmax)
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

def preprocess_for_classification(img, box, padding=1, save_path=None):
    """
    Hàm tiền xử lý ảnh cho mô hình PHÂN LOẠI.
    - Cắt khuôn mặt, thêm padding.
    - (MỚI) Lưu lại ảnh đã cắt nếu có `save_path`.
    - Chuẩn hóa và chuyển thành tensor.
    """
    x1, y1, x2, y2 = map(int, box)
    
    width = x2 - x1
    height = y2 - y1
    pad_x = int(width * padding)
    pad_y = int(height * padding)
    
    x1_pad = max(0, x1 - pad_x)
    y1_pad = max(0, y1 - pad_y)
    x2_pad = min(img.shape[1], x2 + pad_x)
    y2_pad = min(img.shape[0], y2 + pad_y)

    # Cắt khuôn mặt với padding
    face_padded = img[y1_pad:y2_pad, x1_pad:x2_pad]
    
    # ==================== THAY ĐỔI Ở ĐÂY ====================
    # Lưu lại ảnh đã được cắt và padding để kiểm tra
    if save_path:
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, face_padded)
    # ========================================================

    # Các bước tiền xử lý còn lại giữ nguyên
    face = cv2.resize(face_padded, (SIZE[1], SIZE[0]))
    face = (face[..., ::-1] / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    
    tensor = torch.from_numpy(face.transpose(2, 0, 1)).unsqueeze(0).float()
    
    return tensor

def preprocess_for_landmark(img, box):
    """
    Hàm tiền xử lý ảnh cho mô hình LANDMARK.
    - Cắt khuôn mặt theo đúng bounding box gốc (KHÔNG padding).
    - Chuẩn hóa và chuyển thành tensor.
    - Trả về center và scale cho hàm get_final_preds.
    """
    x1, y1, x2, y2 = map(int, box)
    
    # Cắt khuôn mặt chính xác theo box của YOLO
    face = img[y1:y2, x1:x2]
    
    # Tiền xử lý
    face = cv2.resize(face, (SIZE[1], SIZE[0]))
    face = (face[..., ::-1] / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    
    # Chuyển thành tensor
    tensor = torch.from_numpy(face.transpose(2, 0, 1)).unsqueeze(0).float()
    
    # Trả về các giá trị cần thiết cho get_final_preds
    # Center là góc trên bên trái, Scale là kích thước (width, height)
    center = (x1, y1)
    scale = (x2 - x1, y2 - y1)

    return tensor, center, scale

def process_image(img_path, yolo, model, device, out_path, f2f, source_paths, model_fair_7, collect_results=None):
    # BƯỚC 1: Đọc ảnh gốc
    img = cv2.imread(img_path)

    # BƯỚC 2: Dùng YOLO phát hiện các khuôn mặt
    res = yolo(img)[0]
    boxes = res.boxes.xyxy.cpu().numpy() if res.boxes else np.array([])
    scores = res.boxes.conf.cpu().numpy() if res.boxes else np.array([])

    if boxes is None or boxes.size == 0:
        print(f"Không phát hiện khuôn mặt trong ảnh: {os.path.basename(img_path)}")
        return

    total = 0
    t0 = time.time()
    landmark_data = []
    swapped_img = img.copy() # Bắt đầu với một bản sao của ảnh gốc
    did_swap = False # Cờ để kiểm tra xem có lần swap nào thành công không

    # Bước 3: Duyệt qua từng mặt đã phát hiện
    for idx, (box, conf) in enumerate(zip(boxes, scores)):
        if conf < CONF_THRESH:
            continue

        # --- PHẦN PHÂN LOẠI (CLASSIFY) ---
        # Tiền xử lý mặt với PADDING để phân loại
        # Tạo đường dẫn để lưu ảnh mặt đã padding
        debug_save_dir = os.path.join(OUT_DIR, 'padded_faces_for_debug')
        # Lấy tên file gốc không có đuôi
        base_filename = os.path.splitext(os.path.basename(img_path))[0]
        # Tạo tên file mới
        save_debug_path = os.path.join(debug_save_dir, f"{base_filename}_face_{idx}.jpg")
        
        # Tiền xử lý mặt với PADDING và truyền đường dẫn LƯU ẢNH
        tensor_for_classify = preprocess_for_classification(img, box, padding=0.5, save_path=save_debug_path)
        # Tiến hành phân loại
        race_pred, gender_pred, age_pred = classify_face(model_fair_7, tensor_for_classify, device)

        # Lựa chọn thư mục nguồn
        selected_source = select_source_based_on_classification(gender_pred, age_pred, race_pred, source_paths)
        if selected_source is None:
            print(f"Không tìm thấy face nguồn phù hợp cho {age_pred}_{gender_pred}_{race_pred}")
            continue
        else:
            print(f"Face nguồn được chọn cho {age_pred}_{gender_pred}_{race_pred}: {selected_source}")

        # Chọn ngẫu nhiên một ảnh từ thư mục nguồn
        selected_image = get_random_image_from_folder(selected_source)
        if selected_image is None:
            print(f"Không có ảnh nguồn trong thư mục {selected_source}.")
            continue

        # --- PHẦN HOÁN ĐỔI MẶT (SWAP) ---
        # Hoán đổi mặt, kết quả được ghi đè lên swapped_img
        # Lưu ý: Hàm swap của bạn có thể cần điều chỉnh để nhận và trả về ảnh (mảng numpy) thay vì đường dẫn
        # Ở đây giả sử nó trả về ảnh đã swap
        temp_swapped_img = f2f.swap_img_to_img(selected_image, swapped_img)
        
        if temp_swapped_img is None:
            print(f"Không thể hoán đổi mặt cho box {idx} trong ảnh: {os.path.basename(img_path)}")
            continue
        
        swapped_img = temp_swapped_img # Cập nhật ảnh đã swap thành công
        did_swap = True

        # --- PHẦN TÌM LANDMARK ---
        # Bước 5: Dự đoán landmarks trên mặt đã hoán đổi
        with torch.no_grad():
            # Tiền xử lý mặt đã swap, KHÔNG PADDING
            tensor, center, scale = preprocess_for_landmark(swapped_img, box)
            
            out = model(tensor.to(device))
            preds, _ = get_final_preds(out.cpu(), torch.tensor([center]), torch.tensor([scale]))
            
            landmarks = preds[0].tolist()
            total += len(landmarks)
            landmark_data.append({
                "box": [int(i) for i in box],
                "score": float(conf),
                "landmarks": landmarks
            })

    # Lưu kết quả cuối cùng nếu có ít nhất một lần swap thành công
    if did_swap:
        elapsed = time.time() - t0
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, swapped_img)

        if collect_results is not None:
            collect_results.append({
                "image": os.path.relpath(out_path, OUT_DIR),
                "faces": landmark_data
            })
        print(f"{os.path.basename(img_path)} ({total} điểm, {elapsed:.2f}s)")
    else:
        print(f"Không có khuôn mặt nào được hoán đổi cho ảnh: {os.path.basename(img_path)}")


def parse_metadata_from_folder(folder_name):
    # Giả sử tên thư mục có định dạng "age_pred_gender_pred_race_pred"
    parts = folder_name.split('_')
    if len(parts) == 2:
        age_pred = parts[0]  # Ví dụ: "0-2"
        gender_pred = parts[1]  # Ví dụ: "Female"
        #race_pred = parts[2]   # Ví dụ: "Indian"
        race_pred = None
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
        #if source_gender == gender_pred and source_race == race_pred and is_age_in_range(age_pred, source_age):
        if source_gender == gender_pred and is_age_in_range(age_pred, source_age):
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
        f2f = Face2Face(device_id=None)  # Dùng CPU
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
