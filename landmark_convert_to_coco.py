import json
import os

# ====== CẤU HÌNH ======
input_json_path = r"D:\DeepLearning\Learning\Do research\replace_face\landmark_face\landmark-detection-model\out_landmarks\all_results.json"
output_coco_path = "all_results_coco_new.json"

output_coco = {
    "info": {
        "year": 2025,
        "version": "1.0",
        "description": "MY CROWD",
        "contributor": "AI Studio",
        "url": "www.ai-studio.co.kr",
        "date_created": "Tue May 13 10:03:33 ICT 2025"
    },
    "images": [],
    "annotations": [],
    "categories": [
        {
            "id": 0,
            "name": "face",
            "supercategory": "face"
        }
    ],
    "licenses": [],
    "pointTags": []
}

cnt_id = 0
cnt_obj = 0

# ====== XỬ LÝ FILE JSON ======
with open(input_json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

for item in data:
    file_name = os.path.basename(item["image"])

    # Lấy thông tin bbox đầu tiên để ước lượng width, height nếu cần
    width = 1920
    height = 1080
    if item["faces"]:
        x1, y1, x2, y2 = item["faces"][0]["box"]
        width = max(width, x2)
        height = max(height, y2)

    # Thêm ảnh
    image = {
        "id": cnt_id,
        "width": width,
        "height": height,
        "file_name": file_name
    }
    output_coco["images"].append(image)

    # Thêm annotation
    for face in item.get("faces", []):
        try:
            box = face["box"]
            landmarks = face["landmarks"]

            x_min, y_min = int(box[0]), int(box[1])
            x_max, y_max = int(box[2]), int(box[3])
            box_w = x_max - x_min
            box_h = y_max - y_min

            list_seg = [int(p) for pt in landmarks for p in pt]

            annotation = {
                "id": cnt_obj,
                "image_id": cnt_id,
                "category_id": 0,
                "segmentation": [list_seg],
                "area": box_w * box_h,
                "bbox": [x_min, y_min, box_w, box_h],
                "type": "polygon"
            }

            output_coco["annotations"].append(annotation)
            cnt_obj += 1
        except Exception as e:
            print(f"[ERROR] {file_name} - {e}")
            continue

    cnt_id += 1

# ====== LƯU KẾT QUẢ ======
with open(output_coco_path, 'w', encoding='utf-8') as f:
    json.dump(output_coco, f, ensure_ascii=False, indent=4)

print(f"✅ Đã xuất file COCO JSON tại: {output_coco_path}")
