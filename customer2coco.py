import json
import os


def get_json_files(folder_path):
    json_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files


list_class = []
root_packet_folder = "Mobis_JG1_SD_pack3"
list_json_file = get_json_files(root_packet_folder)
# get list class
for json_file in list_json_file:
    with open(json_file, 'r') as f:
        data = json.load(f)
        for item in data["shapes"]:
            if item["label"] not in list_class:
                list_class.append(item["label"])

output_coco_path = root_packet_folder + "_coco_import.json"
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
    "categories": [],
    "licenses": [],
    "pointTags": []
}

categories = [{"id": idx, "name": item} for idx, item in enumerate(list_class)]

output_coco["categories"] = categories

cnt_id = 0
cnt_obj = 0
for json_file in list_json_file:
    with open(json_file, 'r') as f:
        data = json.load(f)
        # create output images
        file_name = data["imagePath"]
        image = {
            "id": cnt_id,
            "width": data["imageWidth"],
            "height": data["imageHeight"],
            "file_name": file_name
        }
        output_coco["images"].append(image)
        # cnt_id += 1

        # get annotations
        # list_objects = data["frames"][0]["objects"]
        list_objects = data["shapes"]
        for obj in list_objects:
            try:
                annotation = {
                    "id": cnt_obj,
                    "image_id": cnt_id,
                    "category_id": list_class.index(obj["label"]),
                    "segmentation": [],
                    "area": 0,
                    "bbox":[],
                    "type": obj["shape_type"]
                }
                # get point
                list_x = []
                list_y = []
                list_seg = []
                for point in obj["points"]:
                    list_seg.append(int(point[0]))
                    list_x.append(int(point[0]))
                    list_seg.append(int(point[1]))
                    list_y.append(int(point[1]))
                # get bbox
                annotation["bbox"] = [min(list_x), min(list_y), max(list_x)-min(list_x), max(list_y)-min(list_y)]
                annotation["area"] = (annotation["bbox"][2]- annotation["bbox"][0])* (annotation["bbox"][3]-annotation["bbox"][1])
                annotation["segmentation"] = [list_seg]
                output_coco["annotations"].append(annotation)
                cnt_obj += 1
            except:
                print("ERROR")
                continue
    cnt_id += 1
with open(output_coco_path, 'w', encoding='utf-8') as f:
    json.dump(output_coco, f, ensure_ascii=False, indent=4)
