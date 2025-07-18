import cv2
import os
import random
from face2face import Face2Face

# ==============================================================================
# PHẦN CẤU HÌNH
# ==============================================================================

source_folder = r"D:\DeepLearning\Learning\Do research\replace_face\face2face_project\replace_face\image_source"
input_folder = r"D:\DeepLearning\Learning\Do research\replace_face\data_test\20230707_2"
output_folder = "output_results"
image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')

# ==============================================================================
# PHẦN XỬ LÝ CHÍNH
# ==============================================================================

print("Đang khởi tạo Face2Face...")
try:
    f2f = Face2Face(device_id=0)
    print("Face2Face đã khởi tạo trên GPU.")
except Exception as e:
    print(f"Không thể khởi tạo trên GPU, đang thử với CPU. Lỗi: {e}")
    f2f = Face2Face()
    print("Face2Face đã khởi tạo trên CPU.")

print(f"Đang kiểm tra thư mục đầu ra: '{output_folder}'")
os.makedirs(output_folder, exist_ok=True)

print(f"Đang tải danh sách các ảnh nguồn từ: '{source_folder}'")
try:
    source_filenames = os.listdir(source_folder)
    source_image_paths = [
        os.path.join(source_folder, f)
        for f in source_filenames if f.lower().endswith(image_extensions)
    ]
    if not source_image_paths:
        print(f"Lỗi: Không tìm thấy ảnh nguồn hợp lệ trong '{source_folder}'")
        exit()
    print(f"Tìm thấy {len(source_image_paths)} ảnh nguồn hợp lệ.")
except FileNotFoundError:
    print(f"Lỗi: Thư mục nguồn '{source_folder}' không tồn tại.")
    exit()

print(f"Đang tìm ảnh trong thư mục con của: '{input_folder}'")
# Duyệt toàn bộ file ảnh trong thư mục con
all_target_images = []
for root, _, files in os.walk(input_folder):
    for file in files:
        if file.lower().endswith(image_extensions):
            full_input_path = os.path.join(root, file)
            # Tính đường dẫn output tương ứng
            relative_path = os.path.relpath(full_input_path, input_folder)
            full_output_path = os.path.join(output_folder, relative_path)
            all_target_images.append((full_input_path, full_output_path))

if not all_target_images:
    print("Lỗi: Không tìm thấy ảnh nào trong các thư mục con.")
    exit()

print(f"Tổng cộng {len(all_target_images)} ảnh đích sẽ được xử lý.")

# --- Bắt đầu xử lý từng ảnh ---
print("\nBắt đầu quá trình hoán đổi khuôn mặt hàng loạt...")
for full_input_path, full_output_path in all_target_images:
    filename = os.path.basename(full_input_path)
    print(f"\nĐang xử lý ảnh đích: '{filename}'")

    try:
        random_source_path = random.choice(source_image_paths)
        print(f" -> Sử dụng ảnh nguồn: '{os.path.basename(random_source_path)}'")

        swapped_image = f2f.swap_img_to_img(random_source_path, full_input_path)

        if swapped_image is not None:
            os.makedirs(os.path.dirname(full_output_path), exist_ok=True)
            cv2.imwrite(full_output_path, swapped_image)
            print(f" -> Hoán đổi thành công! Lưu tại: '{full_output_path}'")
        else:
            print(" -> Không thể hoán đổi: Không phát hiện được khuôn mặt.")

    except Exception as e:
        print(f" -> Lỗi khi xử lý '{filename}': {e}")

print("\n=========================================")
print("Quá trình xử lý hàng loạt đã hoàn tất!")
print(f"Kiểm tra kết quả trong thư mục: '{output_folder}'")
print("=========================================")
