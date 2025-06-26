import cv2
from face2face import Face2Face

# 1. Khởi tạo đối tượng Face2Face
# Đảm bảo bạn đã cài đặt thư viện: pip install socaity-face2face
# device_id=0 có nghĩa là sử dụng GPU đầu tiên. Thư viện sẽ tự động dùng CPU nếu không có GPU.
try:
    f2f = Face2Face(device_id=0)
    print("Face2Face được khởi tạo thành công trên GPU.")
except Exception as e:
    print(f"Không thể khởi tạo trên GPU, đang thử với CPU. Lỗi: {e}")
    f2f = Face2Face() # Khởi tạo mặc định trên CPU
    print("Face2Face được khởi tạo thành công trên CPU.")

# 2. Xác định đường dẫn đến ảnh nguồn và ảnh đích
# Ảnh nguồn (src): chứa khuôn mặt bạn muốn lấy để hoán đổi.
# Ảnh đích (target): chứa khuôn mặt sẽ bị thay thế.
source_image_path = "INF2404001.jpg"
target_image_path = "anh_Infiniq.jpg"

print(f"Đang hoán đổi khuôn mặt từ '{source_image_path}' sang '{target_image_path}'...")

# 3. Thực hiện hoán đổi
# Phương thức swap_img_to_img trả về một đối tượng ảnh ở định dạng numpy array (BGR).
# Đây là định dạng tiêu chuẩn được sử dụng bởi OpenCV.
swapped_image = f2f.swap_img_to_img(source_image_path, target_image_path)

# 4. Lưu hoặc hiển thị kết quả
if swapped_image is not None:
    output_path = "result2.jpg"
    # Lưu ảnh kết quả ra đĩa
    cv2.imwrite(output_path, swapped_image)
    print(f"Hoán đổi thành công! Kết quả đã được lưu tại: {output_path}")

    # (Tùy chọn) Hiển thị ảnh kết quả trên màn hình
    cv2.imshow("Swapped Face", swapped_image)
    print("Nhấn phím bất kỳ để đóng cửa sổ xem ảnh.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Hoán đổi thất bại. Không thể phát hiện khuôn mặt trong ảnh nguồn hoặc ảnh đích.")