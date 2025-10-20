import os
import glob
from PIL import Image
from tqdm import tqdm # Thư viện giúp tạo thanh tiến trình (pip install tqdm)

# ==============================================================================
# CÁC THAM SỐ CẤU HÌNH
# ==============================================================================
# Thư mục chứa 131 ảnh gốc của bạn
ORIGINAL_IMAGE_DIR = './'  

# Thư mục để lưu ảnh sau khi đã resize
PROCESSED_IMAGE_DIR = './'

# Kích thước ảnh mục tiêu theo đề thi
TARGET_WIDTH = 600
TARGET_HEIGHT = 360

# ==============================================================================
# HÀM TIỀN XỬ LÝ
# ==============================================================================
def preprocess_and_save_images(input_dir, output_dir, size):
    """
    Đọc tất cả ảnh từ input_dir, resize về kích thước `size`, và lưu vào output_dir.
    """
    # Tạo thư mục output nếu nó chưa tồn tại
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Đã tạo thư mục output: {output_dir}")

    # Tìm tất cả file ảnh
    image_paths = glob.glob(os.path.join(input_dir, '*.jpg')) + glob.glob(os.path.join(input_dir, '*.png'))
    
    if not image_paths:
        raise ValueError(f"Không tìm thấy file ảnh .jpg hoặc .png nào trong thư mục: {input_dir}")

    print(f"Bắt đầu tiền xử lý {len(image_paths)} ảnh. Từ '{input_dir}' -> '{output_dir}'")

    # Sử dụng tqdm để hiển thị thanh tiến trình
    for path in tqdm(image_paths, desc="Resizing images"):
        try:
            # Mở, chuyển sang RGB và resize
            img = Image.open(path).convert('RGB')
            resized_img = img.resize(size, Image.Resampling.LANCZOS) # Dùng bộ lọc chất lượng cao
            
            # Tạo đường dẫn file output
            filename = os.path.basename(path)
            output_path = os.path.join(output_dir, filename)
            
            # Lưu ảnh đã resize
            resized_img.save(output_path)

        except Exception as e:
            print(f"\nLỗi khi xử lý ảnh {path}: {e}. Bỏ qua file này.")

    print(f"\n✅ Hoàn tất! Đã resize và lưu {len(image_paths)} ảnh vào thư mục '{output_dir}'.")


# ==============================================================================
# HÀM MAIN
# ==============================================================================
if __name__ == '__main__':
    preprocess_and_save_images(ORIGINAL_IMAGE_DIR, PROCESSED_IMAGE_DIR, (TARGET_WIDTH, TARGET_HEIGHT))