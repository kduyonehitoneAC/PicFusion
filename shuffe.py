import os
import glob
import random
from PIL import Image
from tqdm import tqdm

# ==== CẤU HÌNH ====
INPUT_DIR = "./"       # thư mục chứa ảnh gốc
OUTPUT_DIR = "./"      # thư mục lưu ảnh đã tạo
ROWS, COLS = 3, 5                # chia ảnh thành 3x5 = 15 mảnh
NUM_SHUFFLES = 10              # số ảnh xáo trộn cần sinh ra / mỗi ảnh

os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_shuffled_versions(image_path, num_versions=1000):
    """Cắt ảnh thành 15 mảnh, xáo trộn rồi ghép lại nhiều lần"""
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    patch_w, patch_h = W // COLS, H // ROWS

    # Cắt ảnh gốc thành các mảnh nhỏ
    patches = []
    for r in range(ROWS):
        for c in range(COLS):
            left, upper = c * patch_w, r * patch_h
            right, lower = left + patch_w, upper + patch_h
            patches.append(img.crop((left, upper, right, lower)))

    # Lưu ảnh gốc (đã resize và chuẩn kích thước)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    orig_save_path = os.path.join(OUTPUT_DIR, f"{base_name}_original.jpg")
    img.save(orig_save_path)

    # Tạo 1000 phiên bản xáo trộn
    for i in range(num_versions):
        random.shuffle(patches)
        shuffled_img = Image.new("RGB", (W, H))
        for idx, patch in enumerate(patches):
            r, c = divmod(idx, COLS)
            shuffled_img.paste(patch, (c * patch_w, r * patch_h))

        save_path = os.path.join(OUTPUT_DIR, f"{base_name}_shuffle_{i+1:04d}.jpg")
        shuffled_img.save(save_path)

    print(f"✅ {base_name}: đã tạo {num_versions} ảnh xáo trộn + ảnh gốc.")

# ==== CHẠY TRÊN TOÀN BỘ ẢNH TRONG THƯ MỤC ====
if __name__ == "__main__":
    image_paths = glob.glob(os.path.join(INPUT_DIR, "*.jpg")) + \
                  glob.glob(os.path.join(INPUT_DIR, "*.png"))

    if not image_paths:
        raise ValueError(f"Không tìm thấy ảnh trong {INPUT_DIR}")

    print(f"Tìm thấy {len(image_paths)} ảnh trong {INPUT_DIR}.")
    print(f"👉 Sẽ tạo {NUM_SHUFFLES} ảnh xáo trộn cho mỗi ảnh (tổng ~{len(image_paths) * NUM_SHUFFLES} ảnh).")

    for path in tqdm(image_paths, desc="Processing images"):
        create_shuffled_versions(path, NUM_SHUFFLES)

    print("\n🎯 Hoàn tất tạo dữ liệu train trong thư mục:", OUTPUT_DIR)
