# Các bước chạy để train và giải jigsaw puzzle
## Xử lý scale ảnh về 360 x 600: 
``` python
python process_img.py
```

## Sau khi scale ảnh thì chia ảnh thành 15 mảnh và shuffle để tạo data:
``` python
python shuffle.py
```

## Train mạng CNN để trích xuất đặc trưng cạnh cho ảnh:
``` python
python train.py
```

## Sử dụng thuật toán Deepzzle để giải quyết bài toán:
``` python
python solver.py
```
