# Pipeline Guide

Tài liệu này mô tả luồng xử lý hiện tại của repo trên nhánh `main`.

## 1. Hai cách chạy chính

Repo hiện có hai entry point thực tế:

1. Python pipeline chính: [`../main.py`](../main.py)
2. Native integration harness: [`../native_core/run_native_harness.py`](../native_core/run_native_harness.py)

Hai luồng này dùng chung nhiều thành phần như:

- `warp_engine/` để căn chỉnh ảnh về template space
- `config/` để lấy ROI, marker layout, thresholds
- `postprocess_engine/` để đọc Student ID / Quiz ID và xuất handwritten review

Khác nhau chính là:

- `main.py` dùng ORM Python ở [`../orm_engine/orm.py`](../orm_engine/orm.py)
- `run_native_harness.py` dùng native core qua [`../native_core/native_api.py`](../native_core/native_api.py)

## 2. Python pipeline trong `main.py`

File: [`../main.py`](../main.py)

Đây là luồng chạy đơn giản nhất để hiểu repo.

### Các bước chính

1. Đọc ảnh đầu vào
   - mặc định dùng `samples/1photo5.jpg`

2. Khởi tạo `WarpEngine`
   - file chính: [`../warp_engine/engine.py`](../warp_engine/engine.py)

3. Warp ảnh đầu vào về đúng không gian template
   - dùng marker detection
   - fit global transform
   - có thể có refine để bù méo cục bộ

4. Chạy ORM Python trên ảnh đã căn chỉnh
   - file chính: [`../orm_engine/orm.py`](../orm_engine/orm.py)
   - đầu ra hiện tại trên `main` vẫn là danh sách `answers`

5. Đọc các bubble field như Student ID / Quiz ID
   - file chính: [`../postprocess_engine/bubble_field_reader.py`](../postprocess_engine/bubble_field_reader.py)

6. Ghi output
   - ảnh chấm điểm
   - `results/pipeline_result.json`

### Lệnh chạy

```powershell
$env:PYTHONPATH='.'; python main.py
```

### Khi nào nên dùng `main.py`

- Khi muốn hiểu pipeline Python hiện tại
- Khi muốn chỉnh ROI / thresholds nhanh
- Khi muốn debug OMR Python trước khi đụng sang native

## 3. Native harness trong `native_core/run_native_harness.py`

File: [`../native_core/run_native_harness.py`](../native_core/run_native_harness.py)

Đây là script tích hợp linh hoạt nhất của repo hiện tại.

### Nó có thể chạy những mode nào

- Raw native warp + native grading
- Ảnh đã aligned sẵn + native grading
- Python warp trước rồi mới gọi native grading
- Python prep grayscale trước rồi mới gọi native grading
- Đọc Student ID / Quiz ID
- Xuất handwritten review

### Các file liên quan

- Python bridge tới DLL: [`../native_core/native_api.py`](../native_core/native_api.py)
- Adapter config Python -> native: [`../native_core/python_adapter.py`](../native_core/python_adapter.py)
- C API header: [`../native_core/include/omr_api.h`](../native_core/include/omr_api.h)
- Native implementation: [`../native_core/src/omr_api.cpp`](../native_core/src/omr_api.cpp)

### Các lệnh chạy điển hình

Raw native:

```powershell
$env:PYTHONPATH='.'; python native_core/run_native_harness.py --input samples/1photo5.jpg
```

Python warp trước, native grading sau:

```powershell
$env:PYTHONPATH='.'; python native_core/run_native_harness.py --input samples/1photo5.jpg --python-warp-first --python-prep-gray-first
```

### Khi nào nên dùng harness

- Khi muốn test native DLL
- Khi muốn benchmark / parity giữa Python và native
- Khi muốn thử output handwritten review hoặc bubble field output
- Khi muốn gọi flow gần với tích hợp thực tế hơn `main.py`

## 4. Vai trò của từng lớp trong pipeline

### Warp

Thư mục: [`../warp_engine`](../warp_engine)

Vai trò:

- detect marker trên ảnh đầu vào
- fit phép biến đổi toàn cục
- tinh chỉnh warp bằng global IDW hoặc local refine
- tạo ảnh aligned hoặc ảnh review theo template

File trung tâm:

- [`../warp_engine/engine.py`](../warp_engine/engine.py)
- [`../warp_engine/detector.py`](../warp_engine/detector.py)
- [`../warp_engine/global_homography.py`](../warp_engine/global_homography.py)
- [`../warp_engine/region_warp.py`](../warp_engine/region_warp.py)

### ORM Python

Thư mục: [`../orm_engine`](../orm_engine)

Vai trò:

- đọc bubble answer từ ảnh đã aligned
- tính score từng bubble
- chọn đáp án cuối cùng theo ngưỡng
- vẽ overlay chấm điểm

File trung tâm:

- [`../orm_engine/orm.py`](../orm_engine/orm.py)

Lưu ý quan trọng trên `main`:

- ORM Python hiện vẫn theo model đơn giản hơn
- output chính vẫn là `answers`
- chưa có các field giàu semantics như `question_statuses`

### Native core

Thư mục: [`../native_core`](../native_core)

Vai trò:

- xử lý ảnh và grading ở C/C++
- cung cấp C API để Python gọi qua `ctypes`
- đọc bubble field metadata trong native path

### Post-processing

Thư mục: [`../postprocess_engine`](../postprocess_engine)

Vai trò:

- đọc Student ID / Quiz ID bằng bubble field layout
- crop/merge handwritten review
- điều khiển artifact output qua config

## 5. Luồng dữ liệu hiện tại

### Luồng Python

```text
input image
-> WarpEngine
-> aligned image
-> OMRProcessor (Python)
-> answers / score
-> bubble field reader
-> pipeline_result.json + scored image
```

### Luồng native harness

```text
input image
-> Python adapter normalizes config
-> native DLL processes image
-> native result
-> Python formats JSON / optional review artifacts
```

## 6. Những giới hạn cần biết trên `main`

- Python ORM trên `main` vẫn là nhánh đơn giản hơn nhánh ORM đang phát triển riêng.
- Native harness phong phú hơn `main.py`, nhưng không phải tất cả tính năng đang được app chính dùng làm mặc định.
- Tài liệu này mô tả đúng mã nguồn đang có trên `main`, không mô tả các thay đổi chỉ có ở nhánh khác.

## 7. Đọc tiếp gì sau tài liệu này

- Nếu muốn biết repo tổ chức ra sao: đọc [`codebase-structure.md`](codebase-structure.md)
- Nếu muốn biết config nào điều khiển phần nào: đọc [`config-reference.md`](config-reference.md)
- Nếu muốn biết JSON output hiện có những field nào: đọc [`output-reference.md`](output-reference.md)
