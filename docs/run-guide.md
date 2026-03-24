# Run Guide

Tài liệu này hướng dẫn cách chạy code trong repo trên nhánh `main`.

Nó tập trung vào:

- chuẩn bị môi trường
- chạy Python pipeline
- chạy native harness
- build/test native core
- xem output sau khi chạy

Lưu ý:

- Với các script nằm trong thư mục package như `native_core/` hoặc `postprocess_engine/`, nên ưu tiên chạy bằng `python -m ...` từ thư mục gốc của repo.
- Cách này tránh lỗi import kiểu `ModuleNotFoundError: No module named 'native_core'`.

## 1. Chuẩn bị môi trường Python

### Cách nhanh nhất với `.venv`

Tạo môi trường ảo:

```powershell
python -m venv .venv
```

Kích hoạt môi trường ảo:

```powershell
.\.venv\Scripts\Activate.ps1
```

Nâng cấp `pip`:

```powershell
python -m pip install --upgrade pip
```

Cài thư viện:

```powershell
python -m pip install -r requirements-parity.txt
```

### Nếu không dùng `.venv`

Bạn vẫn có thể cài trực tiếp vào Python hiện tại:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements-parity.txt
```

### Kiểm tra nhanh môi trường

```powershell
python --version
python -c "import cv2; print(cv2.__version__)"
```

## 2. Chạy Python pipeline chính

Entry point:

- [`../main.py`](../main.py)

### Lệnh chạy tối thiểu

```powershell
python main.py
```

### Luồng này sẽ làm gì

1. Đọc ảnh đầu vào
2. Khởi tạo `WarpEngine`
3. Warp ảnh về template
4. Chạy ORM Python
5. Đọc Student ID / Quiz ID
6. Ghi ảnh chấm điểm và JSON summary

### Input mặc định hiện tại

Trong [`../main.py`](../main.py), các giá trị mặc định là:

- input image: `samples/1photo5.jpg`
- template image: `samples/template_scan1.png`
- output dir: `results`

### Output thường thấy

- `results/orm_3_scored.png`
- `results/pipeline_result.json`
- `results/id_bubble_fields.png`
- `results/id_bubble_values.json`

### Chạy lại nhanh nhiều lần

```powershell
python main.py
Get-Content results\pipeline_result.json
```

## 3. Chạy native harness

Entry point:

- [`../native_core/run_native_harness.py`](../native_core/run_native_harness.py)

Đây là script tích hợp native linh hoạt nhất trong repo.

Lưu ý quan trọng trên `main` hiện tại:

- Sau khi test lại trên `main`, các command gọi trực tiếp `run_native_harness.py` hiện chưa ổn định để đưa vào runbook chính.
- Các lỗi mình gặp thực tế gồm:
  - `ROI circle exceeds image bounds`
  - crash/exit code bất thường ở một số mode raw native
- Vì vậy, trong tài liệu chạy code chính thức trên `main`, mình không khuyến nghị dùng các command harness như lệnh “copy-paste là chạy”.
- Nếu bạn đang phát triển native core, nên dùng:
  - [`../scripts/run_native_ci_checks.ps1`](../scripts/run_native_ci_checks.ps1)
  - [`../native_core/tests/adapter_contract_check.py`](../native_core/tests/adapter_contract_check.py)

### 3.1. Khi nào vẫn nên dùng harness

Bạn vẫn nên đọc file này khi:

- cần hiểu native integration flow
- cần debug Python adapter -> native DLL boundary
- cần sửa argument parsing hoặc output formatting

Nhưng để chạy kiểm tra trên `main`, hãy ưu tiên nhóm lệnh ở phần `Build native core` và `Chạy native tests/checks`.

## 4. Build native core

### 4.1. Build bằng script chuẩn của repo

Nếu bạn chỉ cần build/test đúng kiểu repo đang dùng, nên chạy:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_native_ci_checks.ps1
```

### 4.2. Build bằng CMake + Visual Studio generator mặc định

```powershell
cmake -S native_core -B build/native_core
cmake --build build/native_core --config Release
```

Lưu ý:

- Nếu vừa mới thêm `cmake` vào `PATH`, hãy mở terminal mới trước khi chạy lệnh này.

### 4.3. Build bằng Ninja

```powershell
cmake -S native_core -B build/native_core -G Ninja
cmake --build build/native_core
```

### 4.4. File build quan trọng

Sau khi build, các file thường nằm trong:

- `build/native_core/`

Ví dụ:

- `build/native_core/omr_core.dll`
- `build/native_core/omr_unit_tests.exe`

## 5. Chạy native tests/checks

### 5.1. Chạy full native checks nhanh

Script:

- [`../scripts/run_native_ci_checks.ps1`](../scripts/run_native_ci_checks.ps1)

Lệnh:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_native_ci_checks.ps1
```

Script này hiện làm:

1. build `native_core`
2. chạy C++ unit tests
3. chạy synthetic parity
4. chạy best-path real-image parity

### 5.2. Chạy C++ unit tests trực tiếp

```powershell
.\build\native_core\omr_unit_tests.exe
```

### 5.3. Chạy adapter contract check

```powershell
python -m native_core.tests.adapter_contract_check
```

## 6. Chạy handwritten review riêng

### Demo một ảnh

```powershell
python -m postprocess_engine.run_handwritten_review_demo
```

### Chạy batch trên sample

```powershell
python -m postprocess_engine.run_handwritten_review_batch
```

## 7. Xem kết quả ở đâu

### Python pipeline

Xem trong:

- `results/`

Các file đáng xem:

- `results/pipeline_result.json`
- `results/orm_3_scored.png`
- `results/id_bubble_values.json`

Đọc nhanh JSON:

```powershell
Get-Content results\pipeline_result.json
```

## 8. Khi nào nên chạy lệnh nào

### Nếu chỉ muốn biết repo còn chạy được không

```powershell
python main.py
```

### Nếu đang làm native core

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_native_ci_checks.ps1
```

### Nếu đang debug handwritten review

```powershell
python -m postprocess_engine.run_handwritten_review_demo
```

### Nếu đang debug bubble field / Student ID / Quiz ID

```powershell
python main.py
Get-Content results\id_bubble_values.json
```

## 9. Đọc tiếp gì sau tài liệu này

- muốn hiểu pipeline: đọc [`pipeline-guide.md`](pipeline-guide.md)
- muốn hiểu config: đọc [`config-reference.md`](config-reference.md)
- muốn hiểu output: đọc [`output-reference.md`](output-reference.md)
- muốn hiểu cấu trúc repo: đọc [`codebase-structure.md`](codebase-structure.md)
