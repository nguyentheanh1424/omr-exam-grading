# Documentation

Tài liệu trong thư mục `docs/` được viết theo đúng mã nguồn hiện có trên nhánh `main`.

Repo hiện có hai luồng chạy thực tế:

- Python pipeline chính ở [`../main.py`](../main.py)
- Native integration harness ở [`../native_core/run_native_harness.py`](../native_core/run_native_harness.py)

Điểm cần nhớ khi đọc docs:

- Python pipeline trên `main` vẫn dùng ORM Python trong [`../orm_engine/orm.py`](../orm_engine/orm.py).
- Native harness trên `main` đã hỗ trợ native raw warp, native grading, đọc Student ID / Quiz ID, và handwritten review.
- Tên file config trên `main` vẫn là:
  - `config/circle_rois.json`
  - `config/circle_grid_preset.json`
  - `config/template_marker_layout.json`
  - `config/omr_thresholds.json`

## Tài liệu nên đọc theo thứ tự

1. [`pipeline-guide.md`](pipeline-guide.md)  
   Giải thích toàn bộ pipeline hiện tại, từ ảnh đầu vào đến kết quả đầu ra.

2. [`codebase-structure.md`](codebase-structure.md)  
   Mô tả cấu trúc thư mục, vai trò từng module, và file nào nên đọc trước.

3. [`run-guide.md`](run-guide.md)  
   Hướng dẫn cài môi trường và chạy từng flow bằng lệnh cụ thể.

4. [`config-reference.md`](config-reference.md)  
   Giải thích từng file JSON trong `config/` và ý nghĩa các field quan trọng.

5. [`output-reference.md`](output-reference.md)  
   Mô tả các file output hiện tại và ý nghĩa của từng trường JSON.

6. [`native-c-api.md`](native-c-api.md)  
   Tài liệu C API cho phần native core.

## Khi nào đọc tài liệu nào

- Nếu bạn chỉ muốn chạy repo: đọc [`pipeline-guide.md`](pipeline-guide.md)
- Nếu bạn cần lệnh chạy cụ thể: đọc [`run-guide.md`](run-guide.md)
- Nếu bạn muốn sửa config: đọc [`config-reference.md`](config-reference.md)
- Nếu bạn muốn nối app/backend vào output JSON: đọc [`output-reference.md`](output-reference.md)
- Nếu bạn muốn gọi DLL native trực tiếp: đọc [`native-c-api.md`](native-c-api.md)
- Nếu bạn mới vào repo và chưa biết nên bắt đầu ở đâu: đọc [`codebase-structure.md`](codebase-structure.md)
