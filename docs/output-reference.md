# Output Reference

Tài liệu này mô tả các file output chính của repo trên nhánh `main`.

## 1. Python pipeline output

Python pipeline trong [`../main.py`](../main.py) ghi kết quả chính vào:

- `results/pipeline_result.json`

### Field hiện có

- `input`
- `mode`
- `score`
- `graded_questions`
- `total_questions`
- `answers`
- `student_id`
- `quiz_id`
- `thresholds`
- `bubble_fields`

### Ý nghĩa từng field

- `input`: ảnh đầu vào đã dùng
- `mode`: mô tả mode chạy
- `score`: số câu đúng
- `graded_questions`: số câu được dùng để chấm
- `total_questions`: tổng số câu trong answer key
- `answers`: đáp án đọc được theo từng câu
- `student_id`: chuỗi giải mã từ bubble field Student ID
- `quiz_id`: chuỗi giải mã từ bubble field Quiz ID
- `thresholds`: các ngưỡng OMR đã dùng
- `bubble_fields`: dữ liệu chi tiết hơn cho các field metadata

### Lưu ý

Trên `main`, Python pipeline chưa trả các field giàu semantics như:

- `question_statuses`
- `selected_options`

Nghĩa là downstream hiện vẫn chủ yếu đọc `answers`.

## 2. Native harness output

Native harness trong [`../native_core/run_native_harness.py`](../native_core/run_native_harness.py) ghi kết quả chính vào:

- `results/.../native_result.json`

Ví dụ thường gặp:

- `results/native_harness/native_result.json`

### Field thường thấy

- `input`
- `mode`
- `python_warp_first`
- `python_prep_gray_first`
- `adapter`
- `score`
- `graded_questions`
- `total_questions`
- `used_abs_th`
- `used_rel_th`
- `configured_abs_th`
- `configured_rel_th`
- `answers`
- `detected_marker_count`
- `marker_source`
- `student_id`
- `quiz_id`

### Ý nghĩa

- `python_warp_first`: có dùng Python warp trước native không
- `python_prep_gray_first`: có dùng Python preprocessing trước native không
- `adapter`: thông tin config sau khi normalize
- `used_abs_th` / `used_rel_th`: threshold cuối cùng thật sự dùng để chấm
- `configured_abs_th` / `configured_rel_th`: threshold lấy từ config
- `detected_marker_count`: số marker được detect
- `marker_source`: marker đến từ native detect hay nguồn khác

## 3. Bubble field output

Khi bật đọc Student ID / Quiz ID, repo có thể ghi:

- `results/id_bubble_values.json`
- `results/.../bubble_fields/bubble_field_values.json`
- `results/.../bubble_fields/aligned_bubble_fields.png`

### Mục đích

- cho biết cột nào đang chọn hàng nào
- cho biết chuỗi decode cuối cùng là gì
- cho phép xem overlay vùng bubble field để debug

## 4. Handwritten review output

Khi bật handwritten review từ native harness, output có thể gồm:

- `aligned_source_img.png`
- `aligned_source_regions.png`
- `template_merged_img.png`
- `template_merged_regions.png`
- `review_merged_template.png`
- `review_merged_template_ink_mask.png`
- `review_manifest.json`

### Ý nghĩa

- `aligned_source_img.png`: ảnh source sau khi đã căn chỉnh
- `aligned_source_regions.png`: overlay các vùng handwritten trên ảnh aligned
- `template_merged_img.png`: ảnh template đã được merge patch handwritten
- `template_merged_regions.png`: overlay vùng trên ảnh merge
- `review_merged_template.png`: ảnh review kiểu replace rect
- `review_merged_template_ink_mask.png`: ảnh review kiểu ink mask
- `review_manifest.json`: mô tả file nào đã được sinh ra và vùng nào đã dùng

## 5. Output artifact config điều khiển gì

Việc có ghi các file trên hay không phụ thuộc vào:

- [`../config/pipeline_outputs.json`](../config/pipeline_outputs.json)
- hoặc các preset:
  - [`../config/pipeline_outputs.minimal.json`](../config/pipeline_outputs.minimal.json)
  - [`../config/pipeline_outputs.review.json`](../config/pipeline_outputs.review.json)
  - [`../config/pipeline_outputs.debug_full.json`](../config/pipeline_outputs.debug_full.json)

## 6. Downstream nên đọc gì

### Nếu chỉ cần điểm và đáp án

Đọc:

- `score`
- `answers`
- `graded_questions`
- `total_questions`

### Nếu cần metadata

Đọc thêm:

- `student_id`
- `quiz_id`

### Nếu cần debug

Đọc thêm:

- `used_abs_th`
- `used_rel_th`
- `detected_marker_count`
- `marker_source`
- các ảnh overlay trong thư mục `results/`
