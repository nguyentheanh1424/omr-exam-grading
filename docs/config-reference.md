# Config Reference

Tài liệu này mô tả các file cấu hình JSON đang được dùng trên nhánh `main`.

## Tổng quan

Phần lớn config runtime nằm trong thư mục [`../config`](../config).

Các nhóm config chính:

- ROI và layout cho answer bubbles
- Marker layout cho warp
- Threshold cho OMR
- Metadata bubble fields như Student ID / Quiz ID
- Handwritten review regions
- Output artifact toggles

## 1. `config/circle_rois.json`

File: [`../config/circle_rois.json`](../config/circle_rois.json)

### Vai trò

Đây là file ROI quan trọng nhất cho phần OMR.

Nó chứa danh sách bubble thật sự được dùng khi chấm bài. Mỗi item tương ứng với một ô tròn trên phiếu.

### Dùng ở đâu

- [`../main.py`](../main.py)
- [`../orm_engine/orm.py`](../orm_engine/orm.py)
- [`../native_core/python_adapter.py`](../native_core/python_adapter.py)

### Field chính của mỗi ROI

- `cx`: tọa độ tâm theo trục X trong template space
- `cy`: tọa độ tâm theo trục Y trong template space
- `r`: bán kính bubble
- `question`: số câu hỏi
- `option`: chỉ số đáp án trong câu đó

### Ý nghĩa thực tế

- File này là “layout runtime”
- Khi warp xong, ORM sẽ dùng đúng các tọa độ này để cắt bubble và tính score

### Khi nào cần sửa

- Khi vị trí ROI lệch so với template
- Khi số câu / số option thay đổi
- Khi dùng form mới có layout bubble khác

## 2. `config/circle_grid_preset.json`

File: [`../config/circle_grid_preset.json`](../config/circle_grid_preset.json)

### Vai trò

Đây không phải file ROI runtime cuối cùng.

Nó là preset để dựng hoặc chỉnh lưới ROI, thường phục vụ editor hoặc bước sinh layout ban đầu.

### Ý nghĩa

- mô tả form có bao nhiêu câu
- khoảng cách các cột / hàng
- cách lặp pattern bubble

### Khác gì với `circle_rois.json`

- `circle_grid_preset.json`: mô tả quy luật sinh lưới
- `circle_rois.json`: mô tả từng bubble cụ thể được dùng khi chạy thật

## 3. `config/template_marker_layout.json`

File: [`../config/template_marker_layout.json`](../config/template_marker_layout.json)

### Vai trò

Mô tả vị trí marker chuẩn trong không gian template.

### Dùng ở đâu

- [`../warp_engine/config.py`](../warp_engine/config.py)
- [`../native_core/python_adapter.py`](../native_core/python_adapter.py)

### Tác dụng

- giúp warp engine biết marker nào nằm ở đâu trên template
- giúp native adapter xây đúng `template marker layout` cho DLL

### Khi nào cần sửa

- khi thay template
- khi thay marker design hoặc marker positions

## 4. `config/omr_thresholds.json`

File: [`../config/omr_thresholds.json`](../config/omr_thresholds.json)

### Vai trò

Chứa các threshold chính cho việc quyết định bubble nào được coi là được tô.

### Dùng ở đâu

- [`../orm_engine/orm.py`](../orm_engine/orm.py)
- [`../native_core/python_adapter.py`](../native_core/python_adapter.py)

### Field chính

- `abs_th`
- `rel_th`

### Ý nghĩa thực tế

- `abs_th` thấp quá: dễ nhiễu bởi viền ô, ký tự in, nền scan
- `abs_th` cao quá: dễ bỏ sót bubble tô nhẹ
- `rel_th` thấp quá: dễ chọn nhầm khi hai ô gần nhau
- `rel_th` cao quá: dễ ra `-1` cho các câu sát ngưỡng

## 5. `config/id_bubble_fields.json`

File: [`../config/id_bubble_fields.json`](../config/id_bubble_fields.json)

### Vai trò

Mô tả các bubble field dạng mã số, ví dụ:

- Student ID
- Quiz ID

### Dùng ở đâu

- [`../main.py`](../main.py)
- [`../native_core/run_native_harness.py`](../native_core/run_native_harness.py)
- [`../postprocess_engine/bubble_field_reader.py`](../postprocess_engine/bubble_field_reader.py)

### Field thường gặp

- `id`
- `label`
- `origin`
- `dx`
- `dy`
- `n_cols`
- `n_rows`
- `radius`
- `row_values`

### Cách hiểu

- `origin`: bubble đầu tiên của field
- `dx`: bước dịch ngang giữa các cột
- `dy`: bước dịch dọc giữa các hàng
- `n_cols`: số cột, thường là số chữ số cần đọc
- `n_rows`: số lựa chọn theo chiều dọc cho mỗi cột
- `row_values`: giá trị ứng với mỗi hàng

## 6. `config/handwritten_regions.json`

File: [`../config/handwritten_regions.json`](../config/handwritten_regions.json)

### Vai trò

Mô tả các vùng chữ viết tay cần crop hoặc merge để review.

### Dùng ở đâu

- [`../native_core/run_native_harness.py`](../native_core/run_native_harness.py)
- [`../postprocess_engine/handwritten_review.py`](../postprocess_engine/handwritten_review.py)

### Field chính

- `id`
- `label`
- `rect`
- `padding_px`
- `merge_mode`
- `save_patch`

### Ý nghĩa

- `rect`: vùng chữ nhật trên template
- `padding_px`: nới rộng thêm khi crop
- `merge_mode`: cách đặt lại patch vào review image
- `save_patch`: có lưu patch riêng từng vùng hay không

## 7. `config/pipeline_outputs.json`

File: [`../config/pipeline_outputs.json`](../config/pipeline_outputs.json)

### Vai trò

Điều khiển artifact nào sẽ được ghi ra `results/`.

### Dùng ở đâu

- [`../main.py`](../main.py)
- [`../native_core/run_native_harness.py`](../native_core/run_native_harness.py)
- [`../postprocess_engine/output_artifacts.py`](../postprocess_engine/output_artifacts.py)

### Nhóm field chính

- `debug_intermediate`
- `summary_json`
- `scored_image`
- `bubble_fields`
- `handwritten_review`

### Các preset đi kèm

- [`../config/pipeline_outputs.minimal.json`](../config/pipeline_outputs.minimal.json)
- [`../config/pipeline_outputs.review.json`](../config/pipeline_outputs.review.json)
- [`../config/pipeline_outputs.debug_full.json`](../config/pipeline_outputs.debug_full.json)

### Khi nào chỉnh file này

- khi muốn chỉ lấy JSON, không ghi nhiều ảnh
- khi muốn bật handwritten review
- khi muốn debug nhiều artifact hơn

## 8. Cách đọc config theo nhiệm vụ

Nếu bạn đang làm việc về:

- căn chỉnh answer bubbles: xem `circle_rois.json`
- sinh/chỉnh lưới ROI: xem `circle_grid_preset.json`
- warp/alignment: xem `template_marker_layout.json`
- tuning chấm OMR: xem `omr_thresholds.json`
- Student ID / Quiz ID: xem `id_bubble_fields.json`
- handwritten review: xem `handwritten_regions.json`
- output artifact: xem `pipeline_outputs*.json`
