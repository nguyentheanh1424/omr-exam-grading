# ORM Pipeline Guide

## Mục đích

`ORM` trong repo này dùng để đọc các ô tròn đã tô trên ảnh phiếu sau khi ảnh đã được warp về không gian template.

Pipeline chuẩn trong [main.py](D:\Theanh\Learn\20251\Pr3\omr-exam-grading\main.py) là:

1. Đọc ảnh đầu vào.
2. Warp ảnh về template bằng `WarpEngine`.
3. Chạy `OMRProcessor` để đọc đáp án.
4. Đọc thêm `Student ID` / `Quiz ID`.
5. Ghi kết quả ra `results/pipeline_result.json`.

## Cấu hình ROI

File ROI hiện tại là [omr_bubble_layout.json](D:\Theanh\Learn\20251\Pr3\omr-exam-grading\config\omr_bubble_layout.json).

Mỗi phần tử mô tả một bubble:

```json
{
  "cx": 803,
  "cy": 1198,
  "r": 22,
  "question": 1,
  "option": 0,
  "selection_mode": "single"
}
```

Các trường:

- `cx`, `cy`, `r`: vị trí và bán kính bubble trong không gian template.
- `question`: số câu hỏi, bắt đầu từ `1`.
- `option`: chỉ số lựa chọn, bắt đầu từ `0`.
- `selection_mode`: loại câu hỏi.

## selection_mode

`selection_mode` được khai báo ngay trong ROI và áp dụng cho toàn bộ bubble của cùng một câu.

Các giá trị hiện hỗ trợ:

- `single`
- `multiple`

Nếu `selection_mode` không có trong ROI, hệ thống mặc định là `single`.

Lưu ý:

- Tất cả ROI của cùng một câu phải có cùng `selection_mode`.
- Nếu một câu bị khai báo lẫn `single` và `multiple`, `OMRProcessor` sẽ báo lỗi cấu hình.

## Cách ORM hoạt động

ORM hiện được tách thành 2 tầng trong [orm.py](D:\Theanh\Learn\20251\Pr3\omr-exam-grading\orm_engine\orm.py):

1. Tầng bubble:
   - `_detect_filled_options(...)`
   - quyết định bubble nào được xem là đã tô.

2. Tầng question:
   - `_resolve_question_selection(...)`
   - từ các bubble đã tô, suy ra trạng thái cuối của câu.

Thiết kế này giúp phân biệt rõ:

- câu để trống
- câu chọn một đáp án
- câu tô nhiều đáp án hợp lệ
- câu single nhưng bị tô nhiều ô
- câu sát ngưỡng, chưa đủ chắc chắn

## question_statuses

`OMRProcessor.run()` trả về `question_statuses` cho từng câu.

Các trạng thái hiện có:

- `blank`: không có bubble nào đủ mạnh.
- `single`: đúng 1 lựa chọn hợp lệ.
- `multiple`: câu ở mode `multiple` và có từ 2 bubble được tô.
- `invalid_multiple_on_single`: câu ở mode `single` nhưng bị tô nhiều ô.
- `uncertain`: có dấu hiệu tô nhưng chưa đủ chắc để chọn 1 đáp án hợp lệ.

## answers và selected_options

Output chính của ORM gồm:

- `answers`
- `selected_options`
- `question_statuses`
- `question_selection_modes`

Ý nghĩa:

- `selected_options`: danh sách các ô được xem là đã tô cho từng câu.
- `question_statuses`: trạng thái cuối của câu.
- `question_selection_modes`: mode của từng câu, lấy từ ROI.
- `answers`: đầu ra tương thích ngược cho grading.

Quy ước của `answers`:

- Nếu câu là `single` hợp lệ, `answers[i]` là option được chọn.
- Nếu câu là `blank`, `multiple`, `invalid_multiple_on_single`, hoặc `uncertain`, `answers[i] = -1`.

Điều này giúp pipeline cũ không bị vỡ, trong khi downstream mới có thể đọc `selected_options` và `question_statuses` để xử lý đầy đủ hơn.

## Output JSON của pipeline

File output chính là [pipeline_result.json](D:\Theanh\Learn\20251\Pr3\omr-exam-grading\results\pipeline_result.json).

Các field quan trọng liên quan đến ORM:

- `answers`
- `selected_options`
- `question_statuses`
- `question_selection_modes`
- `multiple_questions`
- `invalid_multiple_on_single_questions`
- `uncertain_questions`
- `blank_questions`
- `single_questions`
- `status_counts`

Hiểu nhanh:

- `multiple_questions`: chỉ các câu mode `multiple` và đang có nhiều lựa chọn hợp lệ.
- `invalid_multiple_on_single_questions`: các câu mode `single` nhưng bị tô nhiều ô.

## Xử lý input ảnh

`OMRProcessor` hiện nhận được:

- ảnh grayscale
- ảnh single-channel
- ảnh BGR

Nếu `debug=True` thì bắt buộc phải có `output`, nếu không hệ thống sẽ báo lỗi rõ ràng.

## Script debug multiple

Để kiểm tra trực quan các câu đang bị coi là tô nhiều ô, dùng:

```powershell
$env:PYTHONPATH='.'; python orm_engine/debug_multiple_questions.py
```

Script sẽ sinh artifact tại:

- [results/orm_multiple_debug](D:\Theanh\Learn\20251\Pr3\omr-exam-grading\results\orm_multiple_debug)

Bao gồm:

- `summary.json`
- `qXX/crop.png`
- `qXX/crop_overlay.png`

Mặc định script lọc các câu có trạng thái `multiple`.

## Gợi ý sử dụng

Nếu đề thi của bạn:

- chỉ có câu chọn một đáp án:
  - để toàn bộ ROI ở `selection_mode = "single"`

- có lẫn câu single và multiple:
  - cập nhật `selection_mode` theo từng câu ngay trong ROI

Downstream nên đọc:

- `question_statuses`
- `selected_options`

thay vì chỉ đọc `answers`, vì `answers` chỉ còn là đầu ra tương thích ngược cho grading.
