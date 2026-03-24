# Native C API

Tài liệu này mô tả C API hiện tại của native core trên nhánh `main`.

Header chính:

- [`../native_core/include/omr_api.h`](../native_core/include/omr_api.h)

Implementation chính:

- [`../native_core/src/omr_api.cpp`](../native_core/src/omr_api.cpp)

## 1. Mục tiêu của C API

C API là lớp giao tiếp ổn định giữa:

- native core viết bằng C/C++
- Python bridge trong [`../native_core/native_api.py`](../native_core/native_api.py)

API này giúp Python không phải gọi trực tiếp vào code C++ nội bộ, mà chỉ làm việc với struct C rõ ràng.

## 2. Vòng đời gọi API

Các hàm chính:

- `omr_create`
- `omr_destroy`
- `omr_init_result`
- `omr_process`
- `omr_free_result`

Luồng dùng điển hình:

1. tạo handle bằng `omr_create`
2. chuẩn bị input image + form spec + params
3. gọi `omr_init_result`
4. gọi `omr_process`
5. đọc `OMR_Result`
6. giải phóng bằng `omr_free_result`
7. hủy handle bằng `omr_destroy`

## 3. Kiểu dữ liệu đầu vào chính

### `OMR_ImageView`

Mô tả buffer ảnh đầu vào:

- `width`
- `height`
- `stride`
- `channels`
- `data`

Giá trị `channels` thường dùng:

- `1`: grayscale
- `3`: BGR

### `OMR_FormSpec`

Đây là struct mô tả toàn bộ form chấm.

Nó thường chứa:

- kích thước output/template
- marker layout
- region windows
- circle ROIs
- answer key
- metadata field layout

Python adapter sẽ build struct này từ các file JSON hiện có trong `config/`.

### `OMR_CircleROI`

Trên `main`, mỗi bubble ROI native hiện có các field:

- `cx`
- `cy`
- `r`
- `question`
- `option`

Lưu ý:

- `question` ở native là `0-based`
- `option` ở native cũng là `0-based`

### `OMR_MetadataField` và `OMR_MetadataBubble`

Hai struct này dùng cho các bubble field như:

- Student ID
- Quiz ID

Chúng mô tả:

- field có bao nhiêu cột
- mỗi cột có bao nhiêu lựa chọn
- bubble nào tương ứng với giá trị nào

## 4. Nhóm tham số điều khiển

### `OMR_WarpParams`

Điều khiển phần warp:

- marker matching threshold
- global IDW
- local region refine
- patch refine parameters

Nhóm này ảnh hưởng trực tiếp đến việc ảnh đầu vào có được căn chỉnh đúng về template hay không.

### `OMR_BinarizeParams`

Điều khiển tiền xử lý / nhị phân hóa:

- blur
- percentile
- thinning / denoise

Nhóm này tác động nhiều đến chất lượng bubble score.

### `OMR_GradingParams`

Điều khiển logic grading:

- `abs_th`
- `rel_th`
- auto-threshold calibration
- gray preprocessing parameters
- annulus scoring parameters

### `OMR_RuntimeOptions`

Điều khiển cách chạy:

- input có phải ảnh aligned sẵn không
- có trả scored image hay không
- debug level

## 5. Kết quả đầu ra: `OMR_Result`

Trên `main`, `OMR_Result` hiện trả các nhóm dữ liệu chính sau:

- lỗi và thông điệp lỗi
- score tổng
- số câu
- mảng `answers`
- metadata rows đã chọn
- thresholds thật sự đã dùng
- optional scored image

Các field thường gặp:

- `err_code`
- `error_message`
- `score`
- `total_questions`
- `graded_questions`
- `answers`
- `metadata_selected_rows`
- `used_abs_th`
- `used_rel_th`

Nếu bật trả ảnh, result còn có buffer ảnh scored.

## 6. Quản lý bộ nhớ

Điểm quan trọng nhất:

- Bộ nhớ nằm trong `OMR_Result` phải được giải phóng bằng `omr_free_result`

Không nên:

- tự `free()` các con trỏ trong result từ phía Python
- giữ con trỏ sang dữ liệu đã được free

## 7. Xử lý lỗi

Các lỗi được phản ánh qua:

- `err_code`
- `error_message`
- `omr_error_code_to_string(...)`

Nhóm lỗi thường gặp:

- input image không hợp lệ
- config form không hợp lệ
- marker không đủ để warp
- ROI layout sai
- cấp phát bộ nhớ thất bại

Khi tích hợp, nên luôn kiểm tra `err_code` trước khi đọc các field output khác.

## 8. Python bridge đang dùng API này như thế nào

### `native_api.py`

File: [`../native_core/native_api.py`](../native_core/native_api.py)

Vai trò:

- load DLL
- khai báo `ctypes` cho các struct và hàm C
- gọi `omr_process`
- đổi `OMR_Result` sang dict/Python object dễ dùng hơn

### `python_adapter.py`

File: [`../native_core/python_adapter.py`](../native_core/python_adapter.py)

Vai trò:

- đọc config từ repo
- normalize layout và thresholds
- chuyển từ JSON/Python data sang native form spec

## 9. Nên đọc tiếp gì sau tài liệu này

- muốn hiểu luồng tích hợp: đọc [`pipeline-guide.md`](pipeline-guide.md)
- muốn hiểu config nào map vào native: đọc [`config-reference.md`](config-reference.md)
- muốn xem thư mục native core tổ chức ra sao: đọc [`codebase-structure.md`](codebase-structure.md)
