# Codebase Structure

Tài liệu này mô tả cấu trúc thư mục hiện tại của repo trên nhánh `main`, vai trò của từng phần, và file nào chịu trách nhiệm cho việc gì.

Mục tiêu của tài liệu này là để:

- người mới vào repo biết nên đọc từ đâu
- người đang sửa code biết nên chạm vào thư mục nào
- tránh nhầm giữa Python pipeline, native core, và post-processing

## 1. Nhìn từ trên xuống

Ở mức top-level, repo có thể chia thành 6 khối chính:

1. entry point và orchestration
2. config và sample data
3. warp engine
4. ORM / answer reading
5. post-processing
6. native core

## 2. Top-level files và thư mục

### `main.py`

File: [`../main.py`](../main.py)

Đây là entry point Python đơn giản nhất của repo.

Nó chịu trách nhiệm:

- đọc ảnh đầu vào
- load config
- gọi warp engine
- gọi ORM Python
- đọc Student ID / Quiz ID
- ghi kết quả ra `results/`

Nếu bạn muốn hiểu repo chạy thế nào từ đầu đến cuối, đây là file nên đọc đầu tiên.

### `config/`

Thư mục: [`../config`](../config)

Chứa JSON config dùng lúc chạy.

Đây là nơi mô tả:

- bubble ROI layout
- marker positions
- threshold
- metadata bubble fields
- handwritten review regions
- output toggles

Nếu bạn đang sửa hình học của form hoặc đổi hành vi output, khả năng cao bạn sẽ chạm vào đây.

### `samples/`

Thư mục: [`../samples`](../samples)

Chứa ảnh mẫu để dev/test.

Các file ở đây thường gồm:

- ảnh chụp đề thi thật, ví dụ `1photo*.jpg`
- ảnh template như `template_scan1.png`

Đây là nguồn dữ liệu nhỏ nhưng quan trọng để:

- debug warp
- test grading
- chạy parity với native

### `results/`

Thư mục: [`../results`](../results)

Chứa artifact sinh ra khi chạy repo.

Ví dụ:

- ảnh chấm điểm
- JSON summary
- benchmark report
- parity report
- handwritten review output

Thư mục này là output runtime, không phải source code.

### `build/`

Thư mục: [`../build`](../build)

Chứa build output cho native core.

Thường có:

- DLL
- executable test
- CMake/Ninja artifact

Nếu bạn đang làm với `native_core`, đây là nơi sẽ sinh ra file build.

### `scripts/`

Thư mục: [`../scripts`](../scripts)

Chứa script repo-level như:

- build/check native
- smoke test kiểu CI local

### `.idea/` và `.vscode/`

Thư mục:

- [`../.idea`](../.idea)
- [`../.vscode`](../.vscode)

Đây là config IDE/editor cục bộ, không phải logic pipeline.

## 3. Warp engine

Thư mục: [`../warp_engine`](../warp_engine)

Đây là khối chịu trách nhiệm biến ảnh chụp thô thành ảnh aligned theo template.

### Mục tiêu của module này

- detect marker trên phiếu
- tính biến đổi ảnh về template
- tinh chỉnh warp nếu có méo cục bộ
- trả ảnh aligned để ORM có thể đọc bubble đúng vị trí

### File trung tâm

#### `warp_engine/engine.py`

File: [`../warp_engine/engine.py`](../warp_engine/engine.py)

Đây là high-level orchestrator của phần warp Python.

Nó thường là nơi:

- nhận input image
- gọi detector
- fit global homography
- apply refine
- trả image/result object

Nếu bạn chỉ đọc một file trong `warp_engine`, nên đọc file này trước.

#### `warp_engine/detector.py`

File: [`../warp_engine/detector.py`](../warp_engine/detector.py)

Chịu trách nhiệm tìm marker trên ảnh chụp.

Nếu marker detect lệch, mọi bước sau thường đều bị ảnh hưởng.

#### `warp_engine/global_homography.py`

File: [`../warp_engine/global_homography.py`](../warp_engine/global_homography.py)

Chứa logic fit biến đổi toàn cục từ marker ảnh chụp sang marker template.

Đây là lớp sửa “méo lớn”.

#### `warp_engine/idw_refine.py`

File: [`../warp_engine/idw_refine.py`](../warp_engine/idw_refine.py)

Dùng cho refine toàn cục bằng IDW sau bước global warp.

#### `warp_engine/region_warp.py`

File: [`../warp_engine/region_warp.py`](../warp_engine/region_warp.py)

Xử lý refine theo từng vùng, hữu ích khi méo không đều trên toàn tờ giấy.

#### `warp_engine/refine_idw_patch.py`

File: [`../warp_engine/refine_idw_patch.py`](../warp_engine/refine_idw_patch.py)

Chứa helper để nội suy/refine patch cục bộ.

#### `warp_engine/template.py`

File: [`../warp_engine/template.py`](../warp_engine/template.py)

Hỗ trợ làm việc với template image và dữ liệu template-space.

#### `warp_engine/config.py`

File: [`../warp_engine/config.py`](../warp_engine/config.py)

Là nơi gom các hằng số, path config, và một phần wiring của warp.

#### `warp_engine/binarize.py`

File: [`../warp_engine/binarize.py`](../warp_engine/binarize.py)

Các hàm nhị phân hóa/phụ trợ tiền xử lý cho ảnh.

#### `warp_engine/types.py`

File: [`../warp_engine/types.py`](../warp_engine/types.py)

Các dataclass hoặc structured type giúp warp code dễ đọc hơn.

#### `warp_engine/utils.py`

File: [`../warp_engine/utils.py`](../warp_engine/utils.py)

Helper dùng chung cho nhiều file trong module.

### Khi nào sửa `warp_engine`

- ROI đúng nhưng bubble vẫn lệch vì ảnh chưa thẳng
- marker detect kém
- warp parity với native chưa ổn
- muốn xuất thêm aligned image hoặc debug overlay

## 4. ORM engine

Thư mục: [`../orm_engine`](../orm_engine)

Đây là phần đọc đáp án từ ảnh đã aligned.

### Mục tiêu của module này

- nhận ảnh đã căn chỉnh
- cắt bubble theo ROI
- tính score từng ô
- chọn đáp án cho từng câu
- vẽ overlay chấm điểm

### File chính

#### `orm_engine/orm.py`

File: [`../orm_engine/orm.py`](../orm_engine/orm.py)

Đây là implementation ORM Python hiện tại trên `main`.

Nó thường làm:

- grayscale prep
- bubble scoring
- answer selection bằng `abs_th` và `rel_th`
- draw scored overlay

Trên `main`, đây vẫn là nhánh ORM Python đơn giản hơn, chưa phải phiên bản semantics-rich ở nhánh riêng.

#### `orm_engine/roi_editor.py`

File: [`../orm_engine/roi_editor.py`](../orm_engine/roi_editor.py)

Tool/editor hỗ trợ chỉnh lưới ROI hoặc thao tác với bubble layout.

#### `orm_engine/tests/`

Thư mục: [`../orm_engine/tests`](../orm_engine/tests)

Chứa test cho phần ORM.

### Khi nào sửa `orm_engine`

- bubble ROI đúng nhưng logic chọn đáp án chưa đúng
- cần chỉnh threshold / scoring
- cần đổi cách draw overlay
- cần support thêm semantics answer reading trong Python path

## 5. Post-processing engine

Thư mục: [`../postprocess_engine`](../postprocess_engine)

Đây là nơi đặt những bước xử lý sau grading hoặc song song với grading.

### Hai nhiệm vụ lớn

1. đọc bubble field dạng metadata
2. sinh artifact review cho handwritten content

### File chính

#### `postprocess_engine/bubble_field_reader.py`

File: [`../postprocess_engine/bubble_field_reader.py`](../postprocess_engine/bubble_field_reader.py)

Đọc các field như:

- Student ID
- Quiz ID

Nó dùng layout trong `config/id_bubble_fields.json`.

#### `postprocess_engine/handwritten_review.py`

File: [`../postprocess_engine/handwritten_review.py`](../postprocess_engine/handwritten_review.py)

Chịu trách nhiệm:

- crop các vùng handwritten
- merge lên ảnh review
- sinh manifest / overlay liên quan

#### `postprocess_engine/output_artifacts.py`

File: [`../postprocess_engine/output_artifacts.py`](../postprocess_engine/output_artifacts.py)

Giúp kiểm soát artifact nào nên được ghi ra disk.

#### `postprocess_engine/run_handwritten_review_demo.py`

File: [`../postprocess_engine/run_handwritten_review_demo.py`](../postprocess_engine/run_handwritten_review_demo.py)

Script demo một ảnh cho handwritten review.

#### `postprocess_engine/run_handwritten_review_batch.py`

File: [`../postprocess_engine/run_handwritten_review_batch.py`](../postprocess_engine/run_handwritten_review_batch.py)

Script batch để chạy handwritten review trên nhiều sample.

#### `postprocess_engine/tests/`

Thư mục: [`../postprocess_engine/tests`](../postprocess_engine/tests)

Test cho phần post-processing.

### Khi nào sửa `postprocess_engine`

- muốn đổi cách đọc Student ID / Quiz ID
- muốn thêm field metadata mới
- muốn thêm vùng handwritten review
- muốn thay đổi output artifact

## 6. Native core

Thư mục: [`../native_core`](../native_core)

Đây là khối C/C++ native của repo.

Mục tiêu của nó là:

- xử lý ảnh nặng ở native
- cung cấp C API có thể gọi từ Python
- hỗ trợ warp, grading, và metadata bubble reading

### File top-level quan trọng

#### `native_core/CMakeLists.txt`

File: [`../native_core/CMakeLists.txt`](../native_core/CMakeLists.txt)

Cấu hình build của native core.

#### `native_core/native_api.py`

File: [`../native_core/native_api.py`](../native_core/native_api.py)

Bridge `ctypes` từ Python sang DLL native.

Nó thường làm:

- load `omr_core.dll`
- định nghĩa struct Python tương ứng struct C
- gọi hàm native
- đổi result native sang dict/Python object

#### `native_core/python_adapter.py`

File: [`../native_core/python_adapter.py`](../native_core/python_adapter.py)

Là lớp normalize config Python sang native form spec.

Đây là file quan trọng nếu bạn làm ở ranh giới:

- config JSON của repo
- struct đầu vào của native DLL

#### `native_core/run_native_harness.py`

File: [`../native_core/run_native_harness.py`](../native_core/run_native_harness.py)

Script tích hợp native mạnh nhất hiện có trong repo.

Nếu bạn muốn:

- test raw native warp
- chạy native grading
- so parity với Python
- bật handwritten review
- bật bubble field output

thì đây là file nên đọc.

#### `native_core/README.md`

File: [`../native_core/README.md`](../native_core/README.md)

README cục bộ cho module native.

### Header exported

#### `native_core/include/omr_api.h`

File: [`../native_core/include/omr_api.h`](../native_core/include/omr_api.h)

Đây là source of truth cho C API exported.

Nếu bạn làm với:

- ctypes binding
- external integration
- memory ownership
- result fields

thì đây là file bắt buộc phải đọc.

### Native source chính

#### `native_core/src/omr_api.cpp`

File: [`../native_core/src/omr_api.cpp`](../native_core/src/omr_api.cpp)

Đây là entry point xử lý chính bên native.

Nó thường chịu trách nhiệm:

- validate input
- orchestration warp
- grading
- metadata extraction
- result packaging

#### `native_core/src/warp_global.cpp`

File: [`../native_core/src/warp_global.cpp`](../native_core/src/warp_global.cpp)

Logic global warp phía native.

#### `native_core/src/marker_detect.cpp`

File: [`../native_core/src/marker_detect.cpp`](../native_core/src/marker_detect.cpp)

Marker detection bên native.

#### `native_core/src/idw_refine.cpp`

File: [`../native_core/src/idw_refine.cpp`](../native_core/src/idw_refine.cpp)

Global IDW refinement bên native.

#### `native_core/src/region_refine.cpp`

File: [`../native_core/src/region_refine.cpp`](../native_core/src/region_refine.cpp)

Refine cục bộ theo region bên native.

### Test và diagnostics

#### `native_core/tests/`

Thư mục: [`../native_core/tests`](../native_core/tests)

Đây là nơi có:

- parity scripts Python/native
- benchmark script
- real-image validation
- synthetic grading tests
- C++ runner phụ trợ

#### `native_core/examples/`

Thư mục: [`../native_core/examples`](../native_core/examples)

Chứa ví dụ gọi native API ở mức tối thiểu.

## 7. `docs/`

Thư mục: [`../docs`](../docs)

Đây là bộ tài liệu bạn đang đọc.

Mục tiêu là mô tả đúng mã nguồn trên `main`, không trộn lẫn với những thay đổi chỉ có ở nhánh khác.

## 8. Nên đọc file nào khi làm từng việc

### Muốn hiểu toàn pipeline

Đọc theo thứ tự:

1. [`../main.py`](../main.py)
2. [`pipeline-guide.md`](pipeline-guide.md)
3. [`../warp_engine/engine.py`](../warp_engine/engine.py)
4. [`../orm_engine/orm.py`](../orm_engine/orm.py)
5. [`../postprocess_engine/bubble_field_reader.py`](../postprocess_engine/bubble_field_reader.py)

### Muốn sửa warp

Đọc:

1. [`../warp_engine/engine.py`](../warp_engine/engine.py)
2. [`../warp_engine/detector.py`](../warp_engine/detector.py)
3. [`../warp_engine/global_homography.py`](../warp_engine/global_homography.py)
4. [`../warp_engine/region_warp.py`](../warp_engine/region_warp.py)

### Muốn sửa answer reading Python

Đọc:

1. [`../orm_engine/orm.py`](../orm_engine/orm.py)
2. [`../config/circle_rois.json`](../config/circle_rois.json)
3. [`../config/omr_thresholds.json`](../config/omr_thresholds.json)

### Muốn sửa Student ID / Quiz ID

Đọc:

1. [`../postprocess_engine/bubble_field_reader.py`](../postprocess_engine/bubble_field_reader.py)
2. [`../config/id_bubble_fields.json`](../config/id_bubble_fields.json)

### Muốn sửa handwritten review

Đọc:

1. [`../postprocess_engine/handwritten_review.py`](../postprocess_engine/handwritten_review.py)
2. [`../config/handwritten_regions.json`](../config/handwritten_regions.json)
3. [`../config/pipeline_outputs.json`](../config/pipeline_outputs.json)

### Muốn sửa native integration

Đọc:

1. [`../native_core/run_native_harness.py`](../native_core/run_native_harness.py)
2. [`../native_core/native_api.py`](../native_core/native_api.py)
3. [`../native_core/python_adapter.py`](../native_core/python_adapter.py)
4. [`../native_core/include/omr_api.h`](../native_core/include/omr_api.h)
5. [`../native_core/src/omr_api.cpp`](../native_core/src/omr_api.cpp)

## 9. Tóm tắt ngắn

Nếu nhìn repo theo đúng chức năng:

- `main.py`: entry point Python đơn giản
- `warp_engine/`: căn chỉnh ảnh
- `orm_engine/`: đọc bubble answer bằng Python
- `postprocess_engine/`: Student ID / Quiz ID và handwritten review
- `native_core/`: DLL native + bridge Python
- `config/`: toàn bộ layout/threshold/output config
- `samples/`: ảnh mẫu
- `results/`: artifact sinh ra khi chạy
