#include "omr_api.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

struct TestContext {
    int passed = 0;
    int failed = 0;
};

void assert_true(TestContext& ctx, bool condition, const char* message) {
    if (condition) {
        ctx.passed += 1;
    } else {
        ctx.failed += 1;
        std::printf("[FAIL] %s\n", message);
    }
}

void draw_dark_annulus(
    std::vector<uint8_t>& img,
    int width,
    int height,
    int stride,
    int cx,
    int cy,
    int r_inner,
    int r_outer,
    uint8_t value
) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const int dx = x - cx;
            const int dy = y - cy;
            const int d2 = dx * dx + dy * dy;
            if (d2 >= r_inner * r_inner && d2 <= r_outer * r_outer) {
                img[static_cast<size_t>(y) * static_cast<size_t>(stride) + static_cast<size_t>(x)] = value;
            }
        }
    }
}

void draw_square_ring(
    std::vector<uint8_t>& img,
    int width,
    int height,
    int stride,
    int cx,
    int cy,
    int half_size,
    int thickness,
    uint8_t value
) {
    const int x0 = std::max(0, cx - half_size);
    const int x1 = std::min(width - 1, cx + half_size);
    const int y0 = std::max(0, cy - half_size);
    const int y1 = std::min(height - 1, cy + half_size);
    for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
            const bool border =
                (x - x0 < thickness) || (x1 - x < thickness) ||
                (y - y0 < thickness) || (y1 - y < thickness);
            if (border) {
                img[static_cast<size_t>(y) * static_cast<size_t>(stride) + static_cast<size_t>(x)] = value;
            }
        }
    }
}

OMR_ImageView make_image(std::vector<uint8_t>* storage, int width, int height) {
    const int channels = 1;
    const int stride = width * channels;
    storage->assign(static_cast<size_t>(stride) * static_cast<size_t>(height), 255u);
    OMR_ImageView image{};
    image.width = width;
    image.height = height;
    image.stride = stride;
    image.channels = channels;
    image.data = storage->data();
    return image;
}

void setup_defaults(
    OMR_WarpParams* warp,
    OMR_BinarizeParams* bin,
    OMR_GradingParams* grading,
    OMR_RuntimeOptions* runtime
) {
    omr_init_default_warp_params(warp);
    omr_init_default_binarize_params(bin);
    omr_init_default_grading_params(grading);
    omr_init_default_runtime_options(runtime);
    runtime->assume_aligned_input = 1;
    runtime->return_scored_image = 0;
}

void test_basic_detect(TestContext& ctx) {
    std::vector<uint8_t> image_storage;
    OMR_ImageView image = make_image(&image_storage, 260, 160);

    draw_dark_annulus(image_storage, image.width, image.height, image.stride, 180, 80, 7, 14, 30u);

    OMR_CircleROI rois[2]{};
    rois[0] = {80, 80, 16, 0, 0};
    rois[1] = {180, 80, 16, 0, 1};
    int32_t key[1] = {1};

    OMR_FormSpec form{};
    form.output_width = image.width;
    form.output_height = image.height;
    form.circle_rois = rois;
    form.n_circle_rois = 2;
    form.n_questions = 1;
    form.n_options_per_question = 2;
    form.answer_key = key;
    form.n_answer_key = 1;

    OMR_WarpParams warp{};
    OMR_BinarizeParams bin{};
    OMR_GradingParams grading{};
    OMR_RuntimeOptions runtime{};
    setup_defaults(&warp, &bin, &grading, &runtime);

    OMR_Result out{};
    omr_init_result(&out);
    OMR_Handle* h = omr_create();

    const int32_t rc = omr_process(h, &image, &form, &warp, &bin, &grading, &runtime, &out);
    assert_true(ctx, rc == OMR_OK, "basic detect should succeed");
    assert_true(ctx, out.n_answers == 1, "basic detect answers size");
    assert_true(ctx, out.answers != nullptr, "basic detect answers buffer");
    if (out.answers != nullptr) {
        assert_true(ctx, out.answers[0] == 1, "basic detect predicted option should be 1");
    }
    assert_true(ctx, out.score == 1, "basic detect score should be 1");

    omr_free_result(&out);
    omr_destroy(h);
}

void test_ambiguous_answer(TestContext& ctx) {
    std::vector<uint8_t> image_storage;
    OMR_ImageView image = make_image(&image_storage, 260, 160);

    draw_dark_annulus(image_storage, image.width, image.height, image.stride, 80, 80, 7, 14, 40u);
    draw_dark_annulus(image_storage, image.width, image.height, image.stride, 180, 80, 7, 14, 40u);

    OMR_CircleROI rois[2]{};
    rois[0] = {80, 80, 16, 0, 0};
    rois[1] = {180, 80, 16, 0, 1};
    int32_t key[1] = {1};

    OMR_FormSpec form{};
    form.output_width = image.width;
    form.output_height = image.height;
    form.circle_rois = rois;
    form.n_circle_rois = 2;
    form.n_questions = 1;
    form.n_options_per_question = 2;
    form.answer_key = key;
    form.n_answer_key = 1;

    OMR_WarpParams warp{};
    OMR_BinarizeParams bin{};
    OMR_GradingParams grading{};
    OMR_RuntimeOptions runtime{};
    setup_defaults(&warp, &bin, &grading, &runtime);
    grading.rel_th = 0.10f;

    OMR_Result out{};
    omr_init_result(&out);
    OMR_Handle* h = omr_create();

    const int32_t rc = omr_process(h, &image, &form, &warp, &bin, &grading, &runtime, &out);
    assert_true(ctx, rc == OMR_OK, "ambiguous detect should succeed");
    if (out.answers != nullptr) {
        assert_true(ctx, out.answers[0] == -1, "ambiguous detect should be unanswered");
    }
    assert_true(ctx, out.score == 0, "ambiguous detect score should be 0");

    omr_free_result(&out);
    omr_destroy(h);
}

void test_answer_key_skip(TestContext& ctx) {
    std::vector<uint8_t> image_storage;
    OMR_ImageView image = make_image(&image_storage, 260, 160);

    OMR_CircleROI rois[2]{};
    rois[0] = {80, 80, 16, 0, 0};
    rois[1] = {180, 80, 16, 0, 1};
    int32_t key[1] = {-1};

    OMR_FormSpec form{};
    form.output_width = image.width;
    form.output_height = image.height;
    form.circle_rois = rois;
    form.n_circle_rois = 2;
    form.n_questions = 1;
    form.n_options_per_question = 2;
    form.answer_key = key;
    form.n_answer_key = 1;

    OMR_WarpParams warp{};
    OMR_BinarizeParams bin{};
    OMR_GradingParams grading{};
    OMR_RuntimeOptions runtime{};
    setup_defaults(&warp, &bin, &grading, &runtime);

    OMR_Result out{};
    omr_init_result(&out);
    OMR_Handle* h = omr_create();

    const int32_t rc = omr_process(h, &image, &form, &warp, &bin, &grading, &runtime, &out);
    assert_true(ctx, rc == OMR_OK, "skip key should succeed");
    assert_true(ctx, out.graded_questions == 0, "graded_questions should be 0 when key is -1");

    omr_free_result(&out);
    omr_destroy(h);
}

void test_duplicate_roi_fails(TestContext& ctx) {
    std::vector<uint8_t> image_storage;
    OMR_ImageView image = make_image(&image_storage, 260, 160);

    OMR_CircleROI rois[2]{};
    rois[0] = {80, 80, 16, 0, 0};
    rois[1] = {100, 80, 16, 0, 0};  // duplicate (q,opt)
    int32_t key[1] = {0};

    OMR_FormSpec form{};
    form.output_width = image.width;
    form.output_height = image.height;
    form.circle_rois = rois;
    form.n_circle_rois = 2;
    form.n_questions = 1;
    form.n_options_per_question = 2;
    form.answer_key = key;
    form.n_answer_key = 1;

    OMR_WarpParams warp{};
    OMR_BinarizeParams bin{};
    OMR_GradingParams grading{};
    OMR_RuntimeOptions runtime{};
    setup_defaults(&warp, &bin, &grading, &runtime);

    OMR_Result out{};
    omr_init_result(&out);
    OMR_Handle* h = omr_create();

    const int32_t rc = omr_process(h, &image, &form, &warp, &bin, &grading, &runtime, &out);
    assert_true(ctx, rc == OMR_ERR_INVALID_ROI_LAYOUT, "duplicate ROI should fail");

    omr_free_result(&out);
    omr_destroy(h);
}

void test_incomplete_question_roi_fails(TestContext& ctx) {
    std::vector<uint8_t> image_storage;
    OMR_ImageView image = make_image(&image_storage, 260, 160);

    OMR_CircleROI rois[1]{};
    rois[0] = {80, 80, 16, 0, 0};
    int32_t key[1] = {0};

    OMR_FormSpec form{};
    form.output_width = image.width;
    form.output_height = image.height;
    form.circle_rois = rois;
    form.n_circle_rois = 1;
    form.n_questions = 1;
    form.n_options_per_question = 2;
    form.answer_key = key;
    form.n_answer_key = 1;

    OMR_WarpParams warp{};
    OMR_BinarizeParams bin{};
    OMR_GradingParams grading{};
    OMR_RuntimeOptions runtime{};
    setup_defaults(&warp, &bin, &grading, &runtime);

    OMR_Result out{};
    omr_init_result(&out);
    OMR_Handle* h = omr_create();

    const int32_t rc = omr_process(h, &image, &form, &warp, &bin, &grading, &runtime, &out);
    assert_true(ctx, rc == OMR_ERR_INVALID_ROI_LAYOUT, "incomplete question ROI should fail");

    omr_free_result(&out);
    omr_destroy(h);
}

void test_warp_missing_markers_fails(TestContext& ctx) {
    std::vector<uint8_t> image_storage;
    OMR_ImageView image = make_image(&image_storage, 260, 160);

    OMR_CircleROI rois[2]{};
    rois[0] = {80, 80, 16, 0, 0};
    rois[1] = {180, 80, 16, 0, 1};
    int32_t key[1] = {0};

    OMR_MarkerTemplate markers[4]{};
    markers[0] = {1, 10.f, 10.f};
    markers[1] = {2, 10.f, 120.f};
    markers[2] = {3, 240.f, 10.f};
    markers[3] = {4, 240.f, 120.f};

    OMR_RegionWindow windows[1]{};
    windows[0].marker_ids[0] = 1;
    windows[0].marker_ids[1] = 2;
    windows[0].marker_ids[2] = 3;
    windows[0].marker_ids[3] = 4;
    windows[0].n_marker_ids = 4;

    OMR_FormSpec form{};
    form.output_width = image.width;
    form.output_height = image.height;
    form.template_markers = markers;
    form.n_template_markers = 4;
    form.region_windows = windows;
    form.n_region_windows = 1;
    form.circle_rois = rois;
    form.n_circle_rois = 2;
    form.n_questions = 1;
    form.n_options_per_question = 2;
    form.answer_key = key;
    form.n_answer_key = 1;

    OMR_WarpParams warp{};
    OMR_BinarizeParams bin{};
    OMR_GradingParams grading{};
    OMR_RuntimeOptions runtime{};
    setup_defaults(&warp, &bin, &grading, &runtime);
    runtime.assume_aligned_input = 0;

    OMR_Result out{};
    omr_init_result(&out);
    OMR_Handle* h = omr_create();

    const int32_t rc = omr_process(h, &image, &form, &warp, &bin, &grading, &runtime, &out);
    assert_true(ctx, rc == OMR_ERR_INSUFFICIENT_MARKERS, "warp path without detectable markers should fail");

    omr_free_result(&out);
    omr_destroy(h);
}

void test_warp_identity_success(TestContext& ctx) {
    std::vector<uint8_t> image_storage;
    OMR_ImageView image = make_image(&image_storage, 260, 160);

    draw_dark_annulus(image_storage, image.width, image.height, image.stride, 180, 80, 7, 14, 30u);

    OMR_CircleROI rois[2]{};
    rois[0] = {80, 80, 16, 0, 0};
    rois[1] = {180, 80, 16, 0, 1};
    int32_t key[1] = {1};

    OMR_MarkerTemplate markers[4]{};
    markers[0] = {1, 20.f, 20.f};
    markers[1] = {2, 20.f, 140.f};
    markers[2] = {3, 240.f, 20.f};
    markers[3] = {4, 240.f, 140.f};

    OMR_DetectedMarker detected[4]{};
    detected[0] = {1, 20.f, 20.f};
    detected[1] = {2, 20.f, 140.f};
    detected[2] = {3, 240.f, 20.f};
    detected[3] = {4, 240.f, 140.f};

    OMR_RegionWindow windows[1]{};
    windows[0].marker_ids[0] = 1;
    windows[0].marker_ids[1] = 2;
    windows[0].marker_ids[2] = 3;
    windows[0].marker_ids[3] = 4;
    windows[0].n_marker_ids = 4;

    OMR_FormSpec form{};
    form.output_width = image.width;
    form.output_height = image.height;
    form.template_markers = markers;
    form.n_template_markers = 4;
    form.detected_markers = detected;
    form.n_detected_markers = 4;
    form.region_windows = windows;
    form.n_region_windows = 1;
    form.circle_rois = rois;
    form.n_circle_rois = 2;
    form.n_questions = 1;
    form.n_options_per_question = 2;
    form.answer_key = key;
    form.n_answer_key = 1;

    OMR_WarpParams warp{};
    OMR_BinarizeParams bin{};
    OMR_GradingParams grading{};
    OMR_RuntimeOptions runtime{};
    setup_defaults(&warp, &bin, &grading, &runtime);
    runtime.assume_aligned_input = 0;
    warp.use_global_idw = 0;
    warp.use_region_refine = 0;

    OMR_Result out{};
    omr_init_result(&out);
    OMR_Handle* h = omr_create();

    const int32_t rc = omr_process(h, &image, &form, &warp, &bin, &grading, &runtime, &out);
    assert_true(ctx, rc == OMR_OK, "warp identity should succeed");
    if (out.answers != nullptr) {
        assert_true(ctx, out.answers[0] == 1, "warp identity should preserve answer");
    }

    omr_free_result(&out);
    omr_destroy(h);
}

void test_warp_auto_detect_success(TestContext& ctx) {
    std::vector<uint8_t> image_storage;
    OMR_ImageView image = make_image(&image_storage, 260, 160);

    draw_dark_annulus(image_storage, image.width, image.height, image.stride, 180, 80, 7, 14, 30u);
    draw_square_ring(image_storage, image.width, image.height, image.stride, 20, 20, 12, 3, 0u);
    draw_square_ring(image_storage, image.width, image.height, image.stride, 20, 140, 12, 3, 0u);
    draw_square_ring(image_storage, image.width, image.height, image.stride, 240, 20, 12, 3, 0u);
    draw_square_ring(image_storage, image.width, image.height, image.stride, 240, 140, 12, 3, 0u);

    OMR_CircleROI rois[2]{};
    rois[0] = {80, 80, 16, 0, 0};
    rois[1] = {180, 80, 16, 0, 1};
    int32_t key[1] = {1};

    OMR_MarkerTemplate markers[4]{};
    markers[0] = {1, 20.f, 20.f};
    markers[1] = {2, 20.f, 140.f};
    markers[2] = {3, 240.f, 20.f};
    markers[3] = {4, 240.f, 140.f};

    OMR_RegionWindow windows[1]{};
    windows[0].marker_ids[0] = 1;
    windows[0].marker_ids[1] = 2;
    windows[0].marker_ids[2] = 3;
    windows[0].marker_ids[3] = 4;
    windows[0].n_marker_ids = 4;

    OMR_FormSpec form{};
    form.output_width = image.width;
    form.output_height = image.height;
    form.template_markers = markers;
    form.n_template_markers = 4;
    form.region_windows = windows;
    form.n_region_windows = 1;
    form.circle_rois = rois;
    form.n_circle_rois = 2;
    form.n_questions = 1;
    form.n_options_per_question = 2;
    form.answer_key = key;
    form.n_answer_key = 1;

    OMR_WarpParams warp{};
    OMR_BinarizeParams bin{};
    OMR_GradingParams grading{};
    OMR_RuntimeOptions runtime{};
    setup_defaults(&warp, &bin, &grading, &runtime);
    runtime.assume_aligned_input = 0;
    warp.use_global_idw = 0;
    warp.use_region_refine = 0;

    OMR_Result out{};
    omr_init_result(&out);
    OMR_Handle* h = omr_create();

    const int32_t rc = omr_process(h, &image, &form, &warp, &bin, &grading, &runtime, &out);
    assert_true(ctx, rc == OMR_OK, "warp auto detect should succeed on synthetic square markers");
    if (out.answers != nullptr) {
        assert_true(ctx, out.answers[0] == 1, "warp auto detect should preserve answer");
    }

    omr_free_result(&out);
    omr_destroy(h);
}

void test_warp_with_global_idw_smoke(TestContext& ctx) {
    std::vector<uint8_t> image_storage;
    OMR_ImageView image = make_image(&image_storage, 260, 160);

    draw_dark_annulus(image_storage, image.width, image.height, image.stride, 180, 80, 7, 14, 30u);
    draw_square_ring(image_storage, image.width, image.height, image.stride, 20, 20, 12, 3, 0u);
    draw_square_ring(image_storage, image.width, image.height, image.stride, 20, 140, 12, 3, 0u);
    draw_square_ring(image_storage, image.width, image.height, image.stride, 240, 20, 12, 3, 0u);
    draw_square_ring(image_storage, image.width, image.height, image.stride, 240, 140, 12, 3, 0u);

    OMR_CircleROI rois[2]{};
    rois[0] = {80, 80, 16, 0, 0};
    rois[1] = {180, 80, 16, 0, 1};
    int32_t key[1] = {1};

    OMR_MarkerTemplate markers[4]{};
    markers[0] = {1, 20.f, 20.f};
    markers[1] = {2, 20.f, 140.f};
    markers[2] = {3, 240.f, 20.f};
    markers[3] = {4, 240.f, 140.f};

    OMR_RegionWindow windows[1]{};
    windows[0].marker_ids[0] = 1;
    windows[0].marker_ids[1] = 2;
    windows[0].marker_ids[2] = 3;
    windows[0].marker_ids[3] = 4;
    windows[0].n_marker_ids = 4;

    OMR_FormSpec form{};
    form.output_width = image.width;
    form.output_height = image.height;
    form.template_markers = markers;
    form.n_template_markers = 4;
    form.region_windows = windows;
    form.n_region_windows = 1;
    form.circle_rois = rois;
    form.n_circle_rois = 2;
    form.n_questions = 1;
    form.n_options_per_question = 2;
    form.answer_key = key;
    form.n_answer_key = 1;

    OMR_WarpParams warp{};
    OMR_BinarizeParams bin{};
    OMR_GradingParams grading{};
    OMR_RuntimeOptions runtime{};
    setup_defaults(&warp, &bin, &grading, &runtime);
    runtime.assume_aligned_input = 0;
    warp.use_global_idw = 1;
    warp.use_region_refine = 0;

    OMR_Result out{};
    omr_init_result(&out);
    OMR_Handle* h = omr_create();

    const int32_t rc = omr_process(h, &image, &form, &warp, &bin, &grading, &runtime, &out);
    assert_true(ctx, rc == OMR_OK, "warp with global idw should succeed");
    if (out.answers != nullptr) {
        assert_true(ctx, out.answers[0] == 1, "warp with global idw should preserve answer");
    }

    omr_free_result(&out);
    omr_destroy(h);
}

void test_region_refine_smoke(TestContext& ctx) {
    std::vector<uint8_t> image_storage;
    OMR_ImageView image = make_image(&image_storage, 260, 160);

    draw_dark_annulus(image_storage, image.width, image.height, image.stride, 180, 80, 7, 14, 30u);
    draw_square_ring(image_storage, image.width, image.height, image.stride, 20, 20, 12, 3, 0u);
    draw_square_ring(image_storage, image.width, image.height, image.stride, 20, 140, 12, 3, 0u);
    draw_square_ring(image_storage, image.width, image.height, image.stride, 240, 20, 12, 3, 0u);
    draw_square_ring(image_storage, image.width, image.height, image.stride, 240, 140, 12, 3, 0u);

    OMR_CircleROI rois[2]{};
    rois[0] = {80, 80, 16, 0, 0};
    rois[1] = {180, 80, 16, 0, 1};
    int32_t key[1] = {1};

    OMR_MarkerTemplate markers[4]{};
    markers[0] = {1, 20.f, 20.f};
    markers[1] = {2, 20.f, 140.f};
    markers[2] = {3, 240.f, 20.f};
    markers[3] = {4, 240.f, 140.f};

    OMR_DetectedMarker detected[4]{};
    detected[0] = {1, 20.f, 20.f};
    detected[1] = {2, 20.f, 140.f};
    detected[2] = {3, 240.f, 20.f};
    detected[3] = {4, 240.f, 140.f};

    OMR_RegionWindow windows[1]{};
    windows[0].marker_ids[0] = 1;
    windows[0].marker_ids[1] = 2;
    windows[0].marker_ids[2] = 3;
    windows[0].marker_ids[3] = 4;
    windows[0].n_marker_ids = 4;

    OMR_FormSpec form{};
    form.output_width = image.width;
    form.output_height = image.height;
    form.template_markers = markers;
    form.n_template_markers = 4;
    form.detected_markers = detected;
    form.n_detected_markers = 4;
    form.region_windows = windows;
    form.n_region_windows = 1;
    form.circle_rois = rois;
    form.n_circle_rois = 2;
    form.n_questions = 1;
    form.n_options_per_question = 2;
    form.answer_key = key;
    form.n_answer_key = 1;

    OMR_WarpParams warp{};
    OMR_BinarizeParams bin{};
    OMR_GradingParams grading{};
    OMR_RuntimeOptions runtime{};
    setup_defaults(&warp, &bin, &grading, &runtime);
    runtime.assume_aligned_input = 0;
    warp.use_global_idw = 0;
    warp.use_region_refine = 1;

    OMR_Result out{};
    omr_init_result(&out);
    OMR_Handle* h = omr_create();

    const int32_t rc = omr_process(h, &image, &form, &warp, &bin, &grading, &runtime, &out);
    assert_true(ctx, rc == OMR_OK, "region refine should succeed");
    if (out.answers != nullptr) {
        assert_true(ctx, out.answers[0] == 1, "region refine should preserve answer");
    }

    omr_free_result(&out);
    omr_destroy(h);
}

}  // namespace

int main() {
    TestContext ctx{};

    assert_true(ctx, omr_api_version() == OMR_API_VERSION, "api version should match header");
    test_basic_detect(ctx);
    test_ambiguous_answer(ctx);
    test_answer_key_skip(ctx);
    test_duplicate_roi_fails(ctx);
    test_incomplete_question_roi_fails(ctx);
    test_warp_missing_markers_fails(ctx);
    test_warp_identity_success(ctx);
    test_warp_auto_detect_success(ctx);
    test_warp_with_global_idw_smoke(ctx);
    test_region_refine_smoke(ctx);

    std::printf("[RESULT] passed=%d failed=%d\n", ctx.passed, ctx.failed);
    return (ctx.failed == 0) ? 0 : 1;
}
