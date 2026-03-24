#ifndef OMR_API_H
#define OMR_API_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define OMR_API_VERSION 1u
#define OMR_ERROR_MESSAGE_CAPACITY 256

#if defined(_WIN32)
    #if defined(OMR_BUILD_DLL)
        #define OMR_API __declspec(dllexport)
    #elif defined(OMR_USE_DLL)
        #define OMR_API __declspec(dllimport)
    #else
        #define OMR_API
    #endif
#else
    #define OMR_API
#endif

typedef struct OMR_Handle OMR_Handle;

typedef enum OMR_ErrorCode {
    OMR_OK = 0,
    OMR_ERR_NULL_ARGUMENT = 1,
    OMR_ERR_INVALID_HANDLE = 2,
    OMR_ERR_BAD_IMAGE = 3,
    OMR_ERR_BAD_CONFIG = 4,
    OMR_ERR_INVALID_ROI_LAYOUT = 5,
    OMR_ERR_ALLOCATION_FAILED = 6,
    OMR_ERR_INSUFFICIENT_MARKERS = 7,
    OMR_ERR_WARP_FAILED = 8,
    OMR_ERR_NOT_IMPLEMENTED = 9,
    OMR_ERR_INTERNAL = 10
} OMR_ErrorCode;

typedef enum OMR_AprilTagDict {
    OMR_TAG_DICT_APRILTAG_16H5 = 0
} OMR_AprilTagDict;

typedef enum OMR_BinarizeMethod {
    OMR_BINARIZE_DUAL = 0,
    OMR_BINARIZE_SKELETON = 1
} OMR_BinarizeMethod;

typedef enum OMR_SelectionMode {
    OMR_SELECTION_SINGLE = 0,
    OMR_SELECTION_MULTIPLE = 1
} OMR_SelectionMode;

typedef enum OMR_QuestionStatus {
    OMR_STATUS_BLANK = 0,
    OMR_STATUS_SINGLE = 1,
    OMR_STATUS_MULTIPLE = 2,
    OMR_STATUS_INVALID_MULTIPLE_ON_SINGLE = 3,
    OMR_STATUS_UNCERTAIN = 4
} OMR_QuestionStatus;

typedef struct OMR_ImageView {
    int32_t width;
    int32_t height;
    int32_t stride;
    int32_t channels;
    const uint8_t* data;
} OMR_ImageView;

typedef struct OMR_MarkerTemplate {
    int32_t id;
    float x;
    float y;
} OMR_MarkerTemplate;

typedef struct OMR_DetectedMarker {
    int32_t id;
    float x;
    float y;
} OMR_DetectedMarker;

typedef struct OMR_RegionWindow {
    int32_t marker_ids[4];
    int32_t n_marker_ids;
} OMR_RegionWindow;

typedef struct OMR_CircleROI {
    int32_t cx;
    int32_t cy;
    int32_t r;
    int32_t question;
    int32_t option;
    int32_t selection_mode;
} OMR_CircleROI;

typedef struct OMR_MetadataField {
    int32_t field_id;
    int32_t n_columns;
    int32_t n_rows;
} OMR_MetadataField;

typedef struct OMR_MetadataBubble {
    int32_t field_id;
    int32_t column;
    int32_t row;
    int32_t cx;
    int32_t cy;
    int32_t r;
} OMR_MetadataBubble;

typedef struct OMR_FormSpec {
    int32_t output_width;
    int32_t output_height;

    const OMR_MarkerTemplate* template_markers;
    int32_t n_template_markers;

    const OMR_DetectedMarker* detected_markers;
    int32_t n_detected_markers;

    const OMR_RegionWindow* region_windows;
    int32_t n_region_windows;

    const OMR_CircleROI* circle_rois;
    int32_t n_circle_rois;

    const OMR_MetadataField* metadata_fields;
    int32_t n_metadata_fields;

    const OMR_MetadataBubble* metadata_bubbles;
    int32_t n_metadata_bubbles;

    int32_t n_questions;
    int32_t n_options_per_question;

    const int32_t* answer_key;
    int32_t n_answer_key;
} OMR_FormSpec;

typedef struct OMR_WarpParams {
    int32_t apriltag_dict;
    float global_h_ransac_thresh;
    float local_h_ransac_thresh;
    int32_t region_bbox_margin_px;

    int32_t use_global_idw;
    int32_t use_region_refine;

    int32_t global_idw_grid_w;
    int32_t global_idw_grid_h;
    float global_idw_power;
    float global_idw_eps;

    int32_t patch_idw_grid_w;
    int32_t patch_idw_grid_h;
    float patch_idw_power;
    float patch_idw_eps;

    float skip_idw_if_residual_lt_px;
    float residual_breakpoints_px[3];
    float residual_factors[4];

    int32_t reserved[8];
} OMR_WarpParams;

typedef struct OMR_BinarizeParams {
    int32_t method;
    int32_t blur_ksize;
    float fill_percentile;
    int32_t thin_iterations;
    int32_t denoise_kernel_size;

    int32_t reserved[8];
} OMR_BinarizeParams;

typedef struct OMR_GradingParams {
    float abs_th;
    float rel_th;
    int32_t auto_threshold;

    float clahe_clip_limit;
    int32_t clahe_tile_w;
    int32_t clahe_tile_h;
    int32_t gaussian_ksize_w;
    int32_t gaussian_ksize_h;
    float gaussian_sigma;

    float patch_radius_multiplier;
    float fill_inner_ratio;
    float fill_outer_ratio;
    float bg_inner_ratio;
    float bg_outer_ratio;
    int32_t min_valid_pixels;

    int32_t min_questions_for_calibration;
    float calibration_percentile;
    float abs_th_mad_multiplier;
    float rel_th_mad_multiplier;
    float abs_th_baseline_offset;
    float rel_th_baseline_offset;
    float abs_th_min;
    float abs_th_max;
    float rel_th_min;
    float rel_th_max;

    int32_t reserved[8];
} OMR_GradingParams;

typedef struct OMR_RuntimeOptions {
    int32_t assume_aligned_input;
    int32_t return_scored_image;
    int32_t return_intermediate_steps;
    int32_t debug_level;

    int32_t reserved[8];
} OMR_RuntimeOptions;

typedef struct OMR_Result {
    int32_t err_code;
    char error_message[OMR_ERROR_MESSAGE_CAPACITY];

    int32_t score;
    int32_t total_questions;
    int32_t graded_questions;

    int32_t n_answers;
    int32_t* answers;

    int32_t n_selected_option_flags;
    int32_t* selected_option_flags;

    int32_t n_question_statuses;
    int32_t* question_statuses;

    int32_t n_metadata_selected_rows;
    int32_t* metadata_selected_rows;

    float used_abs_th;
    float used_rel_th;

    uint8_t* scored_image_data;
    int32_t scored_image_width;
    int32_t scored_image_height;
    int32_t scored_image_stride;
    int32_t scored_image_channels;
} OMR_Result;

OMR_API uint32_t omr_api_version(void);

OMR_API OMR_Handle* omr_create(void);
OMR_API void omr_destroy(OMR_Handle* handle);

OMR_API void omr_init_result(OMR_Result* out_result);
OMR_API void omr_free_result(OMR_Result* out_result);

OMR_API void omr_init_default_warp_params(OMR_WarpParams* out_params);
OMR_API void omr_init_default_binarize_params(OMR_BinarizeParams* out_params);
OMR_API void omr_init_default_grading_params(OMR_GradingParams* out_params);
OMR_API void omr_init_default_runtime_options(OMR_RuntimeOptions* out_options);

OMR_API int32_t omr_process(
    OMR_Handle* handle,
    const OMR_ImageView* image,
    const OMR_FormSpec* form,
    const OMR_WarpParams* warp_params,
    const OMR_BinarizeParams* bin_params,
    const OMR_GradingParams* grading_params,
    const OMR_RuntimeOptions* runtime_options,
    OMR_Result* out_result
);

OMR_API const char* omr_error_code_to_string(int32_t code);

#ifdef __cplusplus
}
#endif

#endif
