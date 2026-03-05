#include "omr_api.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void draw_dark_annulus(
    unsigned char* img,
    int width,
    int height,
    int stride,
    int cx,
    int cy,
    int r_inner,
    int r_outer,
    unsigned char value
) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int dx = x - cx;
            int dy = y - cy;
            int d2 = dx * dx + dy * dy;
            if (d2 >= r_inner * r_inner && d2 <= r_outer * r_outer) {
                img[y * stride + x] = value;
            }
        }
    }
}

int main(void) {
    const int width = 220;
    const int height = 120;
    const int channels = 1;
    const int stride = width * channels;

    unsigned char* image = (unsigned char*)malloc((size_t)stride * (size_t)height);
    if (image == NULL) {
        fprintf(stderr, "failed to allocate image\n");
        return 1;
    }
    memset(image, 255, (size_t)stride * (size_t)height);

    draw_dark_annulus(image, width, height, stride, 150, 60, 6, 12, 30);

    OMR_ImageView img = {width, height, stride, channels, image};

    OMR_CircleROI rois[2];
    rois[0].cx = 70; rois[0].cy = 60; rois[0].r = 15; rois[0].question = 0; rois[0].option = 0;
    rois[1].cx = 150; rois[1].cy = 60; rois[1].r = 15; rois[1].question = 0; rois[1].option = 1;

    int32_t answer_key[1] = {1};

    OMR_FormSpec form;
    memset(&form, 0, sizeof(form));
    form.output_width = width;
    form.output_height = height;
    form.circle_rois = rois;
    form.n_circle_rois = 2;
    form.n_questions = 1;
    form.n_options_per_question = 2;
    form.answer_key = answer_key;
    form.n_answer_key = 1;

    OMR_WarpParams warp_params;
    OMR_BinarizeParams bin_params;
    OMR_GradingParams grading_params;
    OMR_RuntimeOptions runtime_options;
    OMR_Result result;

    omr_init_default_warp_params(&warp_params);
    omr_init_default_binarize_params(&bin_params);
    omr_init_default_grading_params(&grading_params);
    omr_init_default_runtime_options(&runtime_options);
    omr_init_result(&result);

    grading_params.min_valid_pixels = 5;
    runtime_options.assume_aligned_input = 1;
    runtime_options.return_scored_image = 0;

    OMR_Handle* handle = omr_create();
    if (handle == NULL) {
        fprintf(stderr, "omr_create failed\n");
        free(image);
        return 1;
    }

    int32_t rc = omr_process(
        handle,
        &img,
        &form,
        &warp_params,
        &bin_params,
        &grading_params,
        &runtime_options,
        &result
    );

    if (rc != OMR_OK) {
        fprintf(stderr, "omr_process failed: %s (%s)\n",
                omr_error_code_to_string(rc),
                result.error_message);
    } else {
        printf("score: %d/%d\n", result.score, result.graded_questions);
        if (result.n_answers > 0 && result.answers != NULL) {
            printf("answer[0]=%d\n", result.answers[0]);
        }
    }

    omr_free_result(&result);
    omr_destroy(handle);
    free(image);
    return (rc == OMR_OK) ? 0 : 2;
}
