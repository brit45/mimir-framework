#pragma once

#include <vector>

namespace RuntimeLossGrad {

struct GlobalSSIM {
    double loss = 0.0;
    std::vector<float> grad;
};

struct LossWithGrad {
    double loss = 0.0;
    std::vector<float> grad;
};

float sigmoid_scalar(float x);

GlobalSSIM ssim_global_hwc(
    const std::vector<float>& pred,
    const std::vector<float>& target,
    int w,
    int h,
    int c,
    float k1,
    float k2,
    float L
);

void avgpool2x2_hwc(
    const std::vector<float>& in,
    int w,
    int h,
    int c,
    std::vector<float>& out,
    int& out_w,
    int& out_h
);

void avgpool2x2_back_hwc(
    const std::vector<float>& grad_out,
    int in_w,
    int in_h,
    int c,
    std::vector<float>& grad_in
);

LossWithGrad spectral_dct_l1_hwc(
    const std::vector<float>& pred,
    const std::vector<float>& target,
    int w,
    int h,
    int c
);

} // namespace RuntimeLossGrad
