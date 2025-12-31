#pragma once
#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include "BaseData.h"

class RandomPerspective : public BaseTransform
{
public:
    explicit RandomPerspective(VariantConfigs* pargs,
        int imgsz, 
        int dataset_num);

    virtual void getSpecialSets() override;
    virtual void doTransform(BaseDataset* pdataset, size_t index, VariantData& pInput) override;
private:
    std::tuple<cv::Mat, torch::Tensor, float> affine_transform(cv::Mat img, std::vector<float> border);

    torch::Tensor apply_bboxes(VariantData& labels, torch::Tensor M);
    std::tuple<torch::Tensor, torch::Tensor> apply_segments(VariantData& labels, torch::Tensor M);
private:
    float m_degrees = 0.0f;
    float m_translate = 0.1f;
    float m_scale = 0.5f;
    float m_shear = 0.0;
    float m_perspective = 0.0f;

    std::vector<float> m_size;

    torch::Tensor box_candidates(const torch::Tensor box1,
        const torch::Tensor box2,
        float area_thr = 0.1f,
        float wh_thr = 2.0f,
        float ar_thr = 100.0f,
        float eps = 1e-16f);
};
