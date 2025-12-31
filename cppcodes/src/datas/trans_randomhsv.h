#pragma once
#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include "BaseData.h"

class RandomHSV : public BaseTransform
{
public:
    explicit RandomHSV(VariantConfigs* pargs,
        int imgsz, 
        int dataset_num);

    virtual void getSpecialSets() override;
    virtual void doTransform(BaseDataset* pdataset, size_t index, VariantData& pInput) override;
private:
    float m_hgain = 0.5f;
    float m_sgain = 0.5f;
    float m_vgain = 0.5f;

    cv::Mat lut_hue, lut_sat, lut_val;
protected:

};