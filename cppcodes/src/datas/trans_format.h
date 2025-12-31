#pragma once
#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include "BaseData.h"
/*
* 放在最后，输出训练所需要的格式
*/
class FormatOutput : public BaseTransform
{
public:
    explicit FormatOutput(VariantConfigs* pargs,
        int imgsz, 
        int dataset_num);

    virtual void getSpecialSets() override;
    virtual void doTransform(BaseDataset* pdataset, size_t index, VariantData& pInput) override;
private:
    bool m_overlayMask = true;
    float m_maskRatio = 4;

private:
    torch::Tensor convertImageToTensor(cv::Mat image);
};
