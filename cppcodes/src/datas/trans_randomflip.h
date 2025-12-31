#pragma once
#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include "BaseData.h"

class RandomFlip : public BaseTransform
{
public:
    explicit RandomFlip(VariantConfigs* pargs,
        int imgsz, 
        int dataset_num);

    virtual void getSpecialSets() override;
    virtual void doTransform(BaseDataset* pdataset, size_t index, VariantData& pInput) override;
private:
    float m_flipudProb = 0.5f; // Probability for applying vertical
    float m_fliplrProb = 0.5f; // Probability for applying Horizontal 

protected:

};