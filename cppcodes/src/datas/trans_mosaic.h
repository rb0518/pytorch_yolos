#pragma once
#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include "BaseData.h"

class Mosaic : public BaseTransform
{
public:
    explicit Mosaic(VariantConfigs* pargs,
        int imgsz, 
        int dataset_num);

    virtual void getSpecialSets();
    virtual void doTransform(BaseDataset* pdataset, size_t index, VariantData& pInput);
private:
    std::vector<int> getIndexes(size_t index);
//    void updateLabels(VariantData& label, int w, int h, int padw, int padh);

protected:
    float m_mosaicProb =  1.0f;
    std::vector<float> m_border;
};
