#pragma once
#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include "BaseData.h"

class LetterBox : public BaseTransform
{
public:
    explicit LetterBox(VariantConfigs* pargs,
        int imgsz, 
        int dataset_num);

    virtual void getSpecialSets();
    virtual void doTransform(BaseDataset* pdataset, size_t index, VariantData& pInput);
private:

protected:

};
