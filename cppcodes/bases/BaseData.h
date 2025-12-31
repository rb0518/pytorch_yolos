#pragma once
#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include <vector>
#include <string>
#include <tuple>

#include "globaldef.h"

enum class VDK {    //VariantData Map's key
    FILENAME,
    INDEX,
    SRC_SIZE,
    IMAGE,
    IMG,
    CLS,
    BBOXES,
    BATCH_IDX,
    RATIO,
    PAD,
    RESIZED,
    SEGMENTS,
    MOSAIC_BORDER,
    MASK
};

std::string GET_VDK(VDK key);

using DataComplex = std::variant<
                                // 最终传递变量类型
                                torch::Tensor,                          // img, cls, bboxes, batch_idx, masks, segments
                                std::vector<std::tuple<float, float>>,  // src_size, resized
                                std::vector<std::string>,               // filename 
                                // 临时变量类型
                                cv::Mat,
                                std::vector<std::vector<float>>,        // bboxes, segments 
                                std::vector<float>,                     // mosaic border

                                std::vector<torch::Tensor>,
                                std::vector<int>,
                                int>;                                   // index, xyxy_flags
using VariantData = std::map<std::string, DataComplex>;

class BaseTransformArray;

class BaseDataset 
{
public:
    explicit BaseDataset(const std::string& path,
        VariantConfigs* pargs, 
        int stride);

    void initFinal();
    bool getAugment() { return m_bAugment; }
    int getSamplesCount() { return m_imgFiles.size(); }
    VariantData getImageAndLabel(size_t index);
public:
    virtual void getSpecialSettins() = 0;
    virtual VariantData getOneSample(size_t index) = 0;

protected:
    void getSettings();

    std::tuple<cv::Mat, std::tuple<float, float>, std::tuple<float, float>> loadImage(size_t index);
    std::vector<std::vector<float>> getLabel(size_t index);

    std::tuple<std::string, std::string> getImageLabelLists(const std::string& img_path,
        std::vector<std::string>& img_files, std::vector<std::string>& label_files);

    std::vector<std::vector<float>> readLabelsFromFile(const std::string& filename);
    VariantData transLabelsToBbox(std::vector<std::vector<float>>& labels, VariantData& data);
protected:
    std::string m_path;
    VariantConfigs* m_pArgs;

    int m_stride = 32;
    int m_imgSize = 640;

    int m_totalSamples = 0;

    bool m_bRect = false;
    bool m_bMosaic = false;
    bool m_bAugment = false;
    bool m_bSingleClass = false;

    float m_fPad = 0.0f;

    std::vector<std::string> m_imgFiles;
    std::vector<std::string> m_labelFiles;
    std::string m_imagesDir;
    std::string m_labelsDir;
    std::vector<int> m_indices;
};

class BaseDataloader
{
public:
    explicit BaseDataloader(const std::string& root_path,
        VariantConfigs& args_input,
        int stride = 32,
        bool is_val = false
    );

    bool getIsVal() { return m_isVal; }
    std::string getPath() { return m_datasetsPath; }
    int getTotalSamples() { return m_totalNumber; }
    int getBatchSize() { return m_batchSize; }
    int getNumberOfBatch();
    void closeMosaic(bool bFlag);

    void initDatas();
public:
    /*
        train 中利用getBegin()取得iterator，通过for(itr = XX.getBegin(); itr != XX.getEnd(); itr++)
    */
    virtual void createDataloader() = 0;
    virtual torch::data::Iterator<VariantData> getBegin() = 0;
    virtual torch::data::Iterator<VariantData> getEnd() = 0;

protected:
    void getDataSets(const std::string root);
protected:
    bool m_isVal = false;
    std::string m_datasetsPath;
    VariantConfigs m_args;
    int m_stride = 32;
    int m_totalNumber = 0;
    int m_batchSize = 16;
    std::vector<std::string> m_names;
};

class BaseTransform
{
public:
    BaseTransform( 
        VariantConfigs* pargs,
        int imgsz, 
        int dataset_num)
        : m_pArgs(pargs), m_imageSize(imgsz), m_datasetSize(dataset_num)
    {}
    virtual void doTransform(BaseDataset* pdataset, size_t index, VariantData& pInput) = 0;
    virtual void getSpecialSets() = 0;

    void initialize() {
        getSpecialSets();
    }
protected:

    void updateLabels(VariantData& label, int w, int h, int padw, int padh);
public:
    int m_imageSize = 640;
    int m_datasetSize;
    VariantConfigs* m_pArgs;
};

class BaseTransformArray
{
public:
    BaseTransformArray(VariantConfigs* args, int imgsz, int dataset_size, bool augment);
    
    void initAllTrans();
    void doTransforms(BaseDataset* pdataset, size_t index, VariantData& pData);
    int getTransCount() { return m_trans.size(); }

    virtual void buildTranforms(bool bAugment) = 0;
protected:
    int m_imageSize = 640;
    VariantConfigs* m_pArgs;
    int m_datasetSize;
    bool m_bAugment = true;
protected:
    std::vector<std::shared_ptr<BaseTransform>> m_trans;
};