#include "pred.h"
#include "yolo.h"
#include "nms.h"
#include "utils.h"
#include "plots.h"

#include <filesystem>
namespace fs = std::filesystem;

DetectPredict::DetectPredict(VariantConfigs& args_input, std::string root_dir)
	: BasePredict(args_input, root_dir)
{
}

void DetectPredict::readSpecialSettings()
{
}

std::shared_ptr<BaseModelImpl> DetectPredict::createModel(std::string model_cfg)
{
	return std::make_shared<ModelImpl>(model_cfg, 3, &m_args);
}

void DetectPredict::loadWeights(std::string model_cfg)
{
    std::cout << ColorString("Load weights: ") << model_cfg << std::endl;
    torch::serialize::InputArchive ckpt_in;
    ckpt_in.load_from(model_cfg);
    m_modelPtr->load_weights(ckpt_in);
    std::cout << " over." << std::endl;
}

void DetectPredict::setupSpecialObjects()
{
}

std::vector<torch::Tensor> DetectPredict::postprocess(torch::Tensor preds)
{
	return non_max_suppression(preds);
}

void DetectPredict::plotsResultToMat(cv::Mat src_image, std::string filename, std::vector<torch::Tensor> bboxes, torch::Tensor protos)
{
    // 1 save source image
    auto filename_path = fs::path(filename);
    std::string source_filename = fs::path(m_saveDir).append(filename_path.filename().string()).string();
    cv::imwrite(source_filename, src_image);

    // draw bbox result to image
    torch::Tensor boxs = bboxes[0];  // predict batch_size = 1
    int nobjs = boxs.size(0);
    if(boxs.size(0) == 0)
        return;


    drawBoxsToMatImage(src_image, std::make_tuple(float(m_imageSize), float(m_imageSize)),
        boxs, m_bScaleFill);

    std::string pred_filename = fs::path(m_saveDir).append(filename_path.stem().string() + "_pred.jpg").string();
    
    if(m_bshowPred)
    {
        cv::imshow(filename_path.stem().string(), src_image);
        cv::waitKey();
    }
    cv::imwrite(pred_filename, src_image);
}
