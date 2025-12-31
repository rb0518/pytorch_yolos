#pragma once
#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include "BaseValidate.h"
#include "BaseModel.h"
#include "BaseData.h"

class DetectValidator : public BaseValidate
{
public:
    DetectValidator(VariantConfigs& args_input, std::string root_dir,
        std::shared_ptr<BaseModelImpl> model_ptr = nullptr, 
        std::shared_ptr<BaseLoss> loss_ptr = nullptr);

public:
    virtual void readSpecialSettins() override;
    virtual std::shared_ptr<BaseModelImpl> createModel(std::string model_cfg) override;
    virtual void loadWeights(std::string model_cfg) override;
    virtual void createDataloader() override;
    virtual void createSpecialOjects() override;

    virtual void beforeEpoch() override;             
    virtual void preprocessBatch(VariantData& batch) override;         
    virtual std::vector<torch::Tensor> postprocess(torch::Tensor preds) override;
    virtual void update_metrics(std::vector<torch::Tensor> preds, VariantData batch) override;

    virtual std::map<std::string, torch::Tensor> get_stats() override;

    virtual std::string get_desc() override;
    virtual void print_results(std::map<std::string, torch::Tensor> stats) override;


protected:
    VariantData _prepare_batch(int si, VariantData batch);
    torch::Tensor _prepare_pred(torch::Tensor pred, VariantData pbatch);
    torch::Tensor _process_batch(torch::Tensor detections, torch::Tensor gt_bboxes, torch::Tensor gt_cls);

    torch::Tensor match_predictions(torch::Tensor pred_classes, torch::Tensor true_classes, torch::Tensor iou);

    VariantData cat_stats();
protected:
    float m_conf = 0.01f;
    float m_iouthreshold = 0.30f;
    int m_seen = 0;

    std::vector<float> m_iouv;

    std::vector<VariantData> m_stats;
    int m_instance = 0;
};