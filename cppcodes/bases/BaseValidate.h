#pragma once

#include <torch/torch.h>

#include "BaseModel.h"
#include "BaseData.h"
#include "BaseLoss.h"
class BaseValidate
{
public:
    explicit BaseValidate(VariantConfigs& args_input, std::string root_dir,
        std::shared_ptr<BaseModelImpl> model_ptr = nullptr, std::shared_ptr<BaseLoss> loss_ptr = nullptr);
    void setupValidate();
    void doValidateTask();

protected:        
    void readSettings();
    void initDevice();
    void initDirs();

    void setupModel();
    void setupDataloader();
    void setupSpecialObjects();

    void save_batch_sample_image(int epoch, VariantData& batch);
    void save_batch_preds_image(int epoch, std::vector<torch::Tensor>& preds, VariantData& batch);

protected:
    virtual void readSpecialSettins() = 0;
    virtual std::shared_ptr<BaseModelImpl> createModel(std::string model_cfg) = 0;
    virtual void loadWeights(std::string model_cfg) = 0;
    virtual void createDataloader() = 0;
    virtual void createSpecialOjects() = 0;

    virtual void beforeEpoch() = 0;             // 每次epoch前的准备工作或清零处理等
    virtual void preprocessBatch(VariantData& batch) = 0;         // 预处理batch数据

    virtual std::vector<torch::Tensor> postprocess(torch::Tensor preds) = 0;
    virtual void update_metrics(std::vector<torch::Tensor> preds, VariantData batch) = 0;

    virtual std::map<std::string, torch::Tensor> get_stats() = 0;
    virtual std::string get_desc() = 0;
    virtual void print_results(std::map<std::string, torch::Tensor> stats) = 0;
public:
    bool m_isTraining = false;
    std::string m_rootDir;
    VariantConfigs m_args;
    std::shared_ptr<BaseModelImpl> m_modelPtr = { nullptr };
    std::shared_ptr<BaseLoss> m_lossPtr = { nullptr };

    torch::Device m_device = torch::Device(torch::kCPU);
    std::string m_saveDir;

    bool m_bPlotsEnable = true;
    std::shared_ptr<BaseDataloader> m_valLoader;

    bool m_bSingle = false;

    int nc = 80;
    int imgsz = 640;
    int seen;
};