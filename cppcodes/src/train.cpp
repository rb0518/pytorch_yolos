
#include <ATen/autocast_mode.h>     // autocast
#include <filesystem>
#include <tuple>
#include <fstream>
#include <cmath>

#include "train.h"
#include "utils.h"

#include "yolo.h"
#include "general.h"

#include "loss.h"
#include "lambda_lr.h"

#include "progressbar.h"
#include "plots.h"
#include "yaml_load.h"
#include "torch_utils.h"
#include "nms.h"

#include "augmentations.h"

#include "BaseData.h"
#include "detect_data.h"
#include "val.h"

#include "SpecTest.h"

#ifdef WIN32
typedef struct my_autocast {
    void _enter(torch::DeviceType deviceType)
    {
        devicetype = deviceType;
        pre_cache_enable = at::autocast::is_autocast_cache_enabled();
        pre_enable = at::autocast::is_autocast_enabled(devicetype);
        pre_fastdtype = at::autocast::get_autocast_dtype(devicetype);

        torch::autocast::set_autocast_enabled(devicetype, true);
        torch::autocast::set_autocast_dtype(devicetype, pre_fastdtype);
        torch::autocast::increment_nesting();
        torch::autocast::set_autocast_cache_enabled(pre_cache_enable);

        
    }
    void _exit() { 
        if (at::autocast::decrement_nesting() == 0)
            at::autocast::clear_cache();
        at::autocast::set_autocast_enabled(devicetype, pre_enable);
        at::autocast::set_autocast_dtype(devicetype, pre_fastdtype);
        at::autocast::set_autocast_cache_enabled(pre_cache_enable);
        ; }

    torch::DeviceType devicetype;
    bool pre_enable;
    bool pre_cache_enable;
    c10::ScalarType pre_fastdtype;
}MY_AUTOCAST;
#endif

// ---------------------  DetectionTrainder --------------------------------

void DetectionTrainer::setupModel()
{
    auto tmp_cfg_file_path = std::filesystem::path(m_rootDir).append(std::get<std::string>(m_args["model"]));
    std::string cfg_file = "";
    if (std::filesystem::exists(tmp_cfg_file_path))
        cfg_file = tmp_cfg_file_path.string();
    else
    {
        LOG(ERROR) << "set model cfg yaml file: " << tmp_cfg_file_path.string() << "not exists.";
        exit(-1);
    }
    model = std::dynamic_pointer_cast<BaseModelImpl>(std::make_shared<ModelImpl>(cfg_file, 3, &m_args));
    model->show_modelinfo(getVariantValue(&m_args, "debug_show_info", false));
    model->to(m_device);
}

void DetectionTrainer::setupDataloader()
{
//    train_loader_old = std::make_shared<DataloaderBase>(root_dir, args, 32, false, false);
    train_loader = std::dynamic_pointer_cast<BaseDataloader>(std::make_shared<DetectDataload>(m_rootDir, m_args, 32, false));
    train_loader->initDatas();

}

void DetectionTrainer::write_envsets_to_pt(torch::serialize::OutputArchive& out_archive)
{
    c10::IValue i_task(std::string("detect"));
    out_archive.write("task", i_task);
}

void DetectionTrainer::preprocess_batch(VariantData& batch)
{
    bool non_blocking = m_device.type() == c10::DeviceType::CUDA;

    auto img = std::get<torch::Tensor>(batch[GET_VDK(VDK::IMG)]);
    img = img.to(torch::kFloat).div(255.f);
    img = img.to(m_device, non_blocking);
    batch[GET_VDK(VDK::IMG)] = img;

    batch[GET_VDK(VDK::CLS)] = std::get<torch::Tensor>(batch[GET_VDK(VDK::CLS)]).to(m_device, non_blocking);
    batch[GET_VDK(VDK::BATCH_IDX)] = std::get<torch::Tensor>(batch[GET_VDK(VDK::BATCH_IDX)]).to(m_device, non_blocking);
    batch[GET_VDK(VDK::BBOXES)] = std::get<torch::Tensor>(batch[GET_VDK(VDK::BBOXES)]).to(m_device, non_blocking);
}


void DetectionTrainer::specialTreatWhenEpochOver()		
{
}		


void DetectionTrainer::specialTreatWhenTrainover()
{
    test_model(model);
}

void DetectionTrainer::createComputeLoss()
{
    m_compute_loss = std::dynamic_pointer_cast<BaseLoss>(std::make_shared<v8DetectionLossImpl>(model, m_device));
}

void DetectionTrainer::createValidator()
{
    if(m_bVal)
    {
        m_validator = std::dynamic_pointer_cast<BaseValidate>(
            std::make_shared<DetectValidator>(m_args, m_rootDir, model, m_compute_loss));

        m_validator->setupValidate();
    }
}
