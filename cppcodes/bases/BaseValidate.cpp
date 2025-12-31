#include "BaseValidate.h"
#include "utils.h"
#include "torch_utils.h"
#include "general.h"

#include "progressbar.h"
#include "plots.h"

#include <filesystem>
#include <variant>

namespace fs = std::filesystem;

BaseValidate::BaseValidate(VariantConfigs& args_input, std::string root_dir, 
    std::shared_ptr<BaseModelImpl> model_ptr/*= nullptr*/,
    std::shared_ptr<BaseLoss> loss_ptr/*= nullptr*/)
{
    m_args = args_input;
    m_rootDir = root_dir;
    m_modelPtr = model_ptr;
    m_lossPtr = loss_ptr;
    m_isTraining = m_modelPtr != nullptr ? true : false;

    readSettings();

    initDevice();
    initDirs();
}

void BaseValidate::setupValidate()
{
    readSpecialSettins();

    setupModel();
    setupDataloader();
    setupSpecialObjects();
}

void BaseValidate::readSettings()
{
    m_bPlotsEnable = getVariantValue<bool>(&m_args, "plots", true);
    m_bSingle = getVariantValue<bool>(&m_args, "single", false) || nc == 1;
}

void BaseValidate::initDevice()
{
    if (m_isTraining)
    {
        m_device = m_modelPtr->parameters().begin()->device();
    }
    else
    {
        std::string device_name = getVariantValue<std::string>(&m_args, "device", "cpu");
        std::cout << "input device type: " << device_name << "\n";
        m_device = get_device(device_name);
        std::cout << ColorString("Device: ") << m_device.type() << std::endl;
    }
}

void BaseValidate::initDirs()
{
    if (m_isTraining)
    {
        m_saveDir = fs::path(m_rootDir).append(getVariantValue<std::string>(&m_args, "save_dir", "val"));
    }
    else
    {
        std::tie(m_saveDir, std::ignore, std::ignore, std::ignore) = getSaveDirAndFilenames(m_args, m_rootDir, true);
    }
}

void BaseValidate::setupModel()
{
    if (m_isTraining)
        return;

    std::string model_set = getVariantValue<std::string>(&m_args, "model", "");
    auto tmp_cfg_file_path = fs::path(m_rootDir).append(model_set);
    std::string cfg_file = "";
    if (fs::exists(tmp_cfg_file_path))
        cfg_file = tmp_cfg_file_path.string();
    else
    {
        LOG(ERROR) << "set model cfg yaml file: " << tmp_cfg_file_path.string() << "not exists.";
        exit(-1);
    }

    m_modelPtr = createModel(cfg_file);
    m_modelPtr->show_modelinfo(std::get<bool>(m_args["debug_show_info"]));

    loadWeights(cfg_file);

    m_modelPtr->to(m_device);
}

void BaseValidate::setupDataloader()
{
    createDataloader();
}

void BaseValidate::setupSpecialObjects()
{
    createSpecialOjects(); 
}

void BaseValidate::doValidateTask()
{
    int batch_num = m_valLoader->getNumberOfBatch();
    progressbar pbar(batch_num, 18);
    pbar.set_done_char(std::string("█"));
    pbar.set_prefix(get_desc());
    beforeEpoch();

    if (m_isTraining)
        m_lossPtr->reset();

    m_modelPtr->eval();
    int idx = 0;
    
    torch::NoGradGuard nograd;
    for (auto itr = m_valLoader->getBegin(); itr != m_valLoader->getEnd(); ++itr)
    {
        auto batch = *itr;
        
        preprocessBatch(batch);
        torch::Tensor input = std::get<torch::Tensor>(batch[GET_VDK(VDK::IMG)]);

        // 1 run model forward
        auto [pred, train_vect, masks] = m_modelPtr->forward(input);
        // 2 run compute loss
        if (m_isTraining)
        {
            auto [loss, lossitem] = m_lossPtr->forward(train_vect, batch);
            m_lossPtr->addNewItems(loss, lossitem);
        }

        // 3 run postprocess
        auto rets = this->postprocess(pred);

        // 4 record metrics
        update_metrics(rets, batch);

        // 5 save samples and predict to image
        bool bshow = getVariantValue<bool>(&m_args, "debug_show_preds", false);
        if(m_bPlotsEnable && idx < 3)
            save_batch_sample_image(idx, batch);

        if (bshow || m_bPlotsEnable)
        {
            save_batch_preds_image(idx, rets, batch);
        }

        idx += 1;
        pbar.update();
    }
    std::cout << std::endl;
    std::map<std::string, torch::Tensor> result_dict = get_stats();
    print_results(result_dict);
}


void BaseValidate::save_batch_sample_image(int epoch, VariantData& batch)
{
    torch::Tensor batch_idx = std::get<torch::Tensor>(batch[GET_VDK(VDK::BATCH_IDX)]);
    torch::Tensor cls = std::get<torch::Tensor>(batch[GET_VDK(VDK::CLS)]);
    torch::Tensor bboxes = std::get<torch::Tensor>(batch[GET_VDK(VDK::BBOXES)]);
    torch::Tensor img = std::get<torch::Tensor>(batch[GET_VDK(VDK::IMG)]);
    auto targets = torch::cat({ batch_idx.view({-1, 1}),
                                cls.view({-1, 1}),
                                bboxes }, 1);

    auto image = img.clone();
    image = image.to(torch::kCPU);
    auto labels = targets.clone();
    labels = labels.to(torch::kCPU);

    Singleton_PlotBatchImages::getInstance(m_saveDir)->push_data(image, labels, "val_batch%d_labels", epoch);
}

void BaseValidate::save_batch_preds_image(int epoch, std::vector<torch::Tensor>& preds, VariantData& batch)
{
    torch::Tensor imgs = std::get<torch::Tensor>(batch[GET_VDK(VDK::IMG)]);
    auto images = imgs.clone();
    images = images.to(torch::kCPU);

    bool normalise = images.max().item().toFloat() <= 1.0;

    int batch_size = images.size(0);
    int img_width = images.size(3);
    int img_height = images.size(2);
    int channels = images.size(1);

    if (channels != 3)
    {
        LOG(WARNING) << "only support channels as 1 or 3. program will do nothing.";
        return;
    }

    if (normalise)
        images = images * 255.f;

    bool bDebugShow = false;
    cv::Mat mat_mosaic = cv::Mat(img_height * 2, img_width * 2, CV_8UC3);
    for (int i = 0; i < std::min(16, batch_size); i++)
    {
        auto img = images[i];
        auto mat_one = convert_tensor_to_mat(img, img_width, img_height, channels);

        torch::Tensor boxs = preds[i];
        for (int j = 0; j < boxs.size(0); j++)
        {
            bDebugShow = true;
            auto x1 = boxs[j][0].item().toFloat();
            auto y1 = boxs[j][1].item().toFloat();
            auto x2 = boxs[j][2].item().toFloat();
            auto y2 = boxs[j][3].item().toFloat();
            auto score = boxs[j][4].item().toFloat();
            auto cls_id = boxs[j][5].item().toInt();

            auto typecolor = SingletonColors::getInstance()->get_color_scalar(cls_id);

            cv::rectangle(mat_one, cv::Point(int(x1), int(y1)), cv::Point(int(x2), int(y2)), typecolor, 2);
            cv::putText(mat_one, std::to_string(cls_id), cv::Point(x1, std::max(0, int(y1) - 2)), 0, 1.,
                typecolor, 3);
        }
        cv::Mat mat_half;
        cv::resize(mat_one, mat_half, cv::Size(img_width / 2, img_height / 2));
        int row_id = i % 4;
        int col_id = i > row_id ? (i - row_id) / 4 : 0;
        auto y1_row = row_id * (img_height / 2);
        auto x1_col = col_id * (img_width / 2);
        cv::Mat roi = mat_mosaic(cv::Rect(x1_col, y1_row, img_width / 2, img_height / 2));
        mat_half.copyTo(roi);
    }

    if (bDebugShow)
    {
        cv::imshow("pred", mat_mosaic);
        cv::waitKey(500);
    }

    if(m_bPlotsEnable)
    {
        std::string filename = "val_batch" + std::to_string(epoch) + "_pred.jpg";
        std::string fname = fs::path(m_saveDir).append(filename).string();
        cv::imwrite(fname, mat_mosaic);
    }
}
