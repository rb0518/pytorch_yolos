#include "utils.h"

#include <filesystem>
#include <opencv2/dnn/dnn.hpp>  // nms要用到

#include <torch/script.h>       // load torchscript file
#include <regex>                // for std::regex_replace
#include <algorithm>
#include <fstream>
#include <random>
#include <algorithm> // std::shuffle
#include <chrono>

#include "plots.h"

torch::Tensor xyxy2xywh(const torch::Tensor& x)
{
    // 检查输入张量的最后一个维度是否为4
    TORCH_CHECK(x.size(-1) == 4, "input shape last dimension expected 4 but input shape is ", x.sizes());
    // 创建与输入相同类型的输出张量
    auto y = torch::empty_like(x);

    y.index_put_({"...", 0}, (x.index({"...", 0}) + x.index({"...", 2})) / 2);
    y.index_put_({"...", 1}, (x.index({"...", 1}) + x.index({"...", 3})) / 2);
    y.index_put_({"...", 2}, x.index({"...", 2}) - x.index({"...", 0}));
    y.index_put_({"...", 3}, x.index({"...", 3}) - x.index({"...", 1}));
    return y;
}

torch::Tensor xywh2xyxy(const torch::Tensor& x) 
{
    TORCH_CHECK(x.size(-1) == 4, "input shape last dimension expected 4 but input shape is ", x.sizes());
    
    auto y = torch::empty_like(x);
   
    // 计算左上角坐标
    y.index_put_({"...", 0}, x.index({"...", 0}) - x.index({"...", 2}) / 2);
    y.index_put_({"...", 1}, x.index({"...", 1}) - x.index({"...", 3}) / 2);
    // 计算右下角坐标
    y.index_put_({"...", 2}, x.index({"...", 0}) + x.index({"...", 2}) / 2);
    y.index_put_({"...", 3}, x.index({"...", 1}) + x.index({"...", 3}) / 2);
    return y;
}

torch::Tensor xywhn2xyxy(const torch::Tensor& x, int w /*= 640*/, int h /*= 640*/, int padw /*= 0*/, int padh /*= 0*/) 
{
    TORCH_CHECK(x.size(-1) == 4, "input shape last dimension expected 4 but input shape is ", x.sizes());
    auto y = torch::empty_like(x);
    
    // 计算转换后的坐标
    y.index_put_({"...", 0}, w * (x.index({"...", 0}) - x.index({"...", 2}) / 2) + padw);  // top left x
    y.index_put_({"...", 1}, h * (x.index({"...", 1}) - x.index({"...", 3}) / 2) + padh);  // top left y
    y.index_put_({"...", 2}, w * (x.index({"...", 0}) + x.index({"...", 2}) / 2) + padw);  // bottom right x
    y.index_put_({"...", 3}, h * (x.index({"...", 1}) + x.index({"...", 3}) / 2) + padh);  // bottom right y
    
    return y;
}

torch::Tensor xyxy2xywhn(const torch::Tensor& x, int w /*= 640*/, int h /*= 640*/, bool clip /*= false*/, float eps /*= 0.0*/) {
    // 检查输入张量的最后一个维度是否为4
    TORCH_CHECK(x.size(-1) == 4, "input shape last dimension expected 4 but input shape is ", x.sizes());
    
    // 创建与输入相同类型的输出张量
    auto y = torch::empty_like(x);
    
    // 如果需要裁剪到图像边界，前期简化，暂时不用，未完善
    if (clip) {
//        auto clipped_x = clip_boxes(x, {h - eps, w - eps});
//        x = clipped_x;
    }
    
    // 计算归一化中心坐标和宽高
    y.index_put_({"...", 0}, ((x.index({"...", 0}) + x.index({"...", 2})) / 2) / w);  // x center
    y.index_put_({"...", 1}, ((x.index({"...", 1}) + x.index({"...", 3})) / 2) / h);  // y center
    y.index_put_({"...", 2}, (x.index({"...", 2}) - x.index({"...", 0})) / w);        // width
    y.index_put_({"...", 3}, (x.index({"...", 3}) - x.index({"...", 1})) / h);        // height
    
    return y;
}

int searchfiles_in_folder(const std::string& folder, const std::string& exttype, 
        std::vector<std::string>& lists)
{
    lists.clear();
    for(const auto& entry : std::filesystem::directory_iterator(folder))
    {
        if (entry.is_regular_file())
        {
            auto filename = entry.path().stem().string();
            auto fileext = entry.path().extension().string();
            if (fileext == exttype)
                lists.push_back(filename);
        }
    }
    return static_cast<int>(lists.size());
}

std::tuple<float, bool> ConvertToNumber(const std::string& str)
{
    float f_r;
    auto result = std::from_chars(str.data(), str.data() + str.size(), f_r);
    return std::make_tuple(f_r, result.ec == std::errc());
}

/*
    python原代码
    def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nm=0,  # number of masks

    pred 网络在eval状态下，输出tuple的第一个elements
    conf_thres : confience threshold 置信度阈值
    iou_trehs  : IOU 阈值
    其它的参数暂时不支持，classes 只保留的类别，默认为None，所有的都保留；agnostic_nms:nms去除类别无关的；max_det，保留的最大检测目标数

*/
int non_max_suppression_old(torch::Tensor prediction, float conf_thres, float iou_thres, std::vector<Detection>& output)
{
    const int item_attr_size = 5;

    int bs = prediction.size(0);    // 取得batch_size
    if (bs != 1)
    {
        std::cout << "program only support detect one image ervery time.";
        return 0;
    }

    auto nc = prediction.size(2) - item_attr_size;

    bool multi_label = nc > 1;  

    // 通过conf_threshold过滤不合格的grid, unsqueeze(2)
    auto conf_mask = prediction.index({ torch::indexing::Ellipsis, 4 }).ge(conf_thres);   // {torch::indexing::Ellipsis, 4} = {"...", 4}
    //std::cout << "conf_mask 0 : " << conf_mask.sizes() << std::endl;
    conf_mask = conf_mask.unsqueeze(2);
    //std::cout << "conf_mask 1 : " << conf_mask.sizes() << " prediction: " << prediction.sizes() << std::endl;

    auto det = torch::masked_select(prediction[0], conf_mask[0]).view({ -1, nc + item_attr_size });
    //std::cout << "conf select result: det size = " << det.sizes() << std::endl;

    if (0 == det.size(0))
    {
        std::cout << "none detections remain after compare with conf_thres." << std::endl;
        return 0;
    }

    // compute overalll score obj_conf * cls_conf x[:5 : 85] *  
    det.slice(1, item_attr_size, nc+ item_attr_size) *= det.select(1, 4).unsqueeze(1);

    auto box = xywh2xyxy(det.slice(1, 0, 4));   // 最后一维85个变量前四个存着的是xc, yc, w, h ==>x1, y1, x2, y2
    
    std::tuple<torch::Tensor, torch::Tensor> max_classes = torch::max(det.slice(1, item_attr_size, nc + item_attr_size), 1);  //从第6个位置找到那一个class的score值最大

    auto max_conf_score = std::get<0>(max_classes);     // 类置信度
    auto max_conf_index = std::get<1>(max_classes);     // 位置=类别索引值 

    max_conf_score = max_conf_score.to(torch::kFloat).unsqueeze(1);
    max_conf_index = max_conf_index.to(torch::kFloat).unsqueeze(1);

    // shape: n * 6, top-left x/y (0,1), bottom-right x/y (2,3), score(4), class index(5)
    det = torch::cat({ box.slice(1, 0, 4), max_conf_score, max_conf_index }, 1);
    //std::cout << "det selected boxes size: " << det.sizes() << std::endl;

    constexpr int max_wh = 4096;
    auto c = det.slice(1, item_attr_size, item_attr_size + 1) * max_wh;
    auto offset_box = det.slice(1, 0, 4) + c;
    //std::cout << "offset_box " << offset_box.sizes() << " c " << c.sizes() << std::endl;

    // 将数据拷贝到CPU，因为nms计算在CPU上
    auto offset_box_cpu = offset_box.cpu();
    auto det_cpu = det.cpu();


    const auto det_cpu_array = det_cpu.accessor<float, 2>();
    auto Tensor2Detection = [](const at::TensorAccessor<float, 2>& offset_boxes,
        const at::TensorAccessor<float, 2>& det,
        std::vector<cv::Rect>& offset_box_vec,
        std::vector<float>& score_vec) {
            for (int i = 0; i < offset_boxes.size(0); i++) {
                offset_box_vec.emplace_back(
                    cv::Rect(cv::Point(offset_boxes[i][0], offset_boxes[i][1]),
                        cv::Point(offset_boxes[i][2], offset_boxes[i][3]))
                );
                score_vec.emplace_back(det[i][4]);
            }
        };
    // use accessor to access tensor elements efficiently
    std::vector<cv::Rect> offset_box_vec;
    std::vector<float> score_vec;
    Tensor2Detection(offset_box_cpu.accessor<float, 2>(), det_cpu_array, offset_box_vec, score_vec);

    // run NMS
    std::vector<int> nms_indices;
    cv::dnn::NMSBoxes(offset_box_vec, score_vec, conf_thres, iou_thres, nms_indices);
    std::cout << "After cv::dnn::NMSBoxes: " << nms_indices.size() << std::endl;

    for (int index : nms_indices) {
        Detection t;
        const auto& b = det_cpu_array[index];

        std::cout << " index: " << index << std::endl;
        std::cout << "     box " << b[0] << " " << b[1] << " - " << b[2] << " " << b[3] 
                    << " score: " << det_cpu_array[index][4] 
                    << " object id: " << det_cpu_array[index][5] << std::endl;

        t.bbox =
            cv::Rect(cv::Point(b[0], b[1]),
                cv::Point(b[2], b[3]));
        t.score = det_cpu_array[index][4];
        t.class_idx = det_cpu_array[index][5];
        output.emplace_back(t);
    }

    return output.size();
}

/*
    python 原代码函数注释内容
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
*/
std::vector<torch::Tensor> non_max_suppression(
    torch::Tensor prediction,
    float conf_thres,
    float iou_thres,
    const std::vector<std::vector<int>>& classes,
    bool agnostic,
    bool multi_label,
    const std::vector<std::vector<float>>& labels
) 
{
    int nc = prediction.size(2) - 5;  // number of classes
    auto xc = prediction.index({"...", 4}) > conf_thres; // candidates
    //std::cout << "input: " << prediction.sizes() << " " << xc.sizes() << std::endl;

    // 参数初始化
    constexpr int min_wh = 2;
    constexpr int max_wh = 4096;
    constexpr int max_det = 300;
    constexpr int max_nms = 30000;
    constexpr float time_limit = 10.0;
    bool redundant = true;
    bool merge = false;

    multi_label &= (nc > 1);

    auto t = std::chrono::steady_clock::now();
    std::vector<torch::Tensor> output;
    // auto output = std::vector<torch::Tensor>(prediction.size(0), torch::zeros({ 0, 6 }, prediction.options()));

    // 主处理循环
    for (int xi = 0; xi < prediction.size(0); ++xi) 
    {
        auto x = prediction[xi];
//        std::cout << xc[xi].sizes() << xc[xi].sum().item().toInt() << std::endl;

        // 应用置信度筛选
        x = x.index({ xc[xi] });
//        std::cout << xi << " x " << x.sizes() << std::endl;
        // ???添加先验标签，先不处理
        /*
        if (labels.size() > 0 && labels[xi].size() > 0) {
            auto l = torch::from_blob(labels[xi].data(), { labels[xi].size(), 5 }, torch::kFloat32);
            auto v = torch::zeros({ l.size(0), x.size(1) }, x.options());
            v.index({ torch::indexing::Slice(), torch::indexing::Slice(0, 4) }) = l.index({ torch::indexing::Slice(), torch::indexing::Slice(1, 5) });
            v.index({ torch::indexing::Slice(), 4 }) = 1.0;
            v.index({ torch::indexing::Slice(0, l.size(0)), torch::indexing::Slice(5, x.size(1)) }) = l.index({ torch::indexing::Slice(), torch::indexing::Slice(0, x.size(1) - 5) });
            x = torch::cat({ x, v }, 0);
        }
        */
        // 如果无剩余框则跳过
        if (x.size(0) == 0)
        {
//            std::cout << xi << " after x[xc[xi]] return 0 " << std::endl;
            output.push_back(torch::zeros({0, 6}));
            continue;
        }

        // 计算最终置信度
        x.index_put_(
            { torch::indexing::Slice(), torch::indexing::Slice(5) },
            x.index({ torch::indexing::Slice(), torch::indexing::Slice(5)}) * x.index({ torch::indexing::Slice(), 4 }).unsqueeze(1)
            );

        // 坐标转换
        auto box = xywh2xyxy(x.index({ torch::indexing::Slice(), torch::indexing::Slice(0, 4) }));
//        std::cout << "box " << box.sizes() << "multi_label: " << multi_label << std::endl;
        // 多标签处理
        if (multi_label) {
            auto ijcls = x.index({ torch::indexing::Slice(), torch::indexing::Slice(5) }) > conf_thres;
            auto ijcls_t = ijcls.nonzero().t();
            auto i = ijcls_t[0];
            auto j = ijcls_t[1];
            auto box_i = box.index({ i });
            auto x_ij = x.index({i, j+5}).unsqueeze(-1);
            x = torch::cat({ box_i, x_ij, j.unsqueeze(-1).to(torch::kFloat) }, 1);
        }
        else {
            auto [conf, j] = x.index({ torch::indexing::Slice(), torch::indexing::Slice(5) }).max(1, true);
            auto x_filtered = torch::cat({ box, conf, j.to(torch::kFloat)}, 1);
            auto conf_sel = conf > conf_thres;
//            std::cout << "x_filtered: " << x_filtered.sizes() << " conf " << conf_sel.sizes() << std::endl;
            x = x_filtered.index({conf_sel.squeeze(1)});
        }

        // 按类别筛选
        // if (!classes.empty()) {
        //     auto mask = torch::zeros({ x.size(0), 1 }, x.options());
        //     for (auto cls : classes[xi]) {
        //         mask |= (x.index({ torch::indexing::Slice(), 5 }) == cls).view(-1, 1);
        //     }
        //     x = x[mask];
        // }
        if(x.size(0) ==  0)
        {
//            std::cout << " no boxes " << std::endl;
            output.push_back(torch::zeros({0, 6}));
            continue;
        }

        // 应用限制条件
        if (x.size(0) > max_nms) {
            x = x[x.index({ torch::indexing::Slice(), 4 }).argsort(0, true).slice(0, max_nms)];
        }
//        std::cout << "before nms x " << x.sizes() << std::endl;
        // batched nms
        auto x_cpu = x.cpu();

        auto c = x.index({torch::indexing::Slice(), 5});
        if(agnostic)
            c = c * 0;
        else 
            c = c * max_wh;
        
        auto boxes = x.index({torch::indexing::Slice(), torch::indexing::Slice(0, 4)}) + c.unsqueeze(1);
        auto scores = x.index({torch::indexing::Slice(), 4});

        std::vector<cv::Rect2d> cv_boxes;
        std::vector<float> cv_scores;
        for(int i = 0; i < x.size(0); i++)
        {
            auto x1 = boxes[i][0].item().toDouble();
            auto y1 = boxes[i][1].item().toDouble();
            auto x2 = boxes[i][2].item().toDouble();
            auto y2 = boxes[i][3].item().toDouble();

            cv_boxes.push_back(cv::Rect2d(cv::Point2d(x1, y1), cv::Point2d(x2, y2)));
            cv_scores.push_back(scores[i].item().toFloat());
        }

        // run cv::dnn::NMSBoxes
        std::vector<int> nms_indices;
        cv::dnn::NMSBoxes(cv_boxes, cv_scores, conf_thres, iou_thres, nms_indices);
//        std::cout << "After cv::dnn::NMSBoxes: " << nms_indices.size() << std::endl;

        // 构造返回函数，压入vector
        int box_count = nms_indices.size();
        if (nms_indices.size() > 0)
        {
            auto box_sel = torch::tensor(nms_indices);
            auto ret_box = x_cpu.index({box_sel});
            output.push_back(ret_box);

//            std::cout << xi << " return " << ret_box.sizes() << std::endl;
        }
        else
        {
            output.push_back(torch::zeros({0, 6}));
        }
 
        // 检查时间限制
        if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - t).count() > time_limit) {
            std::cout << "WARNING: NMS time limit " << time_limit << "s exceeded" << std::endl;
            break;
        }
    }

    return output;
}

int LoadWeightFromJitScript(const std::string strScriptfile, torch::nn::Module& model, bool b_showinfo)
{
    bool load_is_ok = false;
    torch::Device device = model.parameters().begin()->device();
    torch::NoGradGuard nograd;
    torch::jit::script::Module jit_model;
    try {
        jit_model = torch::jit::load(strScriptfile);
        std::cout << "torch::jit::load over..." << std::endl;
        // 1、转存时已经将网上的yolov5s.pt从half-->float, 并指定在kCUDA上
        // 2、目前是用的trace，由于有可能有动态控制流，是否要用torch.jit.script，待测
        // 3、register_module注册子模块以保持参数名一致性
        jit_model.to(device);   // 与model的device保持一致
        jit_model.to(torch::kFloat);
        load_is_ok = true;
    }
    catch (const c10::Error& e)
    {
        std::cerr << e.what() << "\n";
        //std::exit(EXIT_FAILURE);
        return 0;
    }
    catch (...) {
        std::cerr << "torch::jit::load other exception \n";
        //std::exit(EXIT_FAILURE);
        return 0;
    }

    // 从pytorch的模型中加载预训练模型
    // 加载TorchScript模型

    // 提取参数到自定义模型
    // 将torch::jit_name_parameters_list重新组织为unordered_map，方便查找
    std::unordered_map<std::string, torch::Tensor> jit_params;
    for (const auto& param : jit_model.named_parameters())
    {
        jit_params[param.name] = param.value;
        //std::cout << "jit_model:  " << param.name << std::endl;
    }
    std::cout << "load jit weigth, params count " << jit_params.size() << std::endl;
    int n_params = 0;
    int count_changes = 0;

    for (auto& param : model.named_parameters())
    {
        torch::AutoGradMode enable_grad(false);
        //torch::NoGradGuard no_grad;   // 对模板进行操作时，不能是目标与源tensor一个有grad,一个无grad      
        std::string str_name = param.key();
        
        //  "model-1.m-1.bn1.bias" ==> "model.1.m.1.bn1.bias
        str_name = std::regex_replace(str_name, std::regex("-"), ".");
        
        if (jit_params.find(str_name) != jit_params.end())
        {
            if (jit_params[str_name].sizes() == param.value().sizes())
            {
                //std::cout << "trans ok : " << str_name << std::endl;
                count_changes += 1;
                param.value().data().copy_(jit_params[str_name].data());
            }
            else if (b_showinfo)
            {   // 找到了，但尺寸不匹配，显示方便后续调试要检测目标尺寸与当前shape是否一致
                std::cout << "jit tensor " << jit_params[str_name].sizes() << " " << jit_params[str_name].dtype() << std::endl;
                std::cout << "model tensor " << param.value().sizes() << " " << param.value().dtype() << std::endl;
            }
        }
        else{
            std::cout <<"miss name: " << str_name << std::endl;
        }
        n_params += 1;
    }
    //if (b_showinfo)
    std::cout << "Total model parameters: " << n_params << " loaded " << count_changes << " paramterss." << std::endl;
    return count_changes;
}

// void save_checkpoint(
//     torch::nn::Module& model,
//     torch::optim::Optimizer& optimizer,
//     int epoch,
//     const std::string& path) 
// {
//     //torch::serialize::OutputArchive ckpt;

//     auto dir = std::filesystem::path(path).has_parent_path() ? std::filesystem::path(path).parent_path().string() : "./";

//     //torch::serialize::OutputArchive model_out;
//     if(model.parameters().begin()->device().type()!=torch::kCPU)
//     {
//         std::cout << "Model not run in CPU, save model stipulation as CPU" << std::endl;
//         model.to(torch::kCPU);
//     }
//     //model.save(model_out);
//     //model_out.save_to(path);
//     torch::save(model, path);
//     // 据说序列化存储优化器时，再次调入后在调用step()时会报错
    
    
//     // torch::serialize::OutputArchive optim_out;
//     // optimizer.save(optim_out);
//     // auto optim_file = std::filesystem::path(dir).append("last_optm.pt").string();
//     // optim_out.save_to(optim_file);

//     auto epoch_file = std::filesystem::path(dir).append("last_epoch.txt").string();
//     std::ofstream fs(epoch_file);
//     if (fs.is_open())
//     {
//         fs.clear();
//         fs << epoch;
//         fs.close();
//     }
// }

// bool load_checkpoint(const std::string& path, torch::nn::Module* model,
//     torch::optim::Optimizer* optimizer,
//     int epoch)
// {
//     auto check_file_exists = [](std::string file) {
//         return std::filesystem::exists(std::filesystem::path(file));
//         };

//     auto dir = std::filesystem::path(path).has_parent_path() ? std::filesystem::path(path).parent_path().string() : "./";
//     auto optim_file = std::filesystem::path(dir).append("last_optm.pt").string();
//     auto epoch_file = std::filesystem::path(dir).append("last_epoch.txt").string();

//     if (!check_file_exists(path) || !check_file_exists(optim_file) || !check_file_exists(epoch_file))
//     {
//         std::cout << "model, optimizer, epoch file not found in dir: " << dir << std::endl;
//         return false;
//     }


//     try {

//         //torch::serialize::InputArchive model_in;
//         //model_in.load_from(path);
//         //std::cout << "load model file " << path <<" over." << std::endl;
//         if(model->parameters().begin()->device().type()!=torch::kCPU)
//         {
//             std::cout << "Model not run in CPU, save model stipulation as CPU" << std::endl;
//             model->to(torch::kCPU);
//         }
//         //model->load(model_in);
//         torch::load(model, path);
//         std::cout << "load model parameters from " << path << " ok." << std::endl;

//         // 据说序列化存储优化器时，再次调入后在调用step()时会报错
//         /*
//         torch::serialize::InputArchive optim_in;
//         optim_in.load_from(optim_file);
//         optimizer->load(optim_in);
//         std::cout << "load optimizer parameters from " << path << " ok." << std::endl;
//         */
//         std::ifstream fs(epoch_file);
//         if (fs.is_open())
//         {
//             fs >> epoch;
//             std::cout << "load last train end epoch : " << epoch << std::endl;
//             fs.close();
//         }
//     }
//     catch (const c10::Error& e)
//     {
//         LOG(ERROR) << "load fail: " << e.what();
//         return false;
//     }

//     return true;
// }

std::tuple<std::string, std::string> get_checkpoint_files(const std::string& path)
{
    auto dir = std::filesystem::path(path).has_parent_path() ? std::filesystem::path(path).parent_path().string() : "./";
    auto optim_file = std::filesystem::path(dir).append("last_optm.pt").string();
    auto epoch_file = std::filesystem::path(dir).append("last_epoch.txt").string();

    return {optim_file, epoch_file};
}

void init_torch_seek(int seed/* = 0*/)
{
    torch::manual_seed(seed);

    if (seed == 0)
    {
        if (torch::cuda::cudnn_is_available())
        {
            torch::globalContext().setBenchmarkCuDNN(false);
            torch::globalContext().setDeterministicCuDNN(true);
        }
        else
            LOG(WARNING) << "torch::cuda::cudnn_is_available: false";
    }
    else
    {
        if (torch::cuda::cudnn_is_available())
        {
            torch::globalContext().setBenchmarkCuDNN(true);
            torch::globalContext().setDeterministicCuDNN(false);
        }
        else
            LOG(WARNING) << "torch::cuda::cudnn_is_available: false";
    }
}

// 进程组ProcessGroup核心实现
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
// 通信相关基础功能
#include <torch/csrc/distributed/c10d/comm.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <torch/csrc/distributed/autograd/context/context.h>
// 太复杂了，暂时不用多卡或多机联合训练
void torch_distribute_zero_first(int local_rank,
    const std::shared_ptr<c10d::ProcessGroup>& pg)
{
    if (local_rank != -1 && local_rank != 0)
    {
        pg->barrier()->wait();
    }
    if (local_rank == 0)
        pg->barrier()->wait();
}

// Generate an incremental queue and randomly shuffle the order
std::vector<int> random_queue(int n) 
{
	std::vector<int> permutation(n);
	std::iota(permutation.begin(), permutation.end(), 0); // create list [1, 2, ..., n-1]

	// shuffle the order
	std::random_device rd;
	std::mt19937 g(rd()); 
	std::shuffle(permutation.begin(), permutation.end(), g); 

	return permutation;
}

float random_beta(float alpha, float beta) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::gamma_distribution<float> gamma_alpha(alpha, 1.0);
	std::gamma_distribution<float> gamma_beta(beta, 1.0);

	float x = gamma_alpha(gen);
	float y = gamma_beta(gen);
	return x / (x + y);
}


float random_uniform(float start/* = 0.0f*/, float end/* = 1.0f*/)
{
	std::random_device rd;  
	std::mt19937 gen(rd()); 

	std::uniform_real_distribution<> dis(start, end);

	return dis(gen);
}


void MatDrawTargets(cv::Mat& img, const torch::Tensor& labels, bool xywh , bool is_scale,  int start_idx)
{
    int nt = labels.size(0);

    auto labels_xyxy = labels;
    if(xywh)
        labels_xyxy  = is_scale ? xywhn2xyxy(labels.index({"...", torch::indexing::Slice(start_idx, start_idx+4)}), img.cols, img.rows) : xywh2xyxy(labels);

    for(int i = 0; i < nt; i++)
    {
        int x1 = labels_xyxy[i][0].item().toInt();
        int y1 = labels_xyxy[i][1].item().toInt();
        int x2 = labels_xyxy[i][2].item().toInt();
        int y2 = labels_xyxy[i][3].item().toInt();
        auto colors = SingletonColors::getInstance();
        rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), colors->get_color_scalar(int(labels[i][start_idx-1].item().toInt())), 2);
    }
}

