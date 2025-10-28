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

torch::Tensor xyn2xy(const torch::Tensor& x, int w, int h, int padw, int padh)
{
    TORCH_CHECK(x.size(-1) == 2, "input shape last dimension expected 2 but input shape is ", x.sizes());
    auto y = torch::empty_like(x);
    
    // 计算转换后的坐标
    y.index_put_({"...", 0}, w * x.index({"...", 0}) + padw);  // top left x
    y.index_put_({"...", 1}, h * x.index({"...", 1}) + padh);  // top left y
   
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
    const std::vector<std::vector<float>>& labels,
    int max_det /*= 300*/,
    int nm /*= 0*/  // number of masks
) 
{
    int bs = prediction.size(0);        // batch size
    int nc = prediction.size(2) - nm - 5;  // number of classes
    auto xc = prediction.index({"...", 4}) > conf_thres; // candidates
    //std::cout << "input: " << prediction.sizes() << " " << xc.sizes() << std::endl;

    // 参数初始化
    //constexpr int min_wh = 2;
    constexpr int max_wh = 7680;    // 10-9 4096==>7680
    constexpr int max_nms = 30000;
    constexpr float time_limit = 30.0;
    bool redundant = true;
    bool merge = false;

    multi_label &= (nc > 1);

    auto t = std::chrono::steady_clock::now();

    int mi = 5 + nc;    // mask start index

    std::vector<torch::Tensor> output;
    // auto output = std::vector<torch::Tensor>(prediction.size(0), torch::zeros({ 0, 6 }, prediction.options()));

    // 主处理循环
    for (int xi = 0; xi < prediction.size(0); ++xi) 
    {
        auto x = prediction[xi];
//        std::cout << xc[xi].sizes() << xc[xi].sum().item().toInt() << std::endl;

        // 应用置信度筛选
        x = x.index({ xc[xi] });
        //std::cout << xi << " x " << x.sizes() << std::endl;
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
            output.push_back(torch::zeros({0, 6 + nm}));
            continue;
        }

        // 计算最终置信度
        x.index_put_(
            { torch::indexing::Slice(), torch::indexing::Slice(5, torch::indexing::None)},
            x.index({ torch::indexing::Slice(), torch::indexing::Slice(5, torch::indexing::None)}) * x.index({ torch::indexing::Slice(), torch::indexing::Slice(4, 5)})
            );

        // 坐标转换
        auto box = xywh2xyxy(x.index({ torch::indexing::Slice(), torch::indexing::Slice(0, 4) }));
        // zero columns if no masks
        auto mask = x.index({torch::indexing::Slice(), torch::indexing::Slice(mi, torch::indexing::None)});
        // std::cout << "box " << box.sizes() << "multi_label: " << multi_label <<" mask: " << mask.sizes() << std::endl;

        // 多标签处理
        if (multi_label) {
            auto ijcls = x.index({ torch::indexing::Slice(), torch::indexing::Slice(5, mi) }) > conf_thres;
            auto ijcls_t = ijcls.nonzero().t();
            auto i = ijcls_t[0];
            auto j = ijcls_t[1];
            auto box_i = box.index({ i });
            auto x_ij = x.index({i, j+5}).unsqueeze(-1);
            auto mask_i = mask.index({i});
            std::cout << "box_i: " << box_i.sizes() << " mask_i " << mask_i.sizes() << std::endl;
            x = torch::cat({ box_i, x_ij, j.unsqueeze(-1).to(torch::kFloat), mask_i}, 1);
        }
        else {
            auto [conf, j] = x.index({ torch::indexing::Slice(), torch::indexing::Slice(5, mi) }).max(1, true);
            auto x_filtered = torch::cat({ box, conf, j.to(torch::kFloat), mask}, 1);
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
            output.push_back(torch::zeros({0, 6 + nm}));
            continue;
        }

        // 应用限制条件
        if (x.size(0) > max_nms) {
            x = x[x.index({ torch::indexing::Slice(), 4 }).argsort(0, true).slice(0, max_nms)];
        }
//        std::cout << "before nms x " << x.sizes() << std::endl;
        // batched nms
        auto x_cpu = x.cpu();

        // Batched NMS
        auto c = x.index({torch::indexing::Slice(), 5});
        if(agnostic)
            c = c * 0;
        else 
            c = c * max_wh;
        
        auto boxes = x.index({torch::indexing::Slice(), torch::indexing::Slice(0, 4)}) + c.unsqueeze(1);
        auto scores = x.index({torch::indexing::Slice(), 4});

        std::vector<cv::Rect> cv_boxes;
        std::vector<float> cv_scores;
        for(int i = 0; i < x.size(0); i++)
        {
            auto x1 = boxes[i][0].item().toDouble();
            auto y1 = boxes[i][1].item().toDouble();
            auto x2 = boxes[i][2].item().toDouble();
            auto y2 = boxes[i][3].item().toDouble();

            cv_boxes.push_back(cv::Rect(cv::Point2d(x1, y1), cv::Point2d(x2, y2)));
            cv_scores.push_back(scores[i].item().toFloat());
        }

        // run cv::dnn::NMSBoxes
        std::vector<int> nms_indices;
#if 0
        std::vector<float> updated_scores;
        cv::dnn::softNMSBoxes(cv_boxes, cv_scores, updated_scores, conf_thres, iou_thres, nms_indices);
#else
        cv::dnn::NMSBoxes(cv_boxes, cv_scores, conf_thres, iou_thres, nms_indices);
#endif        
//        std::cout << "After cv::dnn::NMSBoxes: " << nms_indices.size() << std::endl;

        // 构造返回函数，压入vector
        int box_count = nms_indices.size();
        if (nms_indices.size() > 0)
        {
            auto box_sel = torch::tensor(nms_indices);
            auto ret_box = x_cpu.index({box_sel});
            output.push_back(ret_box);

            std::cout << xi << " return " << ret_box.sizes() << std::endl;
        }
        else
        {
            output.push_back(torch::zeros({0, 6 + nm}));
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
    auto jit_params_count = jit_params.size();
    int n_params = 0;
    int count_changes = 0;

    for (auto& param : model.named_parameters())
    {
        torch::AutoGradMode enable_grad(false);
        //torch::NoGradGuard no_grad;   // 对模板进行操作时，不能是目标与源tensor一个有grad,一个无grad      
        std::string str_name = param.key();
        //std::cout << "model named_parameters: " << str_name << std::endl;
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
    std::cout <<"Load jit parameters total: " << jit_params_count << " Model parameters total: " << n_params << " trans_count: " << count_changes << " paramters." << std::endl;
    return count_changes;
}

void init_torch_seek(int seed/* = 0*/)
{
    torch::manual_seed(seed);
    torch::cuda::manual_seed(seed);
    if (torch::cuda::device_count() > 1)
        torch::cuda::manual_seed_all(seed);

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

std::tuple<bool, int, std::string, std::string> parse_yolo_config(const std::string& filename) 
{
    // 默认值设置
    int version_num;
    std::string size_type = "n";
    std::string task_type = "";

    // 正则表达式模式（支持yolov(n)('n/l/m/x'){-seg/obb/pose/cls}.yaml格式）
    std::regex pattern(R"(^yolo([v])?(\d+)([nlsmx])?(?:-(\w+))?\.yaml$)");
    std::smatch matches;
    bool is_valid = false;

    if (std::regex_match(filename, matches, pattern)) 
    {
        // for (int i = 0; i < matches.size(); i++)
        // {
        //     std::cout << " matches[" << i << "] " << matches[i].str() << "\n";
        // }

        is_valid = true;
        // version_num = matches[1].matched ? std::stoi(matches[1].str()) : 12;
        if (matches[2].matched) { // 必须带版本号
            version_num = std::stoi(matches[2].str());
        } else {
            is_valid = false;
            LOG(ERROR) << "filename not have yolo version: " << filename;
            return std::make_tuple(is_valid, version_num, size_type, task_type);
        }    

        size_type = matches[3].matched ? matches[2].str() : "";
        task_type = matches[4].matched ? matches[3].str() : "";
    }

    return std::make_tuple(is_valid, version_num, size_type, task_type);
}

namespace ops {
std::vector<torch::Tensor> non_max_suppression(
    torch::Tensor prediction,
    float conf_thres/* = 0.25f*/,
    float iou_thres/* = 0.45f*/,
    std::vector<int> classes/* = {}*/,
    bool agnostic/* = false*/,
    bool multi_label/* = false*/,
    std::vector<int>/* = {}*/, // 暂时未用，用std::vector<int>代替
    int max_det/* = 300*/,
    int nc/* = 0*/,
    float max_time_img/* = 0.05f*/,
    int max_nms/* = 30000*/,
    int max_wh/* = 7680*/,
    bool in_place/* = true*/,
    bool rotated /*= false*/)
{
    std::cout << "non_max_suppressio startm, input: " << prediction.sizes() << std::endl;

    if (prediction.size(-1) == 6)
    {
        std::cout << "prediction size(-1) == 6, it's end2ent model i.e. [1,300, 6] \n";
        std::cout << ColorString("ERROR: ", "R") << "not support now.\n";
        exit(-1);
    }

    int bs = prediction.size(0);    // batch size
    if (nc == 0) nc = prediction.size(1) - 4;
    int nm = prediction.size(1) - nc - 4;   // number of masks
    int mi = 4 + nc;    // mask start index

    auto xc = prediction.index({ torch::indexing::Slice(), torch::indexing::Slice(4, mi) }).amax(1) > conf_thres; // condidates
    std::cout << "xc " << xc.sizes() << std::endl;      // [bs, 8400]
    float time_limit = 2.0 + max_time_img * bs; // seconds to quit after
    multi_label &= nc > 1;
    prediction = prediction.transpose(-1, -2);
    std::cout << "pred transpose: " << prediction.sizes() << "\n";

    if(!rotated)
    { 
        if (in_place)
        {
            auto pred_bbox_xyxy = xywh2xyxy(prediction.index(
                { "...", torch::indexing::Slice(0, 4) }));
            std::cout << "in_place pred_bbox_xyxy: " << pred_bbox_xyxy.sizes() << std::endl;
            prediction.index_put_({ "...", torch::indexing::Slice(0,4)},
                pred_bbox_xyxy);
        }
        else
        { 
            auto xyxy_1 = xywh2xyxy(prediction.index({ "...",
                torch::indexing::Slice(0, 4) }));
            prediction = torch::cat({ xyxy_1,
                prediction.index({"...", torch::indexing::Slice(4, torch::indexing::None)}) }
                , - 1);
        }
    }

    std::vector<torch::Tensor> output;
    for (int i = 0; i < bs; i++)
        output.emplace_back(torch::zeros({ 0, 6 + nm },
            torch::TensorOptions().device(prediction.device())));

    for (size_t xi = 0; xi < prediction.size(0); ++xi)
    {
        auto x = prediction[xi];
        x = x.index({ xc[xi] });
        if (x.size(0) == 0)
        {
            std::cout << xi << " after x[xc[xi]] return 0 " << std::endl;
            continue;
        }

        auto xyxy_cls_mask = x.split({ 4, nc, nm }, 1);
        auto box = xyxy_cls_mask[0];
        auto cls = xyxy_cls_mask[1];
        auto mask = xyxy_cls_mask[2];
        std::cout << "cls " << cls.sizes() << " box " << box.sizes()
            << " mask " << mask.sizes() << std::endl;
        if (multi_label)
        {
            auto cls_where = torch::where(cls > conf_thres);
            auto i = cls_where[0];
            auto j = cls_where[1];
            std::cout << "i " << i.sizes() << " j " << j.sizes() << std::endl;
            auto box_i = box.index({ i });
            auto x_ij = x.index({ i, 4 + j, torch::indexing::None });
            auto mask_i = mask.index({ i });
            auto j_float = j.index({torch::indexing::Slice(), 
                torch::indexing::None});
            j_float = j_float.to(torch::kFloat);
            x = torch::cat({ box_i, x_ij, j_float, mask_i }, 1);
        }
        else
        {
            auto [conf, j] = cls.max(1, true);
            x = torch::cat({ box, conf, j.to(torch::kFloat), mask }, 1).index(
                {conf.view(-1)> conf_thres});
        }

        if (classes.size())
        {
            // not complete code
        }

        int n = x.size(0);   // number of boxes
        std::cout << "number of boxes: " << n << std::endl;

        if (n > max_nms) 
        { 
            // 获取第4列（置信度）并按降序排序，取前max_nms个索引
            auto x_argsort = x.index({torch::indexing::Slice(), 4 }).argsort(-1, true);
            x_argsort = x_argsort.index({ torch::indexing::Slice(0, max_nms) });
            x = x.index({ x_argsort });
        }

        torch::Tensor c;
        if (agnostic)
            c = x.index({ torch::indexing::Slice(), torch::indexing::Slice(5, 6) }) * 0;

        else
            c = x.index({ torch::indexing::Slice(), torch::indexing::Slice(5, 6) }) * max_wh;
        auto scores = x.index({ torch::indexing::Slice(), 4 });
        std::vector<int> nms_indices;
        if (rotated)
        {
            std::cout << "rotated not support now. \n";

        }
        else
        {
            auto boxes = x.index({ torch::indexing::Slice(), torch::indexing::Slice(
                0, 4) }).add(c);

            std::vector<cv::Rect> cv_boxes;
            std::vector<float> cv_scores;
            for (int i = 0; i < boxes.size(0); i++)
            {
                auto x1 = boxes[i][0].item().toDouble();
                auto y1 = boxes[i][1].item().toDouble();
                auto x2 = boxes[i][2].item().toDouble();
                auto y2 = boxes[i][3].item().toDouble();

                cv_boxes.push_back(cv::Rect(cv::Point2d(x1, y1), cv::Point2d(x2, y2)));
                cv_scores.push_back(scores[i].item().toFloat());
            }

            // run cv::dnn::NMSBoxes
#if 0
            std::vector<float> updated_scores;
            cv::dnn::softNMSBoxes(cv_boxes, cv_scores, updated_scores, conf_thres, iou_thres, nms_indices);
#else
            cv::dnn::NMSBoxes(cv_boxes, cv_scores, conf_thres, iou_thres, nms_indices);
#endif   
            if (nms_indices.size() > max_det) {
                nms_indices.resize(max_det);
            }
        }
        output[xi] = x.index({ torch::tensor(nms_indices) });
    }

    return output;
}

}   // end namespace ops
