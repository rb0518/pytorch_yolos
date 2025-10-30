#include <filesystem>

#include <torch/torch.h>
#include <torch/script.h>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>  // nms要用到

#define GLOG_USE_GLOG_EXPORT

#include "gflags/gflags.h"
#include "glog/logging.h"

#include "yaml_load.h"
#include "BaseModel.h"
#include "yolo.h"

#include "utils.h"
#include "utils/general.h"
#include "plots.h"
#include <regex>
#include "train.h"
#include "datasets.h"
#include "progressbar.h"
#include "SpecTest.h"

// jit_weights如果设置了，就会先调用pretrain从torchscript,除了代码调试，默认应为""
// 仿yolov5中代码，FLAGS_weights和FLAGS_jit_weights各取一
DEFINE_string(runtype,  "train",                "runtype: [train, predict, jit]");

// 对应Yolov5-5.0代码
DEFINE_string(weights,  "",                     "initial weight path");
DEFINE_string(cfg,      "models/11/yolo11.yaml",  "model.yaml path");
DEFINE_string(data,     "data/coco128.yaml",    "data.yaml path");
DEFINE_string(hyp,      "data/hyp.scratch.yaml","hyperparameters path");

DEFINE_int32(epochs,    300,                    "training epochs");
DEFINE_int32(batch_size,16,                     "load data batch size");
DEFINE_int32(img_size,  640,                    "image sizes");
DEFINE_bool(rect,       false,                  "rectangular training");
DEFINE_bool(resume,     false,                  "resume most recent training");
DEFINE_bool(save,       true,                   "save train checkpoints and predict results");
DEFINE_bool(notest,     true,                   "only test final epoch");

DEFINE_bool(noautoanchor, true,                 "diable autoanchor check");
DEFINE_bool(evolve,     false,                  "evolve hyperparameters");
DEFINE_string(bucket,   "",                     "gsutil bucket");
DEFINE_bool(cache_images, false,                "cache images for faster training");
DEFINE_bool(image_weights, false,               "use weighted image selection for training");
DEFINE_string(device,   "gpu",                  "cpu gpu");
DEFINE_bool(multi_scale,false,                  "vary img-size +/- 50%%");
DEFINE_bool(single_cls, false,                  "train multi-class data as single-class");
DEFINE_string(optimizer,  "auto",               "(str) optimizer to use, choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]");
DEFINE_bool(sync_bn,    false,                  "use SyncBatchNorm, only available in DDP mode");
DEFINE_int32(local_rank,-1,                     "DDP parameter, do not modify");
DEFINE_int32(workers,   8,                      "maximum number of dataloader workers");
DEFINE_string(project,  "runs/train",           "save to project/name");
DEFINE_int32(entity,    -1,                     "W&b entity"); // python None
DEFINE_string(name,     "exp",                  "save to project/name");
DEFINE_bool(exist_ok,   false,                  "existing project/name ok, do not increment");
DEFINE_bool(quad,       false,                  "quad dataloader");
DEFINE_bool(linear_lr,  false,                  "linear LR");
DEFINE_double(label_smoothing, 0.0,            "Label smoothing espilon");
DEFINE_bool(upload_dataset, false,              "Upload dataset as W&B artifact table");
DEFINE_int32(bbox_interval, -1,                 "Set bounding-box image logging interval for W&B");
DEFINE_int32(save_period, -1,                   "log model after every 'save_period' epoch");
DEFINE_string(artifact_alias,                   "latest", "version of dataset artifact to be used");


DEFINE_bool(is_segment, false, "predict or jit runtype");
// 约定script.pt是train export， torchscript.pt是eval导出
DEFINE_string(jit_weights,  "", "path to pytorch pt file");             // load pytorch pretrained weights, must be script export

void parse_args(VariantConfigs& args)
{
    args["weights"] = FLAGS_weights;
    args["model"] = FLAGS_cfg;
    args["data"] = FLAGS_data;
    args["epochs"] = FLAGS_epochs;
    args["batch"] = FLAGS_batch_size;
    args["imgsz"] = FLAGS_img_size;
    args["save"] = FLAGS_save;
    args["save_period"] = FLAGS_save_period;
    args["device"] = FLAGS_device;    
    args["workers"] = FLAGS_workers;

    args["project"] = FLAGS_project;
    args["name"] = FLAGS_name;
    args["exist_ok"] = FLAGS_exist_ok;
    // 后续添加或者直接修改default.yaml

    args["jit_weights"] = FLAGS_jit_weights;
}

int main(int argc, char* argv[])
{
    google::ParseCommandLineFlags(&argc, &argv, false);
    ::google::InitGoogleLogging(argv[0]);
    FLAGS_alsologtostderr = true;

    std::string root_path = get_root_path_string();
    std::string default_yaml_file = std::filesystem::path(root_path).append("cfgs").append("default.yaml").string();
    VariantConfigs args_default = load_cfg_yaml(default_yaml_file);


    auto set_model = [&](){
        std::vector<std::string> runtype = {"train", "val", "predict"};
        if(check_str_in_strs(runtype, FLAGS_runtype))
            args_default["mode"] = FLAGS_runtype;
        else{
            LOG(WARNING) << FLAGS_runtype << " not support. will set default: train";
            args_default["mode"] = "train";
        }
    };
    
    auto get_task = [&](){
        auto [name_ok, version, scale, task] = parse_yolo_config(std::filesystem::path(FLAGS_cfg).filename().string());
        if(name_ok)
        {
            if(version != 5 && version != 11 && version != 12)
            {
                LOG(ERROR) << "only complete support v5, v11, v12.";
                exit(-1);
            }

            if(task == "")
                task = "detect";
            if(scale == "")
                scale = "n";
        }
        else{
            LOG(ERROR) << FLAGS_cfg << " not right, or we can't support you input file.";
            exit(-1);
        }

        args_default["task"] = task;
        args_default["cfg_scale"] = scale;               
    };

    auto check_pretrained = [&](){
        bool fileexists = std::filesystem::exists(std::filesystem::path(root_path).append(FLAGS_weights));
        if(FLAGS_weights=="" || !fileexists)
            args_default["pretrained"] = false;
    };

    parse_args(args_default);

    // 特殊处理
    set_model();
    get_task();

    BaseTrainer newtrain(root_path, args_default);
    if(FLAGS_runtype=="train")
    {
        newtrain.do_train();
        // torch::Device device = torch::cuda::is_available() && FLAGS_device != "cpu" ? torch::Device(torch::kCUDA, 0)
        //     : torch::Device(torch::kCPU); //select_device(opt["device"], opt["batch_size"]);

        // train(root_path, args_default, device, FLAGS_jit_weights);
    }
    else if(FLAGS_runtype=="predict" || FLAGS_runtype == "jit")
    {
        float confidence_threshold = 0.4f;
        float iou_threshold = 0.45f;

        if (FLAGS_jit_weights == "")
        {   
            FLAGS_jit_weights = "weights/yolov5s.script.pt";
            if(FLAGS_is_segment)
                FLAGS_jit_weights = "weights/yolov5s-seg.torchscript.pt";
        }
        std::string prj_and_name = FLAGS_runtype=="jit" ? "runs/detect/jit" : "runs/detect/exp";
        auto search_path = std::filesystem::path(root_path).append(prj_and_name).string();
        prj_and_name = increment_path(search_path, false);
        std::cout << "save dir: " << prj_and_name << std::endl;
        
        if (!std::filesystem::exists(prj_and_name))
            std::filesystem::create_directories(std::filesystem::path(prj_and_name));
        
        std::vector<std::string> img_files;
        std::vector<std::string> img_types={".jpg"};
        listallfiles_withsuffixes(std::filesystem::path(root_path).append("./data/images"), img_files, img_types);

        torch::Device device = torch::cuda::is_available() && FLAGS_device != "cpu" ? torch::Device(torch::kCUDA, 0) : torch::Device(torch::kCPU);

        std::string jit_filename = std::filesystem::path(root_path).append(FLAGS_jit_weights).string();
        std::string model_filename = std::filesystem::path(root_path).append(FLAGS_cfg).string();
        std::string weights_filename = std::filesystem::path(root_path).append(FLAGS_weights).string();
        torch::jit::script::Module jit_model;

        try {
            jit_model = torch::jit::load(jit_filename);
            std::cout << "torch::jit::load over..." << std::endl;
        }
        catch (const c10::Error& e)
        {
            std::cerr << e.what() << "\n";
            std::exit(EXIT_FAILURE);
        }
        catch (...) {
            std::cerr << "torch::jit::load other exception \n";
            std::exit(EXIT_FAILURE);
        }
        jit_model.eval();

        auto model = Model(model_filename, 80, FLAGS_img_size, FLAGS_img_size, 3, false);
        model->eval(); 
        if(FLAGS_runtype == "predict")
        {
#if 1            
            torch::serialize::InputArchive ckpt;
            ckpt.load_from(weights_filename);

            torch::serialize::InputArchive model_in;
            if(ckpt.try_read("model", model_in))
            {
                model->load(model_in);
            }
#else
            model->eval();  // 要用best.pt
            torch::load(model, weights_filename);
#endif
        }
        
        auto test_tensor = torch::ones({1, 3, FLAGS_img_size, FLAGS_img_size});
        test_tensor = test_tensor.to(device);
        model->forward(test_tensor);

        model->eval();
        model->to(device);
        jit_model.to(device);

        // 测试代码，经fuse_conv_and_bn，model与jit 推理模型调入一致，126个参数全一一对应了
        if(0)
        {    
            for (auto& sub_module : model->modules(false))
            {
                if (sub_module->name() == "ConvImpl")
                {
                    std::cout << sub_module->name() << std::endl;
                    sub_module->as<Conv>()->fuse_conv_and_bn(true);
                }
            }            

            std::unordered_map<std::string, torch::Tensor> jit_params;
            for (const auto& param : jit_model.named_parameters())
            {
                jit_params[param.name] = param.value;
                std::cout << "jit_model:  " << param.name << std::endl;
            }
            auto jit_params_count = jit_params.size();
            int n_params = 0;
            int count_changes = 0;

            for (auto& param : model->named_parameters())
            {
                torch::AutoGradMode enable_grad(false);
                //torch::NoGradGuard no_grad;   // 对模板进行操作时，不能是目标与源tensor一个有grad,一个无grad      
                std::string str_name = param.key();
                //std::cout << "model named_parameters: " << str_name << std::endl;
                //  "model-1.m-1.bn1.bias" ==> "model.1.m.1.bn1.bias
                str_name = std::regex_replace(str_name, std::regex("-"), ".");
                std::cout << "model named_parameters: " << str_name << std::endl;
                
                if (jit_params.find(str_name) != jit_params.end())
                {
                    if (jit_params[str_name].sizes() == param.value().sizes())
                    {
                        //std::cout << "trans ok : " << str_name << std::endl;
                        count_changes += 1;
                        param.value().data().copy_(jit_params[str_name].data());
                    }
                    else 
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
        }



        std::vector<std::string> names={"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
                                        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                                        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                                        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                                        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                                        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                                        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                                        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
                                        "hair drier", "toothbrush" };

        for(int i = 0; i < img_files.size(); i++)
        {
            std::string filename = img_files[i];
            cv::Mat src_image = cv::imread(filename);
            int src_width = src_image.cols;
            int src_height = src_image.rows;
            cv::resize(src_image, src_image, cv::Size(FLAGS_img_size, FLAGS_img_size));
            cv::Mat input_image;
            cv::cvtColor(src_image, input_image, cv::COLOR_BGR2RGB);

            input_image.convertTo(input_image, CV_32FC3, 1.0 / 255.0);
            at::Tensor input_tensor = torch::from_blob(input_image.data,
                { 1, 640, 640, 3 }).to(device);    // 补齐batch_size维
            input_tensor = input_tensor.permute({ 0, 3, 1, 2 }).contiguous();
            
            at::Tensor output_tensor;
            at::Tensor output_tensor_proto;
            bool is_segment = FLAGS_is_segment;
            int nm = is_segment ? 32 : 0;
            if(FLAGS_runtype == "jit")
            {
                std::vector<torch::jit::IValue> inputs;
                inputs.emplace_back(input_tensor);
                auto jit_outs = jit_model.forward(inputs);
                if(is_segment)
                {
                    output_tensor = jit_outs.toTuple()->elements()[0].toTensor();    // 保留的目的，如体果是Half模式，这里要调用toFloat从FP16转换到FP32
                    output_tensor_proto = jit_outs.toTuple()->elements()[1].toTensor();
                    std::cout << "pred: " << output_tensor.sizes() << " proto: " << output_tensor_proto.sizes() << std::endl;
                }
                else
                    output_tensor = jit_outs.toTuple()->elements()[0].toTensor();    // 保留的目的，如体果是Half模式，这里要调用toFloat从FP16转换到FP32
            }
            else
            {
                auto [pred , empty_v, pred_mask] = model->forward(input_tensor);
                output_tensor = pred;
                output_tensor_proto = pred_mask;
                std::cout << "pred: " << output_tensor.sizes() << " proto: " << output_tensor_proto.sizes() <<" nm: " << nm << std::endl;            
            }
            std::vector<torch::Tensor> bboxs;
            bboxs= non_max_suppression(output_tensor, confidence_threshold, iou_threshold, {},
                    false, false, {}, 300, nm);

            std::vector<cv::Mat> mask_overlays;

            for(int i = 0; i < bboxs.size(); i++)
            {
                auto boxs = bboxs[i];
                std::cout << i << " box size: " << boxs.sizes() << std::endl;
                if(boxs.size(0)!=0)
                {
                    torch::Tensor masks; 
                    if(is_segment)
                    {
                        auto proto = output_tensor_proto[i];
                        proto = proto.to(boxs.device());
                        auto mask_in = boxs.index({torch::indexing::Slice(), torch::indexing::Slice(6, 6 + nm)});
                        auto bboxes = boxs.index({torch::indexing::Slice(), torch::indexing::Slice(0, 4)});
                        std::vector<int64_t> shape = {input_image.rows, input_image.cols};
                        masks = process_mask(proto, mask_in, bboxes, shape, true);
                        std::cout << "is_segment " << boxs.sizes() << " masks: " << masks.sizes() << " " << masks.dtype() << " " << sizeof(torch::kFloat32) << std::endl;
                        cv::split(src_image, mask_overlays);
                    }                    
                    
                    for(int j = 0; j < boxs.size(0); j++)
                    {
                        auto x1 = boxs[j][0].item().toFloat();
                        auto y1 = boxs[j][1].item().toFloat();
                        auto x2 = boxs[j][2].item().toFloat();
                        auto y2 = boxs[j][3].item().toFloat();
                        auto score = boxs[j][4].item().toFloat();
                        auto cls_id = boxs[j][5].item().toInt();

                        auto typecolor = SingletonColors::getInstance()->get_color_scalar(cls_id);

                        cv::rectangle(src_image, cv::Point(int(x1), int(y1)), cv::Point(int(x2), int(y2)), typecolor, 2);
                        std::cout << i <<" " << j <<" " << x1 << " " <<y1 << " " << x2 << " " << y2 << " cls " << cls_id << " s: " << score << std::endl;

                        std::stringstream ss;
                        ss << names[cls_id] <<" " << std::to_string(score);
                        cv::putText(src_image, ss.str(), cv::Point(x1, std::max(0, int(y1) - 2)), cv::FONT_HERSHEY_PLAIN, 1.,
                            typecolor, 2);


                        if(is_segment)
                        {
                            auto mask_b = masks[j].clone().unsqueeze(0);

                            cv::Mat mat_mask = cv::Mat(mask_b.size(1), mask_b.size(2), CV_32FC1);
                            std::memcpy((void*)mat_mask.data, mask_b.data_ptr(), mask_b.element_size()*mask_b.numel());
                            cv::imshow("seg_"+std::to_string(j), mat_mask);

                            auto [c_r, c_g, c_b] = SingletonColors::getInstance()->get_color_uchars(cls_id);
                            float f_mask_thr = 0.5 + std::max(0.0f, (0.5f - score))/10.f; // 二值化
                            mask_b = mask_b.gt_(f_mask_thr);    
                            mask_b = mask_b.to(torch::kByte);
                            int c = mask_b.size(0);
                            int h = mask_b.size(1);
                            int w = mask_b.size(2);

                            float f_r = float(c_r);
                            float f_g = float(c_g);
                            float f_b = float(c_b);
                            uchar* mask_b_ptr = (uchar*)mask_b.data_ptr();
                            uchar* over_r = (uchar*)mask_overlays[2].data;
                            uchar* over_g = (uchar*)mask_overlays[1].data;
                            uchar* over_b = (uchar*)mask_overlays[0].data;

                            for (int k_pixel = 0; k_pixel < mask_b.numel(); k_pixel++) 
                            {
                                float f_front = float(*mask_b_ptr) * 0.3f;
                                float f_back = 1.f - f_front;

                                float s_r = float(*over_r);
                                float s_g = float(*over_g);
                                float s_b = float(*over_b);

                                float f_r_tmp = s_r * f_back + f_r * f_front;
                                float f_g_tmp = s_g * f_back + f_g * f_front;
                                float f_b_tmp = s_b * f_back + f_b * f_front;

                                *over_r = uchar(f_r_tmp);
                                *over_g = uchar(f_g_tmp);
                                *over_b = uchar(f_b_tmp);


                                mask_b_ptr += 1;
                                over_r += 1;
                                over_g += 1;
                                over_b += 1;
                            }                           
                        } 
                    }
                    cv::resize(src_image, src_image, cv::Size(src_width, src_height));
                    cv::imshow("result", src_image);

                    if (is_segment)
                    {
                        cv::Mat segment_result;
                        cv::merge(mask_overlays, segment_result);
                        cv::resize(segment_result, segment_result, cv::Size(src_width, src_height));                        
                        cv::imshow("seg_result", segment_result);
                    }
                    cv::waitKey();
                    cv::destroyAllWindows();
                
                    auto save_name = std::filesystem::path(prj_and_name).append(std::filesystem::path(filename).filename().string()).string();
                    std::cout << "save result to : " << save_name << std::endl;
                    cv::imwrite(save_name, src_image);
                }
            }
        }
    }
    else if(FLAGS_runtype=="temp_test")
    {
        Test_TorchFlip_Funs();

        auto test_case = [](const std::string& s) {
            auto [b, v, sz, t] = parse_yolo_config(s);
            if(!b)  
                std::cout << "Invalid config: " << s << " not a yolo cfg type name" << std::endl;
            else
                std::cout << "Input: " << s << "\n  -> Version: " << v 
                      << ", Size: " << (sz.empty() ? "[empty]" : sz)<< ", Task: " << (t.empty() ? "[empty]" : t) << "\n";
            };

        test_case("yolov8.yaml");
        test_case("yolov5s-obb.yaml");
        test_case("yolov12m-pose.yaml");
        test_case("yolov3x-cls.yaml");
        test_case("yolo.yaml");
        test_case("yolov5n.txt");
    }

    return 0;
}