#include <filesystem>
#include <regex>


#define GLOG_USE_GLOG_EXPORT
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "yaml_load.h"
#include "utils.h"
#include "utils/general.h"

#include "train.h"
#include "val.h"
#include "pred.h"
#include "SpecTest.h"

// jit_weights如果设置了，就会先调用pretrain从torchscript,除了代码调试，默认应为""
// 仿yolov5中代码，FLAGS_weights和FLAGS_jit_weights各取一
DEFINE_string(runtype,  "train",                "runtype: [train, val, predict, jit]");

// 对应Yolov5-5.0代码
DEFINE_string(weights,  "",                     "initial weight path");
DEFINE_string(model,    "models/11/yolo11.yaml",  "model.yaml path");
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

/*
    测试cublas库是否正确初始化
*/
/*
#include <cublas_v2.h>
#include <cuda_runtime.h>
bool init_cublas()
{
    cudaSetDevice(0);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << " cuBlas initialization failed." << std::endl;
        return false;
    }

    cublasDestroy(handle);
    cudaStreamDestroy(stream);

    return true;
}
*/

bool get_task_from_filename(std::string root, std::string model_sets, 
    std::string& task_s, std::string& scale_id)
{
    auto get_task = [&](std::string filename) -> std::tuple<bool, std::string, std::string> {
        auto [name_ok, version, scale, task] = parse_yolo_config(filename);
        if(name_ok)
        {
            if(version != 5 && version != 11 && version != 12 && version !=13)
            {
                LOG(ERROR) << "only complete support v5, v11, v12.";
                name_ok =false;
            }

            if(task == "")
                task = "detect";
            if(scale == "")
                scale = "n";
        }
        else{
            LOG(ERROR) << FLAGS_model << " not right, or we can't support you input file.";
            name_ok =false;
        }

        return std::make_tuple(name_ok, task, scale);
    };

    bool get_task_ok = true;
    std::string file_ext = std::filesystem::path(model_sets).extension().string();
    std::string task_name;
    std::string scale_key;
    if(file_ext == ".pt")   // 指定的是pt文件 
    {
        if(std::filesystem::exists(std::filesystem::path(root).append(model_sets)))
        {
            torch::serialize::InputArchive in_arch;
            in_arch.load_from(std::filesystem::path(root).append(model_sets).string());
            std::cout << "load ptfile get keys: " << in_arch.keys() << std::endl;
            c10::IValue i_task;
            
            if(in_arch.try_read("task", i_task))
            {
                task_s = i_task.toStringRef();
            }
            else
                get_task_ok = false;

            c10::IValue i_scale;
            if(in_arch.try_read("scale", i_scale))
            {
                scale_id = i_task.toStringRef();
            }
            else
                get_task_ok = false;
        }
        else
        {
            std::cout << "not found file: " << std::filesystem::path(root).append(model_sets).string() << std::endl;
            get_task_ok = false;
        }
    }
    else if(file_ext == ".yaml")
    {   // 从文件名推断yolo任务
        auto [name_ok, task, scale] = get_task(std::filesystem::path(model_sets).filename().string());
        if(name_ok)
        {
            task_s = task;
            scale_id = scale;
        }
        else
            get_task_ok = false;
    }
    else{
        LOG(ERROR) << "Can't get task info from file: " << model_sets;
        get_task_ok = false;
    }

    return get_task_ok;
}


void parse_args(VariantConfigs& args)
{
    args["weights"] = FLAGS_weights;
    args["model"] = FLAGS_model;
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

    // init_cublas(); 测试cublas是否正常调用

    std::string root_path = get_root_path_string();
    std::string default_yaml_file = std::filesystem::path(root_path).append("cfgs").append("default.yaml").string();
    VariantConfigs args_default = load_cfg_yaml(default_yaml_file);


    auto set_mode = [&]() -> std::string {
        std::vector<std::string> runtype = {"train", "val", "predict"};
        if(check_str_in_strs(runtype, FLAGS_runtype))
            args_default["mode"] = FLAGS_runtype;
        else{
            LOG(WARNING) << FLAGS_runtype << " not support. will set default: train";
            args_default["mode"] = "train";
        }
        return std::get<std::string>(args_default["mode"]);
    };
    


    auto check_pretrained = [&](){
        bool fileexists = std::filesystem::exists(std::filesystem::path(root_path).append(FLAGS_weights));
        if(FLAGS_weights=="" || !fileexists)
            args_default["pretrained"] = false;
    };

    parse_args(args_default);
    std::string task_s, scale_id;
    // 根据设置，得到本次运行的任务目标: model, task, 用来生成对应的Trainer类
    std::string mode = set_mode();

    if(false == get_task_from_filename(root_path, FLAGS_model, task_s, scale_id))
    {
        LOG(ERROR) << "get the task from: " << FLAGS_model << " fail.";
        exit(-1);
    }

    std::cout << "mode: " << mode << " task: " << task_s << " scale: " << scale_id << std::endl;
    if (mode == "train")
    {
        std::shared_ptr<BaseTrainer> newtrain = nullptr;
        if (task_s == "detect")
            newtrain = std::make_shared<DetectionTrainer>(args_default, root_path);
        else if (task_s == "segment")
        {

        }


        if (newtrain != nullptr)
        {
            newtrain->setupTrainer();
            newtrain->doTrainTask();
        }
    }
    else if(mode == "val")
    {
        std::shared_ptr<BaseValidate> newValidate = nullptr;
        if(task_s == "detect")
        {
            newValidate = std::make_shared<DetectValidator>(args_default, root_path, nullptr, nullptr);
        }


        if (newValidate != nullptr)
        {
            newValidate->setupValidate();
            newValidate->doValidateTask();
        }
    }
    else if (mode == "predict") {
        std::shared_ptr<BasePredict> newPredict = nullptr;
        if (task_s == "predict")
        {
            newPredict = std::make_shared<DetectPredict>(args_default, root_path);
        }

        if (newPredict != nullptr)
        {
            newPredict->setupPredict();
            newPredict->doPredictTask();
        }
    }
    else if (FLAGS_runtype == "temp_test") 
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