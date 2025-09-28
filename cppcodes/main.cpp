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

// jit_weights如果设置了，就会先调用pretrain从torchscript,除了代码调试，默认应为""
// 仿yolov5中代码，FLAGS_weights和FLAGS_jit_weights各取一
DEFINE_string(runtype,  "train",                "runtype: [train, predict, jit]");

// 对应Yolov5-5.0代码
DEFINE_string(weights,  "",                     "initial weight path");
DEFINE_string(cfg,      "models/yolov5s.yaml",  "model.yaml path");
DEFINE_string(data,     "data/coco128.yaml",    "data.yaml path");
DEFINE_string(hyp,      "data/hyp.scratch.yaml","hyperparameters path");

DEFINE_int32(epochs,    300,                    "training epochs");
DEFINE_int32(batch_size,16,                     "load data batch size");
DEFINE_int32(img_size,  640,                    "image sizes");
DEFINE_bool(rect,       false,                  "rectangular training");
DEFINE_string(resume,   "",                     "resume most recent training");
DEFINE_bool(nosave,     true,                   "only save final epoch");
DEFINE_bool(notest,     true,                   "only test final epoch");

DEFINE_bool(noautoanchor, true,                 "diable autoanchor check");
DEFINE_bool(evolve,     false,                  "evolve hyperparameters");
DEFINE_string(bucket,   "",                     "gsutil bucket");
DEFINE_bool(cache_images, false,                "cache images for faster training");
DEFINE_bool(image_weights, false,               "use weighted image selection for training");
DEFINE_string(device,   "gpu",                  "cpu gpu");
DEFINE_bool(multi_scale,false,                  "vary img-size +/- 50%%");
DEFINE_bool(single_cls, false,                  "train multi-class data as single-class");
DEFINE_bool(adam,       false,                  "use torch.optim.Adam optimizer");
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


DEFINE_string(jit_weights,  "", "path to pytorch pt file");             // load pytorch pretrained weights, must be script export

int main(int argc, char* argv[])
{
    google::ParseCommandLineFlags(&argc, &argv, false);
    ::google::InitGoogleLogging(argv[0]);
    FLAGS_alsologtostderr = true;

    std::string root_path = get_root_path_string();

    if(FLAGS_runtype=="train")
    {
        VariantConfigs opt = set_cfg_opt_default();
        std::string ckpt = "";
        if (FLAGS_resume != "")  // 将从指定的目录中恢复训练
        {
            // if FLAGS_resume 设置的文件不存在，在默认的目录中去找最新的pt文件
            if(std::filesystem::exists(std::filesystem::path(root_path).append(FLAGS_resume)))
            {
                ckpt = std::filesystem::path(root_path).append(FLAGS_resume).string();
            }
            else{
                std::string search_path = std::filesystem::path(root_path).append(std::get<std::string>(opt["project"])).string();
                auto tmp_opt =  get_last_run(search_path);
                if(tmp_opt != "")
                    ckpt = tmp_opt;
            }
            
            if(ckpt == ""){
                LOG(ERROR) << "not set right, check resume settings";
                exit(-1);
            }
            
            auto opt_yaml = std::filesystem::path(ckpt).parent_path().parent_path().append("opt.yaml").string();
            std::cout << "Use last opt: " << opt_yaml << " " << std::filesystem::path(ckpt).parent_path().parent_path().string() << std::endl;
            opt = load_cfg_yaml(opt_yaml);
            //opt.cfg, opt.weights, opt.resume, opt.batch_size, opt.global_rank, opt.local_rank 
            //      = '', ckpt, True, opt.total_batch_size, * apriori  # reinstate
            //opt["cfg"] = ""; //依然采取从cfg中读取网络结构，再调入weights的方法，这里就不要去原目录中的cfg设置了
            // 除以下参数外，其它参数全部沿用resume目录中的配置
            opt["weights"] = std::filesystem::relative(std::filesystem::path(ckpt), std::filesystem::path(root_path)).string();
            opt["resume"] = true;           // 将resume 设置为true
            auto hypfilename = std::filesystem::path(ckpt).parent_path().parent_path().append("hyp.yaml");
            opt["hyp"] = std::filesystem::relative(hypfilename, std::filesystem::path(root_path)).string();
            if (std::get<std::string>(opt["data"]) != FLAGS_data)
                opt["data"] = FLAGS_data;
        }
        else
        {
            // load opt from cfgs/opt.yaml
            load_default_environment(root_path, opt);

            opt["weights"] = FLAGS_weights;
            opt["cfg"] = FLAGS_cfg;
            opt["data"] = FLAGS_data;
            opt["hyp"] = FLAGS_hyp;
            opt["batch_size"] = FLAGS_batch_size;
            opt["img_size"] = std::vector<int>({ FLAGS_img_size, FLAGS_img_size });
            opt["rect"] = FLAGS_rect;
            opt["nosave"] = FLAGS_nosave;
            opt["notest"] = FLAGS_notest;
            opt["noautoanchor"] = FLAGS_noautoanchor;
            opt["evolve"] = FLAGS_evolve;
            opt["device"] = FLAGS_device;
            opt["adam"] = FLAGS_adam;
            opt["workers"] = FLAGS_workers;
            opt["project"] = FLAGS_project;
            opt["name"] = FLAGS_name;
            opt["exist_ok"] = FLAGS_exist_ok;
            opt["quad"] = FLAGS_quad;
            opt["linear_lr"] = FLAGS_linear_lr;
            opt["label_smoothing"] = float(FLAGS_label_smoothing);
            opt["save_period"] = FLAGS_save_period;
            opt["total_batch_size"] = FLAGS_batch_size;
        }

        opt["epochs"] = FLAGS_epochs;
        std::string prj_and_name = std::get<std::string>(opt["project"]) + "/" + std::get<std::string>(opt["name"]);
        auto search_path = std::filesystem::path(root_path).append(prj_and_name).string();
        opt["exist_ok"] = FLAGS_exist_ok;
        prj_and_name = increment_path(search_path, std::get<bool>(opt["exist_ok"]));
        prj_and_name = std::filesystem::relative(std::filesystem::path(prj_and_name), std::filesystem::path(root_path)).string();
        opt["save_dir"] = prj_and_name;


        torch::Device device = torch::cuda::is_available() && FLAGS_device != "cpu" ? torch::Device(torch::kCUDA, 0)
            : torch::Device(torch::kCPU); //select_device(opt["device"], opt["batch_size"]);

        // Hyperparameters
        VariantConfigs hyp;
        if (std::get<std::string>(opt["hyp"]) == "")
        {
            hyp = set_cfg_hyp_default();
        }
        else
        {
            std::string hyp_file = std::filesystem::path(root_path).append(std::get<std::string>(opt["hyp"])).string();
            hyp = load_cfg_yaml(hyp_file);
        }

        train(root_path, hyp, opt, device, FLAGS_jit_weights);
    }
    else if(FLAGS_runtype=="predict" || FLAGS_runtype == "jit")
    {
        float confidence_threshold = 0.4f;
        float iou_threshold = 0.45f;

        if (FLAGS_jit_weights == "")
            FLAGS_jit_weights = "weights/yolov5s.script.pt";
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
        auto test_tensor = torch::ones({1, 3, FLAGS_img_size, FLAGS_img_size});
//        test_tensor = test_tensor.to(device);
        model->forward(test_tensor);
        if(FLAGS_runtype == "predict")
        {
            torch::serialize::InputArchive ckpt;
            ckpt.load_from(weights_filename);

            torch::serialize::InputArchive model_in;
            if(ckpt.try_read("model", model_in))
            {
                model->load(model_in);
            }
        }
        jit_model.to(device);
        model->to(device);
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
            if(FLAGS_runtype == "jit")
            {
                std::vector<torch::jit::IValue> inputs;
                inputs.emplace_back(input_tensor);
                auto jit_outs = jit_model.forward(inputs).toTuple()->elements()[0].toTensor();
                output_tensor = jit_outs;    // 保留的目的，如体果是Half模式，这里要调用toFloat从FP16转换到FP32
            }
            else
            {
                auto [pred , empty_v, pred_mask] = model->forward(input_tensor);
                output_tensor = pred;
            }
            std::vector<torch::Tensor> bboxs;
            bboxs= non_max_suppression(output_tensor, confidence_threshold, iou_threshold, {},
                false, false, {});

            
            

            for(int i = 0; i < bboxs.size(); i++)
            {
                auto boxs = bboxs[i];
                std::cout << i << " box size: " << boxs.sizes() << std::endl;
                if(boxs.size(0)!=0)
                {
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
                    }
                    cv::resize(src_image, src_image, cv::Size(src_width, src_height));
                    cv::imshow("result", src_image);
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
        VariantConfigs opt;
        load_default_environment(root_path, opt);

        opt["weights"] = FLAGS_weights;
        opt["cfg"] = FLAGS_cfg;
        opt["data"] = FLAGS_data;
        opt["hyp"] = FLAGS_hyp;
        opt["batch_size"] = FLAGS_batch_size;
        opt["img_size"] = std::vector<int>({ FLAGS_img_size, FLAGS_img_size });
        opt["rect"] = FLAGS_rect;
        opt["nosave"] = FLAGS_nosave;
        opt["notest"] = FLAGS_notest;
        opt["noautoanchor"] = FLAGS_noautoanchor;
        opt["evolve"] = FLAGS_evolve;
        opt["device"] = FLAGS_device;
        opt["adam"] = FLAGS_adam;
        opt["workers"] = FLAGS_workers;
        opt["project"] = FLAGS_project;
        opt["name"] = FLAGS_name;
        opt["exist_ok"] = FLAGS_exist_ok;
        opt["quad"] = FLAGS_quad;
        opt["linear_lr"] = FLAGS_linear_lr;
        opt["label_smoothing"] = float(FLAGS_label_smoothing);
        opt["save_period"] = FLAGS_save_period;
        opt["total_batch_size"] = FLAGS_batch_size;


        torch::Device device = torch::cuda::is_available() && FLAGS_device != "cpu" ? torch::Device(torch::kCUDA, 0)
            : torch::Device(torch::kCPU); //select_device(opt["device"], opt["batch_size"]);

        int imgsz = FLAGS_img_size;

        std::string data_file = std::filesystem::path(root_path).append(FLAGS_data).string();            
        std::vector<std::string> names;
        std::string train_path;
        std::string val_path;
        if (std::filesystem::exists(std::filesystem::path(data_file)))
        {       
            read_data_yaml(data_file, train_path, val_path, names); // 对应两种yaml定义，list和map
        }
        else
        {
            std::cout << ColorString("not found you offer data yaml file: ", "Error");
        }
        auto is_coco = data_file.substr(data_file.length() - 10, data_file.length() - 1) == "coco.yaml";
        auto nc = (std::get<bool>(opt["single_cls"])) ? 1 : names.size();

        auto cfg_file = std::get<std::string>(opt["cfg"]);
        if(cfg_file=="")
        {
            LOG(WARNING) << "cfg not define, use default sets yolov5s.yaml";
            cfg_file = std::filesystem::path(root_path).append("models/yolov5s.yaml").string();
        }
        else
            cfg_file = std::filesystem::path(root_path).append(cfg_file).string();

        std::cout << ColorString("cfg: ", "Info") << cfg_file << std::endl;
        std::shared_ptr<Model> ptr_model = std::make_shared<Model>(cfg_file, nc, imgsz, imgsz, 3, false); 
        auto model = ptr_model->get();
        model->show_modelinfo();
        model->to(device);

        std::cout << ColorString("Test datasets: ", "G") << val_path << std::endl;    
        VariantConfigs hyp = set_cfg_hyp_default();
        bool augment = true;        // true && rect == false ==> mosaic = true
        bool rect = std::get<bool>(opt["rect"]);
        bool image_weights = false;
        bool cache_images = false;
        int gs = 32;
        float pad = 0.5f;
        int batch_size = FLAGS_batch_size;

        auto [val_dataloader, num_images] = create_dataloader_segment(train_path, imgsz, nc, batch_size, gs,
                                                                opt, hyp, augment, pad, false, false, 1); 
        std::cout << "test dataset, total images: " << num_images << " batch_size: " << batch_size << std::endl;
        std::cout << "init dataloader over..." << std::endl;
        //progressbar pbar(num_images/batch_size, 36);
        int batch_idx = 0;
        for (auto batch : *val_dataloader)
        {
            auto imgs = batch.data;
            auto targets = batch.target;
            auto masks = batch.mask;
            auto paths = batch.path;
            auto shapes = batch.shape;

            std::cout << batch_idx << "  Input images: " << imgs.sizes() << " targets: " << targets.sizes() << " masks: " << masks.sizes() << std::endl;

            int width = imgs.size(3);
            int height = imgs.size(2);
            int channel = imgs.size(1);
            int bs = imgs.size(0);

            imgs = imgs.to(torch::kFloat) / 255.f;
            imgs = imgs.to(device);

            auto [pred, v_train, pred_seg] = model->forward(imgs);

            std::cout << batch_idx << "  Return vector_train: " << v_train[0].sizes() << " pred: " 
                        << pred.sizes() << " pred_seg: " << pred_seg.sizes() << std::endl;
            batch_idx += 1;
            //pbar.update();
        }
        std::cout << std::endl;
    }

    return 0;
}