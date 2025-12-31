#pragma once

#include <torch/torch.h>
#include <opencv2/opencv.hpp>


#include <iomanip>
#include <iostream>
#include <memory>
#include <tuple>
#include <charconv>
#include <filesystem>


#include "globaldef.h"

int searchfiles_in_folder(const std::string& folder, const std::string& exttype, 
        std::vector<std::string>& lists);
void getAllImageInFolder(const std::string& folder, std::vector<std::string>& lists, bool enable_subfolder = false);        
int listallfiles_withsuffixes(const std::filesystem::path& path, std::vector<std::string>& file_lists,
    std::vector<std::string> suffixes, bool enable_subfolder = false);

std::tuple<std::string, std::string, std::string, std::string> getSaveDirAndFilenames(VariantConfigs& args, 
    std::string root_dir, bool create_dir = true);

/*
    从一个 torch script 文件中调入相同的module的权重文件, 其中几个约定：
    1、pytorch中不需要显示注册，会将torch::nn::Module变量名自己作为名字注册，同时会用0x00.0x00来隔离各级名称，并自动添加同名排序
    2、libtorch中必须显示调用register_module来注册变量，同时名字中不能用0x00.0x00，同级不能有同名的，用ModuleList等容器进行自动
       命名管理，但要求就限制太多了，自己手动注册，为了保证能够名字索引，同级多个模板的序号之间用0x00-0x00连接，在索引时将其替换为
       0x00.0x00，所有注册名一定要与python中变量名保持一致
    3、网络上发布的pt文件，多为pytorch文件，module是放在['model'0x00']中，同时要注意是否是half模式，这样模型尺寸会小，考虑到所有的
       模板情况，在pytorch.jit.trace前，将model->float(), model->to(torch::Device(torch::kCPU))
*/
int LoadWeightFromJitScript(const std::string strScriptfile, torch::nn::Module& model, bool b_showinfo = false);


// Generate an incremental queue and randomly shuffle the order
std::vector<int> random_queue(int n) ;
float random_beta(float alpha, float beta);
float random_uniform(float start = 0.0f, float end = 1.0f);

void init_torch_seek(int seed = 0);

inline std::string ColorString(const std::string& info, const std::string& col ="")
{
/*    
    \033[30m  # 设置前景色为黑色
    \033[31m  # 设置前景色为红色
    \033[32m  # 设置前景色为绿色
    \033[33m  # 设置前景色为黄色
    \033[34m  # 设置前景色为蓝色
    \033[35m  # 设置前景色为品红
    \033[36m  # 设置前景色为青色
    \033[37m  # 设置前景色为白色
*/
    if(col == "G" || col =="Info")
        return std::string("\033[32m"+info+"\033[37m");
    else if(col == "Y" || col =="Warning")
        return std::string("\033[33m"+info+"\033[37m");
    else if(col == "R" || col =="Error")
        return std::string("\033[31m"+info+"\033[37m");

    return std::string("\033[34m"+info+"\033[37m");
}

/*
    按新版本python代码，自动适应yolov8.yaml，yolov8m.yaml，缺失为assuming scale='n'
    利用正则式适配分解函数名，有，返回版本、scale和task，如果没有默认返回12 'n' ""
    '''cpp
        auto test_case = [](const std::string& s) {
            auto [b, v, sz, t] = parse_yolo_config(s);
            if(!b)  
                std::cout << "Invalid config: " << s << " not a yolo cfg type name" << std::endl;
            else
                std::cout << "Input: " << s << "\n  -> Version: " << v 
                      << ", Size: " << sz << ", Task: " << (t.empty() ? "[empty]" : t) << "\n";
            };

        test_case("yolov8.yaml");
        test_case("yolov5s-obb.yaml");
        test_case("yolov12m-pose.yaml");
        test_case("yolov3x-cls.yaml");
    '''

*/
std::tuple<bool, int, std::string, std::string> parse_yolo_config(const std::string& filename);

// 2025-10-30 读取pytorch export Tensor文件
/*
    pytorch中torch.save调用pickle_save模式，libtorch中torch::save和
    torch::load是保持与torch::jit::load torch::jit::save一致
*/
torch::Tensor load_tensordata_from_file(std::string filename);

/*
* 目标，将yaml文件读入到内存中，为后续转换为tensor，存入到pt文件，实现读取pt时重构对应的网络
*/
bool readFileToMemory(const std::string& filePath, std::vector<unsigned char>& buffer);
torch::Tensor convertMemoryToTensor(std::vector<unsigned char>& buffer);
void convertTensorToMemory(torch::Tensor x, std::vector<unsigned char>& buffer);

cv::Mat polygon2mask(cv::Size img_size, torch::Tensor& segment, int color = 1, int downsample_ratio = 1);
torch::Tensor polygons2masks(cv::Size img_size, torch::Tensor segments, int color, int downsample_ratio = 1);
std::tuple<torch::Tensor, torch::Tensor> polygons2masks_overlap(cv::Size img_size, torch::Tensor segments, int downsample_ratio = 1);


inline std::vector<std::tuple<float, float>> changeTupleVector(float x, float y) {

    std::vector<std::tuple<float, float>> rets;
    rets.emplace_back(std::make_tuple(x, y));
    return rets;
}