#pragma once

#include <torch/torch.h>
#include <opencv2/opencv.hpp>


#include <iomanip>
#include <iostream>
#include <memory>
#include <tuple>
#include <charconv>

// 函数名定义说明： 函数名中带n，对应的是在图中的坐标
//Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height) format where (x1, y1) is the
//top-left corner and (x2, y2) is the bottom-right corner.
//Args:
//    x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x1, y1, x2, y2) format.
//Returns:
//    y (np.ndarray | torch.Tensor): The bounding box coordinates in (x, y, width, height) format.
torch::Tensor xyxy2xywh(const torch::Tensor& x);


//Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
//top-left corner and (x2, y2) is the bottom-right corner. Note: ops per 2 channels faster than per channel.
//Args:
//    x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.
//Returns:
//    y (np.ndarray | torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
torch::Tensor xywh2xyxy(const torch::Tensor& x);


//Convert normalized bounding box coordinates to pixel coordinates.
//Args:
//    x (np.ndarray | torch.Tensor): The bounding box coordinates.
//    w (int): Width of the image. Defaults to 640
//    h (int): Height of the image. Defaults to 640
//    padw (int): Padding width. Defaults to 0
//    padh (int): Padding height. Defaults to 0
//Returns:
//    y (np.ndarray | torch.Tensor): The coordinates of the bounding box in the format [x1, y1, x2, y2] where
//        x1,y1 is the top-left corner, x2,y2 is the bottom-right corner of the bounding box.
torch::Tensor xywhn2xyxy(const torch::Tensor& x, int w = 640, int h = 640, int padw = 0, int padh = 0);


//Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height, normalized) format. x, y,
//width and height are normalized to image dimensions.
//Args:
//    x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x1, y1, x2, y2) format.
//    w (int): The width of the image. Defaults to 640
//    h (int): The height of the image. Defaults to 640
//    clip (bool): If True, the boxes will be clipped to the image boundaries. Defaults to False
//    eps (float): The minimum value of the box0x00s width and height. Defaults to 0.0
//Returns:
//    y (np.ndarray | torch.Tensor): The bounding box coordinates in (x, y, width, height, normalized) format
torch::Tensor xyxyn2xywh(const torch::Tensor& x, int w = 640, int h = 640, bool clip = false, float eps = 0.0);

torch::Tensor xyn2xy(const torch::Tensor& x, int w = 640, int h = 640, int padw = 0, int padh = 0);

int searchfiles_in_folder(const std::string& folder, const std::string& exttype, 
        std::vector<std::string>& lists);

// try convert str to float(int), if success bool == true else false
std::tuple<float, bool> ConvertToNumber(const std::string& str);

std::vector<torch::Tensor> non_max_suppression(
    torch::Tensor prediction,
    float conf_thres = 0.25,
    float iou_thres = 0.45,
    const std::vector<std::vector<int>>& classes = {},
    bool agnostic = false,
    bool multi_label = false,
    const std::vector<std::vector<float>>& labels = {}
);

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
