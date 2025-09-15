#pragma once

#include <torch/torch.h>
#include <opencv2/opencv.hpp>


#include <iomanip>
#include <iostream>
#include <memory>
#include <tuple>
#include <charconv>

struct BBox_xyxy
{
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    int cls_id;

    void scale(float scale)
    {
        xmin = xmin * scale;
        ymin = ymin * scale;
        xmax = xmax * scale;
        ymax = ymax * scale;
    };

    std::tuple<float, float, float, float> to_xyhw()
    {
        float h = (ymax - ymin);
        float w = (xmax - xmin);
        float x = (xmin + xmax) / 2.0;
        float y = (ymin + ymax) / 2.0;
        return std::make_tuple(x, y, h, w);
    };
};

struct BBox_xyhw
{
    float x;
    float y;
    float h;
    float w;
    int cls_id;

    // 根据放缩率，自动放缩坐标
    void scale(float _x_scale, float _y_scale){
        x *= _x_scale;
        y *= _y_scale;
        h *= _y_scale;
        w *= _x_scale;
    };
    
    // 放缩坐标，传入坐标偏移量是根据整图大小进行的放缩
    void offset_s(float offset_x, float offset_y)
    {
        x += offset_x;
        y += offset_y;
    };

    std::tuple<float, float, float, float> to_xyxy()
    {
        auto xmin = x - w/2.0;
        auto ymin = y - h/2.0;
        auto xmax = xmin + w;
        auto ymax = ymin + h;
        return std::make_tuple(xmin, ymin, xmax, ymax);
    };

    std::tuple<int, int, int, int> getImageRect(int width, int height)
    {
        auto [xmin, ymin, xmax, ymax] = to_xyxy();
        return std::make_tuple(
            xmin * width,
            ymin * height,
            xmax * width,
            ymax * height
        );
    };
};

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


int searchfiles_in_folder(const std::string& folder, const std::string& exttype, 
        std::vector<std::string>& lists);

std::tuple<float, bool> ConvertToNumber(const std::string& str);

struct Detection {
    cv::Rect bbox;
    float score;
    int class_idx;
};

int non_max_suppression_old(torch::Tensor prediction, float conf_thres, float iou_thres, std::vector<Detection>& output);

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
    2、libtorch中必须显示调用register_module来注册变量，同时名字中不能用0x00.0x00，同级不能有同名的，不会自动增加变量序号，所以必须
       手动添加，为了保证能够名字索引，同级多个模板的序号之间用0x00-0x00连接，在索引时将其替换为0x00.0x00
       所有注册名一定要与python中变量名保持一致
    3、网络上发布的pt文件，多为pytorch文件，module是放在['model'0x00']中，同时要注意是否是half模式，这样模型尺寸会小，考虑到所有的
       模板情况，在pytorch.jit.trace前，将model->float(), model->to(torch::Device(torch::kCPU))
*/
int LoadWeightFromJitScript(const std::string strScriptfile, torch::nn::Module& model, bool b_showinfo = false);


std::tuple<std::string, std::string> get_checkpoint_files(const std::string& path);


// Generate an incremental queue and randomly shuffle the order
std::vector<int> random_queue(int n) ;
float random_beta(float alpha, float beta);
float random_uniform(float start = 0.0f, float end = 1.0f);

void init_torch_seek(int seed = 0);

#include "general.h"
void MatDrawTargets(cv::Mat& img, const torch::Tensor& labels, bool xywh=true, bool is_scale = true, int start_idx = 2);
