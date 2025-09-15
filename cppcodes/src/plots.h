#pragma once
#include <mutex>
#include <vector>
#include <queue>
#include <mutex>
#include <chrono>
#include <thread>
#include <condition_variable>
#include <tuple>

#include <opencv2/opencv.hpp>
#include <torch/torch.h>


class SingletonColors {
private:
    inline static SingletonColors* instance = nullptr;
    inline static std::mutex mtx;
    
    std::vector<uint> colors_ = {
        0x00FF3838, 0x00FF9D97, 0x00FF701F, 0x00FFB21D, 0x00CFD231, 0x0048F90A, 0x0092CC17, 0x003DDB86, 0x001A9334, 0x0000D4BB,
        0x002C99A8, 0x0000C2FF, 0x00344593, 0x006473FF, 0x000018EC, 0x008438FF, 0x00520085, 0x00CB38FF, 0x00FF95C8, 0x00FF37C7
    };  
    // 私有构造函数
    SingletonColors() {}
    ~SingletonColors() = default;

public:
    // 删除拷贝构造和赋值操作
    SingletonColors(const SingletonColors&) = delete;
    SingletonColors& operator=(const SingletonColors&) = delete;

    // 获取唯一实例
    static SingletonColors* getInstance() {
        std::lock_guard<std::mutex> lock(SingletonColors::mtx);
        if(instance == nullptr)
            instance = new SingletonColors();
        return instance;
    }

    uint get_color_uint(int idx) {
        return colors_[idx % colors_.size()];
    }

    std::tuple<uchar, uchar, uchar> get_color_uchars(int idx) {
        uint col = colors_[idx % colors_.size()];
        return {
            uchar(col & 0xFF),
            uchar((col >> 8) & 0xFF),
            uchar((col >> 16) & 0xFF)
        };
    }    

    cv::Scalar get_color_scalar(int idx) {
        uint col = colors_[idx % colors_.size()];
        uchar uc1 = uchar(col & 0xFF);
        uchar uc2 = uchar((col >> 8) & 0xFF);
        uchar uc3 = uchar((col >> 16) & 0xFF);
        return cv::Scalar(uc1, uc2, uc3);
    }  
};

class Singleton_PlotBatchImages
{
private:
    const int max_size = 10;
    inline static Singleton_PlotBatchImages* instance = nullptr;
    inline static std::mutex mtx;
    inline static std::mutex mtx_queue;
    bool b_exit_flag = false;
    std::queue<std::tuple<torch::Tensor, torch::Tensor>> data_queue;
    std::condition_variable cv;

    std::string save_dir = ".";
    std::string prefix_name = "train";
    int incremental_counter = 0;

    Singleton_PlotBatchImages(const std::string& dir, const std::string& prefix);
    ~Singleton_PlotBatchImages(){
        b_exit_flag = true;
    }

public:
    Singleton_PlotBatchImages(const SingletonColors&) = delete;
    Singleton_PlotBatchImages& operator=(const Singleton_PlotBatchImages&) = delete;

    static Singleton_PlotBatchImages* getInstance(std::string dir, std::string prefix) {
        std::lock_guard<std::mutex> lock(Singleton_PlotBatchImages::mtx);
        if (instance == nullptr)
        {
            instance = new Singleton_PlotBatchImages(dir, prefix);
        }
        else
        {
            instance->save_dir = dir;
            instance->prefix_name = prefix;
        }
        return instance;
    }

    bool queue_can_push(){return data_queue.size() < max_size;}
    bool queue_have_data(){return !data_queue.empty();}
    void push_data(torch::Tensor imgs, torch::Tensor targets);
    void pop_data();
    void plot_one_batchs(torch::Tensor imgs, torch::Tensor targets);
    void reset_counter() { incremental_counter = 0; }
};

// img [3, h, w] or [1, h, w]， 输入前自行处理
// return cv::Mat 8UC3
cv::Mat convert_tensor_to_mat(torch::Tensor img, int img_width, int img_height, 
    int channels, bool ret_color = true);

void plot_images(torch::Tensor images, torch::Tensor targets, 
    std::string path, std::string fname, 
    bool normalise = false,
    std::vector<std::string> names = {},
    int max_subplots = 16);
