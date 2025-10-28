#include "SpecTest.h"
#include "utils.h"
#include "plots.h"
#include <vector>

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

void Test_TorchFlip_Funs()
{
    /*
    测试结果 
        d1: 0 d2: 15
        d3: 240 d4: 255
        flip lr with contiguous: 
        d1: 15 d2: 0
        d3: 255 d4: 240
        flip ud after lr with contiguous: 
        d1: 255 d2: 240
        d3: 15 d4: 0

        ----------------------- flip without clone: 
        d1: 0 d2: 15
        d3: 240 d4: 255
        flip lr without contiguous: 
        d1: 15 d2: 0
        d3: 255 d4: 240
        flip ud after lr without contiguous: 
        d1: 255 d2: 240
        d3: 15 d4: 0

        -----------------------
        new test: 
        d1: 0 d2: 15
        d3: 240 d4: 255
        flip lr without contiguous: 
        d1: 240 d2: 255
        d3: 0 d4: 15
        flip ud without contiguous: 
        d1: 240 d2: 255
        d3: 0 d4: 15

        torch::fliplr、torch::flipud、tensor.fliplr和tensor.flipud不对  
        {1, 16, 16}不对
        {16， 16}正确
    */

    auto create3ChannelTensor =[](){
        auto t_1d = torch::arange(1, 17, torch::kByte);
        auto t_diag = torch::diag(t_1d).unsqueeze(0);

        auto ret = torch::cat({t_diag, t_diag, t_diag, t_diag, t_diag, t_diag}, 0);
        std::cout << "create3ChannelTensor: " << ret.sizes() << std::endl;
        return ret;
    };

    auto checkTensorMemoryArray = [](torch::Tensor src_t)
        {
            auto t = src_t.detach().to(torch::kByte);

            auto mem_size = t.element_size()*t.numel();
            cv::Mat mat = cv::Mat(t.size(0), t.size(1), CV_8UC1);
            std::memcpy((void*)mat.data, t.data_ptr(), t.element_size()*t.numel());
            uchar* mem_data = (uchar*)mat.data;
            auto d1 = int(mem_data[0]);
            auto d2 = int(mem_data[15]);
            auto d3 = int(mem_data[15*16]);
            auto d4 = int(mem_data[15*16 + 15]);
            std::cout << "d1: " << d1 << " d2: " << d2 << std::endl;
            std::cout << "d3: " << d3 << " d4: " << d4 << std::endl;
        };

    auto t_eye = torch::arange(0, 256).view({1, 16, 16});
    checkTensorMemoryArray(t_eye[0]);
    auto t_eyp_lr_clone = torch::flip(t_eye, { 2 }).contiguous();
    std::cout << "flip lr with contiguous: " << std::endl;
    checkTensorMemoryArray(t_eyp_lr_clone[0]);
    auto t_eyp_ud_clone = torch::flip(t_eyp_lr_clone, {1}).contiguous();
    std::cout << "flip ud after lr with contiguous: " << std::endl;
    checkTensorMemoryArray(t_eyp_ud_clone[0]);

    std::cout << std::endl << "-----------------------" << " flip without clone: " << std::endl;
    checkTensorMemoryArray(t_eye[0]);
    t_eye = torch::flip(t_eye, {2});
    std::cout << "flip lr without contiguous: " << std::endl;
    checkTensorMemoryArray(t_eye[0]);
    t_eye = torch::flip(t_eye, {1});
    std::cout << "flip ud after lr without contiguous: " << std::endl;
    checkTensorMemoryArray(t_eye[0]);  
    
    std::cout << std::endl << "-----------------------" << std::endl;
    //auto t_eye1 = torch::arange(0, 256).view({1, 16, 16});
    

    auto t_eye1 = torch::arange(0, 256).view({16, 16});
    std::cout << "new test: " << std::endl;
    checkTensorMemoryArray(t_eye1);      
    t_eye1 = torch::fliplr(t_eye1).contiguous();
    std::cout << "flip lr without contiguous: " << std::endl;
    checkTensorMemoryArray(t_eye1);      
    t_eye1 = torch::flipud(t_eye1).contiguous();
    std::cout << "flip ud without contiguous: " << std::endl;
    checkTensorMemoryArray(t_eye1);    

    auto check3channelTensorMemoryData = [](at::Tensor srt_t){
        cv::Mat col_mat = convert_tensor_to_mat(srt_t, 16, 16, 3, true);
        uchar* data = col_mat.ptr<uchar>(0);    // 获取第一行的指针
        int step = (int)col_mat.step;           // 获取每行的字节数
        int ch = col_mat.channels();
        for (int y = 0; y < col_mat.rows; y++) {
            std::cout << y << ": ";
            for (int x = 0; x < col_mat.cols; x++) {
                int idx = y * step + x * ch;
                int d1 = int(data[idx]);
                std::cout << d1 <<" ";
            }
            std::cout << std::endl;
        }
    };

    std::cout << std::endl << "----------------------- 3channels--------- " << std::endl;
    auto t_eye2 = create3ChannelTensor();
    checkTensorMemoryArray(t_eye2[0]);
    t_eye2 = torch::flip(t_eye2,{2}).contiguous();
    //t_eye2 = torch::fliplr(t_eye2).contiguous();

    std::cout << "flip lr without contiguous: " << std::endl;
    //check3channelTensorMemoryData(t_eye2);
    checkTensorMemoryArray(t_eye2[0]);
    t_eye2 = torch::flip(t_eye2,{1}).contiguous();
    //t_eye2 = torch::flipud(t_eye2).contiguous();    // 对不是2维的翻转错误
    std::cout << "flip ud-lr without contiguous: " << std::endl;
//        check3channelTensorMemoryData(t_eye2);
    checkTensorMemoryArray(t_eye2[0]);    
}
