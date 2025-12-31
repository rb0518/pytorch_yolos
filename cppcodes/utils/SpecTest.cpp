#include "SpecTest.h"
#include "utils.h"
#include "plots.h"
#include "augmentations.h"
#include <vector>

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include "general.h"
#include "nms.h"

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

void test_model(std::shared_ptr<BaseModelImpl> ptr_model, std::string strfilename /*= "../../data/images/bus.jpg"*/)
{
    auto model = ptr_model;
    float confidence_threshold = 0.2f;
    float iou_threshold = 0.45f;
    int nms_max_bbox_size_ = 4096;  //只保留排前的多少个

    model->eval();

    bool is_segment = false;
    torch::Device device = model->parameters().begin()->device();
    cv::Mat src_image = cv::imread(strfilename);

    std::vector<float> ratio, pad;
    int img_size = 640;
    std::tie(src_image, ratio, pad) = letterbox(src_image, std::make_tuple(img_size, img_size),
			cv::Scalar(114, 114, 114), false, false, false, 32);

    cv::Mat input_image;
    cv::cvtColor(src_image, input_image, cv::COLOR_BGR2RGB);
    
    input_image.convertTo(input_image, CV_32FC3, 1.0 / 255.0);
    at::Tensor input_tensor = torch::from_blob(input_image.data,
        { 1, input_image.rows, input_image.cols, 3 }).to(device);    // 补齐batch_size维
    input_tensor = input_tensor.permute({ 0, 3, 1, 2 }).contiguous();

    auto [preds, output_tensor_vec, protos] = model->forward(input_tensor);// .toTuple()->elements()[0].toTensor();

    std::cout << "model forward return size: " << preds.sizes() << " protos: " << protos.sizes() << std::endl;
    std::cout << "model forward return type: " << preds.dtype() << " protos: " << protos.dtype() << std::endl;
    // [1, 25200, 85]  // new 1, 84, 8400
    int nm = 0;
    if(is_segment)  nm = 32;

    int bs = preds.size(0); // batch_size
    int nc = model->n_classes;
    if(nm != (preds.size(1)- nc - 4))
    {
        LOG(ERROR) << "preds size(1) != nc + 4 + nm ";
        return;
    }

    auto bboxs= non_max_suppression(preds);
    std::vector<cv::Mat> mask_overlays;
    for(int i = 0; i < bboxs.size(); i++)
    {
        auto boxs = bboxs[i];
        std::cout << i << " box size: " << boxs.sizes() << " type: " << boxs.dtype() << std::endl;
        if(boxs.size(0)!=0)
        {
            torch::Tensor masks;            
            if(is_segment)
            {
                auto proto = protos[i];
                proto = proto.to(boxs.device());
                auto mask_in = boxs.index({torch::indexing::Slice(), torch::indexing::Slice(6, 6 + nm)});
                auto bboxes = boxs.index({torch::indexing::Slice(), torch::indexing::Slice(0, 4)});
                std::vector<int64_t> shape = {input_image.rows, input_image.cols};
                masks = process_mask(proto, mask_in, bboxes, shape, true);
                std::cout << "is_segment " << boxs.sizes() << " masks: " << masks.sizes() << " " << masks.dtype() << " " << sizeof(torch::kFloat32) << std::endl;
                cv::split(src_image, mask_overlays);

            }

            // 修改为倒序，置信度高的mask数据后处理
            for(int j = boxs.size(0)-1; j >= 0; j--)
            {
                auto x1 = boxs[j][0].item().toFloat();
                auto y1 = boxs[j][1].item().toFloat();
                auto x2 = boxs[j][2].item().toFloat();
                auto y2 = boxs[j][3].item().toFloat();
                auto score = boxs[j][4].item().toFloat();
                auto cls_id = boxs[j][5].item().toInt();

                auto typecolor = SingletonColors::getInstance()->get_color_scalar(cls_id);

                cv::rectangle(src_image, cv::Point(int(x1), int(y1)), cv::Point(int(x2), int(y2)), typecolor, 2);
                char info_s[250];
                sprintf(info_s, "%3d %3d box: [ %8.2f, %8.2f - %8.2f , %8.2f ] cls: %-3d score: %6.4f", i, j, 
                    x1, y1, x2, y2, cls_id, score);
                std::cout << std::string(info_s) << std::endl;

                if(is_segment)
                {
                    auto mask_b = masks[j].clone().unsqueeze(0);

                    cv::Mat mat_mask = cv::Mat(mask_b.size(1), mask_b.size(2), CV_32FC1);
                    std::memcpy((void*)mat_mask.data, mask_b.data_ptr(), mask_b.element_size()*mask_b.numel());
                    cv::imshow("seg_"+std::to_string(j), mat_mask);

                    auto [c_r, c_g, c_b] = SingletonColors::getInstance()->get_color_uchars(cls_id);
                    float f_mask_thr = 0.5 /*+ std::max(0.0f, (0.5f - score))/10.f*/; // 二值化
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
                        float f_front = float(*mask_b_ptr) * 0.4f;
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

                    
                    // auto m_f = cv::Mat(h, w, CV_8UC1);
                    
                    // std::cout << "h * w: " << (h*w) << " numel: " << mask_b.numel() << " " << sizeof(CV_8UC1) << " " << mask_b.element_size() << std::endl;
                    // std::memcpy((void*)m_f.data, mask_b.data_ptr(), mask_b.element_size() * mask_b.numel());
                    // cv::imshow("seg"+std::to_string(j), m_f);
//                    cv::imwrite("seg" + std::to_string(j) + ".jpg", mat_mask);
                }
            }

            cv::imshow("result", src_image);

            if (is_segment)
            {
                cv::Mat segment_result;
                cv::merge(mask_overlays, segment_result);
                cv::imshow("seg_result", segment_result);
            }

            cv::waitKey();
            cv::destroyAllWindows();
        }
    }
}