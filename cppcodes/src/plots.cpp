#include "plots.h"

#include <cmath>
#include <math.h>
#include <filesystem>

Singleton_PlotBatchImages::Singleton_PlotBatchImages(const std::string& dir, const std::string& prefix)
{
    save_dir = dir;
    prefix_name = prefix;
    std::thread plot_imgs_targets_thread(&Singleton_PlotBatchImages::pop_data, this);
    plot_imgs_targets_thread.detach();
}

void Singleton_PlotBatchImages::push_data(torch::Tensor imgs, torch::Tensor targets)
{
    auto img_cpu = imgs.clone();
    img_cpu = img_cpu.to(torch::kCPU);
    auto targ_cpu = targets.clone();
    targ_cpu = targ_cpu.to(torch::kCPU);

    std::unique_lock<std::mutex> lock(mtx_queue);
    cv.wait(lock, [&](){return this->queue_can_push(); });
    data_queue.push({imgs, targets});

    cv.notify_all();
}

void Singleton_PlotBatchImages::pop_data()
{
    torch::Tensor imgs, targets;
    while(false == b_exit_flag)
    {
        if(this->queue_have_data())
        {
            std::unique_lock<std::mutex> lock(mtx_queue);
            cv.wait(lock, [&]() {return this->queue_have_data(); });
            std::tie(imgs, targets) = this->data_queue.front();
            data_queue.pop();
            plot_one_batchs(imgs, targets);
        }
        else
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }
}

void Singleton_PlotBatchImages::plot_one_batchs(torch::Tensor imgs, torch::Tensor targets)
{
    bool normalise = imgs.max().item().toFloat() <= 1.0;
    std::string fname = prefix_name + "_" + std::to_string(incremental_counter) + ".jpg";
    plot_images(imgs, targets, save_dir, fname, normalise);
    incremental_counter++;
}

cv::Mat convert_tensor_to_mat(torch::Tensor img, 
     int img_width, int img_height, int channels, bool ret_color)
{
    if(channels == 3)
    {
        img = img.detach().permute({1, 2, 0});
        img = img.to(torch::kU8);
    }
    else if(channels == 1)
    {
        img = img.detach();
        img = img.to(torch::kU8);
    }
    else
    {
        LOG(ERROR) << "only support [c, h, w] or [h, w] type tensor";
        if(ret_color)
            return cv::Mat::zeros(img_width, img_height, CV_8UC3);
        return cv::Mat::zeros(img_width, img_height, CV_8UC1);
    }


    //std::cout << "img size: " << img.sizes() << std::endl;
    cv::Mat mat;
    std::vector<cv::Mat> mat_mono;
    if(channels == 1) {
        auto m_c = cv::Mat(img_height, img_width, CV_8UC1);
        std::memcpy((void*)m_c.data, img.data_ptr(), sizeof(torch::kByte)*img.numel());

        if(ret_color == false)
            return m_c;
        
        mat_mono.push_back(m_c);
        mat_mono.push_back(m_c);
        mat_mono.push_back(m_c);    // 转换为RGB，绘图需要color格式
    }
    else{
        for(int i = 0; i < channels; i++)
        {
            auto img_c = img.index({"...", channels -1 -i}); // rgb==>bgr
            auto m_c = cv::Mat(img_height, img_width, CV_8UC1);
            std::memcpy((void*)m_c.data, img_c.data_ptr(), sizeof(torch::kU8) * img_c.numel());
            mat_mono.push_back(m_c);
        }
    }
    cv::merge(mat_mono, mat);
    return mat;
}

void plot_labels(cv::Mat& img, torch::Tensor targets, int img_idx, int line_thickness/*=3*/)
{
    int img_width = img.cols;
    int img_height = img.rows;

    int nt = targets.size(0);
    for (int t = 0; t < nt; t++)
    {
        auto mask = (targets.index({ torch::indexing::Slice(), 0 }) == img_idx);
        auto this_img_labels = targets.index({ mask });

        if (this_img_labels.size(0))
        {
            for (int k = 0; k < this_img_labels.size(0); k++)
            {
                auto label = this_img_labels[k];
                auto w = label[4].item().toFloat();
                auto h = label[5].item().toFloat();
                auto x1 = label[2].item().toFloat() - w / 2;
                auto y1 = label[3].item().toFloat() - h / 2;

                auto x2 = x1 + w;
                auto y2 = y1 + h;

                x1 *= img_width;
                y1 *= img_height;
                x2 *= img_width;
                y2 *= img_height;

                int obj_id = label[1].item().toInt();
                cv::Scalar color = SingletonColors::getInstance()->get_color_scalar(obj_id);
                cv::rectangle(img, cv::Point(int(x1), int(y1)), cv::Point(int(x2), int(y2)),
                    color, line_thickness);
                cv::putText(img, std::to_string(obj_id), cv::Point(x1, std::max(0, int(y1) - 2)), 0, 1., 
                    color, line_thickness);
            }
        }
    }
}

void plot_images(torch::Tensor images, torch::Tensor targets, 
    std::string path, std::string fname, 
    bool normalise /*= false*/,
    std::vector<std::string> names /*= {}*/,
    int max_subplots /*= 16*/)
{
    int batch_size = images.size(0);
    int img_width = images.size(3);
    int img_height = images.size(2);
    int channels = images.size(1);

    if (channels != 1 && channels != 3)
    {
        LOG(WARNING) << "only support channels as 1 or 3. program will do nothing.";
        return;
    }

    bool is_mono = channels == 1 ? true : false;
    if(normalise)
        images = images * 255.f;

    cv::Mat mat_mosaic = cv::Mat(img_height * 2, img_width * 2, CV_8UC3);
    for (int i = 0; i < std::min(max_subplots, batch_size); i++)
    {
        auto img = images[i];
        auto mat_one = convert_tensor_to_mat(img, img_width, img_height, channels);

        plot_labels(mat_one, targets, i);

        cv::resize(mat_one, mat_one, cv::Size(img_width / 2, img_height / 2));
        int row_id = i % 4;
        int col_id = i > row_id ? (i - row_id) / 4 : 0;
        auto y1 = row_id * (img_height / 2);
        auto x1 = col_id * (img_width / 2);
        cv::Mat roi = mat_mosaic(cv::Rect(x1, y1, img_width / 2, img_height / 2));
        mat_one.copyTo(roi);
    }

    if(fname !="")
    {
        auto img_save_name = std::filesystem::path(path).append(fname).string();
        cv::imwrite(img_save_name, mat_mosaic);
    }
    else
    {
        cv::imshow("test", mat_mosaic);
        cv::waitKey();
    }
}

void plot_images_pred(torch::Tensor images, std::vector<torch::Tensor> preds,
    std::string path, std::string fname,
    bool normalise /*= false*/,
    std::vector<std::string> names /*= {}*/,
    int max_subplots /*= 16*/)
{
    int batch_size = images.size(0);
    int img_width = images.size(3);
    int img_height = images.size(2);
    int channels = images.size(1);

    if (channels != 1 && channels != 3)
    {
        LOG(WARNING) << "only support channels as 1 or 3. program will do nothing.";
        return;
    }

    bool is_mono = channels == 1 ? true : false;
    if (normalise)
        images = images * 255.f;

    cv::Mat mat_mosaic = cv::Mat(img_height * 2, img_width * 2, CV_8UC3);

    for (int i = 0; i < std::min(max_subplots, batch_size); i++)
    {
        auto img = images[i];
        auto mat_one = convert_tensor_to_mat(img, img_width, img_height, channels);

        plot_pred(mat_one, preds[i], names);

        cv::resize(mat_one, mat_one, cv::Size(img_width / 2, img_height / 2));
        int row_id = i % 4;
        int col_id = i > row_id ? (i - row_id) / 4 : 0;
        auto y1 = row_id * (img_height / 2);
        auto x1 = col_id * (img_width / 2);
        cv::Mat roi = mat_mosaic(cv::Rect(x1, y1, img_width / 2, img_height / 2));
        mat_one.copyTo(roi);
    }

    if (fname != "")
    {
        auto img_save_name = std::filesystem::path(path).append(fname).string();
        cv::imwrite(img_save_name, mat_mosaic);
    }
    else
    {
        cv::imshow("test", mat_mosaic);
        cv::waitKey();
    }
}

void plot_pred(cv::Mat& img, torch::Tensor pred, std::vector<std::string> names, int line_thickness/* = 3*/)
{
    int img_width = img.cols;
    int img_height = img.rows;

    int nt = pred.size(0);
    for (int t = 0; t < pred.size(0); t++)
    {
        auto label = pred[t];
        auto x1 = label[0].item().toFloat();
        auto y1 = label[1].item().toFloat();
        auto x2 = label[2].item().toFloat();
        auto y2 = label[3].item().toFloat();
        auto score = label[4].item().toFloat();
        auto cls_id = label[5].item().toInt();

        auto typecolor = SingletonColors::getInstance()->get_color_scalar(cls_id);

        cv::rectangle(img, cv::Point(int(x1), int(y1)), cv::Point(int(x2), int(y2)), typecolor, 2);

        std::stringstream ss;
        ss << names[cls_id] << " " << std::to_string(score);
        cv::putText(img, ss.str(), cv::Point(x1, std::max(0, int(y1) - 2)), cv::FONT_HERSHEY_PLAIN, 1.,
            typecolor, 2);
    }
}
