#pragma once
#include <torch/torch.h>
#include <variant>
#include <tuple>
#include <vector>
#include "BaseData.h"

struct DetectCustomCollate : torch::data::transforms::Collation<VariantData>
{
    VariantData apply_batch(std::vector<VariantData> batch) override
    {
        std::vector<torch::Tensor> v_imgs;
        std::vector<torch::Tensor> v_cls;
        std::vector<torch::Tensor> v_bboxes;
        std::vector<torch::Tensor> v_batch_idx;
        std::vector<std::tuple<float, float>> v_ori_shapes;
        std::vector<std::tuple<float, float>> v_resized_shapes;
        std::vector<std::string> v_filenames;
        std::vector<torch::Tensor> v_masks;
		for (int i_idx = 0; i_idx < batch.size(); i_idx++)
		{
			auto item = batch[i_idx];
			v_imgs.push_back(std::get<torch::Tensor>(item[GET_VDK(VDK::IMG)]));
			v_cls.push_back(std::get<torch::Tensor>(item[GET_VDK(VDK::CLS)]));
			v_bboxes.push_back(std::get<torch::Tensor>(item[GET_VDK(VDK::BBOXES)]));
			torch::Tensor t_idx = std::get<torch::Tensor>(item[GET_VDK(VDK::BATCH_IDX)]);
			t_idx.fill_(i_idx);
			v_batch_idx.push_back(t_idx);
            std::vector<std::tuple<float, float>> src_size = std::get<std::vector<std::tuple<float, float>>>(item[GET_VDK(VDK::SRC_SIZE)]);
			v_ori_shapes.push_back(src_size[0]);
            std::tuple<float, float> resized = std::get<std::vector<std::tuple<float, float>>>(item[GET_VDK(VDK::RESIZED)])[0];
			v_resized_shapes.push_back(resized);
            std::vector<std::string> filename = std::get<std::vector<std::string>>(item[GET_VDK(VDK::FILENAME)]);
            v_filenames.push_back(filename[0]);
		}


        VariantData ret_example;
		ret_example[GET_VDK(VDK::SRC_SIZE)] = v_ori_shapes;
        ret_example[GET_VDK(VDK::RESIZED)] = v_resized_shapes;
        ret_example[GET_VDK(VDK::FILENAME)]= v_filenames;

		ret_example[GET_VDK(VDK::IMG)] = torch::stack(v_imgs, 0);		// n:[3, 640, 640] ==> [n, 3, 640, 640];
		ret_example[GET_VDK(VDK::CLS)] = torch::cat(v_cls, 0);		// [[n1], [n2]...[nx] => [n1+n2+...+nx];
        ret_example[GET_VDK(VDK::BBOXES)] = torch::cat(v_bboxes, 0);
		ret_example[GET_VDK(VDK::BATCH_IDX)] = torch::cat(v_batch_idx, 0);

		return ret_example;
    }
};

class DetectTransformsArray : public BaseTransformArray 
{
public:
    DetectTransformsArray(VariantConfigs* args, int imgsz, int dataset_size, bool augment)
        : BaseTransformArray(args, imgsz, dataset_size, augment) {;}
    virtual void buildTranforms(bool bAugment) override;
};

class DetectDataset : public torch::data::Dataset<DetectDataset, VariantData>, public BaseDataset
{
public:
    DetectDataset(const std::string& path,
        VariantConfigs* pargs,
        int stride);

public:
    torch::optional<size_t> size() const override {
        return m_imgFiles.size();
    }
    VariantData get(size_t index);
    void initFinal(){getSpecialSettins();}

    virtual void getSpecialSettins() override;
    virtual VariantData getOneSample(size_t index) override;

protected:
    std::shared_ptr<DetectTransformsArray> m_pTransforms_Augment;
    std::shared_ptr<DetectTransformsArray> m_pTransforms_Normal;       
};

class DetectDataload : public BaseDataloader
{
public:
    explicit DetectDataload(const std::string& root_path,
        VariantConfigs& args_input,
        int stride = 32,
        bool is_val = false
    );

public:
    virtual void createDataloader() override;
    
    torch::data::Iterator<VariantData> getBegin() {
        return m_dataLoader->begin();
    }
    torch::data::Iterator<VariantData> getEnd() {
        return m_dataLoader->end();
    }

public:
    std::unique_ptr<torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<DetectDataset, DetectCustomCollate>,
        torch::data::samplers::RandomSampler>> m_dataLoader = nullptr;
};
