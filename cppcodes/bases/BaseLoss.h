#pragma once

#include <torch/torch.h>
#include "BaseData.h"
 
class BaseLoss 
{
public:
	BaseLoss() = default;
	void addNewItems(torch::Tensor loss, torch::Tensor lossitems);
	void reset();

	virtual void createLossRecord() = 0;
	virtual std::string getLossitemsNames() = 0;
	virtual std::string getLossitemsInfo(int epoch, int epochs, int idx, int num_targets) = 0;

	virtual std::tuple<torch::Tensor, torch::Tensor> forward(std::vector<torch::Tensor>& preds,
		VariantData& batch) = 0;
protected:
	int m_times;
	torch::Tensor m_loss;
	torch::Tensor m_lossitems;
};