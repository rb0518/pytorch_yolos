#pragma once

#include <torch/torch.h>

/*
	"AIFI",									[¡Ì]		// block.h
	"MLP",									[¡Ì]
	"DeformableTransformerDecoder",			[¡Ì]
	"DeformableTransformerDecoderLayer",	[¡Ì]
	"LayerNorm2d",
	"MLPBlock",
	"MSDeformAttn",							[¡Ì]
	"TransformerBlock",
	"TransformerEncoderLayer",
	"TransformerLayer",
*/
// ================ MLP =======================
class MLPImpl : public torch::nn::Module
{
public:
	MLPImpl(int input_dim, int hidden_dim, int output_dim, int num_layers, bool sigmoid = false);
	torch::Tensor forward(torch::Tensor x);
	void init_weights();
	void init_xavier_uniform(size_t layer_id);
private:
	bool m_sigmoid;
	torch::nn::ReLU m_act{ nullptr };
	torch::nn::ModuleList m_layers;
	int m_num_layers;
};
TORCH_MODULE(MLP);


// ================ MSDeformAttn =======================

class MSDeformAttnImpl : public torch::nn::Module
{
public:
	MSDeformAttnImpl(int d_model = 256, int n_levels = 4,
		int n_heads = 8, int n_points = 4);

	torch::Tensor forward(torch::Tensor query,
		torch::Tensor refer_bbox,
		torch::Tensor value,
		std::vector<std::vector<int>> value_shapes,
		std::optional<torch::Tensor> value_mask = std::nullopt
	);
private:
	void reset_parameters();
private:
	int m_im2col_step = 64;
	int m_d_model = 256;
	int m_n_levels = 4;
	int m_n_heads = 8;
	int m_n_points = 4;

	torch::nn::Linear m_sampling_offsets{ nullptr };
	torch::nn::Linear m_attention_weights{ nullptr };
	torch::nn::Linear m_value_proj{ nullptr };
	torch::nn::Linear m_output_proj{ nullptr };


};
TORCH_MODULE(MSDeformAttn);


// ================ DeformableTransformerDecoderLayer ============

class DeformableTransformerDecoderLayerImpl : public torch::nn::Module
{
public:
	DeformableTransformerDecoderLayerImpl(int d_model = 256,
		int n_heads = 8,
		int d_ffn = 1024,
		float dropout = 0.0f,
		int n_levels = 4,
		int n_points = 4);

	torch::Tensor forward(torch::Tensor embed, 
		torch::Tensor refer_bbox,
		torch::Tensor feats,
		std::vector<std::vector<int>> shapes,
		std::optional<torch::Tensor> padding_mask = std::nullopt,
		std::optional<torch::Tensor> attn_mask = std::nullopt,
		std::optional<torch::Tensor> query_pos = std::nullopt
	);
private:
	torch::Tensor forward_ffn(torch::Tensor tgt);
	torch::Tensor with_pos_embed(torch::Tensor self_tensor, std::optional<torch::Tensor> pos = std::nullopt);

protected:
	// self attention
	torch::nn::MultiheadAttention m_self_attn{ nullptr };
	torch::nn::Dropout m_dropout_1{ nullptr };
	torch::nn::LayerNorm m_norm_1{ nullptr };

	// cross attention
	MSDeformAttn m_cross_attn{ nullptr };
	torch::nn::Dropout m_dropout_2{ nullptr };
	torch::nn::LayerNorm m_norm_2{ nullptr };

	// FFN
	torch::nn::Linear m_linear_1{ nullptr };
	torch::nn::ReLU m_act{ nullptr };
	torch::nn::Dropout m_dropout_3{ nullptr };
	torch::nn::Linear m_linear_2{ nullptr };
	torch::nn::Dropout m_dropout_4{ nullptr };
	torch::nn::LayerNorm m_norm_3{ nullptr };
};
TORCH_MODULE(DeformableTransformerDecoderLayer);

//=================== DeformableTransformerDecoder ====================
class DeformableTransformerDecoderImpl : public torch::nn::Module
{
public:
	DeformableTransformerDecoderImpl(int hidden_dim, int num_layers, int eval_idx = -1);
	
	std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor embed, 
		torch::Tensor refer_bbox,
		torch::Tensor feats,
		std::vector<std::vector<int>> shapes,
		torch::nn::ModuleList& bbox_head,
		torch::nn::ModuleList& score_head,
		MLP& pos_mlp,
		std::optional<torch::Tensor> attn_mask = std::nullopt,
		std::optional<torch::Tensor> padding_mask = std::nullopt
	);
	
	void createLayers(int hd, int nh, int d_ffn, float dropout, int nl, int ndp);

private:
	torch::nn::ModuleList m_layers;
	int m_num_layers;
	int m_hidden_dim;
	int m_eval_idx;

};
TORCH_MODULE(DeformableTransformerDecoder);