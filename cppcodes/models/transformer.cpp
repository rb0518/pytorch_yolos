#include <cmath>

#include "transformer.h"
#include "torch_utils.h"

torch::Tensor inverse_sigmoid(torch::Tensor x, float eps = 1e-5)
{
	x = x.clamp(0, 1);
	auto x1 = x.clamp(eps);
	auto x2 = (1 - x).clamp(eps);
	return torch::log(x1 / x2);
}

torch::Tensor multi_scale_deformable_attn_pytorch(
	torch::Tensor value,	// [bs, num_keys, num_head, embed_dims]
	torch::Tensor value_spatial_shapes,	// [num_levels, 2]
	torch::Tensor sampling_locations,	// [bs, num_queries, num_head, num_levels, num_points, 2]
	torch::Tensor attention_weights		// [bs, num_queries, num_head, num_levels, num_points]
)
{
	auto bs = value.size(0);
	auto num_heads = value.size(2);
	auto embed_dims = value.size(3);
	
	auto num_queries = sampling_locations.size(1);
	num_heads = sampling_locations.size(2);
	auto num_levels = sampling_locations.size(3);
	auto num_points = sampling_locations.size(4);

	std::vector<int64_t> split_sizes;
	value_spatial_shapes = value_spatial_shapes.to(torch::kInt64);
	for (int i = 0; i < value_spatial_shapes.size(0); i++)
	{
		split_sizes.push_back(value_spatial_shapes[i][0].item<int64_t>() * value_spatial_shapes[i][1].item<int64_t>());
	}
	auto value_list = value.split(split_sizes, 1);

	auto sampling_grids = 2 * sampling_locations - 1;
	std::vector<torch::Tensor> sampling_value_list;
	auto grid_sample_op = torch::nn::functional::GridSampleFuncOptions().mode(torch::kBilinear).padding_mode(torch::kZeros).align_corners(false);
	for (int level = 0; level < value_spatial_shapes.size(0); level++) {
		auto H_ = value_spatial_shapes[level][0].item<int64_t>();
		auto W_ = value_spatial_shapes[level][1].item<int64_t>();
		auto value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape({bs*num_heads, embed_dims, H_, W_});

		auto sampling_grid_l_ = sampling_grids.index({
			torch::indexing::Slice(),
			torch::indexing::Slice(),
			torch::indexing::Slice(),
			level
			}).transpose(1, 2).flatten(0, 1);

		sampling_grid_l_ = torch::nn::functional::grid_sample(value_l_, sampling_grid_l_, grid_sample_op);

		sampling_value_list.push_back(sampling_grid_l_);
	}
	attention_weights = attention_weights.transpose(1, 2).reshape({
		bs * num_heads, 1, num_queries, num_levels * num_points });

	auto output = (torch::stack(sampling_value_list, -2).flatten(-2) * attention_weights).sum(-1).view({
		bs, num_heads * embed_dims, num_queries });

	return output.transpose(1, 2).contiguous();
}

// ============ MSDeformAttnImpl ============

MLPImpl::MLPImpl(int input_dim, int hidden_dim, int output_dim, int num_layers, bool sigmoid)
{
	m_num_layers = num_layers;
	std::vector<int> in_dim = {input_dim};
	std::vector<int> out_dim;
	for(int i = 0; i < num_layers-1; i++)
	{
		in_dim.emplace_back(hidden_dim);
		out_dim.emplace_back(hidden_dim);
	}
	out_dim.push_back(output_dim);

	m_sigmoid = sigmoid;
	m_act = torch::nn::ReLU();

	for (int i = 0; i < in_dim.size(); i++)
	{
		int n = in_dim[i];
		int k = out_dim[i];
		m_layers->push_back(torch::nn::LinearImpl(n, k));
	}
	register_module("layers", m_layers);
	register_module("act", m_act);
}

torch::Tensor MLPImpl::forward(torch::Tensor x)
{
	for (int i = 0; i < m_num_layers - 1; i++)
	{
		if (auto layer = m_layers[i]->as<torch::nn::LinearImpl>())
		{
			x = m_act->forward(layer->forward(x));
		}
		else
			LOG(ERROR) << "modulelist ==> LinearImpl error.";
	}
	x = (m_layers[m_num_layers - 1]->as<torch::nn::LinearImpl>())->forward(x);

	if (m_sigmoid == false)
		return x.sigmoid();
	return x;
}

void MLPImpl::init_weights()
{
	auto layer = m_layers[m_layers->size() - 1]->as<torch::nn::LinearImpl>();
	if (layer)
	{
		torch::nn::init::constant_(layer->weight, 0.0);
		torch::nn::init::constant_(layer->bias, 0.0);
	}
}

void MLPImpl::init_xavier_uniform(size_t layer_id)
{
	auto layer = m_layers[layer_id]->as<torch::nn::LinearImpl>();
	if (layer)
	{
		torch::nn::init::xavier_uniform_(layer->weight);
	}
}



// ============ MSDeformAttnImpl ============

MSDeformAttnImpl::MSDeformAttnImpl(int d_model, int n_levels, int n_heads, int n_points)
{
	AT_ASSERT((d_model % n_heads) == 0, "d_model must be divisible by n_heads");

	int _d_per_head = d_model / n_heads;

	m_d_model = d_model;
	m_n_levels = n_levels;
	m_n_heads = n_heads;
	m_n_points = n_points;

	m_sampling_offsets = torch::nn::Linear(d_model, n_heads * n_levels * n_points * 2);
	m_attention_weights = torch::nn::Linear(d_model, n_heads * n_levels * n_points);
	m_value_proj = torch::nn::Linear(d_model, d_model);
	m_output_proj = torch::nn::Linear(d_model, d_model);

	register_module("sampling_offsets", m_sampling_offsets);
	register_module("attention_weights", m_attention_weights);
	register_module("value_proj", m_value_proj);
	register_module("output_proj", m_output_proj);


	reset_parameters();
}

torch::Tensor MSDeformAttnImpl::forward(torch::Tensor query, torch::Tensor refer_bbox, torch::Tensor value, 
	std::vector<std::vector<int>> value_shapes, std::optional<torch::Tensor> value_mask)
{
	auto bs = query.size(0);
	auto len_q = query.size(1);
	auto len_v = value.size(1);

	value = m_value_proj->forward(value);

	/*	python codes
	if value_mask is not None:
		value = value.masked_fill(value_mask[..., None], float(0))
	*/
	if (value_mask.has_value())
	{
		auto vm_tmp = value_mask.value();
		vm_tmp = vm_tmp.index({ "...", torch::indexing::None });
		value = value.masked_fill(vm_tmp, 0.0f);
	}

	value = value.view({ bs, len_v, m_n_heads, m_d_model / m_n_heads });
	auto sampling_offsets = m_sampling_offsets->forward(query).view({ bs, len_q, m_n_heads, m_n_levels, m_n_points, 2 });
	auto attention_weights = m_attention_weights->forward(query).view({ bs, len_q, m_n_heads, m_n_levels * m_n_points });
	attention_weights = torch::nn::functional::softmax(attention_weights, torch::nn::functional::SoftmaxFuncOptions(-1)).view(
		{ bs, len_q, m_n_heads, m_n_levels, m_n_points });
	auto num_points = refer_bbox.sizes().back();
	torch::Tensor sampling_locations;

	torch::Tensor offset_normalizer = torch::from_blob(
			value_shapes.data(),  // ����ָ��
			{ static_cast<int64_t>(value_shapes.size()), 2 },  // ��״
			torch::kInt  // ��������
		).to(query.dtype()).to(query.device()).flip({ -1 });
	if (num_points == 2)
	{
		auto add_tmp = sampling_offsets / offset_normalizer.index(
			{torch::indexing::None, torch::indexing::None, torch::indexing::None,
			torch::indexing::Slice(), torch::indexing::None, torch::indexing::Slice()}
		);
		sampling_locations = refer_bbox.index({ torch::indexing::Slice(), torch::indexing::Slice(),
			torch::indexing::None, torch::indexing::Slice(), torch::indexing::None, torch::indexing::Slice() }) + add_tmp;
	}
	else if (num_points == 4)
	{
		auto add_tmp = sampling_offsets / m_n_points * refer_bbox.index({
			torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::None,
			torch::indexing::Slice(), torch::indexing::None, torch::indexing::Slice(2, torch::indexing::None)}) * 0.5;
		sampling_locations = refer_bbox.index({ torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::None,
			torch::indexing::Slice(), torch::indexing::None, torch::indexing::Slice(0, 2) }) + add_tmp;
	}
	else
	{
		throw std::invalid_argument("num_points not equil 2 or 4.");
	}
	auto output = multi_scale_deformable_attn_pytorch(value, offset_normalizer, sampling_locations, attention_weights);
	return torch::Tensor();
}

void MSDeformAttnImpl::reset_parameters()
{
	torch::nn::init::constant_(m_sampling_offsets->weight.data(), 0.0);

	torch::Tensor thetas = torch::arange(m_n_heads, torch::kFloat32)* (2.0 * M_PI / double(m_n_heads));
	auto grid_init = torch::stack({ thetas.cos(), thetas.sin() }, -1);
	torch::Tensor grid_init_max;
	std::tie(grid_init_max, std::ignore) = grid_init.abs().max(-1, true);

	grid_init = (grid_init / grid_init_max).view({ m_n_heads,1,1,2 }).repeat({ 1, m_n_levels, m_n_points, 1 });
	
	for (int i = 0; i < m_n_points; i++)
	{
		grid_init.index_put_({ torch::indexing::Slice(), torch::indexing::Slice(), i, torch::indexing::Slice() },
			grid_init.index({ torch::indexing::Slice(), torch::indexing::Slice(), i, torch::indexing::Slice() }) * (i + 1));
	}

	{
		torch::NoGradGuard nograd;
		m_sampling_offsets->bias.copy_(grid_init.view(-1));
	}

	torch::nn::init::constant_(m_attention_weights->weight.data(), 0.0);
	torch::nn::init::constant_(m_attention_weights->bias.data(), 0.0);
	torch::nn::init::xavier_uniform_(m_value_proj->weight.data());
	torch::nn::init::constant_(m_value_proj->bias.data(), 0.0);
	torch::nn::init::xavier_uniform_(m_output_proj->weight.data());
	torch::nn::init::constant_(m_output_proj->bias.data(), 0.0);

}

// ============ DeformableTransformerDecoderLayer ============

DeformableTransformerDecoderLayerImpl::DeformableTransformerDecoderLayerImpl(int d_model, int n_heads, int d_ffn, float dropout, int n_levels, int n_points)
{
	auto ma_ops = torch::nn::MultiheadAttentionOptions(d_model, n_heads).dropout(dropout);
	m_self_attn = torch::nn::MultiheadAttention(ma_ops);
	m_dropout_1 = torch::nn::Dropout(torch::nn::DropoutOptions().p(dropout));
	m_norm_1 = torch::nn::LayerNorm(torch::nn::LayerNormOptions({ int64_t(d_model) }));
	register_module("self_attn", m_self_attn);
	register_module("dropout1", m_dropout_1);
	register_module("norm1", m_norm_1);

	m_cross_attn = MSDeformAttn(d_model, d_ffn);
	m_dropout_2 = torch::nn::Dropout(torch::nn::DropoutOptions().p(dropout));
	m_norm_2 = torch::nn::LayerNorm(torch::nn::LayerNormOptions({ int64_t(d_model) }));
	register_module("cross_attn", m_cross_attn);
	register_module("dropout2", m_dropout_2);
	register_module("norm2", m_norm_2);

	m_linear_1 = torch::nn::Linear(torch::nn::LinearOptions(d_model, d_ffn));
	m_act = torch::nn::ReLU();
	m_dropout_3 = torch::nn::Dropout(torch::nn::DropoutOptions().p(dropout));
	m_linear_2 = torch::nn::Linear(torch::nn::LinearOptions(d_ffn, d_model));
	m_dropout_4 = torch::nn::Dropout(torch::nn::DropoutOptions().p(dropout));
	m_norm_3 = torch::nn::LayerNorm(torch::nn::LayerNormOptions({ int64_t(d_model) }));
	register_module("linear1", m_linear_1);
	register_module("act", m_act);
	register_module("dropout3", m_dropout_3);
	register_module("linear2", m_linear_2);
	register_module("dropout4", m_dropout_4);
	register_module("norm3", m_norm_3);
}

torch::Tensor DeformableTransformerDecoderLayerImpl::forward(torch::Tensor embed,
	torch::Tensor refer_bbox, 
	torch::Tensor feats, 
	std::vector<std::vector<int>> shapes, 
	std::optional<torch::Tensor> padding_mask, 
	std::optional<torch::Tensor> attn_mask, 
	std::optional<torch::Tensor> query_pos)
{
	auto q = with_pos_embed(embed, query_pos);
	auto k = q;
	auto [tgt, tgt_weights] = m_self_attn->forward(q.transpose(0, 1), k.transpose(0, 1), embed.transpose(0, 1));
	tgt = tgt.transpose(0, 1);
	embed = embed + m_dropout_1->forward(tgt);
	embed = m_norm_1->forward(embed);

	tgt = m_cross_attn->forward(with_pos_embed(embed, query_pos), refer_bbox.unsqueeze(2), feats, shapes, padding_mask);
	embed = embed + m_dropout_2->forward(tgt);
	embed = m_norm_2->forward(tgt);

	return forward_ffn(embed);
}


torch::Tensor DeformableTransformerDecoderLayerImpl::forward_ffn(torch::Tensor tgt)
{
	auto tgt2 = m_linear_2->forward(m_dropout_3->forward(m_act->forward(m_linear_1->forward(tgt))));
	tgt = tgt + m_dropout_4->forward(tgt2);
	return m_norm_3->forward(tgt);
}

torch::Tensor DeformableTransformerDecoderLayerImpl::with_pos_embed(torch::Tensor self_tensor, std::optional<torch::Tensor> pos)
{
	if (pos == std::nullopt)
		return self_tensor;
	return self_tensor + pos.value();
}

//=========================== DeformableTransformerDecoder ==================

DeformableTransformerDecoderImpl::DeformableTransformerDecoderImpl(int hidden_dim, int num_layers, int eval_idx)
{
	m_num_layers = num_layers;
	m_hidden_dim = hidden_dim;
	m_eval_idx = eval_idx >= 0 ? eval_idx : num_layers + eval_idx;
}

std::tuple<torch::Tensor, torch::Tensor> DeformableTransformerDecoderImpl::forward(torch::Tensor embed, torch::Tensor refer_bbox, 
	torch::Tensor feats, std::vector<std::vector<int>> shapes, 
	torch::nn::ModuleList& bbox_head, torch::nn::ModuleList& score_head, MLP& pos_mlp, 
	std::optional<torch::Tensor> attn_mask, std::optional<torch::Tensor> padding_mask)
{
	auto output = embed;
	std::vector<torch::Tensor> dec_bboxes;
	std::vector<torch::Tensor> dec_cls;
	torch::Tensor last_refined_bbox;

	refer_bbox = refer_bbox.sigmoid();

	for (int i = 0; i < m_layers->size(); i++)
	{
		auto layer = m_layers[i]->as<DeformableTransformerDecoderLayer>();
		output = layer->forward(output, refer_bbox, feats, shapes, padding_mask, attn_mask, pos_mlp->forward(refer_bbox));

		torch::Tensor bbox = modulelistForward(bbox_head, output);
		auto refined_bbox = torch::sigmoid(bbox + inverse_sigmoid(refer_bbox));

		if (is_training())
		{
			dec_cls.push_back(score_head[i]->as<torch::nn::AnyModule>()->forward(output));
			if (0 == i)
				dec_bboxes.push_back(refined_bbox);
			else
				dec_bboxes.push_back(torch::sigmoid(bbox + inverse_sigmoid(last_refined_bbox)));
		}
		else if (i == m_eval_idx)
		{
			dec_cls.push_back(modulelistForward(score_head, output));
			dec_bboxes.push_back(refined_bbox);
			break;
		}

		last_refined_bbox = refined_bbox;
		refer_bbox = is_training() ? refined_bbox.detach(): refined_bbox;
	}

	return {torch::stack(dec_bboxes), torch::stack(dec_cls)};
}

void DeformableTransformerDecoderImpl::createLayers(int hd, int nh, int d_ffn, float dropout, int nl, int ndp)
{
	for(int i = 0; i < m_num_layers; i++)
		m_layers->push_back(DeformableTransformerDecoderLayer(hd, nh, dropout, nl, ndp));
	register_module("layers", m_layers);
}


