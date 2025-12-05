#pragma once

#include <tuple>
#include <vector>
#include <variant>
#include <type_traits>
#include <utility>
#include <stdexcept>

#include "BaseModel.h"

template<typename Tuple, size_t N>
struct vector_2_tuple
{
	static void apply(Tuple& t, std::vector<arg_complex> args)
	{
		if ((N - 1) < args.size())
		{
			using TargetType = typename std::tuple_element<N - 1, Tuple>::type;
			std::get<N - 1>(t) = std::get<TargetType>(args[N - 1]);
		}
		vector_2_tuple<Tuple, N - 1>::apply(t, args);
	}
};

template<typename Tuple>
struct vector_2_tuple<Tuple, 0>
{
	static void apply(Tuple& t, std::vector<arg_complex> args) {}
};

template<typename Tuple>
void make_param_from_args(Tuple& defaults, std::vector<arg_complex> args)
{
	constexpr auto tuple_size = std::tuple_size<Tuple>::value;
	if (tuple_size < args.size())
		LOG(ERROR) << "make_param_from_args: tuple size error: " << tuple_size << " args.size: " << args.size();
	
	vector_2_tuple<Tuple, tuple_size>::apply(defaults, args);
	return;
}