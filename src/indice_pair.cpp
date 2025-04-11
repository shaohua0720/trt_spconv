#include "indice_pair.h"
#include <torch/torch.h>
#include <cstring>
#include <stdexcept>

namespace trt_spconv {

IndicePair::IndicePair(torch::Tensor indices, int64_t num_out)
    : num_output(num_out) {
    
    // 检查输入
    if (indices.dim() != 2) {
        throw std::runtime_error("Indices must be 2D tensor");
    }
    if (indices.size(1) != 2) {
        throw std::runtime_error("Indices must have 2 columns (input, output)");
    }
    
    // 分离输入和输出索引
    input_indices = indices.select(1, 0).contiguous();
    output_indices = indices.select(1, 1).contiguous();
    
    // 计算数量
    num_input = input_indices.max().item<int64_t>() + 1;
    num_pairs = indices.size(0);
}

std::vector<char> IndicePair::serialize() const {
    // 计算序列化数据大小
    size_t total_size = sizeof(int64_t) * 3 +  // num_input, num_output, num_pairs
                       input_indices.numel() * sizeof(int64_t) +
                       output_indices.numel() * sizeof(int64_t);
    
    std::vector<char> data(total_size);
    char* ptr = data.data();
    
    // 序列化元数据
    std::memcpy(ptr, &num_input, sizeof(int64_t));
    ptr += sizeof(int64_t);
    std::memcpy(ptr, &num_output, sizeof(int64_t));
    ptr += sizeof(int64_t);
    std::memcpy(ptr, &num_pairs, sizeof(int64_t));
    ptr += sizeof(int64_t);
    
    // 序列化输入索引
    auto input_ptr = input_indices.data_ptr<int64_t>();
    std::memcpy(ptr, input_ptr, input_indices.numel() * sizeof(int64_t));
    ptr += input_indices.numel() * sizeof(int64_t);
    
    // 序列化输出索引
    auto output_ptr = output_indices.data_ptr<int64_t>();
    std::memcpy(ptr, output_ptr, output_indices.numel() * sizeof(int64_t));
    
    return data;
}

std::shared_ptr<IndicePair> IndicePair::deserialize(const std::vector<char>& data) {
    const char* ptr = data.data();
    
    // 反序列化元数据
    int64_t num_input, num_output, num_pairs;
    std::memcpy(&num_input, ptr, sizeof(int64_t));
    ptr += sizeof(int64_t);
    std::memcpy(&num_output, ptr, sizeof(int64_t));
    ptr += sizeof(int64_t);
    std::memcpy(&num_pairs, ptr, sizeof(int64_t));
    ptr += sizeof(int64_t);
    
    // 创建索引张量
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    auto indices = torch::zeros({num_pairs, 2}, options);
    
    // 反序列化输入索引
    auto input_ptr = indices.select(1, 0).data_ptr<int64_t>();
    std::memcpy(input_ptr, ptr, num_pairs * sizeof(int64_t));
    ptr += num_pairs * sizeof(int64_t);
    
    // 反序列化输出索引
    auto output_ptr = indices.select(1, 1).data_ptr<int64_t>();
    std::memcpy(output_ptr, ptr, num_pairs * sizeof(int64_t));
    
    // 创建IndicePair对象
    return std::make_shared<IndicePair>(indices, num_output);
}

} // namespace trt_spconv 